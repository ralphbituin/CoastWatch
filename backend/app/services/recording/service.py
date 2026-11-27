"""Single recording service for timestamped shoreline observations (Phase 7).

Every shoreline measurement — live, archive, operator mask or lab run — is
written through :func:`write_observation`. The function enforces:

* one row per accepted ``(camera_id, observed_at)``;
* ``(recording_id, frame_index)`` idempotency (re-processing updates in place);
* strict per-camera/session timestamp ordering (out-of-order rows are rejected,
  never silently reordered);
* exact duplicate timestamps are dropped and reported, so the rolling detector
  is never fed the same sample twice.

It does not fabricate timestamps or positions. A missing measurement is a
``DATA_UNRELIABLE`` row/reason, never a value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.anomaly import AnomalyEvent
from app.models.camera import Camera
from app.models.shoreline import ShorelineObservation
from app.services.anomaly.rolling_baseline import (
    BASELINE_STATUS_ROLLING,
    DETECTOR_MODE_ROLLING,
    RollingObservation,
    RollingRowResult,
)
from app.services.vision.position import measure_shoreline_position

logger = get_logger(__name__)

STATUS_PERSISTED = "persisted"
STATUS_UPDATED = "updated"
STATUS_DUPLICATE = "duplicate"
STATUS_OUT_OF_ORDER = "out_of_order"
STATUS_REJECTED = "rejected"

# Documented provenance vocabulary. Keep this list in sync with docs/PHASE7.
TIMESTAMP_SOURCES = {
    "derived",  # archive: verified visible-clock anchors
    "visible_clock",  # archive: a clock read on this exact frame
    "container",  # archive: container creation time actually present
    "operator_declared",  # batch: operator start time + interval
    "manifest",  # batch: per-file timestamps
    "exif",  # batch: trustworthy EXIF timestamp
    "stream_wall_clock",  # live: wall clock at frame receipt (<= 0.6)
    "supplied",  # operator-labelled still/mask supplied observed_at
    "unavailable",  # no absolute time can be established
}


@dataclass(slots=True)
class ObservationWrite:
    status: str
    observation: ShorelineObservation | None
    reason: str | None = None
    observed_at: datetime | None = None


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("observed_at must be timezone-aware (UTC)")
    return value


def write_observation(
    session: Session,
    camera: Camera,
    *,
    observed_at: datetime,
    result: Any | None = None,
    calibration_id: int | None = None,
    quality_score: float | None = None,
    quality_flags: list[str] | None = None,
    source_frame_path: str | None = None,
    recording_id: str | None = None,
    frame_index: int | None = None,
    elapsed_seconds: float | None = None,
    stream_session_id: str | None = None,
    ingest_latency_ms: float | None = None,
    timestamp_source: str | None = None,
    timestamp_confidence: float | None = None,
    extraction_status: str | None = None,
    extraction_reason: str | None = None,
    shoreline_geometry: list[Any] | None = None,
    shoreline_segments: list[Any] | None = None,
    shoreline_axis: str | None = None,
    interpolated_ranges: list[Any] | None = None,
    measured_coverage: float | None = None,
    segmentation_confidence: float | None = None,
    shoreline_position_px: float | None = None,
    signed_position_px: float | None = None,
    position_axis: str | None = None,
    position_sign: str | None = None,
    segment_count: int | None = None,
    segment_measurements: list[Any] | None = None,
    class_fractions: dict[str, Any] | None = None,
    inference_time_ms: float | None = None,
    horizon_y_px: float | None = None,
    frame_shape: tuple[int, int] | None = None,
    primary_region_fraction: float | None = 0.5,
    previous_signed_position: float | None = None,
) -> ObservationWrite:
    """Write exactly one canonical observation row.

    ``result`` may be a ``ShorelineObservationResult`` from the production
    extractor. When position fields are not supplied, the service derives them
    with :func:`measure_shoreline_position` — it never invents a value.
    """
    observed_at = _as_utc(observed_at)

    if result is not None:
        shoreline_geometry = shoreline_geometry or list(getattr(result, "points", []) or [])
        shoreline_segments = shoreline_segments or list(getattr(result, "segments", []) or [])
        shoreline_axis = shoreline_axis or getattr(result, "shoreline_axis", None)
        interpolated_ranges = interpolated_ranges or list(
            getattr(result, "interpolated_ranges", []) or []
        )
        measured_coverage = (
            measured_coverage
            if measured_coverage is not None
            else getattr(result, "measured_coverage", None)
        )
        segmentation_confidence = (
            segmentation_confidence
            if segmentation_confidence is not None
            else getattr(result, "segmentation_confidence", None)
        )
        if signed_position_px is None and getattr(result, "shoreline_available", False):
            position = measure_shoreline_position(
                getattr(result, "segments", []) or [],
                getattr(result, "orientation_used", None),
                previous_signed_position=previous_signed_position,
                frame_shape=frame_shape,
                primary_region_fraction=primary_region_fraction,
            )
            if position.available:
                shoreline_position_px = position.position_px
                signed_position_px = position.signed_position_px
                position_axis = position.position_axis
                position_sign = position.position_sign
                segment_count = position.segment_count
                segment_measurements = position.segment_dicts()
                extraction_status = extraction_status or "available"
            else:
                extraction_status = extraction_status or "unavailable"
                extraction_reason = extraction_reason or position.reason
        if extraction_status is None:
            extraction_status = (
                "available" if getattr(result, "shoreline_available", False) else "unavailable"
            )
        if extraction_reason is None:
            extraction_reason = getattr(result, "reason", None)

    # 1. Idempotency: (recording_id, frame_index) updates the existing row.
    if recording_id is not None and frame_index is not None:
        existing_by_frame = session.scalar(
            select(ShorelineObservation).where(
                ShorelineObservation.recording_id == recording_id,
                ShorelineObservation.frame_index == frame_index,
            )
        )
        if existing_by_frame is not None:
            target = existing_by_frame
            values = {
                "observed_at": observed_at,
                "monitoring_station_id": camera.monitoring_station_id,
                "calibration_id": calibration_id,
                "shoreline_geometry": shoreline_geometry or None,
                "shoreline_segments": shoreline_segments or None,
                "shoreline_axis": shoreline_axis,
                "interpolated_ranges": interpolated_ranges or None,
                "measured_coverage": measured_coverage,
                "segmentation_confidence": segmentation_confidence,
                "frame_quality_score": quality_score,
                "frame_quality_flags": list(quality_flags or []),
                "source_frame_path": source_frame_path,
                "extraction_status": extraction_status,
                "extraction_reason": extraction_reason,
                "shoreline_position_px": shoreline_position_px,
                "signed_position_px": signed_position_px,
                "position_axis": position_axis,
                "position_sign": position_sign,
                "segment_count": segment_count or 0,
                "segment_measurements": segment_measurements or None,
                "class_fractions": class_fractions or None,
                "inference_time_ms": inference_time_ms,
                "capture_timestamp_source": timestamp_source,
                "capture_timestamp_confidence": timestamp_confidence,
                "horizon_y_px": horizon_y_px,
                "elapsed_seconds": elapsed_seconds,
                "stream_session_id": stream_session_id,
                "ingest_latency_ms": ingest_latency_ms,
            }
            for attribute, value in values.items():
                setattr(target, attribute, value)
            return ObservationWrite(
                status=STATUS_UPDATED,
                observation=target,
                reason="idempotent_reprocess",
                observed_at=observed_at,
            )

    # 2. Exact duplicate timestamp: drop it, never double-feed the detector.
    existing_at_timestamp = session.scalar(
        select(ShorelineObservation).where(
            ShorelineObservation.camera_id == camera.id,
            ShorelineObservation.observed_at == observed_at,
        )
    )
    if existing_at_timestamp is not None:
        return ObservationWrite(
            status=STATUS_DUPLICATE,
            observation=existing_at_timestamp,
            reason="duplicate_timestamp",
            observed_at=observed_at,
        )

    # 3. Strict ordering per camera/session. Archive rows are sorted upstream;
    #    a genuinely older row is quarantined with a reason, not reordered.
    latest_stmt = select(func.max(ShorelineObservation.observed_at)).where(
        ShorelineObservation.camera_id == camera.id
    )
    if stream_session_id is not None:
        latest_stmt = latest_stmt.where(ShorelineObservation.stream_session_id == stream_session_id)
    latest = session.scalar(latest_stmt)
    if latest is not None and observed_at <= latest:
        return ObservationWrite(
            status=STATUS_OUT_OF_ORDER,
            observation=None,
            reason=f"observed_at {observed_at.isoformat()} is not after {latest.isoformat()}",
            observed_at=observed_at,
        )

    observation = ShorelineObservation(
        camera_id=camera.id,
        observed_at=observed_at,
        monitoring_station_id=camera.monitoring_station_id,
        calibration_id=calibration_id,
        shoreline_geometry=shoreline_geometry or None,
        shoreline_segments=shoreline_segments or None,
        shoreline_axis=shoreline_axis,
        interpolated_ranges=interpolated_ranges or None,
        measured_coverage=measured_coverage,
        calibrated_position_m=getattr(result, "calibrated_position_m", None) if result else None,
        segmentation_confidence=segmentation_confidence,
        frame_quality_score=quality_score,
        frame_quality_flags=list(quality_flags or []),
        source_frame_path=source_frame_path,
        recording_id=recording_id,
        frame_index=frame_index,
        extraction_status=extraction_status,
        extraction_reason=extraction_reason,
        shoreline_position_px=shoreline_position_px,
        signed_position_px=signed_position_px,
        position_axis=position_axis,
        position_sign=position_sign,
        segment_count=segment_count or 0,
        segment_measurements=segment_measurements or None,
        class_fractions=class_fractions or None,
        inference_time_ms=inference_time_ms,
        capture_timestamp_source=timestamp_source,
        capture_timestamp_confidence=timestamp_confidence,
        horizon_y_px=horizon_y_px,
        elapsed_seconds=elapsed_seconds,
        stream_session_id=stream_session_id,
        ingest_latency_ms=ingest_latency_ms,
    )
    session.add(observation)
    return ObservationWrite(
        status=STATUS_PERSISTED,
        observation=observation,
        observed_at=observed_at,
    )


def observation_to_rolling(
    observation: ShorelineObservation,
    *,
    shoreline_available: bool | None = None,
) -> RollingObservation:
    available = (
        shoreline_available
        if shoreline_available is not None
        else (observation.extraction_status == "available")
        and observation.signed_position_px is not None
    )
    return RollingObservation(
        timestamp=observation.observed_at,
        position_px=observation.signed_position_px,
        shoreline_available=available,
        confidence=observation.segmentation_confidence,
        frame_index=observation.frame_index,
        timestamp_confidence=observation.capture_timestamp_confidence,
        reason=observation.extraction_reason,
        segment_measurements=observation.segment_measurements,
        horizon_y_px=observation.horizon_y_px,
    )


def rolling_row_factors(row: RollingRowResult) -> dict[str, Any]:
    """Persisted evidence for an ``AnomalyEvent`` row.

    The dashboard reads these keys from ``contributing_factors``; they are the
    same fields the archive path already writes.
    """
    factors = dict(row.contributing_factors or {})
    factors.update(
        {
            "classification": row.classification,
            "event_max_classification": row.event_max_classification,
            "detector_mode": DETECTOR_MODE_ROLLING,
            "baseline_status": BASELINE_STATUS_ROLLING,
            "tide_adjusted": False,
            "tide_data_available": False,
            "tide_source": "unavailable",
            "units": "pixels; positive shoreline position is seaward",
            "rolling_baseline_px": row.rolling_baseline_px,
            "rolling_scale_px": row.rolling_scale_px,
            "scale_floored": row.scale_floored,
            "cumulative_drawdown_px": row.cumulative_drawdown_px,
            "baseline_frozen": row.baseline_frozen,
            "camera_motion": row.camera_motion_gate,
            "config_hash": row.config_hash,
            "onset_time": None if row.onset_time is None else row.onset_time.isoformat(),
            "detection_time": (
                None if row.detection_time is None else row.detection_time.isoformat()
            ),
            "detection_latency_s": row.detection_latency_s,
            "event_window_seconds": row.event_window_seconds,
            "cooldown_state": row.cooldown_state,
        }
    )
    return factors


def persist_rolling_row(
    session: Session,
    camera: Camera,
    row: RollingRowResult,
    *,
    recording_id: str,
    frame_index: int | None = None,
) -> AnomalyEvent | None:
    """Upsert one rolling output row as an anomaly/quality row.

    Idempotent on ``(recording_id, frame_index)``. No event is invented when
    ``row.timestamp`` is missing.
    """
    if row.timestamp is None:
        return None
    resolved_frame = frame_index if frame_index is not None else row.frame_index
    existing = None
    if resolved_frame is not None:
        existing = session.scalar(
            select(AnomalyEvent).where(
                AnomalyEvent.recording_id == recording_id,
                AnomalyEvent.frame_index == resolved_frame,
            )
        )
    factors = rolling_row_factors(row)
    detection_row = resolved_frame is not None and resolved_frame == row.detection_frame_index
    payload = {
        "detected_at": row.timestamp,
        "monitoring_station_id": camera.monitoring_station_id,
        "camera_id": camera.id,
        "state": row.state,
        "anomaly_score": (
            None
            if not isinstance(factors.get("anomaly_score"), int | float)
            else float(factors["anomaly_score"])
        ),
        "persistence": float(row.persistence_count),
        "explanation": row.reason,
        "contributing_factors": factors,
        "acknowledged": False,
        "recording_id": recording_id,
        "frame_index": resolved_frame,
        "onset_frame_index": row.onset_frame_index if detection_row else None,
        "detector_mode": DETECTOR_MODE_ROLLING,
        "classification": row.classification,
        "baseline_status": BASELINE_STATUS_ROLLING,
        "tide_adjusted": False,
        "tide_data_available": False,
        "tide_source": "unavailable",
        "robust_z": row.robust_z,
        "observed_position_px": row.observed_position_px,
        "expected_position_px": row.rolling_baseline_px,
        "residual_px": row.residual_px,
        "movement_rate_px_s": row.movement_rate_px_s,
        "persistence_count": row.persistence_count,
        "quality_status": row.quality_status,
        "reason": row.reason,
    }
    if existing is None:
        event = AnomalyEvent(**payload)
        session.add(event)
        return event
    for attribute, value in payload.items():
        setattr(existing, attribute, value)
    return existing


def persist_rolling_rows(
    session: Session,
    camera: Camera,
    rows: list[RollingRowResult],
    *,
    recording_id: str,
    frame_indices: dict[datetime, int] | None = None,
) -> int:
    """Persist a batch of rolling rows; returns the written count."""
    written = 0
    for row in rows:
        frame_index = row.frame_index
        if frame_indices is not None and row.timestamp is not None:
            frame_index = frame_indices.get(row.timestamp, frame_index)
        if (
            persist_rolling_row(
                session,
                camera,
                row,
                recording_id=recording_id,
                frame_index=frame_index,
            )
            is not None
        ):
            written += 1
    return written


def load_rolling_history(
    session: Session,
    camera_id: int,
    *,
    window_seconds: float,
    before: datetime | None = None,
    stream_session_id: str | None = None,
    max_rows: int = 5000,
) -> list[ShorelineObservation]:
    """Load the last ``window_seconds`` of stored rows for a warm start.

    Rows are returned oldest-first. Missing/invalid positions stay in the list:
    the detector's own gates decide whether they are usable.
    """
    from datetime import timedelta

    reference = before
    if reference is None:
        reference = session.scalar(
            select(func.max(ShorelineObservation.observed_at)).where(
                ShorelineObservation.camera_id == camera_id
            )
        )
    if reference is None:
        return []
    cutoff = reference - timedelta(seconds=float(window_seconds))
    stmt = (
        select(ShorelineObservation)
        .where(
            ShorelineObservation.camera_id == camera_id,
            ShorelineObservation.observed_at >= cutoff,
            ShorelineObservation.observed_at < reference,
        )
        .order_by(ShorelineObservation.observed_at.desc())
        .limit(max_rows)
    )
    if stream_session_id is not None:
        stmt = stmt.where(ShorelineObservation.stream_session_id == stream_session_id)
    rows = list(session.scalars(stmt).all())
    rows.reverse()
    if before is None:
        latest = session.scalar(
            select(ShorelineObservation)
            .where(ShorelineObservation.camera_id == camera_id)
            .order_by(ShorelineObservation.observed_at.desc())
            .limit(1)
        )
        if latest is not None and (not rows or rows[-1].observed_at < latest.observed_at):
            rows.append(latest)
    return rows


@dataclass(slots=True)
class WriteSummary:
    persisted: int = 0
    updated: int = 0
    duplicates: int = 0
    out_of_order: int = 0
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)

    def add(self, outcome: ObservationWrite) -> None:
        if outcome.status == STATUS_PERSISTED:
            self.persisted += 1
        elif outcome.status == STATUS_UPDATED:
            self.updated += 1
        elif outcome.status == STATUS_DUPLICATE:
            self.duplicates += 1
        elif outcome.status == STATUS_OUT_OF_ORDER:
            self.out_of_order += 1
            if outcome.reason:
                self.reasons.append(outcome.reason)
        else:
            self.rejected += 1
            if outcome.reason:
                self.reasons.append(outcome.reason)
