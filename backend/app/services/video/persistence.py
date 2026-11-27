"""Persist Phase 5/6 prototype analysis through the existing ORM models.

Idempotency key: ``recording_id + frame_index`` for observations and anomaly
events. Reprocessing a recording updates those rows instead of duplicating
them. Baselines are append-only: a new row is inserted and the previous active
row for the camera is deactivated.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import delete, or_, select, update
from sqlalchemy.orm import Session

from app.core.logging import get_logger
from app.models.anomaly import AnomalyEvent
from app.models.camera import Camera
from app.models.shoreline import ShorelineBaseline
from app.models.video import VideoRecording
from app.services.recording.service import write_observation
from app.services.video.pipeline import AnalysisResult

logger = get_logger(__name__)


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def upsert_recording(
    session: Session,
    result: AnalysisResult,
    *,
    camera_id: int | None,
    monitoring_station_id: int | None,
    file_sha256: str | None = None,
    anchors: list[dict[str, Any]] | None = None,
    timezone: str | None = None,
    timestamp_source: str | None = None,
    processing_status: str = "processed",
) -> VideoRecording:
    recording = session.scalar(
        select(VideoRecording).where(VideoRecording.recording_id == result.recording_id)
    )
    if recording is None:
        recording = VideoRecording(recording_id=result.recording_id)
        session.add(recording)
    metadata = result.metadata
    recording.camera_id = camera_id
    recording.monitoring_station_id = monitoring_station_id
    recording.file_path = metadata.path
    recording.file_size_bytes = metadata.file_size_bytes
    recording.file_sha256 = file_sha256
    recording.duration_seconds = metadata.duration_seconds
    recording.fps = metadata.fps
    recording.frame_count = metadata.frame_count
    recording.width = metadata.width
    recording.height = metadata.height
    recording.video_codec = metadata.video_codec
    recording.audio_codec = metadata.audio_codec
    recording.timezone = timezone
    recording.timestamp_source = timestamp_source or ("derived" if anchors else "unavailable")
    recording.timestamp_confidence = next(
        (
            row.get("timestamp_confidence")
            for row in result.observations
            if row.get("timestamp_confidence") is not None
        ),
        None,
    )
    recording.clock_anchors = anchors
    recording.processing_status = processing_status
    recording.processed_frames = result.stats.processed
    recording.sampled_frames = result.stats.sampled
    recording.metadata_json = {
        "container": metadata.as_dict(),
        "stats": result.stats.as_dict(),
        "thresholds": result.threshold_diagnostics,
        "baseline": result.baseline.as_dict(),
        "limitations": result.limitations,
        "detector_mode": result.detector_mode,
        "resolved_detector_mode": result.resolved_detector_mode,
        "mode_reason": result.mode_reason,
        "rolling_baseline": (
            None
            if result.rolling_evaluation is None
            else {
                "config": (
                    None if result.rolling_config is None else result.rolling_config.as_dict()
                ),
                "config_hash": result.rolling_evaluation.config_hash,
                "dropped_duplicate_count": result.rolling_evaluation.dropped_duplicate_count,
                "gate_counts": result.rolling_evaluation.gate_counts,
                "events": [event.as_dict() for event in result.rolling_evaluation.events],
            }
        ),
    }
    return recording


def persist_analysis(
    session: Session,
    result: AnalysisResult,
    *,
    camera: Camera,
    anchors: list[dict[str, Any]] | None = None,
    timezone: str | None = None,
    timestamp_source: str | None = None,
    file_sha256: str | None = None,
) -> dict[str, Any]:
    """Persist recording, observations, baseline and events in one transaction."""
    upsert_recording(
        session,
        result,
        camera_id=camera.id,
        monitoring_station_id=camera.monitoring_station_id,
        file_sha256=file_sha256,
        anchors=anchors,
        timezone=timezone,
        timestamp_source=timestamp_source,
    )

    persisted_observations = 0
    skipped_observations = 0
    duplicate_observations = 0
    out_of_order_observations = 0
    for row in result.observations:
        observed_at = _dt(row.get("capture_timestamp"))
        if observed_at is None:
            skipped_observations += 1
            continue
        outcome = write_observation(
            session,
            camera,
            observed_at=observed_at,
            recording_id=result.recording_id,
            frame_index=row.get("frame_index"),
            elapsed_seconds=row.get("elapsed_seconds"),
            timestamp_source=row.get("timestamp_source"),
            timestamp_confidence=row.get("timestamp_confidence"),
            extraction_status=row.get("extraction_status"),
            extraction_reason=row.get("extraction_reason"),
            shoreline_geometry=row.get("shoreline_geometry") or None,
            shoreline_segments=row.get("shoreline_segments") or None,
            segmentation_confidence=row.get("confidence"),
            quality_flags=["nighttime_out_of_domain"],
            source_frame_path=row.get("frame_path"),
            shoreline_position_px=row.get("shoreline_position_px"),
            signed_position_px=row.get("signed_position_px"),
            position_axis=row.get("position_axis"),
            position_sign=row.get("position_sign"),
            segment_count=row.get("segment_count") or 0,
            segment_measurements=row.get("segment_measurements") or None,
            class_fractions=row.get("class_fractions") or None,
            inference_time_ms=row.get("inference_time_ms"),
            horizon_y_px=row.get("horizon_y_px"),
        )
        if outcome.status in {"persisted", "updated"}:
            persisted_observations += 1
        elif outcome.status == "duplicate":
            duplicate_observations += 1
            skipped_observations += 1
        elif outcome.status == "out_of_order":
            out_of_order_observations += 1
            skipped_observations += 1
            logger.warning(
                "recording %s frame %s rejected as out-of-order: %s",
                result.recording_id,
                row.get("frame_index"),
                outcome.reason,
            )
        else:
            skipped_observations += 1

    # A baseline row is always recorded: either the provisional fit, or an
    # explicit unavailable row with the reason. Never an empty "normal" model.
    session.execute(
        update(ShorelineBaseline)
        .where(
            ShorelineBaseline.camera_id == camera.id,
            ShorelineBaseline.is_active.is_(True),
        )
        .values(is_active=False)
    )
    tide_station_id = None
    if result.tide_selection is not None and result.tide_selection.station is not None:
        tide_station_id = getattr(result.tide_selection.station, "id", None)
    rolling_mode = result.resolved_detector_mode == "rolling_baseline"
    baseline_reason: str | None
    if rolling_mode:
        rolling_config = result.rolling_config
        baseline_name = f"Rolling-median shoreline baseline ({result.recording_id})"
        baseline_model_type = "rolling_median"
        baseline_params = {
            "detector_mode": "rolling_baseline",
            "status": "rolling_unverified",
            "tide_independent": True,
            "config": None if rolling_config is None else rolling_config.as_dict(),
            "config_hash": None if rolling_config is None else rolling_config.config_hash,
            "note": (
                "Self-referential causal rolling median. Not a tide-conditioned "
                "baseline: it cannot establish an absolute expected shoreline, a "
                "tide-adjusted residual, or NORMAL."
            ),
        }
        baseline_status = "rolling_unverified"
        baseline_reason = (
            "No verified tide-conditioned baseline is active. The rolling median is "
            "tide-independent and is labelled rolling_unverified."
        )
        baseline_active = False
        baseline_provisional = True
        baseline_samples = len(result.observations)
        baseline_r2 = None
        baseline_interval = None
        baseline_valid_from = result.baseline.valid_from
        baseline_valid_to = result.baseline.valid_to
    else:
        baseline_name = f"Provisional shoreline baseline ({result.recording_id})"
        baseline_model_type = result.baseline.model_type
        baseline_params = result.baseline.as_dict()
        baseline_status = result.baseline.status
        baseline_reason = result.baseline.reason
        baseline_active = bool(result.baseline.available)
        baseline_provisional = result.baseline.provisional
        baseline_samples = result.baseline.sample_count
        baseline_r2 = result.baseline.r_squared
        baseline_interval = {
            "residual_scale_px": result.baseline.residual_scale_px,
            "residual_mad_px": result.baseline.residual_mad_px,
            "slope_ci95_px_per_m": result.baseline.slope_ci95_px_per_m,
        }
        baseline_valid_from = result.baseline.valid_from
        baseline_valid_to = result.baseline.valid_to

    baseline_row = ShorelineBaseline(
        monitoring_station_id=camera.monitoring_station_id,
        camera_id=camera.id,
        tide_station_id=None if rolling_mode else tide_station_id,
        recording_id=result.recording_id,
        name=baseline_name,
        model_type=baseline_model_type,
        model_params=baseline_params,
        r_squared=baseline_r2,
        residual_mean_m=None,
        residual_std_m=None,
        prediction_interval=baseline_interval,
        sample_count=baseline_samples,
        is_active=baseline_active,
        status=baseline_status,
        provisional=baseline_provisional,
        reason=baseline_reason,
        position_axis=(
            result.observations[0].get("position_axis") if result.observations else None
        ),
        position_sign="seaward_positive",
        position_unit="px",
        detector_mode=result.resolved_detector_mode,
        valid_from=baseline_valid_from,
        valid_to=baseline_valid_to,
        notes=(
            "Experimental prototype baseline. Not validated for anomaly detection; "
            "the official national tsunami warning authority remains authoritative."
        ),
    )
    session.add(baseline_row)

    persisted_events = 0
    skipped_events = 0
    if rolling_mode:
        # One row per evaluated observation (same shape as the legacy path), so
        # the API/frontend can draw the observed position and rolling baseline
        # series. Event identity is carried by onset_frame_index on the
        # detection row only; the partial unique index prevents duplicate event
        # identities. Re-processing deletes only this recording's rolling rows.
        session.execute(
            delete(AnomalyEvent).where(
                AnomalyEvent.recording_id == result.recording_id,
                AnomalyEvent.detector_mode == "rolling_baseline",
            )
        )
        for detection in result.detections:
            if detection.timestamp is None:
                skipped_events += 1
                continue
            factors = dict(detection.contributing_factors or {})
            factors.setdefault("classification", detection.classification)
            factors.setdefault("event_max_classification", detection.event_max_classification)
            factors.setdefault("detector_mode", "rolling_baseline")
            factors.setdefault("baseline_status", "rolling_unverified")
            factors.setdefault("tide_adjusted", False)
            factors.setdefault("tide_data_available", False)
            factors.setdefault("units", "pixels; positive shoreline position is seaward")
            detection_row = (
                detection.frame_index is not None
                and detection.frame_index == detection.detection_frame_index
            )
            session.add(
                AnomalyEvent(
                    detected_at=detection.timestamp,
                    monitoring_station_id=camera.monitoring_station_id,
                    camera_id=camera.id,
                    state=detection.state,
                    anomaly_score=detection.score,
                    persistence=float(detection.persistence_count),
                    explanation=detection.reason,
                    contributing_factors=factors,
                    acknowledged=False,
                    recording_id=result.recording_id,
                    frame_index=detection.frame_index,
                    onset_frame_index=(detection.onset_frame_index if detection_row else None),
                    detector_mode="rolling_baseline",
                    classification=detection.classification,
                    baseline_status="rolling_unverified",
                    tide_adjusted=False,
                    tide_data_available=False,
                    tide_source="unavailable",
                    robust_z=detection.robust_z,
                    observed_position_px=detection.observed_position_px,
                    expected_position_px=detection.rolling_baseline_px,
                    residual_px=detection.residual_px,
                    movement_rate_px_s=detection.movement_rate_px_s,
                    persistence_count=detection.persistence_count,
                    quality_status=detection.quality_status,
                    reason=detection.reason,
                )
            )
            persisted_events += 1
    else:
        # Legacy behaviour: one row per detection, idempotent on recording+frame.
        session.execute(
            delete(AnomalyEvent).where(
                AnomalyEvent.recording_id == result.recording_id,
                or_(
                    AnomalyEvent.detector_mode.is_(None),
                    AnomalyEvent.detector_mode == result.resolved_detector_mode,
                ),
            )
        )
        for detection in result.detections:
            if detection.timestamp is None:
                skipped_events += 1
                continue
            session.add(
                AnomalyEvent(
                    detected_at=detection.timestamp,
                    monitoring_station_id=camera.monitoring_station_id,
                    camera_id=camera.id,
                    state=detection.state,
                    anomaly_score=detection.score,
                    residual_m=None,
                    retreat_velocity_m_s=None,
                    retreat_acceleration_m_s2=None,
                    persistence=float(detection.persistence_count),
                    baseline_uncertainty_m=None,
                    tide_phase=None,
                    tide_height_m=detection.tide_height_m,
                    expected_shoreline_m=None,
                    actual_shoreline_m=None,
                    explanation=detection.reason,
                    contributing_factors={
                        "classification": detection.classification,
                        "baseline_status": detection.baseline_status,
                        "tide_adjusted": detection.tide_adjusted,
                        "robust_z": detection.robust_z,
                        "movement_rate_px_s": detection.movement_rate_px_s,
                        "factors": detection.factors,
                        "thresholds": {
                            "persistence": result.detector_config.persistence,
                            "z_threshold": result.detector_config.z_threshold,
                            "movement_threshold_px_s": (
                                result.detector_config.movement_threshold_px_s
                            ),
                            "drawdown_distance_px": result.detector_config.drawdown_distance_px,
                        },
                        "units": "pixels; positive shoreline position is seaward",
                    },
                    acknowledged=False,
                    recording_id=result.recording_id,
                    frame_index=detection.frame_index,
                    detector_mode=result.resolved_detector_mode or detection.detector_mode,
                    classification=detection.classification,
                    baseline_status=detection.baseline_status,
                    tide_adjusted=detection.tide_adjusted,
                    tide_data_available=detection.tide_available,
                    tide_source=detection.tide_source,
                    robust_z=detection.robust_z,
                    observed_position_px=detection.observed_position_px,
                    expected_position_px=detection.expected_position_px,
                    residual_px=detection.residual_px,
                    movement_rate_px_s=detection.movement_rate_px_s,
                    persistence_count=detection.persistence_count,
                    quality_status=detection.quality_status,
                    reason=detection.reason,
                )
            )
            persisted_events += 1

    session.flush()
    return {
        "recording_id": result.recording_id,
        "observations_persisted": persisted_observations,
        "observations_skipped": skipped_observations,
        "duplicate_observations": duplicate_observations,
        "out_of_order_observations": out_of_order_observations,
        "anomaly_events_persisted": persisted_events,
        "anomaly_events_skipped": skipped_events,
        "baseline_id": session.scalar(
            select(ShorelineBaseline.id)
            .where(
                ShorelineBaseline.recording_id == result.recording_id,
                ShorelineBaseline.monitoring_station_id == camera.monitoring_station_id,
            )
            .order_by(ShorelineBaseline.id.desc())
            .limit(1)
        ),
        "baseline_status": baseline_status,
        "baseline_reason": baseline_reason,
    }
