"""Anomaly-event routes (Phase 6 prototype).

Every event carries the transparent classification and its evidence. An
anomaly is a monitoring signal, never a confirmed tsunami; PHIVOLCS advisories
remain authoritative.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, pagination_params
from app.core.exceptions import DataUnavailableError, NotFoundError
from app.models.anomaly import AnomalyEvent
from app.models.camera import Camera
from app.models.shoreline import ShorelineObservation
from app.models.station import MonitoringStation
from app.schemas.anomalies import (
    AcknowledgeResult,
    AnomalyEventList,
    AnomalyEventRead,
    AnomalySummary,
)
from app.schemas.shorelines import ShorelineObservationDetail

router = APIRouter(tags=["anomalies"])


def _factor(row: AnomalyEvent, key: str) -> Any:
    factors = row.contributing_factors
    if not isinstance(factors, dict):
        return None
    return factors.get(key)


def _factor_bool(row: AnomalyEvent, key: str) -> bool | None:
    value = _factor(row, key)
    return None if value is None else bool(value)


def _factor_time(row: AnomalyEvent, key: str) -> datetime | None:
    value = _factor(row, key)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _read(row: AnomalyEvent) -> AnomalyEventRead:
    return AnomalyEventRead(
        id=row.id,
        detected_at=row.detected_at,
        monitoring_station_id=row.monitoring_station_id,
        camera_id=row.camera_id,
        recording_id=row.recording_id,
        frame_index=row.frame_index,
        state=row.state,
        classification=row.classification,
        anomaly_score=row.anomaly_score,
        baseline_status=row.baseline_status,
        tide_adjusted=row.tide_adjusted,
        tide_data_available=row.tide_data_available,
        tide_source=row.tide_source,
        tide_height_m=row.tide_height_m,
        observed_position_px=row.observed_position_px,
        expected_position_px=row.expected_position_px,
        residual_px=row.residual_px,
        robust_z=row.robust_z,
        movement_rate_px_s=row.movement_rate_px_s,
        persistence_count=row.persistence_count,
        quality_status=row.quality_status,
        reason=row.reason,
        explanation=row.explanation,
        contributing_factors=row.contributing_factors,
        acknowledged=row.acknowledged,
        created_at=row.created_at,
        detector_mode=row.detector_mode,
        resolved_detector_mode=row.detector_mode,
        mode_reason=(
            None
            if row.detector_mode is None
            else (
                "tide-independent rolling-median drawdown detector"
                if row.detector_mode == "rolling_baseline"
                else f"resolved detector mode: {row.detector_mode}"
            )
        ),
        onset_frame_index=row.onset_frame_index,
        rolling_baseline_px=_factor(row, "rolling_baseline_px"),
        rolling_scale_px=_factor(row, "rolling_scale_px"),
        scale_floored=_factor_bool(row, "scale_floored"),
        cumulative_drawdown_px=_factor(row, "cumulative_drawdown_px"),
        baseline_frozen=_factor_bool(row, "baseline_frozen"),
        event_max_classification=_factor(row, "event_max_classification"),
        onset_time=_factor_time(row, "onset_time"),
        detection_time=_factor_time(row, "detection_time"),
        detection_latency_s=_factor(row, "detection_latency_s"),
        event_window_seconds=_factor(row, "event_window_seconds"),
        cooldown_state=_factor(row, "cooldown_state"),
        config_hash=_factor(row, "config_hash"),
    )


@router.get("", response_model=AnomalyEventList, summary="List anomaly/quality events")
def list_events(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    station_id: int | None = None,
    recording_id: str | None = None,
    state: str | None = None,
    classification: str | None = None,
    baseline_status: str | None = None,
    pagination: tuple[int, int] = Depends(pagination_params),
) -> AnomalyEventList:
    limit, offset = pagination
    filters = []
    if camera_id is not None:
        filters.append(AnomalyEvent.camera_id == camera_id)
    if station_id is not None:
        filters.append(AnomalyEvent.monitoring_station_id == station_id)
    if recording_id:
        filters.append(AnomalyEvent.recording_id == recording_id)
    if state:
        filters.append(AnomalyEvent.state == state)
    if classification:
        filters.append(AnomalyEvent.classification == classification)
    if baseline_status:
        filters.append(AnomalyEvent.baseline_status == baseline_status)

    base = select(AnomalyEvent)
    count_stmt = select(func.count()).select_from(AnomalyEvent)
    for condition in filters:
        base = base.where(condition)
        count_stmt = count_stmt.where(condition)
    total = int(db.scalar(count_stmt) or 0)
    rows = list(
        db.scalars(base.order_by(AnomalyEvent.detected_at.desc()).limit(limit).offset(offset)).all()
    )
    return AnomalyEventList(total=total, items=[_read(row) for row in rows])


@router.get("/summary", response_model=AnomalySummary, summary="Counts by classification")
def summary(
    db: Session = Depends(get_db),
    recording_id: str | None = None,
    camera_id: int | None = None,
    station_id: int | None = None,
) -> AnomalySummary:
    filters = []
    if recording_id:
        filters.append(AnomalyEvent.recording_id == recording_id)
    if camera_id is not None:
        filters.append(AnomalyEvent.camera_id == camera_id)
    if station_id is not None:
        filters.append(AnomalyEvent.monitoring_station_id == station_id)

    def grouped(column):
        stmt = select(column, func.count()).group_by(column)
        for condition in filters:
            stmt = stmt.where(condition)
        return {
            str(key if key is not None else "unknown"): int(count)
            for key, count in db.execute(stmt).all()
        }

    total_stmt = select(func.count()).select_from(AnomalyEvent)
    for condition in filters:
        total_stmt = total_stmt.where(condition)
    return AnomalySummary(
        total=int(db.scalar(total_stmt) or 0),
        by_classification=grouped(AnomalyEvent.classification),
        by_state=grouped(AnomalyEvent.state),
        by_baseline_status=grouped(AnomalyEvent.baseline_status),
    )


@router.get(
    "/current",
    response_model=AnomalyEventRead,
    summary="Latest anomaly/quality event, optionally per camera or recording",
)
def current(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    recording_id: str | None = None,
) -> AnomalyEventRead:
    stmt = select(AnomalyEvent).order_by(AnomalyEvent.detected_at.desc()).limit(1)
    if camera_id is not None:
        stmt = stmt.where(AnomalyEvent.camera_id == camera_id)
    if recording_id:
        stmt = stmt.where(AnomalyEvent.recording_id == recording_id)
    row = db.scalar(stmt)
    if row is None:
        raise NotFoundError("No anomaly events recorded yet")
    return _read(row)


@router.get("/{event_id}", response_model=AnomalyEventRead, summary="Anomaly event detail")
def get_event(event_id: int, db: Session = Depends(get_db)) -> AnomalyEventRead:
    row = db.scalar(select(AnomalyEvent).where(AnomalyEvent.id == event_id).limit(1))
    if row is None:
        raise NotFoundError(f"Anomaly event not found: {event_id}")
    return _read(row)


@router.get(
    "/{event_id}/observation",
    response_model=ShorelineObservationDetail,
    summary="Shoreline observation behind an anomaly event",
)
def event_observation(event_id: int, db: Session = Depends(get_db)) -> ShorelineObservationDetail:
    event = db.scalar(select(AnomalyEvent).where(AnomalyEvent.id == event_id).limit(1))
    if event is None:
        raise NotFoundError(f"Anomaly event not found: {event_id}")
    if event.camera_id is None:
        raise DataUnavailableError(
            "The camera for this event has been deleted, so its shoreline "
            "observation is unavailable",
            details={"event_id": event.id},
        )

    base = (
        select(ShorelineObservation, Camera.code, MonitoringStation.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .join(
            MonitoringStation,
            MonitoringStation.id == ShorelineObservation.monitoring_station_id,
        )
        .where(ShorelineObservation.camera_id == event.camera_id)
    )
    row = None
    if event.recording_id and event.frame_index is not None:
        row = db.execute(
            base.where(
                ShorelineObservation.recording_id == event.recording_id,
                ShorelineObservation.frame_index == event.frame_index,
            ).limit(1)
        ).first()
    if row is None:
        row = db.execute(
            base.where(ShorelineObservation.observed_at == event.detected_at).limit(1)
        ).first()
    if row is None:
        # Legacy/rolling events can be stamped a few seconds after the sampled
        # observation. Fall back to the nearest observation for this camera,
        # but only within a bounded window so a different frame is never shown.
        distance = func.abs(
            func.extract(
                "epoch",
                ShorelineObservation.observed_at - event.detected_at,
            )
        )
        candidate = db.execute(base.where(distance <= 600).order_by(distance).limit(1)).first()
        if candidate is not None:
            row = candidate
    if row is None:
        raise DataUnavailableError(
            "No shoreline observation is stored for this event",
            details={
                "event_id": event.id,
                "camera_id": event.camera_id,
                "recording_id": event.recording_id,
                "frame_index": event.frame_index,
            },
        )

    observation, camera_code, station_code = row
    return ShorelineObservationDetail(
        camera_id=observation.camera_id,
        camera_code=camera_code,
        observed_at=observation.observed_at,
        monitoring_station_id=observation.monitoring_station_id,
        station_code=station_code,
        calibration_id=observation.calibration_id,
        shoreline_geometry=observation.shoreline_geometry,
        shoreline_segments=observation.shoreline_segments,
        calibrated_position_m=observation.calibrated_position_m,
        segmentation_confidence=observation.segmentation_confidence,
        frame_quality_score=observation.frame_quality_score,
        frame_quality_flags=list(observation.frame_quality_flags or []),
        source_frame_path=observation.source_frame_path,
        created_at=observation.created_at,
        recording_id=observation.recording_id,
        frame_index=observation.frame_index,
        extraction_status=observation.extraction_status,
        extraction_reason=observation.extraction_reason,
        shoreline_position_px=observation.shoreline_position_px,
        signed_position_px=observation.signed_position_px,
        position_axis=observation.position_axis,
        position_sign=observation.position_sign,
        segment_count=observation.segment_count,
        segment_measurements=observation.segment_measurements,
        class_fractions=observation.class_fractions,
        inference_time_ms=observation.inference_time_ms,
        capture_timestamp_source=observation.capture_timestamp_source,
        capture_timestamp_confidence=observation.capture_timestamp_confidence,
        horizon_y_px=observation.horizon_y_px,
        elapsed_seconds=observation.elapsed_seconds,
        stream_session_id=observation.stream_session_id,
        ingest_latency_ms=observation.ingest_latency_ms,
    )


@router.post(
    "/{event_id}/acknowledge",
    response_model=AcknowledgeResult,
    summary="Acknowledge an event",
)
def acknowledge(event_id: int, db: Session = Depends(get_db)) -> AcknowledgeResult:
    row = db.scalar(select(AnomalyEvent).where(AnomalyEvent.id == event_id).limit(1))
    if row is None:
        raise NotFoundError(f"Anomaly event not found: {event_id}")
    row.acknowledged = True
    db.commit()
    return AcknowledgeResult(id=row.id, acknowledged=row.acknowledged)
