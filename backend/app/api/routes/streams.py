"""Live stream session listing and per-timestamp observations (Phase 7).

``GET /videos/{recording_id}/observations`` remains the archive API. Live
sessions are addressed by ``session_id`` here so the UI can inspect one bounded
run without scanning the whole observation table.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.api.deps import get_db, pagination_params
from app.core.exceptions import ConflictError, NotFoundError
from app.models.anomaly import AnomalyEvent
from app.models.camera import Camera
from app.models.shoreline import ShorelineBaseline, ShorelineObservation
from app.models.stream import StreamSession
from app.schemas.common import Message
from app.schemas.shorelines import ShorelineObservationSummary
from app.schemas.streams import StreamSessionRead
from app.services.live.capture import get_live_manager

router = APIRouter(tags=["streams"])


def _read(row: StreamSession, camera_code: str | None = None) -> StreamSessionRead:
    return StreamSessionRead(
        id=row.id,
        session_id=row.session_id,
        recording_id=row.recording_id,
        camera_id=row.camera_id,
        camera_code=camera_code,
        started_at=row.started_at,
        ended_at=row.ended_at,
        status=row.status,
        source_url_redacted=row.source_url_redacted,
        sample_interval_seconds=row.sample_interval_seconds,
        last_frame_at=row.last_frame_at,
        frames_captured=row.frames_captured,
        frames_persisted=row.frames_persisted,
        frames_rejected=row.frames_rejected,
        reconnect_count=row.reconnect_count,
        last_error=row.last_error,
        last_rolling_classification=row.last_rolling_classification,
        last_event_at=row.last_event_at,
        detector_config=row.detector_config,
        metadata_json=row.metadata_json,
        created_at=row.created_at,
    )


@router.get("", response_model=list[StreamSessionRead], summary="List live stream sessions")
def list_stream_sessions(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    pagination: tuple[int, int] = Depends(pagination_params),
) -> list[StreamSessionRead]:
    limit, offset = pagination
    stmt = (
        select(StreamSession, Camera.code)
        .join(Camera, Camera.id == StreamSession.camera_id)
        .order_by(StreamSession.started_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if camera_id is not None:
        stmt = stmt.where(StreamSession.camera_id == camera_id)
    return [_read(row, code) for row, code in db.execute(stmt).all()]


@router.get("/{session_id}", response_model=StreamSessionRead, summary="Stream session detail")
def get_stream_session(session_id: str, db: Session = Depends(get_db)) -> StreamSessionRead:
    row = db.scalar(select(StreamSession).where(StreamSession.session_id == session_id).limit(1))
    if row is None:
        raise NotFoundError(f"Stream session not found: {session_id}")
    code = db.scalar(select(Camera.code).where(Camera.id == row.camera_id))
    return _read(row, code)


@router.delete(
    "/{session_id}",
    response_model=Message,
    summary="Delete a live stream session and its observation/anomaly rows",
)
def delete_stream_session(session_id: str, db: Session = Depends(get_db)) -> Message:
    row = db.scalar(select(StreamSession).where(StreamSession.session_id == session_id).limit(1))
    if row is None:
        raise NotFoundError(f"Stream session not found: {session_id}")
    live = get_live_manager().status(row.camera_id)
    if live.session_id == row.session_id and live.state in {
        "starting",
        "running",
        "reconnecting",
    }:
        get_live_manager().stop_camera(row.camera_id)
    try:
        db.execute(
            delete(ShorelineObservation).where(
                ShorelineObservation.stream_session_id == row.session_id
            )
        )
        db.execute(
            delete(ShorelineBaseline).where(ShorelineBaseline.recording_id == row.recording_id)
        )
        db.execute(delete(AnomalyEvent).where(AnomalyEvent.recording_id == row.recording_id))
        db.delete(row)
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Could not delete stream session {session_id}: {exc}",
            details={"session_id": session_id},
        ) from exc
    return Message(message=f"live stream session {session_id} deleted")


@router.get(
    "/{session_id}/observations",
    response_model=list[ShorelineObservationSummary],
    summary="Per-timestamp observations for one live session",
)
def stream_observations(
    session_id: str,
    db: Session = Depends(get_db),
    pagination: tuple[int, int] = Depends(pagination_params),
) -> list[ShorelineObservationSummary]:
    session = db.scalar(
        select(StreamSession).where(StreamSession.session_id == session_id).limit(1)
    )
    if session is None:
        raise NotFoundError(f"Stream session not found: {session_id}")
    limit, offset = pagination
    rows = db.execute(
        select(ShorelineObservation, Camera.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .where(ShorelineObservation.stream_session_id == session_id)
        .order_by(ShorelineObservation.observed_at.asc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [
        ShorelineObservationSummary(
            camera_id=row.camera_id,
            camera_code=code,
            observed_at=row.observed_at,
            monitoring_station_id=row.monitoring_station_id,
            station_code=None,
            calibration_id=row.calibration_id,
            calibrated_position_m=row.calibrated_position_m,
            segmentation_confidence=row.segmentation_confidence,
            frame_quality_score=row.frame_quality_score,
            frame_quality_flags=list(row.frame_quality_flags or []),
            created_at=row.created_at,
            recording_id=row.recording_id,
            frame_index=row.frame_index,
            extraction_status=row.extraction_status,
            extraction_reason=row.extraction_reason,
            shoreline_position_px=row.shoreline_position_px,
            signed_position_px=row.signed_position_px,
            position_axis=row.position_axis,
            segment_count=row.segment_count,
            inference_time_ms=row.inference_time_ms,
            horizon_y_px=row.horizon_y_px,
            capture_timestamp_source=row.capture_timestamp_source,
            capture_timestamp_confidence=row.capture_timestamp_confidence,
            elapsed_seconds=row.elapsed_seconds,
            stream_session_id=row.stream_session_id,
            ingest_latency_ms=row.ingest_latency_ms,
        )
        for row, code in rows
    ]
