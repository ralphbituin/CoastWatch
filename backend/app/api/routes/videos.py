"""Archive-video registration and bounded prototype processing (Phase 5/6).

The offline script processes the full 4.3-hour archive; this endpoint is the
database/API integration path and is deliberately bounded so one HTTP request
cannot run for hours.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_db, pagination_params
from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.camera import Camera
from app.models.shoreline import ShorelineObservation
from app.models.video import VideoRecording
from app.schemas.shorelines import ShorelineObservationSummary
from app.schemas.videos import (
    VideoProcessRequest,
    VideoProcessResult,
    VideoRecordingCreate,
    VideoRecordingRead,
)
from app.services.anomaly.rolling_baseline import RollingBaselineConfig
from app.services.camera.video import resolve_video_path
from app.services.video.persistence import persist_analysis
from app.services.video.pipeline import PrototypeConfig, analyze_video
from app.services.video.sampling import probe_video, stable_recording_id
from app.services.video.timestamps import ClockAnchor, anchor_dicts
from app.services.vision.roi import load_active_roi

router = APIRouter(tags=["videos"])


def _read(row: VideoRecording) -> VideoRecordingRead:
    return VideoRecordingRead(
        id=row.id,
        recording_id=row.recording_id,
        camera_id=row.camera_id,
        monitoring_station_id=row.monitoring_station_id,
        file_path=row.file_path,
        file_size_bytes=row.file_size_bytes,
        duration_seconds=row.duration_seconds,
        fps=row.fps,
        frame_count=row.frame_count,
        width=row.width,
        height=row.height,
        video_codec=row.video_codec,
        audio_codec=row.audio_codec,
        timezone=row.timezone,
        timestamp_source=row.timestamp_source,
        timestamp_confidence=row.timestamp_confidence,
        processing_status=row.processing_status,
        processed_frames=row.processed_frames,
        sampled_frames=row.sampled_frames,
        metadata_json=row.metadata_json,
        notes=row.notes,
        created_at=row.created_at,
    )


def _get_recording(db: Session, recording_id: str) -> VideoRecording:
    row = db.scalar(
        select(VideoRecording).where(VideoRecording.recording_id == recording_id).limit(1)
    )
    if row is None:
        raise NotFoundError(f"Video recording not found: {recording_id}")
    return row


def _anchors(
    payload: VideoRecordingCreate | VideoProcessRequest, recording: VideoRecording
) -> list[ClockAnchor]:
    raw = getattr(payload, "clock_anchors", None) or recording.clock_anchors or []
    anchors: list[ClockAnchor] = []
    for item in raw:
        anchors.append(
            ClockAnchor(
                elapsed_seconds=float(item["elapsed_seconds"]),
                timestamp=str(item["timestamp"]),
                source=str(item.get("source", "visible_clock")),
                confidence=float(item.get("confidence", 0.9)),
                verification=item.get("verification"),
            )
        )
    return anchors


@router.get("", response_model=list[VideoRecordingRead], summary="List archive recordings")
def list_recordings(
    db: Session = Depends(get_db),
    pagination: tuple[int, int] = Depends(pagination_params),
) -> list[VideoRecordingRead]:
    limit, offset = pagination
    rows = db.scalars(
        select(VideoRecording)
        .order_by(VideoRecording.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [_read(row) for row in rows]


@router.post(
    "/recordings",
    response_model=VideoRecordingRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register (or refresh) archive metadata without copying the file",
)
def register_recording(
    payload: VideoRecordingCreate, db: Session = Depends(get_db)
) -> VideoRecordingRead:
    settings = get_settings()
    try:
        video_path = resolve_video_path(settings.data_dir, payload.video_path)
    except Exception as exc:
        raise ValidationError(str(exc)) from exc
    if not video_path.is_file():
        raise ValidationError(f"Video not found under DATA_DIR/videos: {payload.video_path}")
    metadata = probe_video(video_path)
    recording_id = stable_recording_id(video_path, metadata)

    camera = None
    if payload.camera_id is not None:
        camera = db.get(Camera, payload.camera_id, options=[joinedload(Camera.station)])
        if camera is None:
            raise NotFoundError(f"Camera not found: {payload.camera_id}")
    station_id = payload.monitoring_station_id or (
        camera.monitoring_station_id if camera is not None else None
    )
    anchors = list(payload.clock_anchors or [])

    row = db.scalar(
        select(VideoRecording).where(VideoRecording.recording_id == recording_id).limit(1)
    )
    if row is None:
        row = VideoRecording(recording_id=recording_id)
        db.add(row)
    row.camera_id = payload.camera_id
    row.monitoring_station_id = station_id
    row.file_path = str(video_path)
    row.file_size_bytes = metadata.file_size_bytes
    row.duration_seconds = metadata.duration_seconds
    row.fps = metadata.fps
    row.frame_count = metadata.frame_count
    row.width = metadata.width
    row.height = metadata.height
    row.video_codec = metadata.video_codec
    row.audio_codec = metadata.audio_codec
    row.timezone = payload.timezone
    row.timestamp_source = payload.timestamp_source or ("derived" if anchors else "unavailable")
    row.timestamp_confidence = payload.timestamp_confidence
    row.clock_anchors = anchors or None
    row.metadata_json = {"container": metadata.as_dict()}
    row.notes = payload.notes
    if row.processing_status is None:
        row.processing_status = "registered"
    db.commit()
    db.refresh(row)
    return _read(row)


@router.get("/{recording_id}", response_model=VideoRecordingRead, summary="Recording detail")
def get_recording(recording_id: str, db: Session = Depends(get_db)) -> VideoRecordingRead:
    return _read(_get_recording(db, recording_id))


@router.get(
    "/{recording_id}/observations",
    response_model=list[ShorelineObservationSummary],
    summary="Observations belonging to one recording",
)
def recording_observations(
    recording_id: str,
    db: Session = Depends(get_db),
    pagination: tuple[int, int] = Depends(pagination_params),
) -> list[ShorelineObservationSummary]:
    _get_recording(db, recording_id)
    limit, offset = pagination
    rows = db.execute(
        select(ShorelineObservation, Camera.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .where(ShorelineObservation.recording_id == recording_id)
        .order_by(ShorelineObservation.frame_index.asc())
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


@router.post(
    "/process",
    response_model=VideoProcessResult,
    summary="Process a bounded portion through the production vision + anomaly path",
)
def process_recording(
    payload: VideoProcessRequest, db: Session = Depends(get_db)
) -> VideoProcessResult:
    # Validate the time-based rolling config before any DB/file lookup so an
    # invalid request is always a 422, not a 404/500.
    rolling_config = RollingBaselineConfig(**payload.rolling.model_dump())
    try:
        rolling_config.validate()
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc

    recording = _get_recording(db, payload.recording_id)
    video_path = Path(recording.file_path)
    if not video_path.is_file():
        raise ValidationError(f"Recording file is no longer available: {recording.file_path}")
    camera_id = payload.camera_id or recording.camera_id
    if camera_id is None:
        raise ValidationError(
            "camera_id is required to persist observations (ShorelineObservation has a camera FK)"
        )
    camera = db.get(Camera, camera_id, options=[joinedload(Camera.station)])
    if camera is None:
        raise NotFoundError(f"Camera not found: {camera_id}")

    latitude = payload.latitude
    longitude = payload.longitude
    if latitude is None:
        latitude = camera.latitude
    if longitude is None:
        longitude = camera.longitude

    anchors = _anchors(payload, recording)
    timezone = recording.timezone or "Asia/Bangkok"
    country_iso = payload.country_iso

    config = PrototypeConfig(
        sample_interval_seconds=payload.sample_interval_seconds,
        start_elapsed=payload.start_elapsed,
        end_elapsed=payload.end_elapsed,
        max_samples=payload.max_samples,
        timezone=timezone,
        country_iso=country_iso,
        latitude=latitude,
        longitude=longitude,
        max_station_distance_km=payload.max_station_distance_km,
        camera_id=camera.id,
        monitoring_station_id=camera.monitoring_station_id,
        persistence=payload.persistence,
        z_threshold=payload.z_threshold,
        save_frames=payload.save_frames,
        detector_mode=payload.detector_mode,
        rolling=rolling_config,
        frame_height=float(recording.height) if recording.height else None,
        roi_polygon=load_active_roi(db, camera),
        frame_dir=(
            str(Path(recording.file_path).parent / "phase5_frames") if payload.save_frames else None
        ),
    )
    try:
        config.rolling.validate()
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    result = analyze_video(
        video_path,
        config,
        anchors=anchors,
    )
    persistence = persist_analysis(
        db,
        result,
        camera=camera,
        anchors=anchor_dicts(anchors),
        timezone=timezone,
        timestamp_source="derived" if anchors else "unavailable",
    )
    db.commit()

    counts: dict[str, int] = {}
    for detection in result.detections:
        counts[detection.classification] = counts.get(detection.classification, 0) + 1
    tide_reason = (
        "tide prediction generation was removed from this build; "
        "analysis rows run without tide association"
    )
    return VideoProcessResult(
        recording_id=result.recording_id,
        processed=result.stats.processed,
        available=result.stats.available,
        unavailable=result.stats.unavailable,
        baseline_status=result.baseline.status,
        baseline_reason=result.baseline.reason,
        tide_data_available=False,
        tide_reason=tide_reason,
        classifications=counts,
        observations_persisted=int(persistence["observations_persisted"]),
        observations_skipped=int(persistence["observations_skipped"]),
        duplicate_observations=int(persistence.get("duplicate_observations", 0)),
        out_of_order_observations=int(persistence.get("out_of_order_observations", 0)),
        anomaly_events_persisted=int(persistence["anomaly_events_persisted"]),
        detector_mode=result.detector_mode,
        resolved_detector_mode=result.resolved_detector_mode,
        mode_reason=result.mode_reason,
        rolling_event_count=(
            0 if result.rolling_evaluation is None else len(result.rolling_evaluation.events)
        ),
    )
