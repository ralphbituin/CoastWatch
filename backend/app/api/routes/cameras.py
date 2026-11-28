"""Camera routes: connection, status, frames and calibration profiles."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import Response
from geoalchemy2.elements import WKTElement
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_db
from app.core.config import get_settings
from app.core.exceptions import (
    ConflictError,
    DataUnavailableError,
    NotFoundError,
    PreviewNotReadyError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.camera import Camera, CameraCalibration
from app.models.station import MonitoringStation
from app.schemas.cameras import (
    CameraCalibrationCreate,
    CameraCalibrationRead,
    CameraCreate,
    CameraDeviceListRead,
    CameraDeviceRead,
    CameraQualityRead,
    CameraRead,
    CameraStatusRead,
    CameraStreamStatusRead,
    CameraTestConnectionResult,
    CameraUpdate,
    RoiPolygon,
    RoiPolygonSave,
)
from app.schemas.common import Message
from app.services.camera import manager
from app.services.camera.base import CameraError, FrameUnavailableError
from app.services.camera.upload import ALLOWED_UPLOAD_EXTENSIONS, camera_upload_dir
from app.services.camera.video import (
    ALLOWED_VIDEO_EXTENSIONS,
    resolve_video_path,
    video_dir,
)
from app.services.camera.youtube import redact_youtube_url, validate_youtube_url
from app.services.live.capture import get_live_manager
from app.services.live.preview_cache import get_preview_cache
from app.services.vision.preprocessing import FrameQuality
from app.services.vision.roi import load_active_roi

logger = get_logger(__name__)

router = APIRouter(tags=["cameras"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB

# Same default palette the model artifact publishes. Used when a preview is
# rendered from the worker cache and no segmenter instance is constructed.
DEFAULT_CLASS_COLOR_MAP = {
    "Beach": [214, 181, 137],
    "Ocean": [32, 112, 190],
    "Sky": [135, 206, 235],
}


def _get_camera(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id, options=[joinedload(Camera.station)])
    if camera is None:
        raise NotFoundError(f"Camera not found: {camera_id}")
    return camera


def _to_read(camera: Camera) -> CameraRead:
    station = camera.station
    return CameraRead(
        id=camera.id,
        code=camera.code,
        name=camera.name,
        monitoring_station_id=camera.monitoring_station_id,
        station_code=station.code if station else None,
        station_name=station.name if station else None,
        camera_type=camera.camera_type,
        rtsp_url_redacted=manager.redact_rtsp_url(camera.rtsp_url),
        device_index=camera.device_index,
        video_path=camera.video_path,
        youtube_url_redacted=manager.redact_camera_source(camera)
        if camera.camera_type == "youtube_live"
        else None,
        stream_quality=camera.stream_quality,
        stream_format=camera.stream_format,
        resolution_width=camera.resolution_width,
        resolution_height=camera.resolution_height,
        fps=camera.fps,
        latitude=camera.latitude,
        longitude=camera.longitude,
        orientation_deg=camera.orientation_deg,
        monitoring_region=camera.monitoring_region,
        enabled=camera.enabled,
        active_calibration_id=camera.active_calibration_id,
        created_at=camera.created_at,
    )


def _quality_read(quality: FrameQuality) -> CameraQualityRead:
    return CameraQualityRead(
        score=quality.score,
        is_black=quality.is_black,
        is_frozen=quality.is_frozen,
        is_blurry=quality.is_blurry,
        is_obstructed=quality.is_obstructed,
        is_timestamp_inconsistent=quality.is_timestamp_inconsistent,
        flags=list(quality.flags),
    )


@router.get("", response_model=list[CameraRead], summary="List cameras")
def list_cameras(db: Session = Depends(get_db)) -> list[CameraRead]:
    cameras = db.scalars(
        select(Camera).options(joinedload(Camera.station)).order_by(Camera.code)
    ).all()
    return [_to_read(c) for c in cameras]


@router.get(
    "/devices",
    response_model=CameraDeviceListRead,
    summary="Detect camera devices (local + host agent)",
)
def list_camera_devices(probe: bool = False) -> CameraDeviceListRead:
    """List camera devices visible to this backend process.

    ``probe=true`` opens each device briefly to report resolution/FPS and
    whether it can actually be used. It is slower and may fail while another
    application is using the camera.

    Docker Desktop on Windows cannot pass a USB webcam into a Linux container.
    Start ``tools/host_camera_agent.py`` on the Windows host and point
    ``HOST_CAMERA_AGENT_URL`` at it to see host webcams here.
    """
    from app.services.camera.discovery import discover_cameras

    result = discover_cameras(probe=probe)
    return CameraDeviceListRead(
        devices=[CameraDeviceRead(**device.as_dict()) for device in result.devices],
        host_agent_configured=result.host_agent_configured,
        host_agent_available=result.host_agent_available,
        warnings=result.warnings,
    )


@router.post(
    "",
    response_model=CameraRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a camera",
)
def create_camera(payload: CameraCreate, db: Session = Depends(get_db)) -> CameraRead:
    station = db.get(MonitoringStation, payload.monitoring_station_id)
    if station is None:
        raise NotFoundError(f"Monitoring station not found: {payload.monitoring_station_id}")
    if payload.camera_type == "video":
        # Normalize + reject path traversal before storing.
        try:
            resolve_video_path(get_settings().data_dir, payload.video_path or "")
        except CameraError as exc:
            raise ValidationError(str(exc), details={"video_path": payload.video_path}) from exc
    location = None
    if payload.latitude is not None and payload.longitude is not None:
        location = WKTElement(f"POINT({payload.longitude} {payload.latitude})", srid=4326)
    camera = Camera(
        monitoring_station_id=station.id,
        code=payload.code,
        name=payload.name,
        camera_type=payload.camera_type,
        rtsp_url=payload.rtsp_url,
        device_index=payload.device_index,
        video_path=payload.video_path,
        youtube_url=payload.youtube_url,
        stream_quality=payload.stream_quality,
        stream_format=payload.stream_format,
        resolution_width=payload.resolution_width,
        resolution_height=payload.resolution_height,
        fps=payload.fps,
        latitude=payload.latitude,
        longitude=payload.longitude,
        location=location,
        orientation_deg=payload.orientation_deg,
        monitoring_region=payload.monitoring_region,
        enabled=payload.enabled,
    )
    db.add(camera)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Camera code already exists or constraint failed: {payload.code}",
            details={"code": payload.code},
        ) from exc
    db.refresh(camera)
    camera.station = station
    return _to_read(camera)


@router.get("/{camera_id}", response_model=CameraRead, summary="Camera detail")
def get_camera(camera_id: int, db: Session = Depends(get_db)) -> CameraRead:
    return _to_read(_get_camera(db, camera_id))


def _validate_camera_transport(values: dict[str, object]) -> None:
    """Validate the transport contract after merging a partial update."""
    camera_type = values.get("camera_type")
    rtsp_url = values.get("rtsp_url")
    device_index = values.get("device_index")
    video_path = values.get("video_path")
    youtube_url = values.get("youtube_url")
    if camera_type == "rtsp" and not rtsp_url:
        raise ValidationError("rtsp_url is required for camera_type='rtsp'")
    if camera_type == "usb" and device_index is None:
        raise ValidationError("device_index is required for camera_type='usb'")
    if camera_type == "upload" and (rtsp_url or device_index is not None or video_path):
        raise ValidationError("upload cameras take neither rtsp_url, device_index nor video_path")
    if camera_type == "video":
        if not video_path:
            raise ValidationError("video_path is required for camera_type='video'")
        if rtsp_url or device_index is not None:
            raise ValidationError("video cameras take neither rtsp_url nor device_index")
        if youtube_url:
            raise ValidationError("video cameras do not take youtube_url")
    if camera_type == "youtube_live":
        if not youtube_url:
            raise ValidationError("youtube_url is required for camera_type='youtube_live'")
        if rtsp_url or device_index is not None or video_path:
            raise ValidationError(
                "youtube_live cameras take neither rtsp_url, device_index nor video_path"
            )
    elif youtube_url:
        raise ValidationError("youtube_url is only valid for camera_type='youtube_live'")


@router.patch("/{camera_id}", response_model=CameraRead, summary="Update a camera")
def update_camera(
    camera_id: int, payload: CameraUpdate, db: Session = Depends(get_db)
) -> CameraRead:
    camera = _get_camera(db, camera_id)
    updates = payload.model_dump(exclude_unset=True)
    if not updates:
        return _to_read(camera)

    if "monitoring_station_id" in updates:
        station = db.get(MonitoringStation, int(updates["monitoring_station_id"]))
        if station is None:
            raise NotFoundError(f"Monitoring station not found: {updates['monitoring_station_id']}")

    merged: dict[str, object] = {
        "camera_type": camera.camera_type,
        "rtsp_url": camera.rtsp_url,
        "device_index": camera.device_index,
        "video_path": camera.video_path,
        "youtube_url": camera.youtube_url,
    }
    merged.update({key: updates[key] for key in merged if key in updates})

    new_type = str(merged["camera_type"])
    if "camera_type" in updates and new_type != camera.camera_type:
        # Drop transport fields that do not belong to the new type so a stale
        # secret cannot survive a type change.
        if new_type != "rtsp":
            merged["rtsp_url"] = None
        if new_type != "usb":
            merged["device_index"] = None
        if new_type != "video":
            merged["video_path"] = None
        if new_type != "youtube_live":
            merged["youtube_url"] = None
        for key, value in merged.items():
            updates[key] = value

    if "video_path" in updates and updates["video_path"]:
        try:
            resolve_video_path(get_settings().data_dir, str(updates["video_path"]))
        except CameraError as exc:
            raise ValidationError(str(exc), details={"video_path": updates["video_path"]}) from exc
    if "youtube_url" in updates and updates["youtube_url"]:
        try:
            updates["youtube_url"] = validate_youtube_url(str(updates["youtube_url"]))
        except ValueError as exc:
            raise ValidationError(str(exc), details={"youtube_url": "invalid"}) from exc

    _validate_camera_transport(merged)

    for key, value in updates.items():
        setattr(camera, key, value)

    if "latitude" in updates or "longitude" in updates:
        if camera.latitude is not None and camera.longitude is not None:
            camera.location = WKTElement(  # type: ignore[assignment]
                f"POINT({camera.longitude} {camera.latitude})", srid=4326
            )
        else:
            camera.location = None

    # A transport or station change invalidates the cached provider.
    manager.drop_provider(camera.id)
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Camera update failed (code already exists or constraint failed): {camera.code}",
            details={"camera_id": camera.id},
        ) from exc
    db.refresh(camera)
    return _to_read(_get_camera(db, camera.id))


@router.delete("/{camera_id}", response_model=Message, summary="Delete a camera")
def delete_camera(camera_id: int, db: Session = Depends(get_db)) -> Message:
    camera = _get_camera(db, camera_id)
    live = get_live_manager().status(camera.id)
    if live.state in {"starting", "running", "reconnecting"}:
        get_live_manager().stop_camera(camera.id)
    manager.drop_provider(camera.id)
    try:
        # Bulk delete relies on the database ON DELETE rules (calibrations,
        # observations and stream sessions cascade; anomaly/video links are set
        # null). ORM relationship nulling would try to write non-nullable FKs.
        db.execute(delete(Camera).where(Camera.id == camera.id))
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Could not delete camera {camera.code}: {exc}",
            details={"camera_id": camera.id},
        ) from exc
    return Message(message=f"camera {camera.code} deleted")


@router.get("/{camera_id}/status", response_model=CameraStatusRead, summary="Camera health status")
def get_camera_status(camera_id: int, db: Session = Depends(get_db)) -> CameraStatusRead:
    camera = _get_camera(db, camera_id)
    status_info = manager.provider_status(camera)
    live = get_live_manager().status(camera.id)
    details = manager.provider_details(camera)
    resolved_format = details.get("resolved_format")
    is_live = details.get("is_live")
    return CameraStatusRead(
        camera_id=camera.id,
        code=camera.code,
        enabled=camera.enabled,
        connected=status_info.connected,
        healthy=status_info.healthy,
        message=status_info.message,
        last_frame_at=status_info.last_frame_at,
        frames_captured=status_info.frames_captured,
        checked_at=datetime.now(UTC),
        is_live=is_live if camera.camera_type == "youtube_live" else None,
        stream_state=live.state,
        stream_session_id=live.session_id,
        stream_recording_id=live.recording_id,
        frames_persisted=live.frames_persisted,
        frames_rejected=live.frames_rejected,
        reconnect_count=live.reconnect_count,
        last_error=live.last_error,
        last_rolling_classification=live.last_rolling_classification,
        last_event_at=live.last_event_at,
        resolved_quality=camera.stream_quality,
        resolved_format=resolved_format or camera.stream_format,
    )


@router.post(
    "/{camera_id}/stream/start",
    response_model=CameraStreamStatusRead,
    summary="Start in-process live capture for a camera",
)
def start_camera_stream(camera_id: int, db: Session = Depends(get_db)) -> CameraStreamStatusRead:
    camera = _get_camera(db, camera_id)
    if not camera.enabled:
        raise ValidationError(f"Camera {camera.code} is disabled", details={"camera_id": camera.id})
    if camera.camera_type not in {"rtsp", "usb", "video", "youtube_live"}:
        raise ValidationError(
            "Only live-capable cameras (rtsp/usb/video/youtube_live) have a stream",
            details={"camera_id": camera.id, "camera_type": camera.camera_type},
        )
    try:
        live = get_live_manager().start_camera(camera.id)
    except Exception as exc:
        raise DataUnavailableError(
            f"could not start live capture: {exc}", details={"camera_id": camera.id}
        ) from exc
    return CameraStreamStatusRead(**live.as_dict())


@router.post(
    "/{camera_id}/stream/stop",
    response_model=CameraStreamStatusRead,
    summary="Stop in-process live capture for a camera",
)
def stop_camera_stream(camera_id: int, db: Session = Depends(get_db)) -> CameraStreamStatusRead:
    _get_camera(db, camera_id)
    live = get_live_manager().stop_camera(camera_id)
    return CameraStreamStatusRead(**live.as_dict())


@router.get(
    "/{camera_id}/stream/status",
    response_model=CameraStreamStatusRead,
    summary="Live capture status for a camera",
)
def camera_stream_status(camera_id: int, db: Session = Depends(get_db)) -> CameraStreamStatusRead:
    _get_camera(db, camera_id)
    live = get_live_manager().status(camera_id)
    return CameraStreamStatusRead(**live.as_dict())


@router.post(
    "/{camera_id}/test-connection",
    response_model=CameraTestConnectionResult,
    summary="Open the camera, read one frame, report stream health",
)
def test_camera_connection(
    camera_id: int, db: Session = Depends(get_db)
) -> CameraTestConnectionResult:
    camera = _get_camera(db, camera_id)
    ok, message, latency_ms, frame, quality = manager.test_connection(camera)
    extra = frame.metadata.extra if frame is not None else {}
    source_redacted = (
        redact_youtube_url(camera.youtube_url)
        if camera.camera_type == "youtube_live"
        else manager.redact_rtsp_url(camera.rtsp_url)
    )
    return CameraTestConnectionResult(
        camera_id=camera.id,
        ok=ok,
        message=message,
        latency_ms=latency_ms,
        frame_width=frame.metadata.width if frame is not None else None,
        frame_height=frame.metadata.height if frame is not None else None,
        captured_at=frame.captured_at if frame is not None else None,
        quality=_quality_read(quality) if quality is not None else None,
        is_live=extra.get("is_live"),
        resolved_format=extra.get("format_id") or extra.get("format_note"),
        resolved_quality=(
            str(extra.get("resolved_height"))
            if extra.get("resolved_height")
            else camera.stream_quality
        ),
        source_url_redacted=source_redacted,
    )


@router.get(
    "/{camera_id}/frame",
    summary="Capture and return the latest frame as JPEG",
    responses={200: {"content": {"image/jpeg": {}}}},
)
def get_camera_frame(camera_id: int, db: Session = Depends(get_db)) -> Response:
    camera = _get_camera(db, camera_id)
    if not camera.enabled:
        raise ValidationError(f"Camera {camera.code} is disabled", details={"camera_id": camera.id})
    try:
        frame, quality = manager.capture_frame(camera)
    except FrameUnavailableError as exc:
        # Unavailable data is never silently treated as a normal frame.
        raise DataUnavailableError(str(exc), details={"camera_id": camera.id}) from exc
    except CameraError as exc:
        raise DataUnavailableError(str(exc), details={"camera_id": camera.id}) from exc
    jpeg = manager.encode_jpeg(frame.image)
    headers = {
        "X-Camera-Code": camera.code,
        "X-Frame-Sequence": str(frame.sequence),
        "X-Frame-Captured-At": frame.captured_at.isoformat(),
        "X-Quality-Score": str(quality.score),
        "X-Quality-Flags": ",".join(quality.flags),
    }
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)


@router.get(
    "/{camera_id}/shoreline-preview",
    summary="Run shoreline segmentation on the latest camera frame and return an overlay",
    responses={200: {"content": {"image/jpeg": {}}}},
)
def get_camera_shoreline_preview(
    camera_id: int,
    mask: bool = False,
    db: Session = Depends(get_db),
) -> Response:
    """Serve the live worker's most recent sampled frame.

    This endpoint is **cache-only**: it never captures a frame and never runs the
    model. The live capture worker samples one frame per configured interval and
    publishes the already-scored frame/mask/shoreline here, so preview latency is
    JPEG encode + network only.

    ``mask=false`` (default) returns the raw frame and exposes the extracted
    Ocean/Beach line in ``X-Shoreline-Polyline`` for the browser to draw.
    ``mask=true`` keeps the server-rendered tint+line image. This endpoint is
    intentionally read-only: it does not persist an observation.
    """
    handler_started = time.perf_counter()
    camera = _get_camera(db, camera_id)
    if not camera.enabled:
        raise ValidationError(f"Camera {camera.code} is disabled", details={"camera_id": camera.id})
    settings = get_settings()
    outline = load_active_roi(db, camera)
    outline_present = outline is not None
    interval = float(settings.live_capture_interval_seconds)
    effective_max_age = max(
        float(settings.preview_cache_max_age_seconds), interval * 2.5
    )

    cache_started = time.perf_counter()
    entry = get_preview_cache().get(camera.id, max_age_seconds=effective_max_age)
    stale = False
    if entry is None:
        # A stopped/failed worker may still have its last frame. Serve it with a
        # stale marker; a running worker that has produced nothing gets a 503.
        # A cache miss never triggers capture or inference.
        last = get_preview_cache().get(camera.id, max_age_seconds=24 * 3600.0)
        if last is not None and camera.id not in get_live_manager().running_camera_ids():
            entry = last
            stale = True
    if entry is None:
        raise PreviewNotReadyError(
            "The live worker has not produced a frame yet",
            details={"camera_id": camera.id, "reason": "preview_not_ready"},
            headers={"Retry-After": str(max(1, min(int(interval), 5)))},
        )

    return _render_preview(
        entry.frame,
        entry.segmentation,
        entry.shoreline,
        mask=mask,
        color_map=DEFAULT_CLASS_COLOR_MAP,
        source="worker",
        cache_age_ms=entry.age_ms(),
        capture_ms=(time.perf_counter() - cache_started) * 1000.0,
        capture_age_ms=entry.captured_age_ms,
        predict_ms=0.0,
        onnx_ms=float(entry.inference_ms or 0.0),
        extract_ms=0.0,
        observation_id=entry.observation_id,
        outline_present=outline_present,
        outline_matches=(entry.monitoring_outline_present == outline_present),
        stale=stale,
        persisted=entry.persisted,
        sample_interval_ms=interval * 1000.0,
        handler_started=handler_started,
    )


def _polyline_header(segments: list[Any], *, max_points: int = 512) -> str:
    """Compact display polyline: ``x,y x,y;x,y ...``.

    Decimated only when a full-resolution line would exceed a safe response
    header size. The scored shoreline geometry remains untouched on disk/API.
    """
    total = sum(len(segment) for segment in segments)
    stride = 1 if total <= max_points else max(1, (total + max_points - 1) // max_points)
    parts: list[str] = []
    for segment in segments:
        if not segment:
            continue
        chosen = segment[::stride]
        parts.append(
            " ".join(
                f"{int(round(float(point[0])))},{int(round(float(point[1])))}" for point in chosen
            )
        )
    return ";".join(parts)


def _render_preview(
    frame: Any,
    segmentation: Any,
    shoreline: Any,
    *,
    mask: bool,
    color_map: dict[str, Any],
    source: str,
    cache_age_ms: float | None,
    capture_ms: float,
    capture_age_ms: float,
    predict_ms: float,
    onnx_ms: float,
    extract_ms: float,
    observation_id: str | None,
    outline_present: bool,
    outline_matches: bool = True,
    stale: bool = False,
    persisted: bool = True,
    sample_interval_ms: float = 0.0,
    handler_started: float,
) -> Response:
    """Render the preview JPEG and attach the standard preview headers.

    ``mask=false`` returns the frame untouched plus the shoreline polyline in
    ``X-Shoreline-Polyline``; the browser draws the line. ``mask=true`` burns
    the class tint and the line into the image (legacy behaviour).
    """
    segments = list(getattr(shoreline, "segments", None) or [])
    shoreline_available = bool(getattr(shoreline, "shoreline_available", False))
    segment_count = len(segments)
    polyline = _polyline_header(segments)
    overlay_started = time.perf_counter()
    if mask:
        overlay = frame.image.copy()
        for class_id, name in segmentation.classes.items():
            pixels = segmentation.class_mask == class_id
            if not np.any(pixels):
                continue
            rgb = np.asarray(color_map.get(name, [128, 128, 128]), dtype=np.float32)
            tint = np.zeros_like(overlay)
            tint[:, :] = rgb[::-1]  # RGB -> BGR
            blended = cv2.addWeighted(overlay, 0.62, tint, 0.38, 0)
            overlay[pixels] = blended[pixels]
        if shoreline_available and segments:
            for segment in segments:
                points = np.asarray(segment, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    overlay,
                    [points],
                    isClosed=False,
                    color=(0, 245, 255),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
    else:
        overlay = frame.image
    overlay_ms = (time.perf_counter() - overlay_started) * 1000.0

    encode_started = time.perf_counter()
    jpeg = manager.encode_jpeg(overlay)
    encode_ms = (time.perf_counter() - encode_started) * 1000.0
    response_age_ms = (datetime.now(UTC) - frame.captured_at).total_seconds() * 1000.0
    total_ms = (time.perf_counter() - handler_started) * 1000.0
    server_timing = ", ".join(
        f"{name};dur={value:.1f}"
        for name, value in (
            ("capture", capture_ms),
            ("capture_age", capture_age_ms),
            ("predict", predict_ms),
            ("onnx", onnx_ms),
            ("extract", extract_ms),
            ("overlay", overlay_ms),
            ("encode", encode_ms),
            ("response_age", response_age_ms),
            ("total", total_ms),
        )
    )

    headers = {
        "X-Segmentation-Confidence": str(segmentation.confidence),
        "X-Inference-Ms": str(segmentation.inference_ms or 0),
        "X-Frame-Captured-At": frame.captured_at.isoformat(),
        "X-Frame-Width": str(frame.image.shape[1]),
        "X-Frame-Height": str(frame.image.shape[0]),
        "X-Model-Architecture": str(segmentation.architecture or ""),
        "X-Model-Encoder": str(segmentation.encoder or ""),
        "X-Model-Checkpoint-Sha256": str(segmentation.checkpoint_sha256 or ""),
        "X-Shoreline-Available": "1" if shoreline_available else "0",
        "X-Shoreline-Segments": str(segment_count),
        "X-Shoreline-Axis": str(getattr(shoreline, "shoreline_axis", None) or ""),
        "X-Shoreline-Interpolated-Ranges": str(
            len(getattr(shoreline, "interpolated_ranges", None) or [])
        ),
        "X-Shoreline-Measured-Coverage": (
            ""
            if getattr(shoreline, "measured_coverage", None) is None
            else str(shoreline.measured_coverage)
        ),
        "X-Shoreline-Reason": str(getattr(shoreline, "reason", None) or ""),
        "X-Shoreline-Polyline": polyline,
        "X-Monitoring-Outline": "1" if outline_present else "0",
        "X-Monitoring-Outline-Matches": "1" if outline_matches else "0",
        "X-Preview-Source": source,
        "X-Preview-Age-Ms": (
            f"{(cache_age_ms if cache_age_ms is not None else response_age_ms):.1f}"
        ),
        "X-Preview-Captured-At": frame.captured_at.isoformat(),
        "X-Preview-Sample-Interval-Ms": f"{sample_interval_ms:.0f}",
        "X-Preview-Stale": "1" if stale else "0",
        "X-Preview-Persisted": "1" if persisted else "0",
        "Server-Timing": server_timing,
        "Access-Control-Expose-Headers": (
            "X-Segmentation-Confidence, X-Inference-Ms, X-Frame-Captured-At, "
            "X-Frame-Width, X-Frame-Height, X-Model-Architecture, X-Model-Encoder, "
            "X-Model-Checkpoint-Sha256, X-Shoreline-Available, X-Shoreline-Segments, "
            "X-Shoreline-Axis, X-Shoreline-Interpolated-Ranges, "
            "X-Shoreline-Measured-Coverage, X-Shoreline-Reason, X-Shoreline-Polyline, "
            "X-Monitoring-Outline, X-Monitoring-Outline-Matches, X-Preview-Source, "
            "X-Preview-Age-Ms, X-Preview-Cache-Age-Ms, X-Preview-Observation-Id, "
            "X-Preview-Captured-At, X-Preview-Sample-Interval-Ms, X-Preview-Stale, "
            "X-Preview-Persisted, Server-Timing"
        ),
    }
    if cache_age_ms is not None:
        headers["X-Preview-Cache-Age-Ms"] = f"{cache_age_ms:.1f}"
    if observation_id is not None:
        headers["X-Preview-Observation-Id"] = str(observation_id)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers=headers,
    )


@router.post(
    "/{camera_id}/upload-frame",
    response_model=Message,
    summary="Upload a still frame (upload-type cameras only)",
)
async def upload_camera_frame(
    camera_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Message:
    camera = _get_camera(db, camera_id)
    if camera.camera_type != "upload":
        raise ValidationError(
            "Only upload-type cameras accept still uploads",
            details={"camera_id": camera.id, "camera_type": camera.camera_type},
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValidationError(
            f"Unsupported upload type {suffix!r}; allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
        )
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise ValidationError(
            f"Upload too large ({len(content)} bytes; max {MAX_UPLOAD_BYTES}).",
            details={"max_bytes": MAX_UPLOAD_BYTES},
        )
    # Verify the payload is a decodable image before persisting it.
    buffer = np.frombuffer(content, dtype=np.uint8)
    decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValidationError("Uploaded file is not a decodable JPEG/PNG image")

    settings = get_settings()
    dest_dir = camera_upload_dir(settings.data_dir, camera.id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    dest = dest_dir / f"{stamp}{suffix}"
    dest.write_bytes(content)
    logger.info("stored uploaded frame for camera %s at %s", camera.code, dest)
    return Message(message=f"frame stored for camera {camera.code}")


@router.post(
    "/{camera_id}/upload-video",
    response_model=Message,
    summary="Upload a video file (video-type cameras only) — becomes the camera feed",
)
async def upload_camera_video(
    camera_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> Message:
    camera = _get_camera(db, camera_id)
    if camera.camera_type != "video":
        raise ValidationError(
            "Only video-type cameras accept video uploads",
            details={"camera_id": camera.id, "camera_type": camera.camera_type},
        )
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValidationError(
            f"Unsupported video type {suffix!r}; allowed: {sorted(ALLOWED_VIDEO_EXTENSIONS)}"
        )
    content = await file.read()
    if len(content) > MAX_VIDEO_UPLOAD_BYTES:
        raise ValidationError(
            f"Upload too large ({len(content)} bytes; max {MAX_VIDEO_UPLOAD_BYTES}).",
            details={"max_bytes": MAX_VIDEO_UPLOAD_BYTES},
        )

    settings = get_settings()
    dest_dir = video_dir(settings.data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
    original = Path(file.filename or f"video{suffix}").stem or "video"
    safe_stem = (
        "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in original)[:80] or "video"
    )
    dest = dest_dir / f"{stamp}_{safe_stem}{suffix}"
    dest.write_bytes(content)
    try:
        dest.resolve().relative_to(dest_dir.resolve())
    except ValueError as exc:  # pragma: no cover - defensive
        dest.unlink(missing_ok=True)
        raise ValidationError("Invalid destination for uploaded video") from exc

    # Drop any cached provider so the next read opens the new file.
    manager.drop_provider(camera.id)
    camera.video_path = dest.name
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        dest.unlink(missing_ok=True)
        raise ConflictError("Could not link uploaded video to the camera") from exc
    logger.info("stored video for camera %s at %s", camera.code, dest)
    return Message(message=f"video stored for camera {camera.code} as {dest.name}")


# ---------------------------------------------------------------------------
# Calibration profiles
# ---------------------------------------------------------------------------


def _calibration_read(cal: CameraCalibration) -> CameraCalibrationRead:
    return CameraCalibrationRead(
        id=cal.id,
        camera_id=cal.camera_id,
        version=cal.version,
        reference_points=cal.reference_points,
        homography_matrix=cal.homography_matrix,
        meters_per_pixel=cal.meters_per_pixel,
        roi_polygon=(RoiPolygon.model_validate(cal.roi_polygon) if cal.roi_polygon else None),
        is_active=cal.is_active,
        valid_from=cal.valid_from,
        valid_to=cal.valid_to,
        notes=cal.notes,
        created_at=cal.created_at,
    )


@router.get(
    "/{camera_id}/calibrations",
    response_model=list[CameraCalibrationRead],
    summary="List calibration profiles for a camera",
)
def list_calibrations(camera_id: int, db: Session = Depends(get_db)) -> list[CameraCalibrationRead]:
    _get_camera(db, camera_id)  # 404 if unknown
    calibrations = db.scalars(
        select(CameraCalibration)
        .where(CameraCalibration.camera_id == camera_id)
        .order_by(CameraCalibration.version.desc())
    ).all()
    return [_calibration_read(c) for c in calibrations]


@router.post(
    "/{camera_id}/calibrations",
    response_model=CameraCalibrationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a calibration profile (next version)",
)
def create_calibration(
    camera_id: int,
    payload: CameraCalibrationCreate,
    db: Session = Depends(get_db),
) -> CameraCalibrationRead:
    camera = _get_camera(db, camera_id)
    latest = db.scalar(
        select(func.max(CameraCalibration.version)).where(CameraCalibration.camera_id == camera.id)
    )
    version = (latest or 0) + 1

    if payload.set_active:
        # Demote existing profiles so exactly one is active.
        db.execute(
            update(CameraCalibration)
            .where(
                CameraCalibration.camera_id == camera.id,
                CameraCalibration.is_active.is_(True),
            )
            .values(is_active=False)
        )

    calibration = CameraCalibration(
        camera_id=camera.id,
        version=version,
        reference_points=payload.reference_points,
        homography_matrix=payload.homography_matrix,
        meters_per_pixel=payload.meters_per_pixel,
        roi_polygon=(payload.roi_polygon.model_dump() if payload.roi_polygon is not None else None),
        valid_from=payload.valid_from,
        valid_to=payload.valid_to,
        notes=payload.notes,
        is_active=payload.set_active,
    )
    db.add(calibration)
    db.flush()  # assign id before pointing the camera at it
    if payload.set_active:
        camera.active_calibration_id = calibration.id
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Could not store calibration for camera {camera.code}",
            details={"camera_id": camera.id},
        ) from exc
    db.refresh(calibration)
    return _calibration_read(calibration)


@router.post(
    "/{camera_id}/calibrations/{calibration_id}/roi",
    response_model=CameraCalibrationRead,
    status_code=status.HTTP_201_CREATED,
    summary="Save a monitoring outline as the next calibration version",
)
def save_calibration_roi(
    camera_id: int,
    calibration_id: int,
    payload: RoiPolygonSave,
    db: Session = Depends(get_db),
) -> CameraCalibrationRead:
    """Append a calibration version carrying the operator's monitoring outline.

    Calibration values are copied from the source profile, so saving an outline
    never discards homography, scale or notes. Sending ``roi_polygon: null``
    creates a version with no outline (full-frame behaviour).
    """
    camera = _get_camera(db, camera_id)
    source = db.get(CameraCalibration, calibration_id)
    if source is None or source.camera_id != camera.id:
        raise NotFoundError(
            f"Calibration {calibration_id} not found for camera {camera.id}",
            details={"camera_id": camera.id, "calibration_id": calibration_id},
        )

    latest = db.scalar(
        select(func.max(CameraCalibration.version)).where(CameraCalibration.camera_id == camera.id)
    )
    version = (latest or 0) + 1
    notes = source.notes
    if payload.notes:
        extra = payload.notes.strip()
        if extra:
            notes = f"{notes}\n{extra}" if notes else extra

    db.execute(
        update(CameraCalibration)
        .where(
            CameraCalibration.camera_id == camera.id,
            CameraCalibration.is_active.is_(True),
        )
        .values(is_active=False)
    )
    calibration = CameraCalibration(
        camera_id=camera.id,
        version=version,
        reference_points=source.reference_points,
        homography_matrix=source.homography_matrix,
        meters_per_pixel=source.meters_per_pixel,
        roi_polygon=(payload.roi_polygon.model_dump() if payload.roi_polygon is not None else None),
        valid_from=source.valid_from,
        valid_to=source.valid_to,
        notes=notes,
        is_active=True,
    )
    db.add(calibration)
    db.flush()
    camera.active_calibration_id = calibration.id
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise ConflictError(
            f"Could not store monitoring outline for camera {camera.code}",
            details={"camera_id": camera.id},
        ) from exc
    db.refresh(calibration)
    logger.info(
        "camera %s calibration v%s monitoring outline saved (%s vertices)",
        camera.code,
        calibration.version,
        0 if calibration.roi_polygon is None else len(calibration.roi_polygon.get("points", [])),
    )
    return _calibration_read(calibration)
