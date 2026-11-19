"""Camera routes: connection, status, frames and calibration profiles."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import Response
from geoalchemy2.elements import WKTElement
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_db
from app.core.config import get_settings
from app.core.exceptions import (
    ConflictError,
    DataUnavailableError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.camera import Camera, CameraCalibration
from app.models.station import MonitoringStation
from app.schemas.cameras import (
    CameraCalibrationCreate,
    CameraCalibrationRead,
    CameraCreate,
    CameraQualityRead,
    CameraRead,
    CameraStatusRead,
    CameraTestConnectionResult,
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
from app.services.vision.preprocessing import FrameQuality

logger = get_logger(__name__)

router = APIRouter(tags=["cameras"])

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB


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


@router.get("/{camera_id}/status", response_model=CameraStatusRead, summary="Camera health status")
def get_camera_status(camera_id: int, db: Session = Depends(get_db)) -> CameraStatusRead:
    camera = _get_camera(db, camera_id)
    status_info = manager.provider_status(camera)
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
    )


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
    return CameraTestConnectionResult(
        camera_id=camera.id,
        ok=ok,
        message=message,
        latency_ms=latency_ms,
        frame_width=frame.metadata.width if frame is not None else None,
        frame_height=frame.metadata.height if frame is not None else None,
        captured_at=frame.captured_at if frame is not None else None,
        quality=_quality_read(quality) if quality is not None else None,
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
