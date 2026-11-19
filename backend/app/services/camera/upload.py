"""Upload camera provider (Phase 3).

For sites where a still image is dropped in by operators (or an upstream
script) instead of a live RTSP/USB feed. Frames are read from a per-camera
upload directory under ``DATA_DIR/uploads/cameras/<camera_id>/``.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import cv2

from app.core.logging import get_logger
from app.services.camera.base import (
    CameraMetadata,
    CameraProvider,
    CameraStatus,
    Frame,
    FrameUnavailableError,
)

logger = get_logger(__name__)

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def camera_upload_dir(data_dir: str | Path, camera_id: int) -> Path:
    return Path(data_dir) / "uploads" / "cameras" / str(camera_id)


def latest_upload(upload_dir: Path) -> Path | None:
    if not upload_dir.is_dir():
        return None
    candidates = [
        p
        for p in upload_dir.iterdir()
        if p.is_file() and p.suffix.lower() in ALLOWED_UPLOAD_EXTENSIONS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


class UploadCameraProvider(CameraProvider):
    """Serves the most recently uploaded still frame for a camera."""

    def __init__(self, upload_dir: Path, *, camera_id: int = 0) -> None:
        self._upload_dir = Path(upload_dir)
        self._camera_id = camera_id
        self._lock = threading.Lock()
        self._frames_captured = 0
        self._last_frame_at: datetime | None = None
        self._sequence = 0

    def connect(self) -> None:
        # Uploads land on disk asynchronously; there is nothing to open.
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info("upload camera %s ready (dir %s)", self._camera_id, self._upload_dir)

    def disconnect(self) -> None:
        return None

    def read_frame(self) -> Frame:
        with self._lock:
            path = latest_upload(self._upload_dir)
            if path is None:
                raise FrameUnavailableError(
                    f"no uploaded frame for camera {self._camera_id} "
                    f"(upload a still to {self._upload_dir})"
                )
            image = cv2.imread(str(path))
            if image is None:
                raise FrameUnavailableError(f"uploaded frame could not be decoded: {path.name}")
            self._frames_captured += 1
            self._sequence += 1
            self._last_frame_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
            height, width = image.shape[:2]
            return Frame(
                image=image,
                captured_at=self._last_frame_at,
                metadata=CameraMetadata(
                    camera_id=self._camera_id,
                    camera_type="upload",
                    width=int(width),
                    height=int(height),
                    extra={"source_file": path.name},
                ),
                sequence=self._sequence,
            )

    def get_status(self) -> CameraStatus:
        with self._lock:
            path = latest_upload(self._upload_dir)
            if path is None:
                return CameraStatus(
                    connected=False,
                    healthy=False,
                    message="no uploaded frames yet",
                    last_frame_at=None,
                    frames_captured=self._frames_captured,
                )
            return CameraStatus(
                connected=True,
                healthy=True,
                message=f"serving latest upload ({path.name})",
                last_frame_at=datetime.fromtimestamp(path.stat().st_mtime, tz=UTC),
                frames_captured=self._frames_captured,
            )

    def get_metadata(self) -> CameraMetadata:
        path = latest_upload(self._upload_dir)
        width = height = None
        if path is not None:
            image = cv2.imread(str(path))
            if image is not None:
                height, width = image.shape[:2]
        return CameraMetadata(
            camera_id=self._camera_id,
            camera_type="upload",
            width=width,
            height=height,
        )
