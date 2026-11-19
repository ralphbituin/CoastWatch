"""Video-file camera provider (Phase 4 dev tooling).

Streams frames from a video file (mp4/mov/avi) as if it were a live camera.
Used to build a "virtual environment" on a workstation: recorded coastal
footage feeds the same provider interface the pipeline and future model
training consume — no physical camera required.

Files live under ``DATA_DIR/videos/`` by convention; the stored path is
relative to that directory. Playback loops by default so a short clip can
stand in for a continuous feed (timestamps advance virtually from a fixed
epoch anchor).
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2

from app.core.logging import get_logger
from app.services.camera.base import (
    CameraConnectionError,
    CameraMetadata,
    CameraProvider,
    CameraStatus,
    Frame,
    FrameUnavailableError,
)

logger = get_logger(__name__)

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def video_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "videos"


def resolve_video_path(data_dir: str | Path, relative_path: str) -> Path:
    """Resolve ``relative_path`` inside ``video_dir``; reject traversal."""
    base = video_dir(data_dir).resolve()
    candidate = (base / relative_path).resolve()
    if not candidate.is_relative_to(base):
        raise CameraConnectionError(f"video_path escapes the videos directory: {relative_path!r}")
    return candidate


class VideoFileCameraProvider(CameraProvider):
    """Loops a video file, emitting one frame per read like a live camera."""

    def __init__(
        self,
        video_path: Path,
        *,
        camera_id: int = 0,
        loop: bool = True,
        fps: float | None = None,
    ) -> None:
        self._path = Path(video_path)
        self._camera_id = camera_id
        self._loop = loop
        self._requested_fps = fps

        self._cap: Any | None = None
        self._lock = threading.Lock()
        self._frames_captured = 0
        self._frame_index = 0
        self._loop_count = 0
        self._eof_reached = False
        self._last_frame_at: datetime | None = None
        self._sequence = 0
        self._last_error: str | None = None
        self._anchor = datetime.now(UTC)

    # ------------------------------------------------------------------
    def connect(self) -> None:
        with self._lock:
            self._open()

    def _open(self) -> None:
        if not self._path.is_file():
            self._last_error = f"video file not found: {self._path}"
            raise CameraConnectionError(self._last_error)
        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            cap.release()
            self._last_error = f"could not open video file: {self._path.name}"
            raise CameraConnectionError(self._last_error)
        self._cap = cap
        self._frame_index = 0
        self._eof_reached = False
        self._last_error = None
        logger.info("video camera %s opened (%s)", self._camera_id, self._path.name)

    def disconnect(self) -> None:
        with self._lock:
            self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _effective_fps(self) -> float:
        if self._requested_fps and self._requested_fps > 0:
            return float(self._requested_fps)
        if self._cap is not None:
            raw = self._cap.get(cv2.CAP_PROP_FPS)
            if raw and raw > 1e-2:
                return float(raw)
        return 1.0

    @property
    def frame_count(self) -> int:
        if self._cap is not None:
            count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            return max(count, 0)
        return 0

    def read_frame(self) -> Frame:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                self._open()
            assert self._cap is not None
            ok, image = self._cap.read()
            if not ok or image is None:
                if self._loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._loop_count += 1
                    self._frame_index = 0
                    ok, image = self._cap.read()
                if not ok or image is None:
                    self._eof_reached = True
                    self._last_error = "end of video stream"
                    raise FrameUnavailableError(
                        f"video file {self._path.name} exhausted and looping is disabled"
                    )
            fps = self._effective_fps()
            # Virtual timestamps: each frame is spaced 1/fps apart from the anchor.
            position = self._loop_count * max(self.frame_count, 1) + self._frame_index
            self._last_frame_at = datetime.fromtimestamp(
                self._anchor.timestamp() + position / fps, tz=UTC
            )
            height, width = image.shape[:2]
            self._frame_index += 1
            self._frames_captured += 1
            self._sequence += 1
            return Frame(
                image=image,
                captured_at=self._last_frame_at,
                metadata=CameraMetadata(
                    camera_id=self._camera_id,
                    camera_type="video",
                    width=int(width),
                    height=int(height),
                    fps=fps,
                    extra={
                        "source_file": self._path.name,
                        "frame_index": self._frame_index - 1,
                        "loop_count": self._loop_count,
                    },
                ),
                sequence=self._sequence,
            )

    def get_status(self) -> CameraStatus:
        with self._lock:
            connected = self._cap is not None and bool(self._cap.isOpened())
            if self._eof_reached:
                message = "end of video stream"
            elif connected:
                total = self.frame_count
                position = min(self._frame_index, total)
                message = (
                    f"playing {self._path.name} "
                    f"(frame {position}/{total or '?'}, loop {self._loop_count})"
                )
            elif self._last_error:
                message = self._last_error
            else:
                message = "not connected"
            return CameraStatus(
                connected=connected,
                healthy=connected,
                message=message,
                last_frame_at=self._last_frame_at,
                frames_captured=self._frames_captured,
            )

    def get_metadata(self) -> CameraMetadata:
        width = height = fps = None
        if self._cap is not None and self._cap.isOpened():
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            fps = self._effective_fps()
        return CameraMetadata(
            camera_id=self._camera_id,
            camera_type="video",
            width=width,
            height=height,
            fps=fps,
            extra={"source_file": self._path.name},
        )
