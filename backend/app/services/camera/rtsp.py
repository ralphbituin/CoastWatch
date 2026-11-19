"""RTSP camera provider (Phase 3).

RTSP URL configuration, connection timeout, automatic reconnection and
stream health monitoring. Frames are acquired via OpenCV's FFMPEG-backed
``VideoCapture``.
"""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
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

# After this many consecutive read failures the provider drops the connection
# and reconnects on the next read/connect call.
_MAX_CONSECUTIVE_FAILURES = 3


class RTSPCameraProvider(CameraProvider):
    def __init__(
        self,
        url: str,
        *,
        camera_id: int = 0,
        timeout_s: float = 5.0,
        max_reconnect_attempts: int = 3,
        reconnect_delay_s: float = 0.5,
    ) -> None:
        if not url or not url.strip():
            raise ValueError("rtsp url must not be blank")
        self._url = url
        self._camera_id = camera_id
        self._timeout_s = timeout_s
        self._max_reconnect_attempts = max(1, max_reconnect_attempts)
        self._reconnect_delay_s = max(0.0, reconnect_delay_s)

        self._cap: Any | None = None
        self._lock = threading.Lock()
        self._frames_captured = 0
        self._last_frame_at: datetime | None = None
        self._consecutive_failures = 0
        self._sequence = 0
        self._last_error: str | None = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------
    def connect(self) -> None:
        with self._lock:
            self._open_capture()

    def disconnect(self) -> None:
        with self._lock:
            self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _open_capture(self) -> None:
        self._release()
        cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        # Best-effort open timeout (supported by the FFMPEG backend on
        # OpenCV >= 4.8). Ignored silently on builds that lack the property.
        try:
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self._timeout_s * 1000))
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self._timeout_s * 1000))
        except Exception:  # pragma: no cover - backend dependent
            logger.debug("rtsp timeout properties not supported by this OpenCV build")
        if not cap.isOpened():
            cap.release()
            self._last_error = f"could not open RTSP stream (timeout {self._timeout_s}s)"
            raise CameraConnectionError(self._last_error)
        self._cap = cap
        self._consecutive_failures = 0
        self._last_error = None
        logger.info("rtsp camera %s connected", self._camera_id)

    def _ensure_connected(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        attempts = 0
        while attempts < self._max_reconnect_attempts:
            attempts += 1
            try:
                self._open_capture()
                return
            except CameraConnectionError:
                if attempts < self._max_reconnect_attempts:
                    time.sleep(self._reconnect_delay_s)
        raise CameraConnectionError(
            self._last_error or f"RTSP stream unavailable after {attempts} attempt(s)"
        )

    # ------------------------------------------------------------------
    # Frame acquisition
    # ------------------------------------------------------------------
    def read_frame(self) -> Frame:
        with self._lock:
            self._ensure_connected()
            assert self._cap is not None
            # Drain one stale frame so live streams return current imagery.
            self._cap.grab()
            ok, image = self._cap.read()
            if not ok or image is None:
                self._consecutive_failures += 1
                self._last_error = "frame read failed"
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    # Drop the stream; the next call reconnects from scratch.
                    self._release()
                raise FrameUnavailableError(
                    f"could not read a frame from the RTSP stream "
                    f"({self._consecutive_failures} consecutive failure(s))"
                )
            self._consecutive_failures = 0
            self._last_error = None
            self._frames_captured += 1
            self._sequence += 1
            self._last_frame_at = datetime.now(UTC)
            return Frame(
                image=image,
                captured_at=self._last_frame_at,
                metadata=self.get_metadata(),
                sequence=self._sequence,
            )

    # ------------------------------------------------------------------
    # Status / metadata
    # ------------------------------------------------------------------
    def get_status(self) -> CameraStatus:
        with self._lock:
            connected = self._cap is not None and bool(self._cap.isOpened())
            healthy = connected and self._consecutive_failures == 0
            if connected and self._last_frame_at is not None:
                message = "stream healthy"
            elif connected:
                message = "connected, no frames read yet"
            elif self._last_error:
                message = self._last_error
            else:
                message = "not connected"
            return CameraStatus(
                connected=connected,
                healthy=healthy,
                message=message,
                last_frame_at=self._last_frame_at,
                frames_captured=self._frames_captured,
            )

    def get_metadata(self) -> CameraMetadata:
        width = height = None
        fps = None
        if self._cap is not None and self._cap.isOpened():
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            fps = float(raw_fps) if raw_fps and raw_fps > 0 else None
        return CameraMetadata(
            camera_id=self._camera_id,
            camera_type="rtsp",
            width=width,
            height=height,
            fps=fps,
        )
