"""USB camera provider (Phase 3).

Device-index selection, resolution / frame-rate configuration and device
monitoring via OpenCV ``VideoCapture``.
"""

from __future__ import annotations

import threading
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

_MAX_CONSECUTIVE_FAILURES = 3


class USBCameraProvider(CameraProvider):
    def __init__(
        self,
        device_index: int,
        *,
        camera_id: int | None = None,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        if device_index is None or device_index < 0:
            raise ValueError("device_index must be a non-negative integer")
        self._device_index = device_index
        self._camera_id = camera_id if camera_id is not None else device_index
        self._width = width
        self._height = height
        self._fps = fps

        self._cap: Any | None = None
        self._lock = threading.Lock()
        self._frames_captured = 0
        self._last_frame_at: datetime | None = None
        self._consecutive_failures = 0
        self._sequence = 0
        self._last_error: str | None = None

    def connect(self) -> None:
        with self._lock:
            self._release()
            cap = cv2.VideoCapture(self._device_index, cv2.CAP_ANY)
            if self._width is not None:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            if self._height is not None:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            if self._fps is not None:
                cap.set(cv2.CAP_PROP_FPS, self._fps)
            if not cap.isOpened():
                cap.release()
                self._last_error = f"USB device index {self._device_index} could not be opened"
                raise CameraConnectionError(self._last_error)
            self._cap = cap
            self._consecutive_failures = 0
            self._last_error = None
            logger.info("usb camera %s (device %s) connected", self._camera_id, self._device_index)

    def disconnect(self) -> None:
        with self._lock:
            self._release()

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    def _ensure_connected(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            return
        self._release()
        cap = cv2.VideoCapture(self._device_index, cv2.CAP_ANY)
        if self._width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        if self._height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps is not None:
            cap.set(cv2.CAP_PROP_FPS, self._fps)
        if not cap.isOpened():
            cap.release()
            self._last_error = f"USB device index {self._device_index} could not be opened"
            raise CameraConnectionError(self._last_error)
        self._cap = cap
        self._consecutive_failures = 0
        self._last_error = None

    def read_frame(self) -> Frame:
        with self._lock:
            self._ensure_connected()
            assert self._cap is not None
            ok, image = self._cap.read()
            if not ok or image is None:
                self._consecutive_failures += 1
                self._last_error = "frame read failed"
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    self._release()
                raise FrameUnavailableError(
                    f"could not read a frame from USB device {self._device_index} "
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

    def get_status(self) -> CameraStatus:
        with self._lock:
            connected = self._cap is not None and bool(self._cap.isOpened())
            healthy = connected and self._consecutive_failures == 0
            if connected and self._last_frame_at is not None:
                message = "device healthy"
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
        width = self._width
        height = self._height
        fps = self._fps
        if self._cap is not None and self._cap.isOpened():
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            raw_fps = self._cap.get(cv2.CAP_PROP_FPS)
            if raw_fps and raw_fps > 0:
                fps = float(raw_fps)
        return CameraMetadata(
            camera_id=self._camera_id,
            camera_type="usb",
            width=width,
            height=height,
            fps=fps,
            extra={"device_index": self._device_index},
        )
