"""USB camera provider (Phase 3).

Planned responsibilities: device-index selection, resolution / frame-rate
configuration and device connection monitoring via OpenCV ``VideoCapture``.
"""

from __future__ import annotations

from app.services.camera.base import CameraMetadata, CameraProvider, CameraStatus, Frame


class USBCameraProvider(CameraProvider):
    def __init__(
        self,
        device_index: int,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
    ) -> None:
        self._device_index = device_index
        self._width = width
        self._height = height
        self._fps = fps

    def connect(self) -> None:
        raise NotImplementedError("USBCameraProvider is implemented in Phase 3.")

    def disconnect(self) -> None:
        raise NotImplementedError("USBCameraProvider is implemented in Phase 3.")

    def read_frame(self) -> Frame | None:
        raise NotImplementedError("USBCameraProvider is implemented in Phase 3.")

    def get_status(self) -> CameraStatus:
        raise NotImplementedError("USBCameraProvider is implemented in Phase 3.")

    def get_metadata(self) -> CameraMetadata:
        return CameraMetadata(
            camera_id=self._device_index,
            camera_type="usb",
            width=self._width,
            height=self._height,
            fps=self._fps,
        )
