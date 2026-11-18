"""RTSP camera provider (Phase 3).

Planned responsibilities: RTSP URL configuration, connection timeout,
automatic reconnection, stream health monitoring and frame acquisition.
"""

from __future__ import annotations

from app.services.camera.base import CameraMetadata, CameraProvider, CameraStatus, Frame


class RTSPCameraProvider(CameraProvider):
    def __init__(self, url: str, *, camera_id: int = 0, timeout_s: float = 5.0) -> None:
        self._url = url
        self._camera_id = camera_id
        self._timeout_s = timeout_s

    def connect(self) -> None:
        raise NotImplementedError("RTSPCameraProvider is implemented in Phase 3.")

    def disconnect(self) -> None:
        raise NotImplementedError("RTSPCameraProvider is implemented in Phase 3.")

    def read_frame(self) -> Frame | None:
        raise NotImplementedError("RTSPCameraProvider is implemented in Phase 3.")

    def get_status(self) -> CameraStatus:
        raise NotImplementedError("RTSPCameraProvider is implemented in Phase 3.")

    def get_metadata(self) -> CameraMetadata:
        return CameraMetadata(camera_id=self._camera_id, camera_type="rtsp")
