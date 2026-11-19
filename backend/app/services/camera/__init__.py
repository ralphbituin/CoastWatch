"""Camera acquisition providers (Phase 3)."""

from app.services.camera.base import (
    CameraConnectionError,
    CameraError,
    CameraMetadata,
    CameraProvider,
    CameraStatus,
    Frame,
    FrameUnavailableError,
)
from app.services.camera.manager import (
    build_provider,
    capture_frame,
    drop_provider,
    provider_status,
    test_connection,
)
from app.services.camera.rtsp import RTSPCameraProvider
from app.services.camera.upload import UploadCameraProvider
from app.services.camera.usb import USBCameraProvider

__all__ = [
    "CameraConnectionError",
    "CameraError",
    "CameraMetadata",
    "CameraProvider",
    "CameraStatus",
    "Frame",
    "FrameUnavailableError",
    "RTSPCameraProvider",
    "USBCameraProvider",
    "UploadCameraProvider",
    "build_provider",
    "capture_frame",
    "drop_provider",
    "provider_status",
    "test_connection",
]
