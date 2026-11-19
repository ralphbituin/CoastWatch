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
    drop_all_providers,
    drop_provider,
    provider_status,
    test_connection,
)
from app.services.camera.rtsp import RTSPCameraProvider
from app.services.camera.upload import UploadCameraProvider
from app.services.camera.usb import USBCameraProvider
from app.services.camera.video import (
    ALLOWED_VIDEO_EXTENSIONS,
    VideoFileCameraProvider,
    resolve_video_path,
    video_dir,
)

__all__ = [
    "ALLOWED_VIDEO_EXTENSIONS",
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
    "VideoFileCameraProvider",
    "build_provider",
    "capture_frame",
    "drop_all_providers",
    "drop_provider",
    "provider_status",
    "resolve_video_path",
    "test_connection",
    "video_dir",
]
