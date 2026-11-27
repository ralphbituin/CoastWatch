"""Camera registry: builds providers from rows and caches live connections.

Route handlers call into this module — they never construct providers
themselves. Providers are cached per camera id so RTSP streams stay warm
across requests within one process; each process (worker) keeps its own
cache. Nothing here talks to the database; callers supply the ORM row.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any

import cv2

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.camera import Camera
from app.services.camera.base import (
    CameraConnectionError,
    CameraProvider,
    CameraStatus,
    Frame,
    FrameUnavailableError,
)
from app.services.camera.rtsp import RTSPCameraProvider
from app.services.camera.upload import UploadCameraProvider, camera_upload_dir
from app.services.camera.usb import USBCameraProvider
from app.services.camera.video import VideoFileCameraProvider, resolve_video_path
from app.services.camera.youtube import YouTubeLiveProvider, redact_youtube_url
from app.services.vision.preprocessing import (
    FrameQuality,
    PreprocessConfig,
    assess_quality,
    preprocess,
)

logger = get_logger(__name__)

_providers: dict[int, CameraProvider] = {}
_previous_frames: dict[int, Any] = {}
_previous_captured_at: dict[int, datetime] = {}
_lock = threading.Lock()


def build_provider(camera: Camera) -> CameraProvider:
    """Construct a fresh provider for ``camera`` (not cached)."""
    settings = get_settings()
    if camera.camera_type == "rtsp":
        if not camera.rtsp_url:
            raise CameraConnectionError(f"camera {camera.code} has no rtsp_url configured")
        return RTSPCameraProvider(camera.rtsp_url, camera_id=camera.id)
    if camera.camera_type == "usb":
        if camera.device_index is None:
            raise CameraConnectionError(f"camera {camera.code} has no device_index configured")
        return USBCameraProvider(
            camera.device_index,
            camera_id=camera.id,
            width=camera.resolution_width,
            height=camera.resolution_height,
            fps=camera.fps,
        )
    if camera.camera_type == "upload":
        return UploadCameraProvider(
            camera_upload_dir(settings.data_dir, camera.id), camera_id=camera.id
        )
    if camera.camera_type == "video":
        if not camera.video_path:
            raise CameraConnectionError(f"camera {camera.code} has no video_path configured")
        source = resolve_video_path(settings.data_dir, camera.video_path)
        return VideoFileCameraProvider(source, camera_id=camera.id, fps=camera.fps)
    if camera.camera_type == "youtube_live":
        if not camera.youtube_url:
            raise CameraConnectionError(f"camera {camera.code} has no youtube_url configured")
        return YouTubeLiveProvider(
            camera.youtube_url,
            camera_id=camera.id,
            quality=camera.stream_quality or "best",
            stream_format=camera.stream_format,
        )
    raise CameraConnectionError(f"unsupported camera_type: {camera.camera_type!r}")


def get_provider(camera: Camera) -> CameraProvider:
    """Return a cached provider for ``camera``, creating it on first use."""
    with _lock:
        provider = _providers.get(camera.id)
        if provider is None:
            provider = build_provider(camera)
            _providers[camera.id] = provider
        return provider


def drop_provider(camera_id: int) -> None:
    """Remove a cached provider (e.g. after transport config changed)."""
    with _lock:
        provider = _providers.pop(camera_id, None)
        _previous_frames.pop(camera_id, None)
        _previous_captured_at.pop(camera_id, None)
    if provider is not None:
        try:
            provider.disconnect()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.warning("failed to disconnect cached provider for camera %s", camera_id)


def drop_all_providers() -> None:
    """Disconnect and forget every cached provider (tests, shutdown)."""
    with _lock:
        providers = list(_providers.values())
        _providers.clear()
        _previous_frames.clear()
        _previous_captured_at.clear()
    for provider in providers:
        try:
            provider.disconnect()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.warning("failed to disconnect cached provider during drop_all")


def provider_status(camera: Camera) -> CameraStatus:
    """Status of the cached provider, or a neutral 'not connected' status."""
    with _lock:
        provider = _providers.get(camera.id)
    if provider is None:
        return CameraStatus(
            connected=False,
            healthy=False,
            message="no live connection in this process (POST test-connection or capture a frame)",
        )
    return provider.get_status()


def provider_details(camera: Camera) -> dict[str, Any]:
    """Optional live-source details (resolved format, is_live) if known."""
    with _lock:
        provider = _providers.get(camera.id)
    if provider is None:
        return {}
    details: dict[str, Any] = {}
    for key in ("is_live", "resolved_format", "reconnect_count"):
        value = getattr(provider, key, None)
        if value is not None:
            details[key] = value
    return details


def test_connection(
    camera: Camera,
) -> tuple[bool, str, float | None, Frame | None, FrameQuality | None]:
    """Open a short-lived connection, read one frame, then disconnect.

    Returns ``(ok, message, latency_ms, frame, quality)``.
    """
    started = time.perf_counter()
    provider = build_provider(camera)  # fresh: never pollute the cached connection
    try:
        provider.connect()
        frame = provider.read_frame()
    except (CameraConnectionError, FrameUnavailableError) as exc:
        return False, str(exc), None, None, None
    except Exception as exc:  # defensive: hardware backends can raise anything
        logger.exception("test-connection failed for camera %s", camera.id)
        return False, f"unexpected error: {exc}", None, None, None
    finally:
        try:
            provider.disconnect()
        except Exception:  # pragma: no cover - best-effort cleanup
            logger.warning("disconnect after test-connection failed for camera %s", camera.id)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    quality = assess_quality(frame.image, captured_at=frame.captured_at)
    return True, "connection ok", latency_ms, frame, quality


def capture_frame(camera: Camera) -> tuple[Frame, FrameQuality]:
    """Acquire a frame via the cached provider and assess its quality."""
    provider = get_provider(camera)
    previous = _previous_frames.get(camera.id)
    previous_at = _previous_captured_at.get(camera.id)
    frame = provider.read_frame()
    quality = assess_quality(
        frame.image,
        previous_frame=previous,
        captured_at=frame.captured_at,
        previous_captured_at=previous_at,
    )
    _previous_frames[camera.id] = frame.image
    _previous_captured_at[camera.id] = frame.captured_at
    return frame, quality


def preprocess_frame(frame: Frame, config: PreprocessConfig | None = None) -> Any:
    return preprocess(frame.image, config)


def encode_jpeg(image: Any) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise FrameUnavailableError("could not encode frame as JPEG")
    return buffer.tobytes()


def redact_rtsp_url(url: str | None) -> str | None:
    """Mask any credentials in an RTSP URL for display/logging."""
    if not url:
        return None
    if "://" in url and "@" in url:
        scheme, rest = url.split("://", 1)
        host = rest.split("@", 1)[1]
        return f"{scheme}://***@{host}"
    return url


def redact_camera_source(camera: Camera) -> str | None:
    """Redacted display URL for a camera source (never tokens/cookies)."""
    if camera.camera_type == "youtube_live":
        return redact_youtube_url(camera.youtube_url)
    return redact_rtsp_url(camera.rtsp_url)
