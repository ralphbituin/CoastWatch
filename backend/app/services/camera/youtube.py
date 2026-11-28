"""YouTube Live camera provider (Phase 7).

Public/authorized streams only. This module does **not** bypass access controls,
read cookies by default, or download an unbounded stream. Resolution is lazy:
importing the module never touches the network.

Resolution adapters, in order:

1. the ``yt_dlp`` Python module (preferred, pinned in ``requirements.txt``);
2. the ``yt-dlp`` CLI on ``PATH`` (documented fallback).

The resolved playable URL is handed to OpenCV's FFMPEG backend, which reads
frames sequentially without buffering the whole stream. YouTube URLs expire, so
the provider re-resolves on every reconnect.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

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

ALLOWED_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
}
ALLOWED_PATHS = ("/watch", "/live/", "/embed/", "/shorts/", "/v/")
_MAX_FORMAT_HEIGHT = 720
_MAX_CONSECUTIVE_FAILURES = 3


class YouTubeError(CameraConnectionError):
    """YouTube source could not be validated or resolved."""


def _looks_like_ip(host: str) -> bool:
    return host.count(".") == 3 and all(part.isdigit() for part in host.split("."))


def validate_youtube_url(url: str) -> str:
    """Validate and normalise a public YouTube Live URL.

    Only ``https://`` URLs on the documented YouTube hosts are accepted. Local
    hosts, bare IPs, credentials, non-HTTPS schemes, filesystem paths and
    non-YouTube hosts are rejected (basic SSRF hardening).
    """
    candidate = (url or "").strip()
    if not candidate:
        raise ValueError("youtube_url must not be blank")
    if len(candidate) > 500:
        raise ValueError("youtube_url is too long")
    parts = urlsplit(candidate)
    if parts.scheme != "https":
        raise ValueError("youtube_url must use https://")
    if parts.username or parts.password or "@" in parts.netloc:
        raise ValueError("youtube_url must not contain credentials")
    host = (parts.hostname or "").lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".local"):
        raise ValueError("youtube_url must not target a local host")
    if _looks_like_ip(host):
        raise ValueError("youtube_url must not target a bare IP address")
    if host not in ALLOWED_HOSTS:
        raise ValueError("youtube_url must be a youtube.com / youtu.be URL")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("youtube_url has an invalid port") from exc
    if port not in (None, 443):
        raise ValueError("youtube_url must use the default HTTPS port")
    path = parts.path or "/"
    if host == "youtu.be":
        video_id = path.strip("/")
        if not video_id or "/" in video_id:
            raise ValueError("youtu.be URL is missing a video id")
    elif not any(path == prefix or path.startswith(prefix) for prefix in ALLOWED_PATHS):
        raise ValueError("youtube_url must point to a watch/live/embed/shorts stream")
    if host != "youtu.be" and path.startswith("/watch"):
        query = parts.query
        if "v=" not in query:
            raise ValueError("youtube.com/watch URL is missing the v parameter")
    return candidate


def redact_youtube_url(url: str | None) -> str | None:
    """Return scheme+host+path only, dropping query strings and fragments.

    YouTube URLs can carry short-lived tokens in the query string; they are
    never returned by the API or written to logs.
    """
    if not url:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if not parts.scheme or not parts.hostname:
        return None
    return f"{parts.scheme}://{parts.hostname}{parts.path or '/'}"


def _pick_format(
    info: dict[str, Any], quality: str | None, stream_format: str | None
) -> dict[str, Any] | None:
    formats = [item for item in (info.get("formats") or []) if item.get("url")]
    if not formats:
        return None
    if stream_format:
        wanted = stream_format.lower()
        for item in formats:
            labels = {
                str(item.get("format_id") or "").lower(),
                str(item.get("format_note") or "").lower(),
                str(item.get("ext") or "").lower(),
            }
            if wanted in labels:
                return item
    wanted_height: int | None = None
    if quality and quality.lower() not in {"best", "worst"}:
        digits = "".join(ch for ch in quality if ch.isdigit())
        if digits:
            wanted_height = int(digits)
    if wanted_height is None:
        wanted_height = _MAX_FORMAT_HEIGHT
    playable = [item for item in formats if item.get("vcodec") != "none"]
    if not playable:
        playable = formats
    bounded = [item for item in playable if (item.get("height") or 0) <= wanted_height]
    pool = bounded or playable
    if quality and quality.lower() == "worst":
        return min(pool, key=lambda item: item.get("height") or 0)
    return max(pool, key=lambda item: item.get("height") or 0)


def _resolved_payload(
    info: dict[str, Any], quality: str | None, stream_format: str | None
) -> dict[str, Any]:
    if info.get("_type") == "playlist" and info.get("entries"):
        entries = [entry for entry in info["entries"] if entry]
        if not entries:
            raise YouTubeError("YouTube playlist has no playable entries")
        info = entries[0]
    is_live = bool(info.get("is_live")) or info.get("live_status") == "is_live"
    if not is_live:
        raise YouTubeError(
            "the YouTube URL is not a live stream (VOD/non-live sources are rejected)"
        )
    chosen = _pick_format(info, quality, stream_format)
    stream_url = (chosen or {}).get("url") or info.get("url")
    if not stream_url:
        raise YouTubeError("YouTube resolution returned no playable stream URL")
    return {
        "stream_url": str(stream_url),
        "is_live": True,
        "format_id": (chosen or {}).get("format_id"),
        "format_note": (chosen or {}).get("format_note"),
        "height": (chosen or {}).get("height"),
        "ext": (chosen or {}).get("ext"),
        "protocol": (chosen or {}).get("protocol"),
        "title": info.get("title"),
        "webpage_url_redacted": redact_youtube_url(info.get("webpage_url")),
    }


def _resolve_with_module(
    url: str, quality: str | None, stream_format: str | None
) -> dict[str, Any]:
    try:
        import yt_dlp  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - dependency-specific
        raise YouTubeError("yt-dlp Python module is not installed") from exc
    options = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "socket_timeout": 20,
    }
    with yt_dlp.YoutubeDL(options) as ydl:  # type: ignore[attr-defined]
        info = ydl.extract_info(url, download=False)
    return _resolved_payload(info or {}, quality, stream_format)


def _resolve_with_cli(url: str, quality: str | None, stream_format: str | None) -> dict[str, Any]:
    executable = shutil.which("yt-dlp")
    if executable is None:
        raise YouTubeError(
            "yt-dlp is not available: install the pinned dependency or add the yt-dlp CLI to PATH"
        )
    command = [
        executable,
        "--no-playlist",
        "--no-warnings",
        "--skip-download",
        "-J",
        "--",
        url,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise YouTubeError(f"yt-dlp could not resolve the stream: {type(exc).__name__}") from exc
    if completed.returncode != 0:
        raise YouTubeError("yt-dlp could not resolve the stream (non-zero exit)")
    try:
        info = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise YouTubeError("yt-dlp returned unreadable metadata") from exc
    return _resolved_payload(info, quality, stream_format)


Resolver = Callable[[str, str | None, str | None], dict[str, Any]]


def resolve_youtube_stream(
    url: str,
    *,
    quality: str | None = None,
    stream_format: str | None = None,
    resolver: Resolver | None = None,
) -> dict[str, Any]:
    """Resolve a live stream URL through the injected or default adapter."""
    safe_url = validate_youtube_url(url)
    if resolver is not None:
        return resolver(safe_url, quality, stream_format)
    try:
        return _resolve_with_module(safe_url, quality, stream_format)
    except YouTubeError as module_error:
        try:
            return _resolve_with_cli(safe_url, quality, stream_format)
        except YouTubeError as cli_error:
            raise YouTubeError(f"{module_error}; CLI fallback failed: {cli_error}") from cli_error


class YouTubeLiveProvider(CameraProvider):
    """Reads frames from a resolved YouTube Live HLS/DASH URL.

    HLS keeps a segment buffer. Reading only one frame per sampling interval
    from a persistent capture makes that backlog grow until the preview replays
    minutes-old video. The provider therefore reconnects from the live edge
    before a sample taken after an idle gap, while keeping the resolved stream
    URL so yt-dlp is not invoked on every sample.
    """

    _REOPEN_AFTER_SECONDS = 2.0

    def __init__(
        self,
        url: str,
        *,
        camera_id: int = 0,
        quality: str | None = "best",
        stream_format: str | None = None,
        resolver: Resolver | None = None,
        capture_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._url = validate_youtube_url(url)
        self._camera_id = camera_id
        self._quality = quality
        self._stream_format = stream_format
        self._resolver = resolver
        self._capture_factory = capture_factory
        self._cap: Any | None = None
        self._lock = threading.RLock()
        self._stream_url: str | None = None
        self._last_read_monotonic: float | None = None
        self._frames_captured = 0
        self._sequence = 0
        self._reconnect_count = 0
        self._consecutive_failures = 0
        self._last_frame_at: datetime | None = None
        self._last_error: str | None = None
        self._is_live = False
        self._resolved: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _resolve_stream(self) -> str:
        info = resolve_youtube_stream(
            self._url,
            quality=self._quality,
            stream_format=self._stream_format,
            resolver=self._resolver,
        )
        stream_url = info.get("stream_url")
        if not stream_url:
            raise YouTubeError("resolved stream URL is empty")
        self._resolved = info
        self._stream_url = str(stream_url)
        self._is_live = True
        return self._stream_url

    def _make_capture(self, stream_url: str) -> Any:
        if self._capture_factory is not None:
            capture = self._capture_factory(stream_url)
        else:
            capture = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
            # Best-effort bounded open/read timeouts (FFMPEG backend).
            try:
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 15000)
                # Keep at most one decoded frame queued inside the backend.
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:  # pragma: no cover - backend dependent
                logger.debug("YouTube stream timeout/buffer properties not supported")
        if capture is None or not capture.isOpened():
            if capture is not None:
                capture.release()
            self._last_error = "could not open the resolved YouTube stream"
            raise CameraConnectionError(self._last_error)
        return capture

    def _open(self) -> None:
        if self._stream_url is None:
            self._resolve_stream()
        assert self._stream_url is not None
        self._cap = self._make_capture(self._stream_url)
        self._consecutive_failures = 0
        self._last_error = None
        self._last_read_monotonic = time.monotonic()
        logger.info(
            "youtube camera %s resolved live stream (format=%s height=%s)",
            self._camera_id,
            self._resolved.get("format_id"),
            self._resolved.get("height"),
        )

    def connect(self) -> None:
        with self._lock:
            self._release()
            self._stream_url = None
            self._open()

    def disconnect(self) -> None:
        with self._lock:
            self._release()
            self._stream_url = None
            self._last_read_monotonic = None
            self._resolved = {}
            self._is_live = False

    def _release(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None

    # ------------------------------------------------------------------
    # Frames
    # ------------------------------------------------------------------
    def read_frame(self) -> Frame:
        with self._lock:
            if self._stream_url is None:
                self._resolve_stream()
            assert self._stream_url is not None
            was_open = self._cap is not None and self._cap.isOpened()
            idle = (
                self._last_read_monotonic is None
                or (time.monotonic() - self._last_read_monotonic) > self._REOPEN_AFTER_SECONDS
            )
            if not was_open or idle:
                # Reconnect at the live edge instead of draining a growing HLS
                # backlog one frame per sampling interval.
                self._release()
                self._cap = self._make_capture(self._stream_url)
                if not was_open:
                    self._reconnect_count += 1
            assert self._cap is not None
            ok, image = self._cap.read()
            if not ok or image is None:
                self._consecutive_failures += 1
                self._last_error = "frame read failed"
                if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    # Expired URL or dropped stream: force a fresh resolve.
                    self._release()
                    self._stream_url = None
                raise FrameUnavailableError(
                    "could not read a frame from the YouTube Live stream "
                    f"({self._consecutive_failures} consecutive failure(s))"
                )
            self._consecutive_failures = 0
            self._last_error = None
            self._last_read_monotonic = time.monotonic()
            self._frames_captured += 1
            self._sequence += 1
            # Wall-clock at frame receipt. YouTube provides no true per-frame
            # capture timestamp; callers must not treat this as one.
            self._last_frame_at = datetime.now(UTC)
            height, width = image.shape[:2]
            return Frame(
                image=image,
                captured_at=self._last_frame_at,
                metadata=CameraMetadata(
                    camera_id=self._camera_id,
                    camera_type="youtube_live",
                    width=int(width),
                    height=int(height),
                    fps=None,
                    extra={
                        "is_live": True,
                        "format_id": self._resolved.get("format_id"),
                        "format_note": self._resolved.get("format_note"),
                        "resolved_height": self._resolved.get("height"),
                        "source_url_redacted": redact_youtube_url(self._url),
                    },
                ),
                sequence=self._sequence,
            )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def get_status(self) -> CameraStatus:
        with self._lock:
            connected = self._cap is not None and bool(self._cap.isOpened())
            if connected:
                message = "YouTube Live stream connected"
            elif self._last_error:
                message = self._last_error
            else:
                message = "not connected"
            return CameraStatus(
                connected=connected,
                healthy=connected and self._consecutive_failures == 0,
                message=message,
                last_frame_at=self._last_frame_at,
                frames_captured=self._frames_captured,
            )

    def get_metadata(self) -> CameraMetadata:
        width = height = None
        if self._cap is not None and self._cap.isOpened():
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or None
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or None
        return CameraMetadata(
            camera_id=self._camera_id,
            camera_type="youtube_live",
            width=width,
            height=height,
            fps=None,
            extra={
                "is_live": self._is_live,
                "format_id": self._resolved.get("format_id"),
                "format_note": self._resolved.get("format_note"),
                "resolved_height": self._resolved.get("height"),
                "source_url_redacted": redact_youtube_url(self._url),
            },
        )

    @property
    def is_live(self) -> bool:
        return self._is_live

    @property
    def reconnect_count(self) -> int:
        return self._reconnect_count

    @property
    def resolved_format(self) -> str | None:
        return (
            self._resolved.get("format_id")
            or self._resolved.get("format_note")
            or self._resolved.get("ext")
        )
