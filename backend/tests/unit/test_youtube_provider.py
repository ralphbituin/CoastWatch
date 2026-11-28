"""Phase 7 YouTube Live provider tests (no network)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.services.camera import youtube as youtube_module
from app.services.camera.base import CameraConnectionError, FrameUnavailableError
from app.services.camera.youtube import (
    YouTubeError,
    YouTubeLiveProvider,
    redact_youtube_url,
    resolve_youtube_stream,
    validate_youtube_url,
)


class FakeCapture:
    instances: list[FakeCapture] = []
    read_ok = True

    def __init__(self, source) -> None:
        self.source = source
        self.released = False
        self._opened = True
        FakeCapture.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2
        return self._opened and not self.released

    def release(self) -> None:
        self.released = True

    def read(self):
        if FakeCapture.read_ok and not self.released:
            return True, np.full((36, 64, 3), 90, dtype=np.uint8)
        return False, None

    def get(self, _prop) -> float:  # noqa: ANN001 - mirrors cv2
        return 0.0


@pytest.fixture(autouse=True)
def _reset() -> None:
    FakeCapture.instances = []
    FakeCapture.read_ok = True


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=abc123",
        "https://youtube.com/live/abc123?feature=share",
        "https://youtu.be/abc123",
        "https://m.youtube.com/watch?v=abc123",
    ],
)
def test_accepts_public_youtube_urls(url: str) -> None:
    assert validate_youtube_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "http://www.youtube.com/watch?v=abc123",
        "https://localhost/watch?v=abc123",
        "https://127.0.0.1/watch?v=abc123",
        "https://example.com/watch?v=abc123",
        "https://www.youtube.com@evil.example/watch?v=abc123",
        "file:///etc/passwd",
        "https://www.youtube.com/channel/abc",
        "https://youtu.be/",
        "https://www.youtube.com:8443/watch?v=abc123",
        "https://user:pass@www.youtube.com/watch?v=abc123",
    ],
)
def test_rejects_ssrf_and_non_youtube_targets(url: str) -> None:
    with pytest.raises(ValueError):
        validate_youtube_url(url)


def test_redaction_drops_query_tokens() -> None:
    assert (
        redact_youtube_url("https://www.youtube.com/watch?v=abc&token=secret")
        == "https://www.youtube.com/watch"
    )
    assert redact_youtube_url(None) is None


def test_resolver_rejects_vod() -> None:
    def vod(_url, _quality, _format):
        return youtube_module._resolved_payload({"is_live": False}, None, None)

    with pytest.raises(YouTubeError):
        resolve_youtube_stream("https://youtu.be/abc123", resolver=vod)


def test_provider_uses_wall_clock_receipt_time_and_redacts() -> None:
    calls = {"resolve": 0}

    def resolver(_url, _quality, _format):
        calls["resolve"] += 1
        return {
            "stream_url": "https://cdn.example/live.m3u8?token=secret",
            "is_live": True,
            "format_id": "96",
            "format_note": "720p",
            "height": 720,
        }

    provider = YouTubeLiveProvider(
        "https://youtu.be/abc123",
        camera_id=9,
        resolver=resolver,
        capture_factory=FakeCapture,
    )
    before = datetime.now(UTC)
    frame = provider.read_frame()
    after = datetime.now(UTC)
    assert before <= frame.captured_at <= after
    assert frame.captured_at.tzinfo is not None
    assert frame.metadata.camera_type == "youtube_live"
    assert frame.metadata.extra["source_url_redacted"] == "https://youtu.be/abc123"
    assert "token" not in str(frame.metadata.extra)
    assert provider.get_status().connected is True
    assert calls["resolve"] == 1


def test_provider_reopens_at_live_edge_after_idle_gap() -> None:
    """A 10 s sampling gap must not replay a growing HLS backlog."""
    calls = {"resolve": 0}

    def resolver(_url, _quality, _format):
        calls["resolve"] += 1
        return {
            "stream_url": "https://cdn.example/live.m3u8",
            "is_live": True,
            "format_id": "96",
        }

    provider = YouTubeLiveProvider(
        "https://youtu.be/abc123",
        resolver=resolver,
        capture_factory=FakeCapture,
    )
    provider.connect()
    provider.read_frame()
    assert len(FakeCapture.instances) == 1
    assert calls["resolve"] == 1

    # Simulate the worker's sampling interval without sleeping.
    provider._last_read_monotonic = time.monotonic() - 10.0  # noqa: SLF001
    provider.read_frame()

    assert len(FakeCapture.instances) == 2
    assert FakeCapture.instances[0].released is True
    assert calls["resolve"] == 1  # cached stream URL, no yt-dlp re-resolution


def test_provider_reconnects_and_re_resolves_expired_url() -> None:
    calls = {"resolve": 0}

    def resolver(_url, _quality, _format):
        calls["resolve"] += 1
        return {
            "stream_url": f"https://cdn.example/live-{calls['resolve']}.m3u8",
            "is_live": True,
            "format_id": "96",
        }

    provider = YouTubeLiveProvider(
        "https://www.youtube.com/watch?v=abc123",
        resolver=resolver,
        capture_factory=FakeCapture,
    )
    provider.connect()
    FakeCapture.read_ok = False
    for _ in range(3):
        with pytest.raises(FrameUnavailableError):
            provider.read_frame()
    FakeCapture.read_ok = True
    frame = provider.read_frame()  # forces a fresh resolve after expiry
    assert frame is not None
    assert calls["resolve"] >= 2
    assert FakeCapture.instances[-1].source == "https://cdn.example/live-2.m3u8"


def test_provider_rejects_non_live_resolution() -> None:
    def resolver(_url, _quality, _format):
        raise YouTubeError("not live")

    provider = YouTubeLiveProvider(
        "https://youtu.be/abc123", resolver=resolver, capture_factory=FakeCapture
    )
    with pytest.raises(CameraConnectionError):
        provider.connect()


def test_yt_dlp_is_pinned_in_requirements() -> None:
    requirements = (Path(__file__).resolve().parents[2] / "requirements.txt").read_text(
        encoding="utf-8"
    )
    assert "yt-dlp==" in requirements
