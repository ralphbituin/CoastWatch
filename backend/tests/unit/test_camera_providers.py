"""Unit tests for the camera providers (hardware mocked out)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import numpy as np
import pytest

from app.services.camera import rtsp as rtsp_module
from app.services.camera import usb as usb_module
from app.services.camera.base import (
    CameraConnectionError,
    FrameUnavailableError,
)
from app.services.camera.rtsp import RTSPCameraProvider
from app.services.camera.upload import UploadCameraProvider, camera_upload_dir
from app.services.camera.usb import USBCameraProvider


class FakeCapture:
    """Configurable stand-in for cv2.VideoCapture."""

    instances: list[FakeCapture] = []
    open_ok = True
    read_ok = True

    def __init__(self, source, *args, **kwargs):
        self.source = source
        self.kwargs = kwargs
        self._opened = FakeCapture.open_ok
        self.props: dict[int, float] = {}
        self.released = False
        self.grabs = 0
        self.retrieves = 0
        FakeCapture.instances.append(self)

    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2 API
        return self._opened and not self.released

    def set(self, prop, value) -> bool:
        self.props[prop] = value
        return True

    def get(self, prop) -> float:
        return self.props.get(prop, 0.0)

    def grab(self) -> bool:
        self.grabs += 1
        return FakeCapture.read_ok

    def retrieve(self):
        self.retrieves += 1
        if FakeCapture.read_ok and self._opened and not self.released:
            return True, np.full((48, 64, 3), 127, dtype=np.uint8)
        return False, None

    def read(self):
        if FakeCapture.read_ok and self._opened and not self.released:
            return True, np.full((48, 64, 3), 127, dtype=np.uint8)
        return False, None

    def release(self) -> None:
        self.released = True


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeCapture.instances = []
    FakeCapture.open_ok = True
    FakeCapture.read_ok = True


def _patch_rtsp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rtsp_module.cv2, "VideoCapture", FakeCapture)


def _patch_usb(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(usb_module.cv2, "VideoCapture", FakeCapture)


# ---------------------------------------------------------------------------
# RTSP provider
# ---------------------------------------------------------------------------


def test_rtsp_connect_and_read_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_rtsp(monkeypatch)
    provider = RTSPCameraProvider("rtsp://cam.local/stream", camera_id=7)
    provider.connect()

    frame = provider.read_frame()
    assert frame.image.shape == (48, 64, 3)
    assert frame.sequence == 1
    assert isinstance(frame.captured_at, datetime)
    assert frame.metadata.camera_type == "rtsp"
    assert frame.metadata.camera_id == 7

    provider.read_frame()
    assert provider.get_status().frames_captured == 2
    assert provider.get_status().healthy is True
    provider.disconnect()


def test_rtsp_connect_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeCapture.open_ok = False
    _patch_rtsp(monkeypatch)
    provider = RTSPCameraProvider("rtsp://cam.local/dead", camera_id=1, reconnect_delay_s=0)
    with pytest.raises(CameraConnectionError):
        provider.connect()


def test_rtsp_read_failure_raises_and_tracks(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_rtsp(monkeypatch)
    provider = RTSPCameraProvider("rtsp://cam.local/stream", camera_id=1)
    provider.connect()
    FakeCapture.read_ok = False
    # The drain thread observes the failure asynchronously.
    deadline = time.monotonic() + 2.0
    while provider.get_status().healthy and time.monotonic() < deadline:
        time.sleep(0.01)
    with pytest.raises(FrameUnavailableError):
        provider.read_frame()
    status = provider.get_status()
    assert status.healthy is False
    provider.disconnect()


def test_rtsp_drain_thread_returns_latest_frame_quickly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_rtsp(monkeypatch)
    provider = RTSPCameraProvider("rtsp://cam.local/stream", camera_id=7, stall_timeout_s=5.0)
    provider.connect()
    time.sleep(0.2)  # let the drain thread grab a few frames
    started = time.perf_counter()
    frame = provider.read_frame()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    cap = FakeCapture.instances[-1]
    gap_ms = (datetime.now(UTC) - frame.captured_at).total_seconds() * 1000.0
    print(
        f"drain: grabbed={cap.grabs} returned={provider.get_status().frames_captured} "
        f"read_frame_ms={elapsed_ms:.2f} newest_frame_age_ms={gap_ms:.2f}"
    )
    assert frame.image.shape == (48, 64, 3)
    assert cap.grabs > 1  # the socket is drained between sampling reads
    assert elapsed_ms < 100.0
    assert gap_ms < 100.0
    assert provider.get_status().healthy is True
    provider.disconnect()


def test_rtsp_status_reports_stream_stalled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_rtsp(monkeypatch)
    provider = RTSPCameraProvider("rtsp://cam.local/stream", camera_id=1, stall_timeout_s=1.0)
    provider.connect()
    provider.read_frame()
    provider._stop_drain()  # noqa: SLF001 - simulate a live socket stuck with no frames
    with provider._lock:  # noqa: SLF001
        provider._latest_monotonic = time.monotonic() - 5.0  # noqa: SLF001
    status = provider.get_status()
    assert status.connected is True
    assert status.healthy is False
    assert status.message == "stream stalled"
    provider.disconnect()


def test_rtsp_blank_url_rejected() -> None:
    with pytest.raises(ValueError):
        RTSPCameraProvider("   ")


def test_rtsp_reconnects_after_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_rtsp(monkeypatch)
    provider = RTSPCameraProvider(
        "rtsp://cam.local/stream", camera_id=1, max_reconnect_attempts=2, reconnect_delay_s=0
    )
    provider.connect()
    FakeCapture.read_ok = False
    deadline = time.monotonic() + 2.0
    while provider.get_status().healthy and time.monotonic() < deadline:
        time.sleep(0.01)
    for _ in range(3):  # exhaust consecutive-failure budget -> stream dropped
        with pytest.raises(FrameUnavailableError):
            provider.read_frame()
    FakeCapture.read_ok = True
    frame = provider.read_frame()  # should auto-reconnect
    assert frame.image is not None
    provider.disconnect()


# ---------------------------------------------------------------------------
# USB provider
# ---------------------------------------------------------------------------


def test_usb_connect_sets_resolution_and_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_usb(monkeypatch)
    provider = USBCameraProvider(2, camera_id=42, width=1280, height=720, fps=15.0)
    provider.connect()

    frame = provider.read_frame()
    assert frame.metadata.camera_type == "usb"
    assert frame.metadata.camera_id == 42
    cap = FakeCapture.instances[-1]
    assert cap.props[provider_usb_width_prop()] == 1280
    assert cap.props[provider_usb_height_prop()] == 720
    assert provider.get_status().connected is True


def provider_usb_width_prop() -> int:
    return usb_module.cv2.CAP_PROP_FRAME_WIDTH


def provider_usb_height_prop() -> int:
    return usb_module.cv2.CAP_PROP_FRAME_HEIGHT


def test_usb_invalid_device_index_rejected() -> None:
    with pytest.raises(ValueError):
        USBCameraProvider(-1)


def test_usb_open_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeCapture.open_ok = False
    _patch_usb(monkeypatch)
    provider = USBCameraProvider(9)
    with pytest.raises(CameraConnectionError):
        provider.connect()


# ---------------------------------------------------------------------------
# Upload provider
# ---------------------------------------------------------------------------


def test_upload_provider_no_frames(tmp_path) -> None:
    provider = UploadCameraProvider(tmp_path / "cam1", camera_id=1)
    provider.connect()
    assert provider.get_status().connected is False
    with pytest.raises(FrameUnavailableError):
        provider.read_frame()


def test_upload_provider_serves_latest_image(tmp_path) -> None:
    import cv2

    upload_dir = camera_upload_dir(tmp_path, 3)
    upload_dir.mkdir(parents=True)
    image = np.full((24, 32, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(upload_dir / "first.jpg"), image)
    provider = UploadCameraProvider(upload_dir, camera_id=3)
    frame = provider.read_frame()
    assert frame.metadata.camera_type == "upload"
    assert frame.metadata.width == 32
    assert frame.metadata.height == 24
    assert provider.get_status().healthy is True


def test_upload_dir_path(tmp_path) -> None:
    assert camera_upload_dir(tmp_path, 5) == tmp_path / "uploads" / "cameras" / "5"
