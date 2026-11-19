"""Unit tests for the video-file virtual camera provider."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.services.camera.base import CameraConnectionError
from app.services.camera.video import (
    VideoFileCameraProvider,
    resolve_video_path,
    video_dir,
)


@pytest.fixture()
def tiny_mp4(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    assert writer.isOpened(), "could not create synthetic mp4 (codec missing)"
    for i in range(9):
        frame = np.full((24, 32, 3), i * 25, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_video_dir_default(tmp_path) -> None:
    assert video_dir(tmp_path) == tmp_path / "videos"


def test_resolve_video_path_rejects_traversal(tmp_path) -> None:
    with pytest.raises(CameraConnectionError):
        resolve_video_path(tmp_path, "../escape.mp4")
    with pytest.raises(CameraConnectionError):
        resolve_video_path(tmp_path, "/abs/path.mp4")


def test_resolve_video_path_ok(tmp_path) -> None:
    resolved = resolve_video_path(tmp_path, "clip.mp4")
    assert resolved == (tmp_path / "videos" / "clip.mp4").resolve()


def test_connect_missing_file_raises(tmp_path) -> None:
    provider = VideoFileCameraProvider(tmp_path / "nope.mp4", camera_id=1)
    with pytest.raises(CameraConnectionError):
        provider.connect()


def test_read_frames_and_loop(tiny_mp4) -> None:
    provider = VideoFileCameraProvider(tiny_mp4, camera_id=1, loop=True)
    provider.connect()
    frames = [provider.read_frame() for _ in range(11)]  # more than the 9 written
    assert frames[0].image.shape == (24, 32, 3)
    assert [f.sequence for f in frames] == list(range(1, 12))
    meta = frames[0].metadata
    assert meta.camera_type == "video"
    assert meta.width == 32
    assert meta.height == 24
    status = provider.get_status()
    assert status.connected is True
    assert status.frames_captured == 11
    # After EOF the provider looped back to the beginning.
    assert frames[9].metadata.extra["frame_index"] == 0
    assert frames[9].metadata.extra["loop_count"] == 1


def test_no_loop_stops_at_eof(tiny_mp4) -> None:
    from app.services.camera.base import FrameUnavailableError

    provider = VideoFileCameraProvider(tiny_mp4, camera_id=1, loop=False)
    provider.connect()
    for _ in range(9):
        provider.read_frame()
    with pytest.raises(FrameUnavailableError):
        provider.read_frame()
    assert provider.get_status().connected is True  # stream open but exhausted
    assert "end of video" in provider.get_status().message


def test_metadata_when_connected(tiny_mp4) -> None:
    provider = VideoFileCameraProvider(tiny_mp4, camera_id=3)
    provider.connect()
    meta = provider.get_metadata()
    assert meta.fps is not None and meta.fps > 0
    assert meta.camera_type == "video"
    provider.disconnect()
    assert provider.get_status().connected is False
