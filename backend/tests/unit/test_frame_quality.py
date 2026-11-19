"""Unit tests for frame preprocessing and quality assessment."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from app.services.vision.preprocessing import (
    FrameQuality,
    PreprocessConfig,
    assess_quality,
    preprocess,
)


def _sharp_frame(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


def test_preprocess_resizes_to_max_dimension() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    out = preprocess(
        frame, PreprocessConfig(max_dimension=320, denoise=False, enhance_contrast=False)
    )
    assert max(out.shape[:2]) == 320


def test_preprocess_explicit_size() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = preprocess(
        frame,
        PreprocessConfig(
            width=50, height=40, max_dimension=None, denoise=False, enhance_contrast=False
        ),
    )
    assert out.shape[:2] == (40, 50)


def test_preprocess_roi_crop() -> None:
    frame = np.arange(100 * 200 * 3, dtype=np.uint8).reshape(100, 200, 3)
    out = preprocess(
        frame,
        PreprocessConfig(
            roi=(10, 20, 50, 30), max_dimension=None, denoise=False, enhance_contrast=False
        ),
    )
    assert out.shape[:2] == (30, 50)


def test_preprocess_roi_out_of_bounds_raises() -> None:
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    try:
        preprocess(frame, PreprocessConfig(roi=(0, 0, 100, 100), max_dimension=None))
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-bounds ROI")


def test_black_frame_flagged() -> None:
    quality = assess_quality(np.zeros((32, 32, 3), dtype=np.uint8))
    assert quality.is_black is True
    assert "black_frame" in quality.flags
    assert quality.score < 0.5


def test_sharp_frame_passes() -> None:
    quality = assess_quality(_sharp_frame())
    assert quality.is_black is False
    assert quality.is_blurry is False
    assert quality.is_obstructed is False
    assert quality.score == 1.0
    assert quality.flags == []


def test_frozen_frame_detected_with_previous() -> None:
    frame = _sharp_frame(seed=1)
    quality = assess_quality(frame, previous_frame=frame.copy())
    assert quality.is_frozen is True
    assert "frozen_frame" in quality.flags


def test_sharp_frame_not_frozen() -> None:
    quality = assess_quality(_sharp_frame(seed=1), previous_frame=_sharp_frame(seed=2))
    assert quality.is_frozen is False


def test_uniform_gray_frame_obstructed() -> None:
    # Flat mid-gray: not black, no detail -> covered lens heuristic.
    quality = assess_quality(np.full((64, 64, 3), 120, dtype=np.uint8))
    assert quality.is_black is False
    assert quality.is_obstructed is True
    assert "obstruction" in quality.flags


def test_timestamp_inconsistent_flagged() -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    earlier = datetime(2025, 12, 31, tzinfo=UTC)
    quality = assess_quality(_sharp_frame(), captured_at=earlier, previous_captured_at=t)
    assert quality.is_timestamp_inconsistent is True
    assert "timestamp_inconsistent" in quality.flags


def test_monotonic_timestamps_ok() -> None:
    t = datetime(2026, 1, 1, tzinfo=UTC)
    later = datetime(2026, 1, 2, tzinfo=UTC)
    quality = assess_quality(_sharp_frame(), captured_at=later, previous_captured_at=t)
    assert quality.is_timestamp_inconsistent is False


def test_quality_score_floor_is_zero() -> None:
    quality = assess_quality(np.zeros((32, 32, 3), dtype=np.uint8))
    assert 0.0 <= quality.score <= 1.0
    assert isinstance(quality, FrameQuality)
