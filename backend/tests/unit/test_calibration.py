"""Unit tests for homography calibration and camera-movement detection."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.vision.calibration import (
    CalibrationError,
    CalibrationProfile,
    apply_homography,
    compute_homography,
    detect_camera_movement,
)


def _references():
    # Four non-collinear points with a known, simple homography: translate+scale.
    return {
        "points": [
            {"pixel": [0, 0], "world_m": [0.0, 0.0]},
            {"pixel": [100, 0], "world_m": [1.0, 0.0]},
            {"pixel": [0, 100], "world_m": [0.0, 1.0]},
            {"pixel": [100, 100], "world_m": [1.0, 1.0]},
        ]
    }


def test_compute_homography_recovers_known_mapping() -> None:
    profile = compute_homography(_references(), camera_id=1, version=2, meters_per_pixel=0.01)
    assert profile.camera_id == 1
    assert profile.version == 2
    assert profile.homography is not None
    assert len(profile.homography) == 3
    assert len(profile.homography[0]) == 3
    # RMS reprojection error should be ~0 for a consistent exact mapping.
    assert profile.rms_error_px is not None
    assert profile.rms_error_px < 1e-6


def test_compute_homography_requires_four_points() -> None:
    refs = {"points": [{"pixel": [0, 0], "world_m": [0.0, 0.0]}]}
    with pytest.raises(CalibrationError):
        compute_homography(refs)


def test_compute_homography_degenerate_points_rejected() -> None:
    refs = {
        "points": [
            {"pixel": [1, 1], "world_m": [0.0, 0.0]},
            {"pixel": [1, 1], "world_m": [1.0, 1.0]},
            {"pixel": [1, 1], "world_m": [2.0, 2.0]},
            {"pixel": [1, 1], "world_m": [3.0, 3.0]},
        ]
    }
    with pytest.raises(CalibrationError):
        compute_homography(refs)


def test_apply_homography_roundtrip() -> None:
    profile = compute_homography(_references())
    world = apply_homography(profile.homography, [(50, 50)])
    assert world.shape == (1, 2)
    np.testing.assert_allclose(world[0], [0.5, 0.5], atol=1e-6)


def test_apply_homography_rejects_bad_shape() -> None:
    with pytest.raises(CalibrationError):
        apply_homography([[1, 0]], [(0, 0)])


def _textured(seed: int = 0, shape: tuple[int, int] = (120, 160)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_static_camera_not_flagged() -> None:
    frame = _textured(seed=7)
    # Same scene, tiny sub-pixel noise — no re-aiming.
    noisy = np.clip(frame.astype(np.int16) + 1, 0, 255).astype(np.uint8)
    assert detect_camera_movement(frame, noisy) is False


def test_shifted_camera_flagged() -> None:
    frame = _textured(seed=7)
    shifted = np.roll(frame, 12, axis=1)  # 12-pixel horizontal re-aim
    assert detect_camera_movement(frame, shifted) is True


def test_flat_frame_cannot_rule_out_movement() -> None:
    flat = np.zeros((64, 64), dtype=np.uint8)
    assert detect_camera_movement(flat, flat.copy()) is True


def test_calibration_profile_defaults() -> None:
    profile = CalibrationProfile(camera_id=9, version=3, homography=None, meters_per_pixel=0.05)
    assert profile.rms_error_px is None
