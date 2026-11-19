"""Unit tests for shoreline (waterline) extraction."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.vision.calibration import CalibrationProfile
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline,
)


def _synthetic_mask(width: int = 200, height: int = 100, sea_top: int = 60) -> np.ndarray:
    """Synthetic coastal scene: water occupies rows >= sea_top."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[sea_top:, :] = 255
    # Add some water speckle above the line to exercise the morphological filter.
    mask[sea_top - 2, min(10, width - 1)] = 255
    mask[sea_top - 3, min(width - 50, width - 1)] = 255
    return mask


def test_extract_shoreline_finds_boundary() -> None:
    mask = _synthetic_mask(sea_top=60)
    result = extract_shoreline(mask, smoothing_window=1)
    assert len(result.points) == mask.shape[1]
    ys = np.array([p[1] for p in result.points])
    assert ys.mean() == pytest.approx(60.0, abs=1.5)
    assert result.calibration_version is None
    assert result.calibrated_position_m is None
    assert result.segmentation_confidence == 1.0  # operator mask treated as ground truth


def test_extract_shoreline_accepts_bgr_mask() -> None:
    mask = np.zeros((80, 120, 3), dtype=np.uint8)
    mask[40:, :, 1] = 255  # "water" in green channel (BGR)
    result = extract_shoreline(mask, smoothing_window=1)
    assert len(result.points) == 120


def test_extract_shoreline_empty_mask_raises() -> None:
    with pytest.raises(ShorelineExtractionError):
        extract_shoreline(np.zeros((50, 50), dtype=np.uint8))


def test_extract_shoreline_roi() -> None:
    mask = _synthetic_mask(width=100, height=80, sea_top=40)
    result = extract_shoreline(mask, roi=(10, 10, 60, 60), smoothing_window=1)
    assert len(result.points) == 60
    # Offsets are added back to the points.
    assert result.points[0][0] == pytest.approx(10.0)
    ys = np.array([p[1] for p in result.points])
    assert ys.mean() == pytest.approx(40.0, abs=1.5)


def test_extract_shoreline_roi_out_of_bounds() -> None:
    mask = _synthetic_mask()
    with pytest.raises(ShorelineExtractionError):
        extract_shoreline(mask, roi=(0, 0, 9999, 9999))


def test_calibrated_position_uses_seaward_axis() -> None:
    mask = _synthetic_mask(width=100, height=100, sea_top=50)
    # Identity homography: world == pixel coordinates.
    calibration = CalibrationProfile(
        camera_id=1,
        version=4,
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        meters_per_pixel=None,
    )
    result = extract_shoreline(
        mask, calibration=calibration, seaward_axis=(0.0, 1.0), smoothing_window=1
    )
    assert result.calibrated_position_m == pytest.approx(50.0, abs=1.0)
    assert result.calibration_version == 4


def test_calibrated_position_requires_axis() -> None:
    mask = _synthetic_mask(width=100, height=100, sea_top=50)
    calibration = CalibrationProfile(
        camera_id=1,
        version=1,
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        meters_per_pixel=0.05,
    )
    result = extract_shoreline(mask, calibration=calibration, smoothing_window=1)
    assert result.calibrated_position_m is None


def test_segmentation_confidence_passthrough() -> None:
    mask = _synthetic_mask(width=60, height=60, sea_top=30)
    result = extract_shoreline(mask, confidence=0.82, smoothing_window=1)
    assert result.segmentation_confidence == pytest.approx(0.82)


def test_zero_axis_rejected() -> None:
    mask = _synthetic_mask(width=60, height=60, sea_top=30)
    calibration = CalibrationProfile(
        camera_id=1,
        version=1,
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        meters_per_pixel=None,
    )
    with pytest.raises(ShorelineExtractionError):
        extract_shoreline(mask, calibration=calibration, seaward_axis=(0.0, 0.0))
