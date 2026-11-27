"""Unit tests for shoreline (waterline) extraction."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.vision.calibration import CalibrationProfile
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline,
    extract_shoreline_from_classes,
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


# ---------------------------------------------------------------------------
# Class-aware (SegFormer) path
# ---------------------------------------------------------------------------


def _synthetic_class_mask(
    width: int = 200, height: int = 120, sky_bottom: int = 30, shoreline: int = 80
) -> np.ndarray:
    """Sky above, Ocean in the middle, Beach below; two distinct boundaries."""
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:sky_bottom] = 2  # Sky
    mask[sky_bottom:shoreline] = 1  # Ocean
    mask[shoreline:] = 0  # Beach
    return mask


def test_class_aware_extracts_ocean_beach_not_sky_ocean() -> None:
    mask = _synthetic_class_mask(sky_bottom=30, shoreline=80)
    result = extract_shoreline_from_classes(
        mask, min_ocean_component_px=100, min_beach_component_px=100
    )
    assert result.shoreline_available is True
    ys = np.array([point[1] for point in result.points])
    assert ys.mean() == pytest.approx(80.0, abs=1.5)
    # The Sky/Ocean horizon is at row 30 and must not be selected.
    assert ys.mean() > 60
    assert result.ocean_pixels and result.ocean_pixels > 0
    assert result.beach_pixels and result.beach_pixels > 0
    assert result.segments


def test_class_aware_preserves_wave_variation() -> None:
    mask = np.full((120, 200), 2, dtype=np.uint8)
    for x in range(200):
        boundary = 70 + int(8 * np.sin(x / 18.0))
        mask[30:boundary, x] = 1
        mask[boundary:, x] = 0
    result = extract_shoreline_from_classes(
        mask, min_ocean_component_px=100, min_beach_component_px=100
    )
    assert result.shoreline_available is True
    ys = np.array([point[1] for point in result.points])
    assert ys.max() - ys.min() >= 10
    assert len(result.segments) >= 1


def test_class_aware_rejects_tiny_disconnected_ocean() -> None:
    mask = np.zeros((80, 80), dtype=np.uint8)
    mask[10:20, 10:20] = 1  # small ocean patch
    mask[20:, :] = 0  # beach
    result = extract_shoreline_from_classes(
        mask, min_ocean_component_px=5_000, min_beach_component_px=10
    )
    assert result.shoreline_available is False
    assert result.points == []
    assert "ocean" in (result.reason or "").lower()


def test_class_aware_returns_no_shoreline_without_contact() -> None:
    # Ocean and Beach are separated by a Sky band: no direct contact exists.
    mask = np.full((80, 120), 2, dtype=np.uint8)
    mask[:20, :] = 1  # Ocean
    mask[-20:, :] = 0  # Beach
    result = extract_shoreline_from_classes(
        mask, min_ocean_component_px=100, min_beach_component_px=100
    )
    assert result.shoreline_available is False
    assert result.points == []
    assert result.calibrated_position_m is None
    assert (
        "beach" in (result.reason or "").lower()
        or "interface" in (result.reason or "").lower()
        or "insufficient_coastline_evidence" in (result.reason or "")
    )


def test_class_aware_calibration_uses_seaward_axis() -> None:
    mask = _synthetic_class_mask(width=100, height=100, sky_bottom=20, shoreline=50)
    calibration = CalibrationProfile(
        camera_id=1,
        version=7,
        homography=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        meters_per_pixel=None,
    )
    result = extract_shoreline_from_classes(
        mask,
        calibration=calibration,
        seaward_axis=(0.0, 1.0),
        min_ocean_component_px=100,
        min_beach_component_px=100,
    )
    assert result.calibrated_position_m == pytest.approx(50.0, abs=1.5)
    assert result.calibration_version == 7


def test_class_aware_rejects_non_2d_mask() -> None:
    with pytest.raises(ShorelineExtractionError):
        extract_shoreline_from_classes(np.zeros((10, 10, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Class-aware extraction: fragmentation and spurious-point regressions
# ---------------------------------------------------------------------------


def _coastal_mask(
    width: int = 640, height: int = 360, *, ocean_top: int = 100, shore: int = 200
) -> np.ndarray:
    """Sky above, Ocean band, Beach below (temporary land_below geometry)."""
    mask = np.full((height, width), 2, dtype=np.uint8)
    mask[ocean_top:shore] = 1
    mask[shore:] = 0
    return mask


def _all_segment_points(result) -> np.ndarray:
    if not result.segments:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate([np.asarray(segment, dtype=np.float64) for segment in result.segments])


def test_thin_vertical_occluder_is_bridged_into_one_segment() -> None:
    # A 9 px Beach-coloured trunk across the Ocean band. The frame-scaled
    # closing kernel (11 px at 640x360) must bridge it; the raw mask is not
    # modified, only the extraction input.
    mask = _coastal_mask()
    occluded = mask.copy()
    occluded[:, 315:324] = 0
    result = extract_shoreline_from_classes(occluded, close_kernel_px=11)
    assert result.shoreline_available is True
    assert len(result.segments) == 1
    assert len(result.segments[0]) >= mask.shape[1] - 10
    assert result.bridge_kernel_px == 11


def test_occluder_wider_than_kernel_produces_one_anchor_line() -> None:
    mask = _coastal_mask()
    occluded = mask.copy()
    occluded[:, 310:330] = 0  # 20 px: wider than the 11 px kernel
    result = extract_shoreline_from_classes(occluded, close_kernel_px=11)
    assert result.shoreline_available is True
    assert len(result.segments) == 1
    assert len(result.points) > 100
    legacy = extract_shoreline_from_classes(occluded, close_kernel_px=11, single_line_mode=False)
    assert len(legacy.segments) == 2


def test_small_beach_island_inside_ocean_contributes_no_points() -> None:
    mask = _coastal_mask()
    island = mask.copy()
    island[140:160, 300:320] = 0
    result = extract_shoreline_from_classes(island)
    assert result.shoreline_available is True
    points = _all_segment_points(result)
    assert points.size > 0
    inside_island = (
        (points[:, 0] >= 300) & (points[:, 0] < 320) & (points[:, 1] >= 140) & (points[:, 1] < 160)
    )
    assert not inside_island.any()
    assert len(result.segments) == 1


def test_border_margin_and_bottom_edge_are_excluded() -> None:
    mask = _coastal_mask()
    # Ocean fills the bottom edge in this column range: the waterline is out of
    # view there and no point may be invented.
    mask[200:, 500:540] = 1
    result = extract_shoreline_from_classes(mask)
    assert result.shoreline_available is True
    assert len(result.segments) == 1
    points = _all_segment_points(result)
    height, width = mask.shape
    margin = result.margin_px or 0
    assert points[:, 0].min() >= margin
    assert points[:, 0].max() < width - margin
    assert points[:, 1].min() >= margin
    assert points[:, 1].max() < height - margin
    assert not np.any((points[:, 0] >= 500) & (points[:, 0] < 540))


def test_sky_ocean_only_returns_empty_never_the_horizon() -> None:
    mask = _coastal_mask()
    mask[mask == 0] = 2  # replace all Beach with Sky
    result = extract_shoreline_from_classes(mask)
    assert result.shoreline_available is False
    assert result.segments == []
    assert result.points == []
    assert "beach" in (result.reason or "").lower()


def test_no_beach_ocean_contact_returns_empty_with_reason() -> None:
    mask = np.full((360, 640), 2, dtype=np.uint8)
    mask[100:200] = 1
    mask[220:] = 0  # Sky band separates Ocean and Beach
    result = extract_shoreline_from_classes(mask)
    assert result.shoreline_available is False
    assert result.points == []
    assert result.segments == []
    assert (
        "insufficient_coastline_evidence" in (result.reason or "")
        or "contact" in (result.reason or "").lower()
    )


def test_segments_are_ordered_bounded_and_legacy_points_match() -> None:
    mask = _coastal_mask()
    for x in range(mask.shape[1]):
        mask[200:, x] = 0
        mask[195 + int(6 * np.sin(x / 25.0)) : 200] = 1
        mask[200 : 205 + int(6 * np.sin(x / 25.0))] = 0
    result = extract_shoreline_from_classes(mask, max_gap_px=25.0, min_segment_px=8)
    assert result.shoreline_available is True
    assert len(result.segments) == 1
    points = np.asarray(result.segments[0], dtype=np.float64)
    assert len(points) >= 8
    # Ordered by scan direction (x), never sorted by y.
    assert np.all(np.diff(points[:, 0]) > 0)
    flattened = _all_segment_points(result)
    assert np.allclose(flattened, np.asarray(result.points, dtype=np.float64))


def test_side_geometry_is_inferred_and_scanned_by_row() -> None:
    # Beach left, Ocean right, Beach right: a CoastSnap-style side view.
    mask = np.full((360, 640), 2, dtype=np.uint8)
    mask[:, :200] = 0
    mask[:, 200:400] = 1
    mask[:, 400:] = 0
    result = extract_shoreline_from_classes(mask)
    assert result.shoreline_axis == "vertical"
    assert result.shoreline_available is True
    assert len(result.segments) == 1
    points = _all_segment_points(result)
    # One point per row, ordered along y, on one of the two Beach/Ocean sides.
    assert np.all(np.diff(points[:, 1]) > 0)
    assert np.all((points[:, 0] >= 195) & (points[:, 0] <= 405))


def test_explicit_orientation_overrides_auto() -> None:
    mask = _coastal_mask()
    result = extract_shoreline_from_classes(mask, orientation="land_below")
    assert result.orientation_used == "land_below"
    assert result.shoreline_available is True


def test_unknown_orientation_is_rejected() -> None:
    mask = _coastal_mask()
    with pytest.raises(ShorelineExtractionError, match="shoreline_orientation"):
        extract_shoreline_from_classes(mask, orientation="diagonal")


def test_extraction_parameters_are_reported_in_original_frame_pixels() -> None:
    mask = _coastal_mask(width=800, height=450)
    result = extract_shoreline_from_classes(mask)
    assert result.shoreline_available is True
    assert result.bridge_kernel_px and result.bridge_kernel_px % 2 == 1
    assert result.median_kernel_px and result.median_kernel_px % 2 == 1
    assert result.max_gap_px and result.max_gap_px > 0
    points = _all_segment_points(result)
    assert points[:, 0].max() <= mask.shape[1]
    assert points[:, 1].max() <= mask.shape[0]
    # Native frame coordinates, not a 1/4-resolution logit grid.
    assert points[:, 0].max() > mask.shape[1] / 2


# ---------------------------------------------------------------------------
# Extent regressions: scan-axis inference and the over-broad edge skip
# ---------------------------------------------------------------------------


def test_orientation_prefers_majority_side_over_land_below_by_default() -> None:
    """A diagonal Beach/Ocean contact must not default to ``land_below``.

    Beach is left of Ocean for every row, but the slope is shallow enough that
    many columns also have Beach below Ocean. The old inference preferred
    ``land_below`` whenever its count was at least 75% of the best side, which
    truncated oblique shorelines (audited 2026-11-26).
    """
    height, width = 120, 200
    yy, xx = np.mgrid[0:height, 0:width]
    line = 20 + 0.8 * yy
    mask = np.full((height, width), 2, dtype=np.uint8)
    mask[xx < line] = 0  # Beach left of the diagonal
    mask[xx >= line] = 1  # Ocean right of it
    result = extract_shoreline_from_classes(
        mask,
        min_ocean_component_px=10,
        min_beach_component_px=10,
        close_kernel_px=1,
    )
    assert result.shoreline_available is True
    assert result.orientation_used == "land_left"


def test_scanline_keeps_contact_when_foreground_ocean_blob_touches_edge() -> None:
    """A wet rock at the bottom edge must not discard the real contact above.

    The foreground Ocean blob is a separate connected component that touches the
    frame bottom. The old whole-scanline edge rule skipped every column it
    touched, even though a genuine Beach-adjacent contact existed higher up.
    """
    mask = np.full((120, 120), 2, dtype=np.uint8)
    mask[20:60, :] = 1  # main Ocean band
    mask[60:, :] = 0  # Beach below
    mask[100:120, 40:80] = 1  # foreground blob touching the bottom edge
    result = extract_shoreline_from_classes(
        mask,
        orientation="land_below",
        min_ocean_component_px=50,
        min_beach_component_px=50,
    )
    assert result.shoreline_available is True
    points = np.asarray(result.points, dtype=np.float64)
    # The real contact at the bottom of the Ocean band is kept.
    assert np.any((points[:, 0] == 50) & (points[:, 1] == 59))
    # No point is invented on the foreground blob near the frame edge.
    assert not np.any(points[:, 1] >= 100)


def test_side_contact_reaches_frame_border_with_zero_margin() -> None:
    """With no margin inset, a shoreline reaching the border is not clipped."""
    mask = np.full((80, 100), 2, dtype=np.uint8)
    mask[:, :50] = 0  # Beach left
    mask[:, 50:] = 1  # Ocean right
    result = extract_shoreline_from_classes(
        mask,
        orientation="land_left",
        margin_px=0,
        close_kernel_px=1,
        min_ocean_component_px=10,
        min_beach_component_px=10,
        min_segment_px=4,
    )
    assert result.shoreline_available is True
    points = np.asarray(result.points, dtype=np.float64)
    assert points[:, 1].max() == 79  # last row of the frame
    assert points[:, 0].min() == 50


def test_legacy_open_water_exiting_bottom_does_not_create_border_points() -> None:
    """Legacy rollback path keeps the old per-segment edge guard."""
    mask = np.full((100, 100), 2, dtype=np.uint8)
    mask[20:60] = 0  # Beach band
    mask[60:80] = 1  # Ocean band
    mask[80:] = 0  # Beach below
    mask[80:, 50:] = 1  # right half: open water exits the frame, no Beach below
    result = extract_shoreline_from_classes(
        mask,
        orientation="land_below",
        margin_px=0,
        close_kernel_px=1,
        min_ocean_component_px=10,
        min_beach_component_px=10,
        min_segment_px=4,
        single_line_mode=False,
    )
    assert result.shoreline_available is True
    points = np.asarray(result.points, dtype=np.float64)
    assert points[:, 1].max() == 79  # the real contact above the out-of-view water
    assert not np.any(points[:, 1] >= 80)
    assert not np.any(points[:, 0] >= 50)  # no point where water exits the frame


def test_pier_standing_in_water_selects_one_anchor_not_two_segments() -> None:
    mask = _coastal_mask()
    # A 30 px non-Ocean pier cuts the Ocean band. Under the single-line
    # contract exactly one side is the anchor; the other is discarded.
    mask[100:200, 300:330] = 2
    result = extract_shoreline_from_classes(mask, close_kernel_px=11)
    assert result.shoreline_available is True
    assert len(result.segments) == 1
    legacy = extract_shoreline_from_classes(mask, close_kernel_px=11, single_line_mode=False)
    assert len(legacy.segments) == 2
