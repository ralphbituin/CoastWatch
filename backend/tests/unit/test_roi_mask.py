"""Monitoring-outline validation and mask enforcement."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.vision.roi import (
    RoiPolygonError,
    apply_roi_to_binary_mask,
    apply_roi_to_class_mask,
    normalize_roi_polygon,
    polygon_area,
    polygon_mask,
)
from app.services.vision.shoreline import extract_shoreline_from_classes


def _roi(points: list[list[float]]) -> dict:
    return normalize_roi_polygon({"points": points, "frame_width": 100, "frame_height": 80})


def _coast_mask(height: int = 80, width: int = 100) -> np.ndarray:
    """Ocean above a Beach band, with a false Ocean/Beach island on the right."""
    mask = np.full((height, width), 2, dtype=np.uint8)  # Sky
    mask[10:40, :] = 1  # Ocean
    mask[40:, :] = 0  # Beach
    # False distant island in the right third: Ocean with a Beach rim below.
    mask[12:22, 80:96] = 1
    mask[22:26, 80:96] = 0
    return mask


def test_outline_removes_false_ocean_beach_component_and_keeps_left() -> None:
    roi = _roi([[0.0, 0.0], [0.65, 0.0], [0.65, 1.0], [0.0, 1.0]])
    before = extract_shoreline_from_classes(_coast_mask(), min_segment_px=1)
    after = extract_shoreline_from_classes(_coast_mask(), min_segment_px=1, monitoring_polygon=roi)

    assert before.shoreline_available
    assert after.shoreline_available
    # The false island cannot contribute a point right of the outline.
    for segment in after.segments:
        for x, _y in segment:
            assert x <= 65
    # The real in-outline shoreline is still measured.
    assert after.valid_columns > 0
    assert after.measured_coverage is not None


def test_apply_roi_returns_copy_and_sets_outside_to_sky() -> None:
    class_mask = _coast_mask()
    original = class_mask.copy()
    roi = _roi([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])

    masked = apply_roi_to_class_mask(class_mask, roi)

    assert masked is not class_mask
    assert np.array_equal(class_mask, original)  # input never mutated
    assert np.all(masked[:, 51:] == 2)
    assert np.array_equal(masked[:, :50], class_mask[:, :50])


def test_no_outline_is_a_no_op() -> None:
    class_mask = _coast_mask()
    assert apply_roi_to_class_mask(class_mask, None) is class_mask
    assert apply_roi_to_binary_mask(np.ones((4, 4), dtype=np.uint8), None) is not None


def test_binary_mask_outside_becomes_zero() -> None:
    water = np.full((10, 10), 255, dtype=np.uint8)
    roi = _roi([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
    masked = apply_roi_to_binary_mask(water, roi)
    assert np.all(masked[:, 6:] == 0)
    assert np.all(masked[:, :4] == 255)


def test_non_square_normalized_scaling() -> None:
    roi = _roi([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
    mask = polygon_mask([(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)], (80, 100))
    assert mask.shape == (80, 100)
    assert mask[0, 0] == 255
    assert mask[79, 99] == 0
    assert mask[39, 49] == 255
    assert mask[41, 51] == 0
    assert roi is not None


def test_concave_polygon_is_accepted() -> None:
    roi = _roi(
        [
            [0.1, 0.1],
            [0.8, 0.1],
            [0.8, 0.8],
            [0.5, 0.4],
            [0.2, 0.8],
        ]
    )
    assert roi is not None
    assert polygon_area([tuple(p) for p in roi["points"]]) > 0


@pytest.mark.parametrize(
    "points, message",
    [
        ([[0.0, 0.0], [1.0, 0.0]], "at least"),
        ([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]], "self-intersecting"),
        ([[0.2, 0.2], [0.2, 0.2], [0.8, 0.2], [0.8, 0.8]], "duplicate"),
        ([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], "area"),
        ([[-0.1, 0.0], [1.0, 0.0], [1.0, 1.0]], "outside"),
        ([[0.0, 0.0], [1.0001, 0.0], [1.0, 1.0]], "outside"),
    ],
)
def test_invalid_polygons_are_rejected(points, message) -> None:
    with pytest.raises(RoiPolygonError, match=message):
        normalize_roi_polygon({"points": points})


def test_boundary_vertices_are_clamped() -> None:
    roi = normalize_roi_polygon({"points": [[-1e-12, 0.0], [1.0 + 1e-12, 0.0], [1.0, 1.0]]})
    assert roi is not None
    assert roi["points"][0] == [0.0, 0.0]
    assert roi["points"][1] == [1.0, 0.0]
