"""Regression tests for false-Ocean component selection.

The real-frame fixtures are 640x360 nearest-neighbour copies of the native class
masks produced by the deployed artifact for frames audited in
``ml/reports/false_ocean_hijack_audit.md``. They are masks only; no model,
weights or calibration data are part of the fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline_from_classes,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "false_ocean"


def _coast(width: int = 240, height: int = 160) -> np.ndarray:
    mask = np.full((height, width), 2, dtype=np.uint8)
    mask[20:60] = 1  # Ocean band
    mask[60:] = 0  # Beach below
    return mask


def _points(result) -> np.ndarray:
    return np.asarray(result.points, dtype=np.float64) if result.points else np.empty((0, 2))


def _points_in_bbox(points: np.ndarray, bbox: list[int]) -> int:
    if points.size == 0:
        return 0
    x, y, width, height = bbox
    return int(
        np.count_nonzero(
            (points[:, 0] >= x)
            & (points[:, 0] < x + width)
            & (points[:, 1] >= y)
            & (points[:, 1] < y + height)
        )
    )


def test_false_ocean_blob_inside_land_is_not_scanned() -> None:
    mask = _coast()
    mask[90:120, 150:200] = 1  # false Ocean island surrounded by Beach

    before = extract_shoreline_from_classes(
        mask, ocean_component_selection="all", single_line_mode=False
    )
    after = extract_shoreline_from_classes(mask)
    false_bbox = [150, 90, 50, 30]

    assert _points_in_bbox(_points(before), false_bbox) > 0
    assert after.shoreline_available is True
    assert _points_in_bbox(_points(after), false_bbox) == 0
    assert len(after.segments) == 1


def test_false_ocean_touching_a_disconnected_beach_is_not_bridged() -> None:
    mask = np.full((160, 240), 2, dtype=np.uint8)
    mask[20:60, 0:120] = 1  # real Ocean
    mask[60:160, 0:120] = 0  # dominant Beach
    mask[100:160, 150:240] = 0  # separate Beach patch
    mask[80:100, 150:220] = 1  # false Ocean touching only the separate Beach

    before = extract_shoreline_from_classes(
        mask, ocean_component_selection="all", single_line_mode=False
    )
    after = extract_shoreline_from_classes(mask)
    false_bbox = [150, 80, 70, 20]

    assert _points_in_bbox(_points(before), false_bbox) > 0
    assert _points_in_bbox(_points(after), false_bbox) == 0
    assert len(after.segments) == 1
    # The real line stays on the dominant coast; it is not extended to x>=150.
    assert _points(after)[:, 0].max() < 120


def test_wide_occluder_produces_one_anchor_line() -> None:
    mask = _coast(width=640, height=360)
    mask[20:60, 310:330] = 2  # Sky-coloured pier wider than the closing kernel

    result = extract_shoreline_from_classes(
        mask, close_kernel_px=11, min_ocean_component_px=100, min_beach_component_px=100
    )

    assert result.shoreline_available is True
    assert len(result.segments) == 1
    legacy = extract_shoreline_from_classes(
        mask,
        close_kernel_px=11,
        min_ocean_component_px=100,
        min_beach_component_px=100,
        single_line_mode=False,
    )
    assert len(legacy.segments) == 2


def test_single_ocean_component_is_unchanged_by_selection() -> None:
    mask = _coast(width=320, height=180)
    before = extract_shoreline_from_classes(
        mask, ocean_component_selection="all", single_line_mode=False
    )
    after = extract_shoreline_from_classes(
        mask, ocean_component_selection="coastline", single_line_mode=False
    )

    assert np.array_equal(_points(before), _points(after))
    assert before.segments == after.segments
    assert after.ocean_components_rejected_geometry == 0


def test_invalid_ocean_component_selection_is_rejected() -> None:
    with pytest.raises(ShorelineExtractionError, match="ocean_component_selection"):
        extract_shoreline_from_classes(_coast(), ocean_component_selection="nearest-largest")


def _diagonal_coast(width: int = 320, height: int = 200) -> np.ndarray:
    mask = np.full((height, width), 2, dtype=np.uint8)
    for x in range(width):
        boundary = int(100 - 0.15 * x)
        mask[30:boundary, x] = 1  # Ocean above a diagonal Beach boundary
        mask[boundary:, x] = 0  # Beach below
    return mask


def test_diagonal_landward_strip_touching_border_is_rejected() -> None:
    mask = _diagonal_coast()
    mask[120:132, 250:320] = 1  # thin landward Ocean strip touching the border

    before = extract_shoreline_from_classes(
        mask, min_branch_depth_ratio=0.0, single_line_mode=False
    )
    after = extract_shoreline_from_classes(mask)
    false_bbox = [250, 120, 70, 12]

    assert _points_in_bbox(_points(before), false_bbox) > 0
    assert _points_in_bbox(_points(after), false_bbox) == 0
    assert len(after.segments) == 1
    assert len(before.segments) == 2


def test_diagonal_pier_split_selects_one_anchor() -> None:
    mask = _diagonal_coast()
    mask[30:95, 150:170] = 2  # Sky pier splitting the real diagonal sea band

    result = extract_shoreline_from_classes(mask)
    legacy = extract_shoreline_from_classes(mask, single_line_mode=False)

    assert result.shoreline_available is True
    assert len(result.segments) == 1
    assert len(legacy.segments) == 2


def test_branch_depth_filter_can_be_disabled() -> None:
    mask = _diagonal_coast()
    mask[120:132, 250:320] = 1
    result = extract_shoreline_from_classes(
        mask, min_branch_depth_ratio=0.0, single_line_mode=False
    )
    assert result.branches_rejected_depth == 0
    assert len(result.segments) == 2


def test_invalid_branch_depth_ratio_is_rejected() -> None:
    with pytest.raises(ShorelineExtractionError, match="min_branch_depth_ratio"):
        extract_shoreline_from_classes(_coast(), min_branch_depth_ratio=1.5)


def test_diagonal_reference_keeps_real_coastline_unchanged() -> None:
    metadata = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))[
        "koh_samui_diagonal_backshore"
    ]
    mask = cv2.imread(str(FIXTURES / "koh_samui_diagonal_backshore.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None
    before = extract_shoreline_from_classes(
        mask, min_branch_depth_ratio=0.0, single_line_mode=False
    )
    after = extract_shoreline_from_classes(mask)
    false_bbox = metadata["false_bbox"]

    assert len(before.segments) > 1
    assert len(after.segments) == 1
    assert _points_in_bbox(_points(after), false_bbox) == 0
    assert after.shoreline_axis == "vertical"
    # The selected real trace stays on the left coast, far from the stairs.
    assert _points(after)[:, 0].max() < 400


def test_single_line_bridges_an_occlusion_gap_with_interpolation() -> None:
    mask = _coast(width=240, height=160)
    # A notch reaches the shoreline but leaves Ocean connected above it.
    mask[40:80, 110:130] = 2

    result = extract_shoreline_from_classes(mask)

    assert result.shoreline_available is True
    assert len(result.segments) == 1
    assert result.interpolated_ranges
    points = _points(result)
    bridged = points[(points[:, 0] >= 110) & (points[:, 0] < 130)]
    assert len(bridged) == 20
    # The bridge is a straight interpolation, so y is constant for this flat coast.
    assert np.allclose(bridged[:, 1], np.median(bridged[:, 1]))


def test_single_line_rejects_sparse_fragmented_evidence() -> None:
    mask = np.full((360, 640), 2, dtype=np.uint8)
    mask[100:150, 300:340] = 1
    mask[150:200, 300:340] = 0

    result = extract_shoreline_from_classes(mask)

    assert result.shoreline_available is False
    assert result.reason == "insufficient_coastline_evidence"
    assert result.points == []


def test_legacy_mode_preserves_segmented_output() -> None:
    mask = _coast(width=640, height=360)
    mask[20:60, 310:330] = 2
    result = extract_shoreline_from_classes(mask, close_kernel_px=11, single_line_mode=False)
    assert len(result.segments) == 2
    assert result.shoreline_axis is None
    assert result.interpolated_ranges == []


@pytest.mark.parametrize(
    "fixture",
    [
        "karon_rock",
        "karon_shadow",
        "karon_reflection",
        "koh_samui_deck_a",
        "koh_samui_deck_b",
        "el_gaucho_chairs",
        "koh_samui_diagonal_backshore",
        "live_daytime_deck",
    ],
)
def test_real_false_ocean_fixture_drops_the_land_side_component(fixture: str) -> None:
    metadata = json.loads((FIXTURES / "fixtures.json").read_text(encoding="utf-8"))[fixture]
    mask = cv2.imread(str(FIXTURES / f"{fixture}.png"), cv2.IMREAD_GRAYSCALE)
    assert mask is not None and mask.shape == (360, 640)

    before = extract_shoreline_from_classes(
        mask, ocean_component_selection="all", single_line_mode=False
    )
    after = extract_shoreline_from_classes(mask)
    bbox = metadata["false_bbox"]

    assert _points_in_bbox(_points(before), bbox) > 0
    assert _points_in_bbox(_points(after), bbox) == 0
    assert len(after.segments) == 1
    assert after.shoreline_axis in ("horizontal", "vertical")
