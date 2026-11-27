"""Tests for the three-class SegFormer dataset, mapping and shoreline modules."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from ml.scripts.prepare_dataset import _build_lut, _convert_mask
from ml.src.metrics import ConfusionMatrix
from ml.src.shoreline_extraction import extract_shoreline


def test_lut_maps_source_order_to_target_order() -> None:
    lut = _build_lut(
        {
            0: {"name": "sky", "target": 2},
            1: {"name": "ocean", "target": 1},
            2: {"name": "beach", "target": 0},
            255: {"name": "ignore", "target": 255},
        }
    )
    assert lut[0] == 2
    assert lut[1] == 1
    assert lut[2] == 0
    assert lut[255] == 255


def test_uncovered_pixels_become_ignore(tmp_path) -> None:
    source = tmp_path / "source.png"
    Image.fromarray(np.full((4, 4), 0, dtype=np.uint8)).save(source)
    destination = tmp_path / "out.png"
    lut = _build_lut({0: {"target": 2}, 255: {"target": 255}})
    region = np.zeros((4, 4), dtype=bool)
    region[:2] = True
    output, counts = _convert_mask(source, destination, lut, valid_region=region)
    assert np.all(output[:2] == 2)
    assert np.all(output[2:] == 255)
    assert counts["2"] == 8
    assert counts["ignore"] == 8


def test_invalid_source_value_is_rejected(tmp_path) -> None:
    source = tmp_path / "source.png"
    Image.fromarray(np.full((2, 2), 7, dtype=np.uint8)).save(source)
    lut = _build_lut({0: {"target": 2}, 255: {"target": 255}})
    with pytest.raises(ValueError, match="unmapped source class ids"):
        _convert_mask(source, tmp_path / "out.png", lut)


def test_metrics_ignore_unlabelled_pixels() -> None:
    target = np.array([[0, 0, 255, 255]])
    prediction = np.array([[0, 1, 2, 2]])
    confusion = ConfusionMatrix()
    confusion.update(prediction, target)
    metrics = confusion.compute()
    assert metrics["supervised_pixels"] == 2
    assert metrics["per_class"]["Beach"]["iou"] == 0.5
    assert metrics["per_class"]["Beach"]["dice"] == pytest.approx(2 / 3)
    assert metrics["per_class"]["Sky"]["support_pixels"] == 0
    assert metrics["per_class"]["Sky"]["iou"] is None


def _synthetic_coast() -> np.ndarray:
    mask = np.zeros((120, 200), dtype=np.uint8)
    mask[:40] = 2  # sky
    # A wavy ocean/beach boundary around row 60-68.
    for x in range(200):
        boundary = 60 + int(8 * np.sin(x / 18.0))
        mask[40:boundary, x] = 1
        mask[boundary:, x] = 0
    return mask


def test_shoreline_preserves_wave_variation() -> None:
    mask = _synthetic_coast()
    result = extract_shoreline(mask, min_ocean_component_px=100, min_beach_component_px=100)
    assert result.metadata["available"] is True
    assert result.valid_columns > 180
    assert result.per_column.shape[1] == 2
    # The polyline must follow the sine wave rather than flatten to one row.
    ys = result.per_column[:, 1]
    assert ys.max() - ys.min() >= 10
    # Points keep the full interface, including wave detail.
    assert result.points.shape[0] >= result.valid_columns


def test_shoreline_rejects_tiny_ocean_patch() -> None:
    mask = np.zeros((60, 60), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    result = extract_shoreline(mask, min_ocean_component_px=5_000)
    assert result.metadata["available"] is False
    assert result.per_column.size == 0


def _component_mask(width: int = 240, height: int = 160) -> np.ndarray:
    mask = np.full((height, width), 2, dtype=np.uint8)
    mask[20:60] = 1  # Ocean band
    mask[60:] = 0  # Beach below
    return mask


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
    mask = _component_mask()
    mask[90:120, 150:200] = 1  # false Ocean island surrounded by Beach
    false_bbox = [150, 90, 50, 30]

    before = extract_shoreline(mask, ocean_component_selection="all", single_line_mode=False)
    after = extract_shoreline(mask)

    assert _points_in_bbox(before.points, false_bbox) > 0
    assert _points_in_bbox(after.points, false_bbox) == 0
    assert after.metadata["available"] is True
    assert len(after.segments) == 1


def test_false_ocean_touching_a_disconnected_beach_is_not_bridged() -> None:
    mask = np.full((160, 240), 2, dtype=np.uint8)
    mask[20:60, 0:120] = 1  # real Ocean
    mask[60:160, 0:120] = 0  # dominant Beach
    mask[100:160, 150:240] = 0  # separate Beach patch
    mask[80:100, 150:220] = 1  # false Ocean touching only the separate Beach
    false_bbox = [150, 80, 70, 20]

    before = extract_shoreline(mask, ocean_component_selection="all", single_line_mode=False)
    after = extract_shoreline(mask)

    assert _points_in_bbox(before.points, false_bbox) > 0
    assert _points_in_bbox(after.points, false_bbox) == 0
    assert len(after.segments) == 1
    assert after.points[:, 0].max() < 120


def test_wide_occluder_selects_one_anchor() -> None:
    mask = _component_mask(width=640, height=360)
    mask[20:60, 310:330] = 2  # Sky-coloured pier wider than the closing kernel

    result = extract_shoreline(
        mask, close_kernel=11, min_ocean_component_px=100, min_beach_component_px=100
    )
    legacy = extract_shoreline(
        mask,
        close_kernel=11,
        min_ocean_component_px=100,
        min_beach_component_px=100,
        single_line_mode=False,
    )

    assert result.metadata["available"] is True
    assert len(result.segments) == 1
    assert len(legacy.segments) == 2


def test_single_ocean_component_is_unchanged_by_selection() -> None:
    mask = _component_mask(width=320, height=180)
    before = extract_shoreline(mask, ocean_component_selection="all", single_line_mode=False)
    after = extract_shoreline(mask, ocean_component_selection="coastline", single_line_mode=False)

    assert np.array_equal(before.points, after.points)
    assert before.metadata["ocean_components_rejected_geometry"] == 0
    assert after.metadata["ocean_components_rejected_geometry"] == 0


def _diagonal_component_mask(width: int = 320, height: int = 200) -> np.ndarray:
    mask = np.full((height, width), 2, dtype=np.uint8)
    for x in range(width):
        boundary = int(100 - 0.15 * x)
        mask[30:boundary, x] = 1
        mask[boundary:, x] = 0
    return mask


def test_diagonal_landward_strip_touching_border_is_rejected() -> None:
    mask = _diagonal_component_mask()
    mask[120:132, 250:320] = 1
    false_bbox = [250, 120, 70, 12]

    before = extract_shoreline(mask, min_branch_depth_ratio=0.0, single_line_mode=False)
    after = extract_shoreline(mask)

    assert _points_in_bbox(before.points, false_bbox) > 0
    assert _points_in_bbox(after.points, false_bbox) == 0
    assert len(after.segments) == 1
    assert len(before.segments) == 2


def test_single_line_bridges_an_occlusion_gap_with_interpolation() -> None:
    mask = _component_mask(width=240, height=160)
    mask[40:80, 110:130] = 2

    result = extract_shoreline(mask)

    assert result.metadata["available"] is True
    assert len(result.segments) == 1
    assert result.metadata["interpolated_ranges"]
    points = result.points
    bridged = points[(points[:, 0] >= 110) & (points[:, 0] < 130)]
    assert len(bridged) == 20
    assert np.allclose(bridged[:, 1], np.median(bridged[:, 1]))


def test_single_line_rejects_sparse_fragmented_evidence() -> None:
    mask = np.full((360, 640), 2, dtype=np.uint8)
    mask[100:150, 300:340] = 1
    mask[150:200, 300:340] = 0

    result = extract_shoreline(mask)

    assert result.metadata["available"] is False
    assert result.metadata["reason"] == "insufficient_coastline_evidence"
    assert result.points.size == 0


def test_diagonal_pier_split_selects_one_anchor() -> None:
    mask = _diagonal_component_mask()
    mask[30:95, 150:170] = 2

    result = extract_shoreline(mask)
    legacy = extract_shoreline(mask, single_line_mode=False)

    assert result.metadata["available"] is True
    assert len(result.segments) == 1
    assert len(legacy.segments) == 2
