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
