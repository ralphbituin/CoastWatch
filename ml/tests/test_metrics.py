"""Metric correctness on synthetic masks with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from ml.src.training.metrics import (
    ConfusionMatrix,
    binary_boundary,
    boundary_f1,
    shoreline_metrics_from_masks,
)


def _two_class_mask(size: int = 64, split: int = 32) -> np.ndarray:
    mask = np.ones((size, size), dtype=np.uint8)
    mask[:split] = 2
    return mask


def test_confusion_matrix_perfect_prediction() -> None:
    target = _two_class_mask()
    confusion = ConfusionMatrix(num_classes=4, ignore_index=255)
    confusion.update(target.copy(), target)
    metrics = confusion.compute()
    assert metrics["mean_iou"] == pytest.approx(1.0)
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)
    assert metrics["per_class"][1]["iou"] == pytest.approx(1.0)
    assert metrics["per_class"][2]["iou"] == pytest.approx(1.0)
    assert metrics["per_class"][3]["iou"] is None  # never present


def test_confusion_matrix_ignores_255() -> None:
    target = _two_class_mask()
    target[40:, :] = 255
    prediction = np.zeros_like(target)
    prediction[:32] = 2
    prediction[32:, :] = 1
    confusion = ConfusionMatrix(num_classes=4, ignore_index=255)
    confusion.update(prediction, target)
    metrics = confusion.compute()
    assert metrics["supervised_pixels"] == 40 * 64  # 32 water rows + 8 land rows
    assert metrics["pixel_accuracy"] == pytest.approx(1.0)


def test_confusion_matrix_counts_only_present_classes_in_mean_iou() -> None:
    target = np.full((10, 10), 1, dtype=np.uint8)
    prediction = np.full((10, 10), 1, dtype=np.uint8)
    prediction[:2] = 2  # 20 % false water, no water in GT
    confusion = ConfusionMatrix(num_classes=4, ignore_index=255)
    confusion.update(prediction, target)
    metrics = confusion.compute()
    assert metrics["per_class"][2]["iou"] == pytest.approx(0.0)
    assert metrics["per_class"][1]["iou"] == pytest.approx(0.8)
    # mean IoU averages classes present in the ground truth, so the
    # never-annotated water class does not dilute the score.
    assert metrics["mean_iou"] == pytest.approx(0.8)


def test_binary_boundary_marks_the_edge() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5:, :] = 1
    boundary = binary_boundary(mask)
    assert boundary[5].all()
    assert not boundary[4].any()
    assert not boundary[6].any()


def test_boundary_f1_is_one_for_identical_masks() -> None:
    mask = _two_class_mask()
    result = boundary_f1(mask, mask, tolerance_px=1)
    assert result["1"]["f1"] == pytest.approx(1.0)
    assert result["2"]["f1"] == pytest.approx(1.0)
    assert result["mean_f1"] == pytest.approx(1.0)


def test_boundary_f1_degrades_when_shifted_beyond_tolerance() -> None:
    target = _two_class_mask(size=64, split=32)
    shifted = _two_class_mask(size=64, split=40)
    result = boundary_f1(shifted, target, tolerance_px=1)
    assert result["2"]["f1"] == pytest.approx(0.0)


def test_shoreline_metrics_reward_a_correct_waterline() -> None:
    target = _two_class_mask(size=64, split=32)
    result = shoreline_metrics_from_masks(target.copy(), target, tolerance_px=1)
    assert result["available"] is True
    assert result["boundary_f1"] == pytest.approx(1.0)
    assert result["mean_boundary_distance_px"] == pytest.approx(0.0, abs=1e-6)


def test_shoreline_metrics_report_displacement_in_pixels() -> None:
    target = _two_class_mask(size=64, split=32)
    prediction = _two_class_mask(size=64, split=36)
    result = shoreline_metrics_from_masks(prediction, target, tolerance_px=1)
    assert result["available"] is True
    assert result["mean_vertical_displacement_px"] == pytest.approx(4.0, abs=0.1)
    assert result["mean_boundary_distance_px"] == pytest.approx(4.0, abs=0.1)


def test_shoreline_metrics_unavailable_without_water() -> None:
    target = np.full((16, 16), 1, dtype=np.uint8)
    result = shoreline_metrics_from_masks(target, target)
    assert result["available"] is False
