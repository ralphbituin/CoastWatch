"""Checkpoint-selection scoring rules."""

from __future__ import annotations

import pytest

from ml.src.training.trainer import _selection_score

CONFIG = {"training": {"selection_metric": "mean_dataset_iou"}}


def _validation() -> dict:
    return {
        "per_dataset": {
            "india_sandy_beach": {"supervised_mean_iou": 0.5, "val_loss": 0.9},
            "sclabels": {"supervised_mean_iou": 0.6, "val_loss": 0.4},
        }
    }


def test_default_is_mean_supervised_iou_over_all_datasets() -> None:
    assert _selection_score(CONFIG, {}, _validation()) == pytest.approx(0.55)


def test_selection_datasets_are_honoured() -> None:
    stage = {"selection_datasets": ["sclabels"]}
    assert _selection_score(CONFIG, stage, _validation()) == pytest.approx(0.6)


def test_negative_val_loss_is_inverted() -> None:
    stage = {"selection_metric": "negative_val_loss"}
    assert _selection_score(CONFIG, stage, _validation()) == -0.65


def test_missing_metrics_are_safe() -> None:
    validation = {"per_dataset": {"x": {"supervised_mean_iou": None, "val_loss": None}}}
    assert _selection_score(CONFIG, {}, validation) == 0.0
    assert _selection_score(CONFIG, {"selection_metric": "negative_val_loss"}, validation) == -1e9
