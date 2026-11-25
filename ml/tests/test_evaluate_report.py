"""Report rendering and architecture selection from synthetic evaluation payloads."""

from __future__ import annotations

import json
from pathlib import Path

from ml.src.evaluation.compare import build_comparison
from ml.src.evaluation.evaluate import render_markdown
from ml.src.utils import load_yaml


def _metrics(dataset: str, iou: float) -> dict:
    return {
        "datasets": {
            dataset: {
                "test": {
                    "dataset": dataset,
                    "split": "test",
                    "images": 3,
                    "supervised_pixels": 1000,
                    "supervised_mean_iou": iou,
                    "mean_iou": iou,
                    "mean_dice": iou,
                    "pixel_accuracy": 0.9,
                    "inference_ms_per_image": 10.0,
                    "per_class": {
                        "0": {
                            "support": 0,
                            "predicted": 0,
                            "iou": None,
                            "dice": None,
                            "precision": None,
                            "recall": None,
                        },
                        "1": {
                            "support": 600,
                            "predicted": 620,
                            "iou": iou,
                            "dice": iou,
                            "precision": 0.95,
                            "recall": 0.9,
                        },
                        "2": {
                            "support": 400,
                            "predicted": 380,
                            "iou": iou,
                            "dice": iou,
                            "precision": 0.9,
                            "recall": 0.95,
                        },
                        "3": {
                            "support": 0,
                            "predicted": 0,
                            "iou": None,
                            "dice": None,
                            "precision": None,
                            "recall": None,
                        },
                    },
                    "boundary": {
                        "1": {"images": 3, "f1": 0.8},
                        "2": {"images": 3, "f1": 0.7},
                        "mean_f1": 0.75,
                    },
                    "shoreline": {
                        "available": True,
                        "images": 3,
                        "boundary_f1": 0.7,
                        "boundary_precision": 0.7,
                        "boundary_recall": 0.7,
                        "mean_boundary_distance_px": 4.0,
                        "mean_vertical_displacement_px": 3.0,
                    },
                    "note": "note",
                }
            }
        },
        "classes": ["unknown", "land", "water", "foam"],
    }


def test_markdown_reports_foam_as_na_and_keeps_pixel_units() -> None:
    results = {
        "architecture": "unet",
        "device": "cpu",
        "split": ["test"],
        "unsupervised_classes": [3],
        "figures": {},
        **_metrics("sclabels", 0.8),
    }
    markdown = render_markdown(results)
    assert "N/A (untrained)" in markdown
    assert "px" in markdown
    assert "shoreline_geometry" not in markdown


def test_compare_selects_the_higher_mean_iou(tmp_path: Path) -> None:
    config = load_yaml("ml/configs/config.yaml")
    config["paths"]["reports"] = str(tmp_path)
    for name, iou in (("unet", 0.82), ("deeplabv3plus", 0.74)):
        (tmp_path / f"evaluation_{name}.json").write_text(
            json.dumps(_metrics("sclabels", iou)), encoding="utf-8"
        )

    result = build_comparison(
        config,
        {
            "unet": tmp_path / "evaluation_unet.json",
            "deeplabv3plus": tmp_path / "evaluation_deeplabv3plus.json",
        },
    )
    assert result["selected"] == "unet"
    report = (tmp_path / "model_selection.md").read_text(encoding="utf-8")
    assert "Selected: `unet`" in report
    assert "Foam is excluded" in report
