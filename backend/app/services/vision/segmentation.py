"""Semantic segmentation (Phase 4) — **no model yet**.

Planned: U-Net / DeepLabV3+ producing a water/land/foam/unknown mask, loaded
from ``MODEL_PATH`` (weights are never committed). ONNX Runtime is used for
optimized inference. A pretrained model is only trusted after validation on
actual coastal footage.

Until a validated artifact exists, :meth:`ShorelineSegmenter.load` raises
:class:`ModelUnavailableError` — predictors never return a fabricated mask.
Operator-labelled masks flow through the same extraction pipeline via
``POST /api/v1/shorelines/observations`` and ``extract_shoreline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import ModelUnavailableError


@dataclass(slots=True)
class SegmentationResult:
    mask: Any  # 2-D integer class mask
    confidence: float
    classes: dict[int, str]


def find_model_artifact(model_path: str | Path | None = None) -> Path | None:
    """Return the path to a usable ONNX artifact, or ``None``.

    Looks for ``<MODEL_PATH>/shoreline-segmentation/model.onnx`` by default —
    the layout prescribed in docs/MODEL_DOCUMENTATION.md. Never downloads or
    invents an artifact.
    """
    base = Path(model_path) if model_path is not None else Path(get_settings().model_path)
    artifact = base / "shoreline-segmentation" / "model.onnx"
    return artifact if artifact.is_file() else None


class ShorelineSegmenter:
    """Loads and runs a segmentation model when one exists."""

    def __init__(self, model_path: str | Path | None = None, *, device: str = "cpu") -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self.device = device

    def load(self) -> None:
        artifact = find_model_artifact(self.model_path)
        if artifact is None:
            raise ModelUnavailableError(
                "No segmentation model artifact found. Train or provision one under "
                "MODEL_PATH/shoreline-segmentation/model.onnx (see "
                "docs/MODEL_DOCUMENTATION.md), or supply operator-labelled masks via "
                "POST /api/v1/shorelines/observations."
            )
        # Artifact present: inference wiring lands with the first validated model.
        raise ModelUnavailableError(
            f"Segmentation artifact present at {artifact} but inference is not wired up "
            "yet — it becomes available after a validated model's config.json is in "
            "place and this path is implemented (Phase 4 model training)."
        )

    def predict(self, frame: Any) -> SegmentationResult:
        self.load()
        raise ModelUnavailableError("segmentation inference is not wired up yet")
