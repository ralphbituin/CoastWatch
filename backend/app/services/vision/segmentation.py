"""Semantic segmentation (Phase 4).

Planned: U-Net / DeepLabV3+ producing a water/land/foam/unknown mask, loaded
from ``MODEL_PATH`` (weights are never committed). ONNX Runtime is used for
optimized inference. A pretrained model is only trusted after validation on
actual coastal footage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SegmentationResult:
    mask: Any  # 2-D integer class mask
    confidence: float
    classes: dict[int, str]


class ShorelineSegmenter:
    """Loads and runs a segmentation model."""

    def __init__(self, model_path: str, *, device: str = "cpu") -> None:
        self.model_path = model_path
        self.device = device

    def load(self) -> None:
        raise NotImplementedError("Segmentation is implemented in Phase 4.")

    def predict(self, frame: Any) -> SegmentationResult:
        raise NotImplementedError("Segmentation is implemented in Phase 4.")
