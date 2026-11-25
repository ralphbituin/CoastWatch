"""Cross-entropy + Dice loss with explicit ignore handling.

Classes with no supervision (foam in the provided datasets) get weight 0 in
the cross-entropy and are skipped by the Dice term. They never receive a
fabricated target, and they do not drag the reported loss around.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


class DiceLoss(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Soft multi-class Dice over non-ignored pixels only."""

    def __init__(self, num_classes: int = 4, ignore_index: int = 255, eps: float = 1e-6) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: Any, target: Any) -> Any:
        probabilities = F.softmax(logits, dim=1)
        valid = target != self.ignore_index
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        one_hot = F.one_hot(safe_target, self.num_classes).permute(0, 3, 1, 2).float()
        valid_f = valid.unsqueeze(1).float()
        probabilities = probabilities * valid_f
        one_hot = one_hot * valid_f

        intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
        denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        dice = (2.0 * intersection + self.eps) / (denominator + self.eps)

        present = denominator > self.eps
        if not bool(present.any()):
            return logits.sum() * 0.0
        return 1.0 - dice[present].mean()


class SegmentationLoss(nn.Module if nn is not None else object):  # type: ignore[misc]
    """Weighted cross-entropy + Dice."""

    def __init__(
        self,
        num_classes: int = 4,
        ignore_index: int = 255,
        class_weights: np.ndarray | list[float] | None = None,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        if ce_weight < 0 or dice_weight < 0 or ce_weight + dice_weight <= 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)
        weight = None
        if class_weights is not None:
            weight = torch.as_tensor(np.asarray(class_weights, dtype=np.float32))
            if weight.numel() != num_classes:
                raise ValueError("class_weights length must equal num_classes")
        self.register_buffer("class_weights", weight if weight is not None else torch.empty(0))
        self.ce = nn.CrossEntropyLoss(weight=None, ignore_index=ignore_index, reduction="mean")
        self.dice = DiceLoss(num_classes=num_classes, ignore_index=ignore_index)

    def forward(self, logits: Any, target: Any) -> tuple[Any, dict[str, float]]:
        valid = target != self.ignore_index
        if not bool(valid.any()):
            # A fully-ignored crop contributes no supervision. Returning a
            # zero loss (with a live graph) keeps the step well-defined instead
            # of producing NaN from a mean over zero pixels.
            zero = logits.sum() * 0.0
            return zero, {"loss": 0.0, "ce": 0.0, "dice": 0.0}
        weight = self.class_weights if self.class_weights.numel() else None
        if weight is not None:
            self.ce.weight = weight.to(logits.device)
        ce = self.ce(logits, target)
        dice = self.dice(logits, target)
        total = self.ce_weight * ce + self.dice_weight * dice
        return total, {
            "loss": float(total.detach()),
            "ce": float(ce.detach()),
            "dice": float(dice.detach()),
        }


def build_loss(
    *,
    num_classes: int,
    ignore_index: int,
    class_weights: np.ndarray | None,
    ce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> SegmentationLoss:
    return SegmentationLoss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_weights=class_weights,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
    )
