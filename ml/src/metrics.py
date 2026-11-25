"""Segmentation metrics for the three-class CoastWatch SegFormer.

Class ids follow ``ml/configs/segformer_b2.yaml``:

    0 = Beach, 1 = Ocean, 2 = Sky, 255 = ignore

The ignore index is removed *before* the confusion matrix is accumulated. A
class with zero ground-truth support is reported as ``None`` rather than as a
misleading zero; mean IoU averages only classes that are present in the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np


CLASS_NAMES: tuple[str, ...] = ("Beach", "Ocean", "Sky")
BEACH, OCEAN, SKY = 0, 1, 2
IGNORE_INDEX = 255


def _none_if_nan(value: float | np.floating[Any]) -> float | None:
    value = float(value)
    return None if np.isnan(value) else value


@dataclass
class ConfusionMatrix:
    num_classes: int = 3
    class_names: tuple[str, ...] = CLASS_NAMES
    ignore_index: int = IGNORE_INDEX
    matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        if self.num_classes != len(self.class_names):
            raise ValueError("num_classes must match class_names")
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction).ravel().astype(np.int64, copy=False)
        target = np.asarray(target).ravel().astype(np.int64, copy=False)
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have the same number of pixels; "
                f"got {prediction.shape} and {target.shape}"
            )
        valid = (
            (target != self.ignore_index)
            & (target >= 0)
            & (target < self.num_classes)
            & (prediction >= 0)
            & (prediction < self.num_classes)
        )
        if not valid.any():
            return
        indices = target[valid] * self.num_classes + prediction[valid]
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def merge(self, other: "ConfusionMatrix") -> None:
        if other.num_classes != self.num_classes:
            raise ValueError("cannot merge confusion matrices with different class counts")
        self.matrix += other.matrix

    def compute(self) -> dict[str, Any]:
        matrix = self.matrix.astype(np.float64)
        true_positive = np.diag(matrix)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        union = support + predicted - true_positive
        iou = np.divide(
            true_positive,
            union,
            out=np.full_like(true_positive, np.nan),
            where=union > 0,
        )
        dice_denom = 2 * true_positive + (predicted - true_positive) + (support - true_positive)
        dice = np.divide(
            2 * true_positive,
            dice_denom,
            out=np.full_like(true_positive, np.nan),
            where=dice_denom > 0,
        )
        precision = np.divide(
            true_positive,
            predicted,
            out=np.full_like(true_positive, np.nan),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.full_like(true_positive, np.nan),
            where=support > 0,
        )
        present = support > 0
        mean_iou = float(np.nanmean(iou[present])) if present.any() else None
        mean_dice = float(np.nanmean(dice[present])) if present.any() else None
        total = matrix.sum()
        pixel_accuracy = float(true_positive.sum() / total) if total else None
        return {
            "class_names": list(self.class_names),
            "ignore_index": self.ignore_index,
            "per_class": {
                name: {
                    "class_id": index,
                    "support_pixels": int(support[index]),
                    "predicted_pixels": int(predicted[index]),
                    "iou": _none_if_nan(iou[index]),
                    "dice": _none_if_nan(dice[index]),
                    "precision": _none_if_nan(precision[index]),
                    "recall": _none_if_nan(recall[index]),
                }
                for index, name in enumerate(self.class_names)
            },
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "pixel_accuracy": pixel_accuracy,
            "supervised_pixels": int(total),
            "confusion_matrix": self.matrix.tolist(),
        }


def class_pixel_counts(
    masks: Iterable[np.ndarray], *, num_classes: int = 3, ignore_index: int = IGNORE_INDEX
) -> np.ndarray:
    """Count valid pixels per class across an iterable of masks."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for mask in masks:
        array = np.asarray(mask).ravel()
        valid = (array != ignore_index) & (array >= 0) & (array < num_classes)
        if valid.any():
            counts += np.bincount(array[valid].astype(np.int64), minlength=num_classes)
    return counts


def inverse_frequency_weights(
    counts: np.ndarray,
    *,
    cap: float = 3.0,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Inverse-frequency class weights, normalised to mean 1 and capped.

    A class with zero training pixels gets weight 1 (it cannot be trained), and
    the caller should not claim metrics for it.
    """
    counts = np.asarray(counts, dtype=np.float64)
    weights = np.ones_like(counts)
    present = counts > 0
    if present.any():
        freq = counts / counts.sum()
        weights[present] = 1.0 / np.maximum(freq[present], 1e-12)
        weights[present] /= weights[present].mean()
    weights = np.clip(weights, 0.0, cap)
    weights[weights == 0] = 1.0
    return weights.astype(np.float32)


def confusion_matrix_figure(
    matrix: np.ndarray,
    class_names: Iterable[str] = CLASS_NAMES,
    *,
    normalize: bool = True,
):
    """Plot a confusion matrix with matplotlib and return the Figure."""
    import matplotlib.pyplot as plt

    matrix = np.asarray(matrix, dtype=np.float64)
    names = list(class_names)
    display = matrix.copy()
    if normalize:
        row_sums = display.sum(axis=1, keepdims=True)
        display = np.divide(display, row_sums, out=np.zeros_like(display), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), dpi=150)
    im = ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0 if normalize else None)
    ax.set_xticks(range(len(names)), names, rotation=20)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    ax.set_title("Confusion matrix" + (" (row-normalised)" if normalize else ""))
    for row in range(display.shape[0]):
        for col in range(display.shape[1]):
            value = display[row, col]
            text = f"{value:.2f}" if normalize else f"{int(value):,}"
            ax.text(
                col,
                row,
                text,
                ha="center",
                va="center",
                color="white" if value > (0.55 if normalize else display.max() * 0.55) else "black",
                fontsize=9,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
