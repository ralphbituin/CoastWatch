"""Segmentation and boundary metrics (NumPy, no torchmetrics dependency)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class ConfusionMatrix:
    num_classes: int = 4
    ignore_index: int = 255
    matrix: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, prediction: np.ndarray, target: np.ndarray) -> None:
        prediction = np.asarray(prediction).ravel().astype(np.int64)
        target = np.asarray(target).ravel().astype(np.int64)
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same number of pixels")
        valid = (target != self.ignore_index) & (target >= 0) & (target < self.num_classes)
        prediction = prediction[valid]
        target = target[valid]
        if target.size == 0:
            return
        indices = target * self.num_classes + prediction
        counts = np.bincount(indices, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes)

    def merge(self, other: ConfusionMatrix) -> None:
        self.matrix += other.matrix

    def compute(self) -> dict[str, Any]:
        matrix = self.matrix.astype(np.float64)
        true_positive = np.diag(matrix)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        union = support + predicted - true_positive
        iou = np.divide(
            true_positive, union, out=np.full_like(true_positive, np.nan), where=union > 0
        )
        dice = np.divide(
            2 * true_positive,
            2 * true_positive + (predicted - true_positive) + (support - true_positive),
            out=np.full_like(true_positive, np.nan),
            where=(2 * true_positive + (predicted - true_positive) + (support - true_positive)) > 0,
        )
        precision = np.divide(
            true_positive, predicted, out=np.full_like(true_positive, np.nan), where=predicted > 0
        )
        recall = np.divide(
            true_positive, support, out=np.full_like(true_positive, np.nan), where=support > 0
        )
        present = support > 0
        mean_iou = float(np.nanmean(iou[present])) if present.any() else None
        mean_dice = float(np.nanmean(dice[present])) if present.any() else None
        total = matrix.sum()
        pixel_accuracy = float(true_positive.sum() / total) if total else None
        return {
            "per_class": {
                index: {
                    "support": int(support[index]),
                    "predicted": int(predicted[index]),
                    "iou": _none_if_nan(iou[index]),
                    "dice": _none_if_nan(dice[index]),
                    "precision": _none_if_nan(precision[index]),
                    "recall": _none_if_nan(recall[index]),
                }
                for index in range(self.num_classes)
            },
            "mean_iou": mean_iou,
            "mean_dice": mean_dice,
            "pixel_accuracy": pixel_accuracy,
            "supervised_pixels": int(total),
            "confusion_matrix": self.matrix.tolist(),
        }


def _none_if_nan(value: float) -> float | None:
    return None if value is None or np.isnan(value) else float(value)


def binary_boundary(mask: np.ndarray) -> np.ndarray:
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=1)
    return (mask - eroded).astype(bool)


def boundary_f1(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    ignore_index: int = 255,
    tolerance_px: int = 2,
    classes: Iterable[int] = (1, 2),
) -> dict[str, Any]:
    """Boundary precision/recall/F1 with a pixel tolerance."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    valid = target != ignore_index
    result: dict[str, Any] = {}
    for class_id in classes:
        gt_mask = (target == class_id) & valid
        pred_mask = (prediction == class_id) & valid
        if not gt_mask.any():
            result[str(class_id)] = {
                "support": 0,
                "precision": None,
                "recall": None,
                "f1": None,
            }
            continue
        gt_boundary = binary_boundary(gt_mask)
        pred_boundary = binary_boundary(pred_mask)
        gt_dilated = cv2.dilate(
            gt_boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=tolerance_px
        ).astype(bool)
        pred_dilated = cv2.dilate(
            pred_boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=tolerance_px
        ).astype(bool)
        precision = (
            float((pred_boundary & gt_dilated).sum() / pred_boundary.sum())
            if pred_boundary.any()
            else 0.0
        )
        recall = (
            float((gt_boundary & pred_dilated).sum() / gt_boundary.sum())
            if gt_boundary.any()
            else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        result[str(class_id)] = {
            "support": int(gt_boundary.sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    values = [row["f1"] for row in result.values() if row["f1"] is not None]
    result["mean_f1"] = float(np.mean(values)) if values else None
    return result


def _dominant_component(mask: np.ndarray) -> np.ndarray:
    """Largest 8-connected component of a boolean mask."""
    array = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(array, connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == largest


def shoreline_metrics_from_masks(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    ignore_index: int = 255,
    tolerance_px: int = 2,
    water_classes: tuple[int, ...] = (2, 3),
) -> dict[str, Any]:
    """Boundary metrics specifically for the land/water interface."""
    pred_water = np.isin(prediction, water_classes)
    gt_water = np.isin(target, water_classes) & (target != ignore_index)
    if not gt_water.any() or not pred_water.any():
        return {
            "available": False,
            "reason": "no water in prediction or ground truth",
        }

    # Only the dominant water body is "the sea": small predicted puddles in the
    # sky or on wet sand are legitimate but their edges are not the shoreline.
    gt_water = _dominant_component(gt_water)
    pred_water = _dominant_component(pred_water)

    # Keep only the *land/water interface*. A plain region boundary also traces
    # the image frame where water runs off the edge, which is not a shoreline
    # and would dominate the distances on rectified CoastSnap frames.
    kernel = np.ones((3, 3), np.uint8)
    gt_land = (target == 1).astype(np.uint8)
    pred_land = (prediction == 1).astype(np.uint8)
    gt_boundary = binary_boundary(gt_water) & cv2.dilate(gt_land, kernel).astype(bool)
    pred_boundary = binary_boundary(pred_water) & cv2.dilate(pred_land, kernel).astype(bool)
    if not gt_boundary.any() or not pred_boundary.any():
        return {
            "available": False,
            "reason": "no land/water interface in prediction or ground truth",
        }
    gt_dilated = cv2.dilate(
        gt_boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=tolerance_px
    ).astype(bool)
    pred_dilated = cv2.dilate(
        pred_boundary.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=tolerance_px
    ).astype(bool)
    precision = float((pred_boundary & gt_dilated).sum() / max(pred_boundary.sum(), 1))
    recall = float((gt_boundary & pred_dilated).sum() / max(gt_boundary.sum(), 1))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    # Shoreline localization error: how far each annotated waterline pixel is
    # from the model's waterline. Measured annotated -> predicted, so extra
    # predicted water in ignored regions (e.g. sky) shows up in precision and
    # the reverse distance rather than inflating this number.
    pred_inverse = (pred_boundary == 0).astype(np.uint8)
    distance_to_pred = cv2.distanceTransform(pred_inverse, cv2.DIST_L2, 5)
    distances = distance_to_pred[gt_boundary]
    mean_distance = float(distances.mean()) if distances.size else None

    gt_inverse = (gt_boundary == 0).astype(np.uint8)
    distance_to_gt = cv2.distanceTransform(gt_inverse, cv2.DIST_L2, 5)
    reverse_distances = distance_to_gt[pred_boundary]
    mean_reverse = float(reverse_distances.mean()) if reverse_distances.size else None

    # Per-column vertical displacement where both masks have a boundary.
    displacement = []
    height, width = gt_water.shape
    for x in range(width):
        gt_rows = np.flatnonzero(gt_boundary[:, x])
        pred_rows = np.flatnonzero(pred_boundary[:, x])
        if gt_rows.size and pred_rows.size:
            displacement.append(abs(float(gt_rows.mean() - pred_rows.mean())))
    return {
        "available": True,
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "mean_boundary_distance_px": mean_distance,
        "mean_boundary_distance_pred_to_gt_px": mean_reverse,
        "mean_vertical_displacement_px": float(np.mean(displacement)) if displacement else None,
        "columns_compared": len(displacement),
        "tolerance_px": tolerance_px,
    }
