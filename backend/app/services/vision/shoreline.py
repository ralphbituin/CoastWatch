"""Shoreline ("waterline") extraction — Phase 4.

BOUNDARY CONTRACT: this module answers **"where is the waterline *now*"**.
It does not know about tides and does not compare against a baseline. It turns
a binary water mask into a shoreline geometry and, when the camera is
calibrated, a calibrated position in metres (positive = seaward).

Input is a **water mask** — a 2-D single-channel image where water pixels are
non-zero. Phase 4 ships no segmentation model (see ``segmentation.py``), so the
mask comes from an operator-labelled upload or, once trained, the U-Net /
DeepLabV3+ model. Either way the mask is treated as ground truth here and
``segmentation_confidence`` defaults to 1.0.

Algorithm: per-column water extent → sea-side detection (whichever half of the
frame carries more water is assumed to be the sea) → water/land boundary at
the far water edge → moving-average smoothing → image-space polyline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.services.vision.calibration import CalibrationProfile, apply_homography


@dataclass(slots=True)
class ShorelineObservationResult:
    points: list[list[float]]  # image-space [[x, y], ...]
    calibrated_position_m: float | None
    segmentation_confidence: float
    calibration_version: int | None = None


class ShorelineExtractionError(ValueError):
    """Raised when a mask yields no usable waterline."""


def _as_water_mask(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.shape[2] == 3 else array[:, :, 0]
    if array.ndim != 2:
        raise ShorelineExtractionError("mask must be a 2-D single-channel image")
    return (array > 127).astype(np.uint8)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < 3:
        return values
    window = min(window, values.size)
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same")


def extract_shoreline(
    mask: Any,
    *,
    calibration: CalibrationProfile | None = None,
    seaward_axis: tuple[float, float] | None = None,
    confidence: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    smoothing_window: int = 5,
    min_water_columns: int = 8,
) -> ShorelineObservationResult:
    """Extract the waterline polyline from a water mask.

    ``calibration`` maps pixel→world metres; when combined with
    ``seaward_axis`` (a unit vector in world coordinates), each waterline
    point is projected onto that axis and the mean projection becomes
    ``calibrated_position_m`` (relative, positive = seaward).
    """
    water = _as_water_mask(mask)
    if roi is not None:
        x, y, w, h = roi
        height, width = water.shape
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > width or y + h > height:
            raise ShorelineExtractionError(
                f"roi (x={x}, y={y}, w={w}, h={h}) lies outside the {width}x{height} mask"
            )
        water = water[y : y + h, x : x + w]
        x_offset, y_offset = x, y
    else:
        x_offset = y_offset = 0

    # Speckle filter: remove isolated water blobs smaller than a few pixels.
    water = cv2.morphologyEx(water, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    height, width = water.shape

    # Per-column water extent.
    column_top = np.full(width, np.nan, dtype=np.float64)
    column_bot = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        ys = np.flatnonzero(water[:, x])
        if ys.size:
            column_top[x] = ys.min()
            column_bot[x] = ys.max()

    valid = ~np.isnan(column_top)
    if int(valid.sum()) < min_water_columns:
        raise ShorelineExtractionError(
            f"mask contains water in only {int(valid.sum())} column(s); "
            f"need at least {min_water_columns}"
        )

    # Sea-side detection: the half carrying more water mass is the sea side;
    # the shoreline is the far water edge on that side.
    top_mass = float(water[: height // 2].sum())
    bottom_mass = float(water[height // 2 :].sum())
    sea_below = bottom_mass >= top_mass
    line = column_top if sea_below else column_bot

    # Interpolate missing columns from neighbours, then smooth.
    xs = np.arange(width, dtype=np.float64)
    line = np.interp(xs, xs[valid], line[valid])
    line = _moving_average(line, smoothing_window)

    points = [[float(x + x_offset), float(y + y_offset)] for x, y in zip(xs, line, strict=True)]

    calibrated_position_m: float | None = None
    if calibration is not None and calibration.homography is not None and seaward_axis is not None:
        axis = np.asarray(seaward_axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            raise ShorelineExtractionError("seaward_axis must be a non-zero vector")
        axis = axis / norm
        world = apply_homography(calibration.homography, [(p[0], p[1]) for p in points])
        calibrated_position_m = float(np.mean(world @ axis))

    segmentation_confidence = 1.0 if confidence is None else float(confidence)
    return ShorelineObservationResult(
        points=points,
        calibrated_position_m=calibrated_position_m,
        segmentation_confidence=segmentation_confidence,
        calibration_version=calibration.version if calibration is not None else None,
    )
