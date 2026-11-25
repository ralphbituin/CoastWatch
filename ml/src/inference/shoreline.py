"""Shoreline point extraction, mirroring CoastWatch's backend algorithm.

The authoritative implementation is
``backend/app/services/vision/shoreline.py``. This is a dependency-light port
used by the ML CLI and by tests; it must stay behaviourally equivalent for the
geometry part (per-column far water edge, smoothing, image-space points).
Keeping it here means ``ml/`` does not need the FastAPI/SQLAlchemy backend to
be importable.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class ShorelineExtractionError(ValueError):
    """Raised when a mask yields no usable waterline."""


def _as_water_mask(mask: Any) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
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


def _open(mask: np.ndarray) -> np.ndarray:
    try:
        import cv2

        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    except ImportError:  # pragma: no cover
        return mask


def extract_shoreline_points(
    mask: Any,
    *,
    smoothing_window: int = 5,
    min_water_columns: int = 8,
) -> list[list[float]]:
    """Return image-space ``[[x, y], ...]`` waterline points."""
    water = _open(_as_water_mask(mask))
    height, width = water.shape

    column_top = np.full(width, np.nan, dtype=np.float64)
    column_bot = np.full(width, np.nan, dtype=np.float64)
    for x in range(width):
        rows = np.flatnonzero(water[:, x])
        if rows.size:
            column_top[x] = rows.min()
            column_bot[x] = rows.max()

    valid = ~np.isnan(column_top)
    if int(valid.sum()) < min_water_columns:
        raise ShorelineExtractionError(
            f"mask contains water in only {int(valid.sum())} column(s); "
            f"need at least {min_water_columns}"
        )

    top_mass = float(water[: height // 2].sum())
    bottom_mass = float(water[height // 2 :].sum())
    sea_below = bottom_mass >= top_mass
    line = column_top if sea_below else column_bot

    xs = np.arange(width, dtype=np.float64)
    line = np.interp(xs, xs[valid], line[valid])
    line = _moving_average(line, smoothing_window)
    return [[float(x), float(y)] for x, y in zip(xs, line, strict=True)]


def max_step_px(points: list[list[float]]) -> float | None:
    """Largest vertical jump between adjacent points (roughness diagnostic)."""
    if len(points) < 2:
        return None
    ys = np.asarray([p[1] for p in points], dtype=np.float64)
    return float(np.max(np.abs(np.diff(ys))))
