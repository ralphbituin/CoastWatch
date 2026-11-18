"""Tide-height interpolation (Phase 2).

Implements linear interpolation as the baseline and PCHIP as the preferred
smooth method. Example: given 14:00 = 0.80 m and 15:00 = 1.00 m, the height at
14:30 is interpolated between the two bracketing samples.

Rules:

* Never extrapolate silently outside the available dataset — ``None`` is
  returned instead, and callers translate it into an explicit ``unavailable``
  status.
* Interpolation is not an independently validated physical tide forecast.
* Timestamps are stored and compared as UTC; conversions to local time are a
  presentation concern.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np


class InterpolationMethod(StrEnum):
    LINEAR = "linear"
    PCHIP = "pchip"


def _cleaned_xy(x: Sequence[float], y: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return strictly increasing, decontaminated arrays.

    Duplicate x positions keep the first sample. Callers receive a copy so the
    input order is unaffected.
    """
    xa = np.asarray(list(x), dtype=np.float64)
    ya = np.asarray(list(y), dtype=np.float64)
    if xa.shape != ya.shape or xa.size == 0:
        raise ValueError("x and y must be equal-length, non-empty sequences")
    order = np.argsort(xa, kind="stable")
    xa, ya = xa[order], ya[order]
    # Keep the first of each duplicate x.
    _, first_idx = np.unique(xa, return_index=True)
    return xa[first_idx], ya[first_idx]


def linear(x: Sequence[float], y: Sequence[float], xq: float) -> float | None:
    """Piecewise-linear interpolation between bracketing samples."""
    xa, ya = _cleaned_xy(x, y)
    if xq < xa[0] or xq > xa[-1]:
        return None  # Never extrapolate.
    return float(np.interp(xq, xa, ya))


def pchip(x: Sequence[float], y: Sequence[float], xq: float) -> float | None:
    """PCHIP (monotone cubic) interpolation between bracketing samples."""
    from scipy.interpolate import PchipInterpolator

    xa, ya = _cleaned_xy(x, y)
    if xq < xa[0] or xq > xa[-1]:
        return None  # Never extrapolate.
    if xa.size < 3:
        # PCHIP needs at least three points; linear is exact for two.
        return float(np.interp(xq, xa, ya))
    return float(PchipInterpolator(xa, ya, extrapolate=False)(xq))


def interpolate_height(
    x: Sequence[float],
    y: Sequence[float],
    target: float,
    *,
    method: InterpolationMethod = InterpolationMethod.PCHIP,
) -> float | None:
    """Interpolate ``y`` at ``target`` using ``method``.

    Returns ``None`` when ``target`` is outside the available dataset.
    """
    if method is InterpolationMethod.LINEAR:
        return linear(x, y, target)
    if method is InterpolationMethod.PCHIP:
        return pchip(x, y, target)
    raise ValueError(f"Unsupported interpolation method: {method}")
