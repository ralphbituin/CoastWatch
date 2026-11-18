"""Tide-height interpolation (Phase 2).

Implements linear interpolation as the baseline and PCHIP as the preferred
smooth method. Example: given 14:00 = 0.80 m and 15:00 = 1.00 m, the height at
14:30 is interpolated between the two bracketing samples.

Rules:

* Never extrapolate silently outside the available dataset — an explicit
  "unavailable" result is returned instead.
* Interpolation is not an independently validated physical tide forecast.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class InterpolationMethod(StrEnum):
    LINEAR = "linear"
    PCHIP = "pchip"


def interpolate_height(
    *args: Any,
    method: InterpolationMethod = InterpolationMethod.LINEAR,
    **kwargs: Any,
) -> float | None:
    raise NotImplementedError("Tide interpolation is implemented in Phase 2.")
