"""Shoreline ("waterline") extraction — Phase 4.

BOUNDARY CONTRACT: this module answers **"where is the waterline *now"**.
It does not know about tides and does not compare against a baseline. It turns
a segmentation mask into a shoreline geometry and, when the camera is
calibrated, a calibrated position in meters (positive = seaward).

Planned pipeline: contour extraction, morphological filtering, boundary
smoothing, ROI filtering and temporal consistency checks. Canny edges are an
optional supporting method only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ShorelineObservationResult:
    points: list[list[float]]  # image-space [[x, y], ...]
    calibrated_position_m: float | None
    segmentation_confidence: float
    calibration_version: int | None = None


def extract_shoreline(*args: Any, **kwargs: Any) -> ShorelineObservationResult:
    raise NotImplementedError("Shoreline extraction is implemented in Phase 4.")
