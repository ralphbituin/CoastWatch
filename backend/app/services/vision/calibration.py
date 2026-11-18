"""Camera calibration (Phase 4).

Homography-based calibration using operator-identified reference points and
known real-world distances, giving shoreline displacement in meters when camera
geometry permits. Calibration profiles are versioned per camera; suspected
camera movement invalidates the active calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CalibrationProfile:
    camera_id: int
    version: int
    homography: list[list[float]] | None
    meters_per_pixel: float | None


def compute_homography(*args: Any, **kwargs: Any) -> CalibrationProfile:
    raise NotImplementedError("Calibration is implemented in Phase 4.")


def detect_camera_movement(*args: Any, **kwargs: Any) -> bool:
    raise NotImplementedError("Camera-movement detection is implemented in Phase 4.")
