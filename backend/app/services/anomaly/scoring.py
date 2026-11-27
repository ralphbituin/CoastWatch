"""Explainable anomaly scoring primitives.

The score is a monitoring signal — it is **not** a validated tsunami
probability. The only implemented statistic here is a robust Z score about the
median of baseline residuals::

    MAD = median(abs(residual - median(residual)))
    robust_z = (residual - median(residual)) / (1.4826 * MAD)

A zero or near-zero MAD is not divided by zero and is not treated as a perfect
model. :func:`robust_scale` reports ``reliable=False`` and callers must emit
``DATA_UNRELIABLE`` rather than an infinite/NaN score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from math import isfinite

import numpy as np

MAD_TO_SIGMA = 1.4826
_MIN_SCALE = 1e-9


@dataclass(slots=True)
class RobustScale:
    median: float | None
    mad: float | None
    scale: float | None
    reliable: bool
    sample_count: int = 0

    def as_dict(self) -> dict[str, float | bool | int | None]:
        return {
            "median": self.median,
            "mad": self.mad,
            "scale": self.scale,
            "reliable": self.reliable,
            "sample_count": self.sample_count,
        }


@dataclass(slots=True)
class AnomalyScore:
    score: float
    factors: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def robust_scale(values: Iterable[float] | None) -> RobustScale:
    cleaned = [
        float(value) for value in (values or []) if value is not None and isfinite(float(value))
    ]
    if not cleaned:
        return RobustScale(median=None, mad=None, scale=None, reliable=False, sample_count=0)
    array = np.asarray(cleaned, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = MAD_TO_SIGMA * mad
    return RobustScale(
        median=median,
        mad=mad,
        scale=scale if scale > _MIN_SCALE else None,
        reliable=bool(scale > _MIN_SCALE),
        sample_count=len(cleaned),
    )


def robust_z(
    value: float | None,
    median: float | None,
    mad: float | None,
) -> float | None:
    """Signed robust Z score, or ``None`` when the scale is unusable."""
    if value is None or median is None or mad is None:
        return None
    if not (isfinite(value) and isfinite(median) and isfinite(mad)):
        return None
    scale = MAD_TO_SIGMA * mad
    if scale <= _MIN_SCALE:
        return None
    return (float(value) - float(median)) / scale


def robust_z_from_scale(value: float | None, scale: RobustScale) -> float | None:
    if not scale.reliable or scale.scale is None or scale.median is None:
        return None
    if value is None or not isfinite(value):
        return None
    return (float(value) - float(scale.median)) / float(scale.scale)


def compute_score(
    *,
    robust_z_value: float | None = None,
    movement_rate: float | None = None,
    movement_threshold: float | None = None,
    persistence: int = 0,
    required_persistence: int = 3,
    quality: float | None = None,
) -> AnomalyScore:
    """Combine transparent factors into a bounded [0, 1] monitoring score.

    This is intentionally simple and inspectable: no learned weights are
    invented. A score of 1.0 does not mean a tsunami is confirmed.
    """
    factors: dict[str, float] = {}
    if robust_z_value is not None:
        factors["robust_z"] = float(robust_z_value)
        factors["position_factor"] = min(1.0, abs(float(robust_z_value)) / 6.0)
    if movement_rate is not None and movement_threshold and movement_threshold > 0:
        factors["movement_rate"] = float(movement_rate)
        factors["movement_factor"] = min(1.0, max(0.0, float(movement_rate) / movement_threshold))
    if required_persistence > 0:
        factors["persistence_factor"] = min(1.0, persistence / float(required_persistence))
    if quality is not None:
        factors["quality"] = float(quality)

    weighted = [
        (factors.get("position_factor", 0.0), 0.5),
        (factors.get("movement_factor", 0.0), 0.35),
        (factors.get("persistence_factor", 0.0), 0.15),
    ]
    score = sum(value * weight for value, weight in weighted)
    if quality is not None:
        score *= max(0.0, min(1.0, float(quality)))
    explanations = []
    if "robust_z" in factors:
        explanations.append(f"robust_z={factors['robust_z']:.2f}")
    if "movement_rate" in factors:
        explanations.append(f"movement_rate={factors['movement_rate']:.3f}px/s")
    explanations.append(f"persistence={persistence}/{required_persistence}")
    return AnomalyScore(
        score=round(float(score), 6),
        factors={key: round(float(value), 6) for key, value in factors.items()},
        explanation="; ".join(explanations),
    )
