"""Explainable anomaly scoring (Phase 6).

Combines shoreline displacement residual, retreat velocity, retreat
acceleration, deviation from tide-expected movement, persistence, baseline
uncertainty and segmentation quality into a normalized, explainable score.

The score is a monitoring signal — it is NOT a validated tsunami probability.
Weights are calibrated against observed data, never hard-coded per station.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class AnomalyScore:
    score: float
    factors: dict[str, float] = field(default_factory=dict)
    explanation: str = ""


def compute_score(*args, **kwargs) -> AnomalyScore:
    raise NotImplementedError("Anomaly scoring is implemented in Phase 6.")
