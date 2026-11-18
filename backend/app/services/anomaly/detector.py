"""Anomaly detector and system-state machine (Phase 6).

Algorithms: Z-score, rolling statistical thresholds and CUSUM initially;
Isolation Forest, One-Class SVM and LSTM later when enough history exists.

System states (see ``app.models.anomaly``):

* ``NORMAL`` — consistent with the learned baseline.
* ``ANOMALY_DETECTED`` — unusual shoreline movement detected.
* ``DATA_UNRELIABLE`` — missing tide data, poor camera quality, invalid
  calibration or low segmentation confidence. Unavailable data is NEVER treated
  as normal conditions.
* ``OFFICIAL_TSUNAMI_ADVISORY`` — an independently verified advisory received.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    state: str
    score: float | None
    explanation: str


class AnomalyDetector:
    def evaluate(self, *args, **kwargs) -> Detection:
        raise NotImplementedError("Anomaly detection is implemented in Phase 6.")
