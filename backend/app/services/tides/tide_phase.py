"""Tide-phase classification (Phase 2).

Identifies high/low water via local extrema and classifies rising/falling/slack
direction from neighbouring prediction values. Uses configurable tolerances for
proximity to high/low water — never universal height thresholds, because tidal
range differs between coastal locations.

This module is pure (no I/O, no pandas) and operates on timestamps stored as
epoch seconds (UTC) plus their heights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
from scipy.signal import find_peaks

from app.services.tides.base import HighLowWater, TideDirection, TideEvent

SECONDS_PER_HOUR = 3600.0


@dataclass(slots=True)
class Extremum:
    index: int
    timestamp: datetime
    height: float
    event: TideEvent


def _to_epoch(timestamps: Sequence[datetime]) -> np.ndarray:
    return np.array([ts.timestamp() for ts in timestamps], dtype=np.float64)


def _aware(ts: datetime) -> datetime:
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def find_extrema(
    timestamps: Sequence[datetime],
    heights: Sequence[float],
    *,
    prominence_m: float = 0.05,
) -> list[Extremum]:
    """Return chronological high/low water extrema via ``scipy.signal.find_peaks``.

    Neighbours of equal sign are merged into a single extremum (the more
    extreme wins); consecutive extrema then alternate high/low, yielding the
    canonical tidal signal.
    """
    if len(timestamps) != len(heights):
        raise ValueError("timestamps and heights must have equal length")
    if len(timestamps) < 3:
        return []

    # Work on a locally time-ordered copy; input order should already be sorted.
    order = np.argsort(_to_epoch(timestamps), kind="stable")
    ts = [timestamps[i] for i in order]
    hts = np.array([heights[i] for i in order], dtype=np.float64)

    peaks, _ = find_peaks(hts, prominence=prominence_m)
    troughs, _ = find_peaks(-hts, prominence=prominence_m)

    extrema = [Extremum(int(i), _aware(ts[i]), float(hts[i]), TideEvent.HIGH) for i in peaks] + [
        Extremum(int(i), _aware(ts[i]), float(hts[i]), TideEvent.LOW) for i in troughs
    ]
    extrema.sort(key=lambda e: e.index)

    # Enforce alternation: collapse runs of the same kind to the most extreme.
    merged: list[Extremum] = []
    for event in extrema:
        if merged and merged[-1].event is event.event:
            keep_high = (
                event.height >= merged[-1].height
                if event.event is TideEvent.HIGH
                else event.height <= merged[-1].height
            )
            if keep_high:
                merged[-1] = event
        else:
            merged.append(event)
    return merged


def estimate_slope_per_hour(
    timestamps: Sequence[datetime],
    heights: Sequence[float],
    at: datetime,
) -> float | None:
    """Slope (m/hour) at ``at`` using the bracketing samples, or ``None``.

    At exact samples the first finite difference is used. Outside the series
    the slope cannot be estimated and ``None`` is returned — the caller maps
    this to ``UNKNOWN`` rather than guessing a direction.
    """
    if len(timestamps) < 2 or len(timestamps) != len(heights):
        return None

    order = np.argsort(_to_epoch(timestamps), kind="stable")
    ts = [timestamps[i] for i in order]
    epoch = _to_epoch(ts)
    hts = np.array([heights[i] for i in order], dtype=np.float64)
    target = _aware(at).timestamp()

    # Index of the first sample strictly after ``at``.
    right = int(np.searchsorted(epoch, target, side="right"))
    left = right - 1

    if left >= 0 and right < len(ts):
        dt_hours = (epoch[right] - epoch[left]) / SECONDS_PER_HOUR
        if dt_hours == 0:
            return 0.0
        return float((hts[right] - hts[left]) / dt_hours)

    # Boundary samples: use the single available finite difference.
    if len(ts) >= 2:
        dt_hours = (epoch[1] - epoch[0]) / SECONDS_PER_HOUR
        if dt_hours == 0:
            return 0.0
        return float((hts[1] - hts[0]) / dt_hours)
    return None


def classify_direction(
    slope_m_per_hour: float | None,
    *,
    slack_tolerance: float = 0.01,
    nearest_event: TideEvent | None = None,
) -> TideDirection:
    """Map a slope to rising/falling/slack.

    ``nearest_event`` is used to label slack water as slack-high vs slack-low;
    without it slack water is reported as ``UNKNOWN`` rather than guessed.
    """
    if slope_m_per_hour is None:
        return TideDirection.UNKNOWN
    if slope_m_per_hour > slack_tolerance:
        return TideDirection.RISING
    if slope_m_per_hour < -slack_tolerance:
        return TideDirection.FALLING
    if nearest_event is TideEvent.HIGH:
        return TideDirection.SLACK_HIGH
    if nearest_event is TideEvent.LOW:
        return TideDirection.SLACK_LOW
    return TideDirection.UNKNOWN


def events_around(
    extrema: Sequence[Extremum],
    at: datetime,
) -> tuple[Extremum | None, Extremum | None]:
    """Return the nearest (previous, next) extremum around ``at``."""
    at_ts = _aware(at)
    previous = [e for e in extrema if e.timestamp < at_ts]
    upcoming = [e for e in extrema if e.timestamp > at_ts]
    prev_event = max(previous, key=lambda e: e.timestamp) if previous else None
    next_event = min(upcoming, key=lambda e: e.timestamp) if upcoming else None
    return prev_event, next_event


def to_high_low_water(extremum: Extremum, *, unit: str, datum: str | None) -> HighLowWater:
    """Convert an internal extremum to the public value object."""
    return HighLowWater(
        timestamp=extremum.timestamp,
        height=extremum.height,
        event=extremum.event,
        unit=unit,
        datum=datum,
    )
