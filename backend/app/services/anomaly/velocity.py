"""Shoreline movement rate and acceleration.

    rate(t) = [position(t) - position(t - dt)] / dt

Positions follow the repository-wide sign convention **positive = seaward**.
Therefore a positive rate means the waterline is moving seaward (drawdown /
retreat of the water edge), and a negative rate means it is moving landward
(wave run-up). This direction is documented rather than assumed by consumers.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def velocity(positions_m: Sequence[float], timestamps_s: Sequence[float]) -> float | None:
    """Simple finite-difference velocity between the last two samples."""
    if len(positions_m) < 2 or len(timestamps_s) < 2:
        return None
    dt = timestamps_s[-1] - timestamps_s[-2]
    if dt == 0:
        return None
    return (positions_m[-1] - positions_m[-2]) / dt


def movement_rate(
    current_position: float | None,
    previous_position: float | None,
    elapsed_seconds: float | None,
) -> float | None:
    """Seaward-positive px/s between two valid observations."""
    if current_position is None or previous_position is None or elapsed_seconds is None:
        return None
    if not (
        isfinite(current_position)
        and isfinite(previous_position)
        and isfinite(elapsed_seconds)
        and elapsed_seconds > 0
    ):
        return None
    return (float(current_position) - float(previous_position)) / float(elapsed_seconds)


def rolling_velocity(
    positions: Sequence[float | None],
    timestamps_s: Sequence[float | None],
    *,
    window: int = 3,
) -> list[float | None]:
    """Finite-difference rates over the last ``window`` valid pairs.

    ``None`` entries break the chain: a missing observation can neither create
    a rate nor count as a zero-movement sample.
    """
    if len(positions) != len(timestamps_s):
        raise ValueError("positions and timestamps must have equal length")
    out: list[float | None] = []
    pairs: list[tuple[float, float]] = []
    for position, timestamp in zip(positions, timestamps_s, strict=False):
        if position is None or timestamp is None:
            out.append(None)
            pairs.clear()
            continue
        pairs.append((float(timestamp), float(position)))
        if len(pairs) > max(2, int(window)):
            pairs.pop(0)
        if len(pairs) < 2:
            out.append(None)
            continue
        (t0, p0), (t1, p1) = pairs[0], pairs[-1]
        out.append(None if t1 <= t0 else (p1 - p0) / (t1 - t0))
    return out


def rolling_acceleration(
    positions: Sequence[float | None],
    timestamps_s: Sequence[float | None],
    *,
    window: int = 3,
) -> list[float | None]:
    """Finite-difference acceleration of the rolling velocity series."""
    rates = rolling_velocity(positions, timestamps_s, window=window)
    out: list[float | None] = []
    previous_rate: float | None = None
    previous_time: float | None = None
    for rate, timestamp in zip(rates, timestamps_s, strict=False):
        if rate is None or timestamp is None or previous_rate is None or previous_time is None:
            out.append(None)
        else:
            dt = float(timestamp) - previous_time
            out.append(None if dt <= 0 else (rate - previous_rate) / dt)
        if rate is None or timestamp is None:
            previous_rate = None
            previous_time = None
        else:
            previous_rate = rate
            previous_time = float(timestamp)
    return out
