"""Retreat velocity and acceleration (Phase 6).

    V(t) = [D_actual(t) - D_actual(t - dt)] / dt

Positive values represent seaward movement. Rolling velocity and acceleration
are computed over a configurable window.
"""

from __future__ import annotations

from collections.abc import Sequence


def velocity(positions_m: Sequence[float], timestamps_s: Sequence[float]) -> float | None:
    """Simple finite-difference velocity between the last two samples."""
    if len(positions_m) < 2 or len(timestamps_s) < 2:
        return None
    dt = timestamps_s[-1] - timestamps_s[-2]
    if dt == 0:
        return None
    return (positions_m[-1] - positions_m[-2]) / dt


def rolling_velocity(*args, **kwargs) -> list[float]:
    raise NotImplementedError("Rolling velocity is implemented in Phase 6.")


def rolling_acceleration(*args, **kwargs) -> list[float]:
    raise NotImplementedError("Rolling acceleration is implemented in Phase 6.")
