"""Shoreline displacement residual (Phase 6).

    R(t) = D_actual(t) - D_expected(t)

Sign convention: positive distance is **seaward**, so a positive residual means
the observed waterline is farther offshore than expected.
"""

from __future__ import annotations


def residual(actual_m: float | None, expected_m: float | None) -> float | None:
    """Return ``actual - expected`` in meters, or ``None`` if either is missing."""
    if actual_m is None or expected_m is None:
        return None
    return actual_m - expected_m
