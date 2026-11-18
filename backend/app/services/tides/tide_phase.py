"""Tide-phase classification (Phase 2).

Identifies high/low water via local extrema and classifies rising/falling
direction from neighbouring prediction values. Uses configurable tolerances for
proximity to high/low water — never universal height thresholds, because tidal
range differs between coastal locations.
"""

from __future__ import annotations

from typing import Any

from app.services.tides.base import HighLowWater, TideDirection


def classify_direction(*args: Any, **kwargs: Any) -> TideDirection:
    raise NotImplementedError("Tide-phase classification is implemented in Phase 2.")


def find_high_low_waters(*args: Any, **kwargs: Any) -> list[HighLowWater]:
    raise NotImplementedError("High/low water detection is implemented in Phase 2.")


def next_event(*args: Any, **kwargs: Any) -> tuple[HighLowWater | None, HighLowWater | None]:
    """Return (previous, next) high/low water events around a timestamp."""
    raise NotImplementedError("Tide-phase classification is implemented in Phase 2.")
