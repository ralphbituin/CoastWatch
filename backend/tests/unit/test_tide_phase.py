"""Unit tests for tide-phase classification (extrema, direction, slack)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from app.services.tides.base import TideDirection, TideEvent
from app.services.tides.tide_phase import (
    classify_direction,
    estimate_slope_per_hour,
    events_around,
    find_extrema,
)


def sine_series(hours: int = 96, start: datetime | None = None):
    """Synthetic 12.4 h periodic tide (labelled synthetic for testing)."""
    t0 = start or datetime(2026, 11, 1, tzinfo=UTC)
    timestamps = [t0 + timedelta(hours=h) for h in range(hours)]
    heights = [1.1 + 0.7 * math.sin(2 * math.pi * h / 12.4) for h in range(hours)]
    return timestamps, heights


def test_extrema_alternate_and_both_found() -> None:
    timestamps, heights = sine_series()
    extrema = find_extrema(timestamps, heights, prominence_m=0.05)
    assert len(extrema) >= 4
    for first, second in zip(extrema, extrema[1:], strict=False):
        assert first.event is not second.event, "extrema must alternate high/low"
    assert any(e.event is TideEvent.HIGH for e in extrema)
    assert any(e.event is TideEvent.LOW for e in extrema)


def test_extrema_chronological_order() -> None:
    timestamps, heights = sine_series()
    extrema = find_extrema(timestamps, heights, prominence_m=0.05)
    assert [e.timestamp for e in extrema] == sorted(e.timestamp for e in extrema)


def test_extrema_prominence_suppresses_noise() -> None:
    timestamps, heights = sine_series()
    noisy = [h + (0.01 if i % 2 == 0 else -0.01) for i, h in enumerate(heights)]
    # With a small bump, prominence 0.05 should still leave one extremum per cycle.
    extrema = find_extrema(timestamps, noisy, prominence_m=0.05)
    events = [e.event for e in extrema]
    assert len(set(events)) == 2


def test_estimate_slope_rising() -> None:
    timestamps, heights = sine_series()
    # h = 1 is rising (peak at ~3.1 h)
    assert estimate_slope_per_hour(timestamps, heights, timestamps[1]) > 0


def test_estimate_slope_falling() -> None:
    timestamps, heights = sine_series()
    # h = 4 is falling (peak at ~3.1 h)
    assert estimate_slope_per_hour(timestamps, heights, timestamps[4]) < 0


def test_estimate_slope_none_for_degenerate_series() -> None:
    assert estimate_slope_per_hour([], [], datetime(2026, 11, 1, tzinfo=UTC)) is None


def test_classify_direction() -> None:
    assert classify_direction(0.5) is TideDirection.RISING
    assert classify_direction(-0.5) is TideDirection.FALLING
    assert classify_direction(0.0, nearest_event=TideEvent.HIGH) is TideDirection.SLACK_HIGH
    assert classify_direction(0.0, nearest_event=TideEvent.LOW) is TideDirection.SLACK_LOW
    assert classify_direction(0.0) is TideDirection.UNKNOWN
    assert classify_direction(None) is TideDirection.UNKNOWN
    assert classify_direction(0.005, slack_tolerance=0.01) is TideDirection.UNKNOWN


def test_events_around_interior() -> None:
    timestamps, heights = sine_series()
    extrema = find_extrema(timestamps, heights, prominence_m=0.05)
    previous, upcoming = events_around(extrema, timestamps[24])
    assert previous is not None
    assert upcoming is not None
    assert previous.timestamp < timestamps[24] < upcoming.timestamp
