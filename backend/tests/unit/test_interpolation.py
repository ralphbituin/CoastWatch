"""Unit tests for tide-height interpolation."""

from __future__ import annotations

import math

import pytest

from app.services.tides.interpolation import InterpolationMethod, interpolate_height


def test_linear_example_from_spec() -> None:
    """14:00 = 0.80 m, 15:00 = 1.00 m, 14:30 -> 0.90 m."""
    assert (
        interpolate_height([0.0, 3600.0], [0.80, 1.00], 1800.0, method=InterpolationMethod.LINEAR)
        == 0.90
    )


def test_pchip_midpoint_exact() -> None:
    # With a symmetric triple, PCHIP passes through the middle sample.
    assert (
        interpolate_height(
            [0.0, 1800.0, 3600.0],
            [0.80, 0.95, 1.00],
            1800.0,
            method=InterpolationMethod.PCHIP,
        )
        == 0.95
    )


def test_out_of_range_returns_none_linear() -> None:
    assert (
        interpolate_height([0.0, 3600.0], [0.8, 1.0], 7200.0, method=InterpolationMethod.LINEAR)
        is None
    )
    assert (
        interpolate_height([0.0, 3600.0], [0.8, 1.0], -1.0, method=InterpolationMethod.LINEAR)
        is None
    )


def test_out_of_range_returns_none_pchip() -> None:
    assert (
        interpolate_height([0.0, 3600.0], [0.8, 1.0], 7200.0, method=InterpolationMethod.PCHIP)
        is None
    )


def test_pchip_follows_sine_between_samples() -> None:
    # A synthetic 12.4 h periodic signal; PCHIP should track it closely.
    x = [3600.0 * h for h in range(0, 25)]
    y = [1.1 + 0.7 * math.sin(2 * math.pi * h / 12.4) for h in range(0, 25)]
    expected = 1.1 + 0.7 * math.sin(2 * math.pi * 6.5 / 12.4)
    interpolated = interpolate_height(x, y, 3600.0 * 6.5, method=InterpolationMethod.PCHIP)
    assert interpolated is not None
    assert interpolated == pytest.approx(expected, abs=0.02)


def test_unsupported_method_raises() -> None:
    with pytest.raises(ValueError):
        interpolate_height([0.0, 1.0], [0.0, 1.0], 0.5, method="cubic-spline")  # type: ignore[arg-type]
