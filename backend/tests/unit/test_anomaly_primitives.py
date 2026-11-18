"""Tests for the displacement residual and velocity primitives."""

from __future__ import annotations

from app.services.anomaly.displacement import residual
from app.services.anomaly.velocity import velocity


def test_residual_positive_means_seaward() -> None:
    # Observed waterline is 0.5 m farther offshore than expected.
    assert residual(1.0, 0.5) == 0.5


def test_residual_negative_means_inland() -> None:
    assert residual(0.2, 0.5) == -0.3


def test_residual_missing_values_are_none() -> None:
    assert residual(None, 0.5) is None
    assert residual(0.5, None) is None


def test_velocity_simple() -> None:
    assert velocity([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == 1.0


def test_velocity_zero_dt_is_none() -> None:
    assert velocity([0.0, 1.0], [5.0, 5.0]) is None


def test_velocity_insufficient_samples_is_none() -> None:
    assert velocity([1.0], [0.0]) is None
