"""Provisional tide-conditioned baseline: refuses bad data, robust to outliers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.baseline.expected_shoreline import (
    BaselineFitConfig,
    PairedObservation,
    expected_shoreline,
    fit_tide_conditioned_baseline,
)


def _series(count: int, *, slope: float = 40.0, intercept: float = 700.0, noise: float = 0.0):
    start = datetime(2026, 11, 19, 21, 30, tzinfo=UTC)
    # A full-ish cycle: rise then fall, so both phases are present.
    observations = []
    for index in range(count):
        phase = index / max(1, count - 1)
        tide = 0.2 + 1.6 * (1.0 - abs(2.0 * phase - 1.0))
        jitter = noise if index % 2 == 0 else -noise
        observations.append(
            PairedObservation(
                timestamp=start + timedelta(seconds=30 * index),
                tide_height_m=tide,
                position_px=intercept + slope * tide + jitter,
            )
        )
    return observations


CONFIG = BaselineFitConfig(min_samples=20, min_tide_range_m=0.2, min_phase_transitions=3)


def test_refuses_to_fit_without_enough_pairs() -> None:
    result = fit_tide_conditioned_baseline(_series(10), CONFIG)
    assert not result.available
    assert result.status == "unavailable"
    assert "insufficient paired" in (result.reason or "")


def test_refuses_to_fit_without_tide_range() -> None:
    start = datetime(2026, 11, 19, tzinfo=UTC)
    observations = [
        PairedObservation(
            timestamp=start + timedelta(seconds=30 * i),
            position_px=700 + i * 0.01,
            tide_height_m=1.0,
        )
        for i in range(30)
    ]
    result = fit_tide_conditioned_baseline(observations, CONFIG)
    assert not result.available
    assert "tide-height range" in (result.reason or "")


def test_refuses_without_both_tide_phases() -> None:
    start = datetime(2026, 11, 19, tzinfo=UTC)
    observations = [
        PairedObservation(
            timestamp=start + timedelta(seconds=30 * i),
            position_px=700 + 40 * (0.2 + i * 0.05),
            tide_height_m=0.2 + i * 0.05,
        )
        for i in range(30)
    ]
    result = fit_tide_conditioned_baseline(observations, CONFIG)
    assert not result.available
    assert "phase" in (result.reason or "")


def test_fits_a_known_line_and_is_outlier_resistant() -> None:
    observations = _series(60, noise=0.4)
    observations[7] = PairedObservation(
        timestamp=observations[7].timestamp,
        position_px=observations[7].position_px + 250,
        tide_height_m=observations[7].tide_height_m,
    )
    result = fit_tide_conditioned_baseline(observations, CONFIG)
    assert result.available
    assert result.slope_px_per_m == pytest.approx(40.0, abs=1.5)
    assert result.intercept_px == pytest.approx(700.0, abs=2.5)
    assert result.sample_count == 60
    assert result.provisional is True
    assert 1.4 <= result.tide_range_m <= 1.6
    assert result.rising_phase_samples > 0
    assert result.falling_phase_samples > 0


def test_zero_mad_is_flagged_unreliable_not_divided_by_zero() -> None:
    observations = _series(40, noise=0.0)
    result = fit_tide_conditioned_baseline(observations, CONFIG)
    assert result.available
    assert result.residual_scale_px is None
    assert result.residual_scale_reliable is False


def test_expected_shoreline_evaluates_only_an_available_fit() -> None:
    fit = fit_tide_conditioned_baseline(_series(40, noise=0.2), CONFIG)
    expected = expected_shoreline(1.0, fit)
    assert expected.position_px == pytest.approx(fit.intercept_px + fit.slope_px_per_m)
    assert expected.position_m is None  # never invent metres without calibration
    assert expected.unit == "px"

    unavailable = expected_shoreline(1.0, {"available": False, "reason": "no tide data"})
    assert unavailable.position_px is None
    assert unavailable.quality == "unavailable"
    assert unavailable.reason == "no tide data"
