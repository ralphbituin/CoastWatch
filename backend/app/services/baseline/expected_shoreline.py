"""Expected-shoreline model — Phase 5 prototype.

BOUNDARY CONTRACT: this module answers **"where *should* the waterline be given
the tide"**. It does not look at a camera frame. It maps a predicted tide height
to an expected shoreline position with uncertainty.

The only implemented model is a robust linear fit::

    expected_position_px = intercept_px + slope_px_per_m * tide_height_m

* Positions are pixels with the repository-wide **seaward-positive** sign.
* The fit is Theil–Sen (median of pairwise slopes), so a handful of bad masks or
  wave splashes cannot dominate it.
* It refuses to fit without enough paired observations, a meaningful tide range,
  and — by default — both rising and falling tide phases. A 4-hour night clip is
  not a tidal cycle and is never presented as a validated baseline.
* No metres are claimed. Metric baselines require a valid camera homography and
  a separate calibrated position; this module keeps pixel units.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

MODEL_THEIL_SEN = "theil_sen_linear"
POSITION_UNIT_PX = "px"
POSITION_SIGN_SEAWARD = "seaward_positive"


@dataclass(frozen=True, slots=True)
class PairedObservation:
    """One shoreline observation paired with a verified tide height."""

    timestamp: datetime
    position_px: float
    tide_height_m: float


@dataclass(slots=True)
class BaselineFitConfig:
    min_samples: int = 30
    min_tide_range_m: float = 0.20
    min_phase_transitions: int = 3
    require_both_phases: bool = True
    robust_z_outlier: float = 3.5
    provisional: bool = True


@dataclass(slots=True)
class BaselineFitResult:
    available: bool
    model_type: str = MODEL_THEIL_SEN
    status: str = "unavailable"
    reason: str | None = None
    intercept_px: float | None = None
    slope_px_per_m: float | None = None
    r_squared: float | None = None
    sample_count: int = 0
    min_samples_required: int = 0
    tide_min_m: float | None = None
    tide_max_m: float | None = None
    tide_range_m: float | None = None
    rising_phase_samples: int = 0
    falling_phase_samples: int = 0
    residual_median_px: float | None = None
    residual_mad_px: float | None = None
    residual_scale_px: float | None = None
    residual_scale_reliable: bool = False
    outlier_count: int = 0
    slope_ci95_px_per_m: tuple[float, float] | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    provisional: bool = True
    position_unit: str = POSITION_UNIT_PX
    position_sign: str = POSITION_SIGN_SEAWARD
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "status": self.status,
            "reason": self.reason,
            "model_type": self.model_type,
            "intercept_px": self.intercept_px,
            "slope_px_per_m": self.slope_px_per_m,
            "r_squared": self.r_squared,
            "sample_count": self.sample_count,
            "min_samples_required": self.min_samples_required,
            "tide_min_m": self.tide_min_m,
            "tide_max_m": self.tide_max_m,
            "tide_range_m": self.tide_range_m,
            "rising_phase_samples": self.rising_phase_samples,
            "falling_phase_samples": self.falling_phase_samples,
            "residual_median_px": self.residual_median_px,
            "residual_mad_px": self.residual_mad_px,
            "residual_scale_px": self.residual_scale_px,
            "residual_scale_reliable": self.residual_scale_reliable,
            "outlier_count": self.outlier_count,
            "slope_ci95_px_per_m": (
                None if self.slope_ci95_px_per_m is None else list(self.slope_ci95_px_per_m)
            ),
            "valid_from": None if self.valid_from is None else self.valid_from.isoformat(),
            "valid_to": None if self.valid_to is None else self.valid_to.isoformat(),
            "provisional": self.provisional,
            "position_unit": self.position_unit,
            "position_sign": self.position_sign,
            "diagnostics": self.diagnostics,
        }


def _unavailable(
    reason: str,
    *,
    config: BaselineFitConfig,
    sample_count: int = 0,
    tide_range: float | None = None,
    rising: int = 0,
    falling: int = 0,
    diagnostics: dict[str, Any] | None = None,
) -> BaselineFitResult:
    return BaselineFitResult(
        available=False,
        status="unavailable",
        reason=reason,
        sample_count=sample_count,
        min_samples_required=config.min_samples,
        tide_range_m=tide_range,
        rising_phase_samples=rising,
        falling_phase_samples=falling,
        provisional=True,
        diagnostics=diagnostics or {},
    )


def _clean(observations: Iterable[PairedObservation]) -> list[PairedObservation]:
    cleaned: list[PairedObservation] = []
    for observation in observations:
        position = float(observation.position_px)
        tide = float(observation.tide_height_m)
        if not (np.isfinite(position) and np.isfinite(tide)):
            continue
        cleaned.append(
            PairedObservation(
                timestamp=observation.timestamp,
                position_px=position,
                tide_height_m=tide,
            )
        )
    cleaned.sort(key=lambda item: item.timestamp)
    return cleaned


def _theil_sen(x: np.ndarray, y: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """Return ``(slope, intercept, slope 95% CI)`` using pairwise slopes."""
    n = len(x)
    slopes: list[float] = []
    for i in range(n - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        valid = np.abs(dx) > 1e-12
        if valid.any():
            slopes.extend((dy[valid] / dx[valid]).tolist())
    slope_array = np.asarray(slopes, dtype=np.float64)
    slope = float(np.median(slope_array))
    intercept = float(np.median(y - slope * x))
    low, high = np.percentile(slope_array, [2.5, 97.5])
    return slope, intercept, (float(low), float(high))


def fit_tide_conditioned_baseline(
    observations: Iterable[PairedObservation],
    config: BaselineFitConfig | None = None,
) -> BaselineFitResult:
    """Fit the provisional tide-conditioned baseline, or refuse honestly."""
    config = config or BaselineFitConfig()
    cleaned = _clean(observations)
    if len(cleaned) < max(2, config.min_samples):
        return _unavailable(
            "insufficient paired shoreline/tide observations",
            config=config,
            sample_count=len(cleaned),
        )

    tide = np.asarray([item.tide_height_m for item in cleaned], dtype=np.float64)
    positions = np.asarray([item.position_px for item in cleaned], dtype=np.float64)
    tide_range = float(tide.max() - tide.min())
    if tide_range < config.min_tide_range_m:
        return _unavailable(
            "insufficient observed tide-height range for a meaningful fit",
            config=config,
            sample_count=len(cleaned),
            tide_range=tide_range,
        )

    rising = falling = 0
    for previous, current in zip(tide[:-1], tide[1:], strict=False):
        delta = float(current - previous)
        if delta > 0:
            rising += 1
        elif delta < 0:
            falling += 1
    if config.require_both_phases and (
        rising < config.min_phase_transitions or falling < config.min_phase_transitions
    ):
        return _unavailable(
            "insufficient rising and falling tide phase coverage",
            config=config,
            sample_count=len(cleaned),
            tide_range=tide_range,
            rising=rising,
            falling=falling,
        )

    slope, intercept, slope_ci = _theil_sen(tide, positions)
    predicted = intercept + slope * tide
    residuals = positions - predicted
    residual_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - residual_median)))
    scale = 1.4826 * mad
    scale_reliable = bool(scale > 1e-9)
    outlier_count = (
        int(np.sum(np.abs(residuals - residual_median) > config.robust_z_outlier * scale))
        if scale_reliable
        else 0
    )
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((positions - positions.mean()) ** 2))
    r_squared = None if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    return BaselineFitResult(
        available=True,
        status="provisional",
        reason=(
            "provisional robust linear fit; insufficient history for a validated "
            "tide-conditioned baseline"
        ),
        intercept_px=round(intercept, 6),
        slope_px_per_m=round(slope, 6),
        r_squared=None if r_squared is None else round(r_squared, 6),
        sample_count=len(cleaned),
        min_samples_required=config.min_samples,
        tide_min_m=round(float(tide.min()), 6),
        tide_max_m=round(float(tide.max()), 6),
        tide_range_m=round(tide_range, 6),
        rising_phase_samples=rising,
        falling_phase_samples=falling,
        residual_median_px=round(residual_median, 6),
        residual_mad_px=round(mad, 6),
        residual_scale_px=round(scale, 6) if scale_reliable else None,
        residual_scale_reliable=scale_reliable,
        outlier_count=outlier_count,
        slope_ci95_px_per_m=(round(slope_ci[0], 6), round(slope_ci[1], 6)),
        valid_from=cleaned[0].timestamp,
        valid_to=cleaned[-1].timestamp,
        provisional=bool(config.provisional),
        diagnostics={
            "position_min_px": round(float(positions.min()), 3),
            "position_max_px": round(float(positions.max()), 3),
            "position_mean_px": round(float(positions.mean()), 3),
            "residual_max_abs_px": round(float(np.abs(residuals).max()), 3),
            "residual_p95_abs_px": round(float(np.percentile(np.abs(residuals), 95)), 3),
            "slope_method": "median_of_pairwise_slopes",
            "intercept_method": "median_of_residual_offsets",
        },
    )


# Backwards-compatible alias for the Phase 5 stub name.
fit_baseline = fit_tide_conditioned_baseline


@dataclass(slots=True)
class ExpectedShoreline:
    position_m: float | None
    uncertainty_m: float | None
    baseline_id: int | None
    quality: str = "unknown"
    position_px: float | None = None
    uncertainty_px: float | None = None
    unit: str = POSITION_UNIT_PX
    reason: str | None = None


def expected_shoreline(
    tide_height_m: float | None,
    baseline: BaselineFitResult | dict[str, Any] | Any,
    *,
    baseline_id: int | None = None,
) -> ExpectedShoreline:
    """Evaluate ``intercept + slope * tide`` without inventing a baseline."""
    status: str | None
    reason: str | None
    if isinstance(baseline, BaselineFitResult):
        available = baseline.available
        intercept = baseline.intercept_px
        slope = baseline.slope_px_per_m
        scale = baseline.residual_scale_px
        status = baseline.status
        reason = baseline.reason
    elif isinstance(baseline, dict):
        available = bool(baseline.get("available"))
        intercept = baseline.get("intercept_px")
        slope = baseline.get("slope_px_per_m")
        scale = baseline.get("residual_scale_px")
        status = baseline.get("status")
        reason = baseline.get("reason")
    else:  # pragma: no cover - defensive for ORM rows
        available = getattr(baseline, "status", None) == "provisional"
        params = getattr(baseline, "model_params", None) or {}
        intercept = params.get("intercept_px")
        slope = params.get("slope_px_per_m")
        scale = params.get("residual_scale_px")
        status = getattr(baseline, "status", None)
        reason = getattr(baseline, "reason", None)
        baseline_id = baseline_id or getattr(baseline, "id", None)

    if not available or intercept is None or slope is None:
        return ExpectedShoreline(
            position_m=None,
            uncertainty_m=None,
            baseline_id=baseline_id,
            quality="unavailable",
            reason=reason or "baseline unavailable",
        )
    if tide_height_m is None or not np.isfinite(tide_height_m):
        return ExpectedShoreline(
            position_m=None,
            uncertainty_m=None,
            baseline_id=baseline_id,
            quality="unavailable",
            reason="tide height unavailable",
        )
    position = float(intercept) + float(slope) * float(tide_height_m)
    return ExpectedShoreline(
        position_m=None,
        uncertainty_m=None,
        baseline_id=baseline_id,
        quality=str(status or "provisional"),
        position_px=round(position, 6),
        uncertainty_px=None if scale is None else round(float(scale), 6),
        reason=None,
    )
