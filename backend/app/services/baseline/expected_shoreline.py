"""Expected-shoreline model — Phase 5.

BOUNDARY CONTRACT: this module answers **"where *should* the waterline be given
the tide"**. It does not look at a camera frame. It maps a predicted tide
(height, and optionally phase/environment) to an expected shoreline position
(positive = seaward) with an uncertainty.

Baseline model:

    Expected Shoreline = f(Tide Height, Tidal Phase, Environmental Conditions)

Starts with linear / polynomial regression and piecewise interpolation; more
advanced models are only evaluated once sufficient data exists. Baselines are
built per monitoring location *and* per camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ExpectedShoreline:
    position_m: float | None
    uncertainty_m: float | None
    baseline_id: int | None
    quality: str = "unknown"


def fit_baseline(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError("Baseline fitting is implemented in Phase 5.")


def expected_shoreline(*args: Any, **kwargs: Any) -> ExpectedShoreline:
    raise NotImplementedError("Expected-shoreline modeling is implemented in Phase 5.")
