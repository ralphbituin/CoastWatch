"""NAMRIA tide provider (Phase 2).

Design notes:

* CoastWatch does **not** assume an official, undocumented NAMRIA API exists.
  Manual CSV import of official station datasets is the primary path.
* If a downloadable dataset arrives as PDF (or any non-CSV format) it must go
  through an explicit parser/conversion step and be validated — it is never
  silently interpreted as CSV.

Official resources:
    https://www.namria.gov.ph/
    https://www.namria.gov.ph/kiosk/namria02.htm
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.services.tides.base import (
    HighLowWater,
    TidePhaseResult,
    TidePoint,
    TideProvider,
    TideStationInfo,
)


class NAMRIATideProvider(TideProvider):
    """Tide provider backed by imported, validated NAMRIA datasets."""

    def get_stations(self) -> list[TideStationInfo]:
        raise NotImplementedError("NAMRIATideProvider is implemented in Phase 2.")

    def import_data(self, source: Any, **options: Any) -> dict[str, Any]:
        raise NotImplementedError("NAMRIATideProvider.import_data is implemented in Phase 2.")

    def get_predictions(self, station_code: str, start: datetime, end: datetime) -> list[TidePoint]:
        raise NotImplementedError("NAMRIATideProvider is implemented in Phase 2.")

    def get_high_low_waters(
        self, station_code: str, start: datetime, end: datetime
    ) -> list[HighLowWater]:
        raise NotImplementedError("NAMRIATideProvider is implemented in Phase 2.")

    def get_prediction_at(self, station_code: str, timestamp: datetime) -> TidePhaseResult:
        raise NotImplementedError("NAMRIATideProvider is implemented in Phase 2.")
