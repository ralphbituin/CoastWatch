"""Tide provider interface and shared value objects (Phase 2).

A ``TideProvider`` abstracts where tide data comes from so that country- and
vendor-specific formats (NAMRIA today, others later) can be added without
touching the anomaly engine.

Value objects carry the datum and unit alongside every height so that data
from different tide stations is never silently blended.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class TideEvent(StrEnum):
    HIGH = "high"
    LOW = "low"


class TideDirection(StrEnum):
    RISING = "rising"
    FALLING = "falling"
    SLACK_HIGH = "slack_high"
    SLACK_LOW = "slack_low"
    UNKNOWN = "unknown"


class DataQuality(StrEnum):
    OK = "ok"
    INTERPOLATED = "interpolated"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"


@dataclass(slots=True)
class TideStationInfo:
    code: str
    name: str
    country_iso: str
    region: str | None
    latitude: float
    longitude: float
    timezone: str
    data_source: str
    vertical_datum: str | None
    measurement_unit: str


@dataclass(slots=True)
class TidePoint:
    timestamp: datetime  # timezone-aware, UTC internally
    height: float
    unit: str
    datum: str | None
    quality: DataQuality = DataQuality.OK


@dataclass(slots=True)
class HighLowWater:
    timestamp: datetime
    height: float
    event: TideEvent
    unit: str
    datum: str | None


@dataclass(slots=True)
class TidePhaseResult:
    timestamp: datetime
    height: float
    unit: str
    datum: str | None
    direction: TideDirection
    quality: DataQuality
    previous_event: HighLowWater | None = None
    next_event: HighLowWater | None = None


class TideProvider(ABC):
    """Interface implemented by every tide data source."""

    @abstractmethod
    def get_stations(self) -> list[TideStationInfo]: ...

    @abstractmethod
    def import_data(self, source: Any, **options: Any) -> dict[str, Any]: ...

    @abstractmethod
    def get_predictions(
        self, station_code: str, start: datetime, end: datetime
    ) -> list[TidePoint]: ...

    @abstractmethod
    def get_high_low_waters(
        self, station_code: str, start: datetime, end: datetime
    ) -> list[HighLowWater]: ...

    @abstractmethod
    def get_prediction_at(self, station_code: str, timestamp: datetime) -> TidePhaseResult: ...
