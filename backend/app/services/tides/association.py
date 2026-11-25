"""Geographically constrained tide association.

The footage location is part of the data contract. A tide height is only
associated with an observation when there is a station that is **both** in the
expected country/time zone and close enough to the site. This prevents the
classic false-association failure: joining Thai footage to a Philippine NAMRIA
station merely because it happens to be the only tide table in the database.

Missing or out-of-coverage tide data yields ``available=False``. It is never
replaced by zero, an interpolated guess, or a station from another country.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any

from app.core.exceptions import DataUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Tokens that identify Philippine tide programmes. A station carrying one of
# these is never a valid source for non-Philippine footage, even if an operator
# mislabels its country in the database.
PHILIPPINE_SOURCE_TOKENS = ("NAMRIA", "PHIVOLCS", "PHILIPPINE", "MANILA")

DEFAULT_MAX_DISTANCE_KM = 120.0


@dataclass(slots=True)
class RejectedStation:
    code: str
    name: str
    distance_km: float | None
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "name": self.name,
            "distance_km": None if self.distance_km is None else round(self.distance_km, 3),
            "reason": self.reason,
        }


@dataclass(slots=True)
class StationSelection:
    station: Any | None
    distance_km: float | None
    candidate_count: int
    rejected: list[RejectedStation] = field(default_factory=list)
    reason: str | None = None


@dataclass(slots=True)
class TideAssociation:
    available: bool
    station_id: int | None = None
    station_code: str | None = None
    station_name: str | None = None
    timestamp: datetime | None = None
    height: float | None = None
    unit: str | None = None
    vertical_datum: str | None = None
    source: str | None = None
    quality: str | None = None
    timezone: str | None = None
    distance_km: float | None = None
    direction: str | None = None
    reason: str | None = None
    candidate_count: int = 0
    rejected_stations: list[dict[str, Any]] = field(default_factory=list)
    interpolation_method: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "tide_data_available": self.available,
            "tide_station_id": self.station_id,
            "tide_station_code": self.station_code,
            "tide_station_name": self.station_name,
            "tide_timestamp": None if self.timestamp is None else self.timestamp.isoformat(),
            "predicted_tide_height": self.height,
            "tide_unit": self.unit,
            "tide_vertical_datum": self.vertical_datum,
            "tide_source": self.source or "unavailable",
            "tide_quality": self.quality,
            "tide_timezone": self.timezone,
            "tide_station_distance_km": (
                None if self.distance_km is None else round(self.distance_km, 3)
            ),
            "tide_direction": self.direction,
            "tide_reason": self.reason,
            "tide_candidate_count": self.candidate_count,
            "tide_rejected_stations": self.rejected_stations,
            "tide_interpolation_method": self.interpolation_method,
        }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    radius_km = 6371.0088
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlambda = radians(lon2 - lon1)
    a = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * asin(sqrt(min(1.0, max(0.0, a))))


def _country_iso(station: Any) -> str | None:
    country = getattr(station, "country", None)
    if country is None:
        return None
    return getattr(country, "iso_code", None)


def _source_text(station: Any) -> str:
    return " ".join(
        str(value)
        for value in (
            getattr(station, "code", ""),
            getattr(station, "name", ""),
            getattr(station, "data_source", ""),
            getattr(station, "region", ""),
        )
        if value
    ).upper()


def select_tide_station(
    session: Any,
    latitude: float,
    longitude: float,
    *,
    country_iso: str | None,
    timezone: str | None = None,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
) -> StationSelection:
    """Choose the nearest geographically appropriate station, or explain why not.

    ``session=None`` is allowed (offline analysis without a database) and
    returns an explicit unavailable reason.
    """
    if session is None:
        return StationSelection(
            station=None,
            distance_km=None,
            candidate_count=0,
            reason="database unavailable; cannot verify a geographically appropriate tide source",
        )

    from sqlalchemy import select

    from app.models.tide import TideStation

    stations = list(session.scalars(select(TideStation)).all())
    rejected: list[RejectedStation] = []
    usable: list[tuple[float, Any]] = []
    expected_country = (country_iso or "").upper() or None
    for station in stations:
        distance = haversine_km(latitude, longitude, station.latitude, station.longitude)
        iso = (_country_iso(station) or "").upper() or None
        text = _source_text(station)
        if expected_country and iso and iso != expected_country:
            rejected.append(
                RejectedStation(
                    station.code,
                    station.name,
                    distance,
                    f"country {iso}!={expected_country}",
                )
            )
            continue
        if not iso and expected_country:
            rejected.append(
                RejectedStation(station.code, station.name, distance, "station country unknown")
            )
            continue
        if any(token in text for token in PHILIPPINE_SOURCE_TOKENS) and expected_country != "PH":
            rejected.append(
                RejectedStation(
                    station.code,
                    station.name,
                    distance,
                    "Philippine tide programme cannot be used for this location",
                )
            )
            continue
        if timezone and station.timezone and station.timezone != timezone:
            rejected.append(
                RejectedStation(
                    station.code,
                    station.name,
                    distance,
                    f"timezone {station.timezone}!={timezone}",
                )
            )
            continue
        if distance > max_distance_km:
            rejected.append(
                RejectedStation(
                    station.code,
                    station.name,
                    distance,
                    f"distance {distance:.1f} km exceeds {max_distance_km:.1f} km",
                )
            )
            continue
        usable.append((distance, station))

    if not usable:
        reason = (
            f"no tide station within {max_distance_km:.0f} km for country "
            f"{expected_country or 'unknown'} / timezone {timezone or 'unknown'}"
        )
        return StationSelection(
            station=None,
            distance_km=None,
            candidate_count=len(stations),
            rejected=rejected,
            reason=reason,
        )
    distance, station = min(usable, key=lambda item: item[0])
    return StationSelection(
        station=station,
        distance_km=distance,
        candidate_count=len(stations),
        rejected=rejected,
    )


def associate_tide(
    session: Any,
    *,
    latitude: float,
    longitude: float,
    at: datetime | None,
    country_iso: str | None,
    timezone: str | None,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    method: str = "pchip",
    provider: Any | None = None,
) -> TideAssociation:
    """Associate one timestamp with a verified, appropriate tide prediction.

    Tide prediction generation was removed from the product, so a provider must
    be supplied explicitly by offline analysis code. Without one this returns an
    explicit unavailable association; it never invents a height.
    """
    selection = select_tide_station(
        session,
        latitude,
        longitude,
        country_iso=country_iso,
        timezone=timezone,
        max_distance_km=max_distance_km,
    )
    rejected = [item.as_dict() for item in selection.rejected]
    if at is None:
        return TideAssociation(
            available=False,
            candidate_count=selection.candidate_count,
            rejected_stations=rejected,
            reason="observation has no verified absolute timestamp",
        )
    if selection.station is None:
        return TideAssociation(
            available=False,
            candidate_count=selection.candidate_count,
            rejected_stations=rejected,
            reason=selection.reason,
        )

    station = selection.station
    if provider is None:
        return TideAssociation(
            available=False,
            station_id=getattr(station, "id", None),
            station_code=station.code,
            station_name=station.name,
            timestamp=at,
            source="unavailable",
            timezone=station.timezone,
            distance_km=selection.distance_km,
            candidate_count=selection.candidate_count,
            rejected_stations=rejected,
            reason="tide prediction generation is not available in this build",
        )
    try:
        result = provider.get_prediction_at(station.code, at, method=method)
    except DataUnavailableError as exc:
        return TideAssociation(
            available=False,
            station_id=getattr(station, "id", None),
            station_code=station.code,
            station_name=station.name,
            timestamp=at,
            source=station.data_source,
            timezone=station.timezone,
            distance_km=selection.distance_km,
            candidate_count=selection.candidate_count,
            rejected_stations=rejected,
            reason=f"tide data unavailable: {exc.message}",
        )
    except Exception as exc:  # pragma: no cover - provider-specific failures
        logger.warning("tide association failed for %s: %s", station.code, exc)
        return TideAssociation(
            available=False,
            station_id=getattr(station, "id", None),
            station_code=station.code,
            station_name=station.name,
            timestamp=at,
            source=station.data_source,
            timezone=station.timezone,
            distance_km=selection.distance_km,
            candidate_count=selection.candidate_count,
            rejected_stations=rejected,
            reason=f"tide lookup failed: {exc}",
        )

    return TideAssociation(
        available=True,
        station_id=getattr(station, "id", None),
        station_code=station.code,
        station_name=station.name,
        timestamp=result.timestamp,
        height=float(result.height),
        unit=result.unit,
        vertical_datum=result.datum,
        source=result.source or station.data_source,
        quality=result.quality.value,
        timezone=station.timezone,
        distance_km=selection.distance_km,
        direction=result.direction.value,
        candidate_count=selection.candidate_count,
        rejected_stations=rejected,
        interpolation_method=str(getattr(method, "value", method)),
    )
