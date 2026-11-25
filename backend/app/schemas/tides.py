"""Schemas for the retained tide-station registry.

Prediction, chart, phase and CSV-import schemas were removed with the tide
prediction feature. Station metadata stays because the tide tables are kept.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class TideStationCreate(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    country_iso: str = Field(default="PH", min_length=2, max_length=2)
    region: str | None = Field(default=None, max_length=120)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone: str = "Asia/Manila"
    data_source: str = Field(default="NAMRIA_CSV", max_length=64)
    vertical_datum: str | None = Field(default=None, max_length=64)
    measurement_unit: str = Field(default="meters", max_length=16)
    monitoring_station_id: int | None = None

    @field_validator("code", "name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("measurement_unit")
    @classmethod
    def _unit(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"meters", "feet", "centimeters"}:
            raise ValueError("unit must be one of: meters, feet, centimeters")
        return value


class TideStationRead(BaseModel):
    id: int
    code: str
    name: str
    region: str | None
    country_iso: str
    country_name: str
    latitude: float
    longitude: float
    timezone: str
    data_source: str
    vertical_datum: str | None
    measurement_unit: str
    prediction_start: datetime | None
    prediction_end: datetime | None
    monitoring_station_id: int | None
    dataset_count: int
