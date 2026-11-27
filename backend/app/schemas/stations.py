"""Schemas for monitoring stations and countries."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class CountryRead(BaseModel):
    id: int
    iso_code: str
    name: str
    default_timezone: str


class MonitoringStationCreate(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    country_iso: str = Field(default="PH", min_length=2, max_length=2)
    region: str | None = Field(default=None, max_length=120)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone: str = "Asia/Manila"
    vertical_datum: str | None = Field(default=None, max_length=64)
    description: str | None = None

    @field_validator("code", "name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class MonitoringStationUpdate(BaseModel):
    """Partial monitoring-station update. Omitted fields are left unchanged."""

    code: str | None = Field(default=None, min_length=1, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    country_iso: str | None = Field(default=None, min_length=2, max_length=2)
    region: str | None = Field(default=None, max_length=120)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    timezone: str | None = Field(default=None, max_length=64)
    vertical_datum: str | None = Field(default=None, max_length=64)
    description: str | None = None

    @field_validator("code", "name")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


class MonitoringStationRead(BaseModel):
    id: int
    code: str
    name: str
    region: str | None
    country_iso: str
    country_name: str
    latitude: float
    longitude: float
    timezone: str
    vertical_datum: str | None
    description: str | None
    tide_station_count: int
    created_at: datetime
