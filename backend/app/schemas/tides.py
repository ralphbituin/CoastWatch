"""Schemas for the tide module (stations, imports, predictions, phase)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

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


class TidePointRead(BaseModel):
    timestamp: datetime
    height: float
    unit: str
    datum: str | None
    quality_status: str


class TideEventRead(BaseModel):
    timestamp: datetime
    height: float
    type: str  # "high" | "low"
    unit: str
    datum: str | None


class ImportPreviewResponse(BaseModel):
    station_code: str | None
    timezone: str
    file_hash: str | None
    columns_detected: list[str]
    column_mapping: dict[str, Any]
    total_records: int
    valid_records: int
    rejected_records: int
    rejection_summary: dict[str, Any]
    rejection_samples: list[dict[str, Any]]
    coverage_start: datetime | None
    coverage_end: datetime | None
    message: str


class ImportCommitResponse(ImportPreviewResponse):
    dataset_id: int | None
    import_id: int
    status: str  # committed | partial | rejected | duplicate
    imported_records: int


class DatasetImportRead(BaseModel):
    id: int
    tide_station_id: int
    station_code: str
    dataset_id: int | None
    source_filename: str | None
    file_hash: str | None
    total_records: int
    imported_records: int
    rejected_records: int
    status: str
    created_at: datetime


class DatasetImportDetail(DatasetImportRead):
    column_mapping: dict[str, Any] | None
    rejection_summary: dict[str, Any] | None
    rejection_samples: list[dict[str, Any]] | None
    message: str | None


_QUALITY_NOTE = {
    "ok": "Exact sample from the imported dataset.",
    "interpolated": "Interpolated between samples (not an independently validated forecast).",
    "estimated": "Estimated value with limited confidence.",
    "unknown": "Value present but unverified.",
    "unavailable": "No data for this timestamp.",
}


def quality_note(token: str) -> str:
    return _QUALITY_NOTE.get(token, f"Unrecognized quality status: {token}")


class TidePhaseResponse(BaseModel):
    """Result for CURRENT TIME and SELECTED TIME prediction modes."""

    station_code: str
    station_name: str
    requested_timestamp: datetime
    local_timestamp: str
    server_time: datetime
    predicted_height_m: float
    unit: str
    vertical_datum: str | None
    direction: str
    quality: str
    quality_note: str
    source: str | None
    dataset_id: int | None
    timezone: str
    interpolation_method: str
    previous_event: TideEventRead | None
    next_event: TideEventRead | None


class TideChartResponse(BaseModel):
    station_code: str
    start: datetime
    end: datetime
    method: str
    count: int
    coverage_start: datetime | None
    coverage_end: datetime | None
    sufficient_for_7d: bool
    points: list[TidePointRead]
    events: list[TideEventRead]


class ImportListResponse(BaseModel):
    imports: list[DatasetImportRead]
    total: int
