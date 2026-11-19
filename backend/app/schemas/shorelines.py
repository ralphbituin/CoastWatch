"""Schemas for shoreline observations (Phase 4)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SystemState = Literal[
    "NORMAL",
    "OBSERVED",  # observation persisted; no tide-baseline/anomaly verdict yet (Phases 5-6)
    "ANOMALY_DETECTED",
    "DATA_UNRELIABLE",
    "OFFICIAL_TSUNAMI_ADVISORY",
]


class ShorelineObservationRead(BaseModel):
    camera_id: int
    camera_code: str | None
    observed_at: datetime
    monitoring_station_id: int
    station_code: str | None
    calibration_id: int | None
    shoreline_geometry: list[list[float]] | None
    calibrated_position_m: float | None
    segmentation_confidence: float | None
    frame_quality_score: float | None
    frame_quality_flags: list[Any] | None
    source_frame_path: str | None
    created_at: datetime


class ShorelineObservationSummary(BaseModel):
    """List view: everything except the (potentially large) geometry."""

    camera_id: int
    camera_code: str | None
    observed_at: datetime
    monitoring_station_id: int
    station_code: str | None
    calibration_id: int | None
    calibrated_position_m: float | None
    segmentation_confidence: float | None
    frame_quality_score: float | None
    frame_quality_flags: list[Any] | None
    created_at: datetime


class ShorelineObservationDetail(ShorelineObservationRead):
    pass


class ShorelineObserveRequest(BaseModel):
    station_id: int
    camera_id: int


class ShorelineObserveResult(BaseModel):
    station_id: int
    camera_id: int
    observed_at: datetime
    actual_shoreline_m: float | None
    expected_shoreline_m: float | None
    residual_m: float | None
    retreat_velocity_m_s: float | None
    anomaly_score: float | None
    state: SystemState
    details: dict[str, Any]


class ShorelineModuleStatus(BaseModel):
    module: str = "shorelines"
    phase: int = 4
    segmentation_model_available: bool
    implemented_endpoints: list[str]
    notes: list[str] = Field(default_factory=list)


class ShorelineCalibrationRead(BaseModel):
    id: int
    camera_id: int
    camera_code: str | None
    version: int
    meters_per_pixel: float | None
    is_active: bool
    valid_from: datetime | None
    valid_to: datetime | None
    notes: str | None
    created_at: datetime


class OperatorMaskUpload(BaseModel):
    """Metadata carried alongside a mask image in the multipart form."""

    camera_id: int = Field(ge=1)
    observed_at: datetime | None = None
    # "dx,dy" world-axis unit vector of the seaward direction for calibration projection.
    seaward_axis: tuple[float, float] | None = None

    @field_validator("seaward_axis")
    @classmethod
    def _non_zero_axis(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and abs(value[0]) + abs(value[1]) == 0.0:
            raise ValueError("seaward_axis must be non-zero")
        return value
