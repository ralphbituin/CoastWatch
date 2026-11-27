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
    # Ordered segments; each segment is an ordered [[x, y], ...] polyline.
    # Optional so pre-existing clients keep working and historical rows remain
    # readable (NULL means the observation predates segment persistence).
    shoreline_segments: list[list[list[float]]] | None = None
    # Single-line contract diagnostics.
    shoreline_axis: str | None = None
    interpolated_ranges: list[list[int]] | None = None
    measured_coverage: float | None = None
    calibrated_position_m: float | None
    segmentation_confidence: float | None
    frame_quality_score: float | None
    frame_quality_flags: list[Any] | None
    source_frame_path: str | None
    created_at: datetime
    # Phase 5 video provenance (optional for historical rows).
    recording_id: str | None = None
    frame_index: int | None = None
    extraction_status: str | None = None
    extraction_reason: str | None = None
    shoreline_position_px: float | None = None
    signed_position_px: float | None = None
    position_axis: str | None = None
    position_sign: str | None = None
    segment_count: int | None = None
    segment_measurements: list[Any] | None = None
    class_fractions: dict[str, Any] | None = None
    inference_time_ms: float | None = None
    capture_timestamp_source: str | None = None
    capture_timestamp_confidence: float | None = None
    horizon_y_px: float | None = None
    elapsed_seconds: float | None = None
    stream_session_id: str | None = None
    ingest_latency_ms: float | None = None


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
    recording_id: str | None = None
    frame_index: int | None = None
    extraction_status: str | None = None
    extraction_reason: str | None = None
    shoreline_position_px: float | None = None
    signed_position_px: float | None = None
    position_axis: str | None = None
    shoreline_axis: str | None = None
    measured_coverage: float | None = None
    segment_count: int | None = None
    inference_time_ms: float | None = None
    horizon_y_px: float | None = None
    capture_timestamp_source: str | None = None
    capture_timestamp_confidence: float | None = None
    elapsed_seconds: float | None = None
    stream_session_id: str | None = None
    ingest_latency_ms: float | None = None


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
    # Model identity / contract. Optional so older clients keep working when
    # no artifact is provisioned.
    architecture: str | None = None
    encoder: str | None = None
    model_version: str | None = None
    class_names: list[str] | None = None
    class_ids: dict[str, int] | None = None
    water_classes: list[int] | None = None
    ignore_index: int | None = None
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    providers: list[str] = Field(default_factory=list)
    precision: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)


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


class ShorelineBaselineRead(BaseModel):
    id: int
    monitoring_station_id: int
    camera_id: int | None
    tide_station_id: int | None
    recording_id: str | None
    name: str
    model_type: str
    status: str | None
    provisional: bool | None
    reason: str | None
    sample_count: int
    is_active: bool
    position_axis: str | None
    position_sign: str | None
    position_unit: str | None
    detector_mode: str | None = None
    r_squared: float | None
    model_params: dict[str, Any] | None
    prediction_interval: dict[str, Any] | None
    valid_from: datetime | None
    valid_to: datetime | None
    notes: str | None
    created_at: datetime


class ShorelineBaselineStatus(BaseModel):
    """Explicit availability, so "no baseline" can never look like a normal one."""

    available: bool
    status: str
    reason: str | None = None
    detector_mode: str | None = None
    parameters: dict[str, Any] | None = None
    experimental_notice: str = (
        "Experimental baseline — insufficient recording history for validated "
        "anomaly detection. A rolling baseline is not tide-aware."
    )
    baselines: list[ShorelineBaselineRead] = Field(default_factory=list)
