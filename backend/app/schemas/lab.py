"""Schemas for the Predictor + Anomaly Test Lab (Phase 7)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from app.schemas.videos import RollingBaselineOptions

EXPERIMENTAL_NOTICE = (
    "Experimental rolling-baseline drawdown detector — not validated for "
    "tsunami detection. Follow the official national tsunami warning authority."
)


class LabRunRead(BaseModel):
    run_id: str
    kind: str
    status: str
    created_at: datetime
    updated_at: datetime
    timezone: str
    source: dict[str, Any]
    progress: dict[str, Any]
    row_count: int
    event_count: int
    errors: list[str]
    warnings: list[str]
    artifacts: list[str]
    persist: bool
    camera_id: int | None
    recording_id: str | None
    experimental_notice: str = EXPERIMENTAL_NOTICE


class LabRowsResponse(BaseModel):
    run_id: str
    total: int
    limit: int
    offset: int
    items: list[dict[str, Any]]
    experimental_notice: str = EXPERIMENTAL_NOTICE


class LabEventRead(BaseModel):
    onset_time: datetime
    detection_time: datetime
    onset_frame_index: int | None
    detection_frame_index: int | None
    detection_latency_s: float
    persistence: int
    max_classification: str
    peak_cumulative_drawdown_px: float | None = None
    max_robust_z: float | None = None
    end_reason: str | None = None


class LabStreamRunRequest(BaseModel):
    camera_id: int | None = Field(default=None, ge=1)
    youtube_url: str | None = Field(default=None, max_length=500)
    sample_interval_seconds: float = Field(default=30.0, ge=1.0, le=3600.0)
    max_samples: int = Field(default=12, ge=1, le=1000)
    max_duration_seconds: float = Field(default=300.0, ge=5.0, le=86400.0)
    timezone: str = Field(default="Asia/Bangkok", max_length=64)
    detector_mode: Literal["rolling_baseline"] = "rolling_baseline"
    rolling: RollingBaselineOptions = Field(default_factory=RollingBaselineOptions)
    persist: bool = False

    @model_validator(mode="after")
    def _one_source(self) -> LabStreamRunRequest:
        if bool(self.camera_id) == bool(self.youtube_url):
            raise ValueError("provide exactly one of camera_id or youtube_url")
        if self.youtube_url:
            from app.services.camera.youtube import validate_youtube_url

            try:
                self.youtube_url = validate_youtube_url(self.youtube_url)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        if self.persist and self.camera_id is None:
            raise ValueError("persist=true requires a registered camera_id")
        return self
