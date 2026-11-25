"""Schemas for the Phase 6 anomaly-event API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnomalyEventRead(BaseModel):
    id: int
    detected_at: datetime
    monitoring_station_id: int
    camera_id: int | None
    recording_id: str | None
    frame_index: int | None
    state: str
    classification: str | None
    anomaly_score: float | None
    baseline_status: str | None
    tide_adjusted: bool | None
    tide_data_available: bool | None
    tide_source: str | None
    tide_height_m: float | None
    observed_position_px: float | None
    expected_position_px: float | None
    residual_px: float | None
    robust_z: float | None
    movement_rate_px_s: float | None
    persistence_count: int | None
    quality_status: str | None
    reason: str | None
    explanation: str | None
    contributing_factors: dict[str, Any] | None
    acknowledged: bool
    created_at: datetime
    # Phase 6 rev 2 rolling-baseline fields (optional; legacy rows unaffected).
    detector_mode: str | None = None
    resolved_detector_mode: str | None = None
    mode_reason: str | None = None
    onset_frame_index: int | None = None
    rolling_baseline_px: float | None = None
    rolling_scale_px: float | None = None
    scale_floored: bool | None = None
    cumulative_drawdown_px: float | None = None
    baseline_frozen: bool | None = None
    event_max_classification: str | None = None
    onset_time: datetime | None = None
    detection_time: datetime | None = None
    detection_latency_s: float | None = None
    event_window_seconds: float | None = None
    cooldown_state: str | None = None
    config_hash: str | None = None


class AnomalyEventList(BaseModel):
    total: int
    items: list[AnomalyEventRead] = Field(default_factory=list)
    experimental_notice: str = (
        "An anomaly is a monitoring signal, not confirmed tsunami activity. "
        "Follow the official national tsunami warning authority."
    )


class AnomalySummary(BaseModel):
    total: int
    by_classification: dict[str, int] = Field(default_factory=dict)
    by_state: dict[str, int] = Field(default_factory=dict)
    by_baseline_status: dict[str, int] = Field(default_factory=dict)
    experimental_notice: str = (
        "Rolling-baseline classifications are exploratory and are never NORMAL. "
        "An anomaly is not a confirmed tsunami."
    )


class AcknowledgeResult(BaseModel):
    id: int
    acknowledged: bool
