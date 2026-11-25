"""Schemas for archive-video registration and prototype processing."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class VideoRecordingCreate(BaseModel):
    # Path relative to DATA_DIR/videos/. No absolute paths / traversal.
    video_path: str = Field(min_length=1, max_length=500)
    camera_id: int | None = None
    monitoring_station_id: int | None = None
    timezone: str = "Asia/Bangkok"
    # Verified visible-clock anchors. Manual verification is acceptable; no OCR
    # confidence is implied unless the operator provides it.
    clock_anchors: list[dict[str, Any]] | None = None
    timestamp_source: str | None = None
    timestamp_confidence: float | None = None
    notes: str | None = None


class VideoRecordingRead(BaseModel):
    id: int
    recording_id: str
    camera_id: int | None
    monitoring_station_id: int | None
    file_path: str
    file_size_bytes: int | None
    duration_seconds: float | None
    fps: float | None
    frame_count: int | None
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None
    timezone: str | None
    timestamp_source: str | None
    timestamp_confidence: float | None
    processing_status: str | None
    processed_frames: int | None
    sampled_frames: int | None
    metadata_json: dict[str, Any] | None
    notes: str | None
    created_at: datetime


class RollingBaselineOptions(BaseModel):
    """Time-based options for the tide-independent rolling detector."""

    window_seconds: float = Field(default=1200.0, gt=0.0, le=86400.0)
    min_history_fraction: float = Field(default=0.75, gt=0.0, le=1.0)
    k_seconds: float = Field(default=90.0, gt=0.0, le=3600.0)
    z_threshold: float = Field(default=3.5, gt=0.0, le=20.0)
    sigma_k_multiplier: float = Field(default=5.0, gt=0.0, le=50.0)
    drawdown_floor_frac: float = Field(default=150.0 / 1080.0, gt=0.0, le=1.0)
    scale_floor_frac: float = Field(default=0.004, gt=0.0, le=1.0)
    persistence: int = Field(default=3, ge=2, le=20)
    max_gap_seconds: float = Field(default=45.0, gt=0.0, le=3600.0)
    exit_z: float = Field(default=2.0, gt=0.0, le=20.0)
    exit_count: int = Field(default=2, ge=1, le=20)
    max_hold_seconds: float = Field(default=3600.0, gt=0.0, le=86400.0)
    cooldown_seconds: float = Field(default=600.0, ge=0.0, le=86400.0)
    persistent_seconds: float = Field(default=600.0, gt=0.0, le=86400.0)
    confidence_floor: float = Field(default=0.80, ge=0.0, le=1.0)
    stale_run_length: int = Field(default=10, ge=2, le=1000)
    require_segment_agreement: bool = False
    freeze_baseline: bool = True
    camera_motion_shift_px: float = Field(default=20.0, gt=0.0, le=1000.0)
    min_timestamp_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class VideoProcessRequest(BaseModel):
    recording_id: str = Field(min_length=1, max_length=64)
    # Persistence requires a registered camera (observations have a camera FK).
    camera_id: int | None = None
    sample_interval_seconds: float = Field(default=60.0, ge=1.0, le=3600.0)
    start_elapsed: float = Field(default=0.0, ge=0.0)
    end_elapsed: float | None = Field(default=None, ge=0.0)
    # A synchronous HTTP request must be bounded. The offline script processes
    # the full archive; this endpoint is a smoke/portion runner.
    max_samples: int = Field(default=12, ge=1, le=120)
    country_iso: str | None = Field(default="TH", max_length=2)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    max_station_distance_km: float = Field(default=120.0, gt=0.0, le=2000.0)
    persistence: int = Field(default=3, ge=1, le=20)
    z_threshold: float = Field(default=3.5, gt=0.0, le=20.0)
    save_frames: bool = False
    detector_mode: Literal[
        "auto", "rolling_baseline", "tide_conditioned", "movement_only"
    ] = "auto"
    rolling: RollingBaselineOptions = Field(default_factory=RollingBaselineOptions)


class VideoProcessResult(BaseModel):
    recording_id: str
    processed: int
    available: int
    unavailable: int
    baseline_status: str
    baseline_reason: str | None
    tide_data_available: bool
    tide_reason: str | None
    classifications: dict[str, int]
    observations_persisted: int
    observations_skipped: int = 0
    duplicate_observations: int = 0
    out_of_order_observations: int = 0
    anomaly_events_persisted: int
    detector_mode: str = "auto"
    resolved_detector_mode: str = "movement_only"
    mode_reason: str = ""
    rolling_event_count: int = 0
    experimental_notice: str = (
        "Experimental prototype. Nighttime masks are out-of-domain and anomalies "
        "do not confirm tsunami activity. Follow the official national tsunami "
        "warning authority."
    )
