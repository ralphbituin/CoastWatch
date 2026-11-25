"""Schemas for live stream sessions (Phase 7)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel


class StreamSessionRead(BaseModel):
    id: int
    session_id: str
    recording_id: str
    camera_id: int
    camera_code: str | None = None
    started_at: datetime
    ended_at: datetime | None
    status: str
    source_url_redacted: str | None
    sample_interval_seconds: float | None
    last_frame_at: datetime | None
    frames_captured: int
    frames_persisted: int
    frames_rejected: int
    reconnect_count: int
    last_error: str | None
    last_rolling_classification: str | None
    last_event_at: datetime | None
    detector_config: dict[str, Any] | None
    metadata_json: dict[str, Any] | None
    created_at: datetime
    experimental_notice: str = (
        "Live rows use wall-clock receipt timestamps (confidence <= 0.6) and are "
        "an experimental monitoring signal, not a validated tsunami detector."
    )
