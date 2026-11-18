"""Health-check response schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    timestamp: datetime


class DatabaseHealthResponse(BaseModel):
    status: str
    database: str
    reachable: bool
    latency_ms: float | None = None
    server_version: str | None = None
    timescaledb_version: str | None = None
    postgis_version: str | None = None
    error: str | None = None
