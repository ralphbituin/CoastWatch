"""Shared schema helpers: the error envelope and generic payloads."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Consistent error envelope returned by every failing endpoint."""

    error: ErrorDetail


class Message(BaseModel):
    message: str


class NotImplementedPayload(BaseModel):
    """Returned by module routes whose implementation lands in a later phase."""

    module: str
    phase: int
    implemented: bool = False
    message: str
    planned_endpoints: list[str] = Field(default_factory=list)
