"""Schemas for camera management and calibration profiles (Phase 3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

CameraType = Literal["rtsp", "usb", "upload"]


class CameraCreate(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    monitoring_station_id: int
    camera_type: CameraType
    # RTSP transport. Never logged with credentials.
    rtsp_url: str | None = Field(default=None, max_length=500)
    # USB device index (OpenCV VideoCapture index).
    device_index: int | None = Field(default=None, ge=0)
    resolution_width: int | None = Field(default=None, ge=1, le=16384)
    resolution_height: int | None = Field(default=None, ge=1, le=16384)
    fps: float | None = Field(default=None, gt=0, le=1000)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    orientation_deg: float | None = Field(default=None, ge=0.0, lt=360.0)
    monitoring_region: dict[str, Any] | None = None
    enabled: bool = True

    @field_validator("code", "name")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @model_validator(mode="after")
    def _check_transport(self) -> CameraCreate:
        if self.camera_type == "rtsp" and not self.rtsp_url:
            raise ValueError("rtsp_url is required for camera_type='rtsp'")
        if self.camera_type == "usb" and self.device_index is None:
            raise ValueError("device_index is required for camera_type='usb'")
        if self.camera_type == "upload" and (self.rtsp_url or self.device_index is not None):
            raise ValueError("upload cameras take neither rtsp_url nor device_index")
        return self


class CameraRead(BaseModel):
    id: int
    code: str
    name: str
    monitoring_station_id: int
    station_code: str | None
    station_name: str | None
    camera_type: str
    # RTSP URLs may carry credentials — only the scheme and host are returned.
    rtsp_url_redacted: str | None
    device_index: int | None
    resolution_width: int | None
    resolution_height: int | None
    fps: float | None
    latitude: float | None
    longitude: float | None
    orientation_deg: float | None
    monitoring_region: dict[str, Any] | None
    enabled: bool
    active_calibration_id: int | None
    created_at: datetime


class CameraStatusRead(BaseModel):
    camera_id: int
    code: str
    enabled: bool
    connected: bool
    healthy: bool
    message: str
    last_frame_at: datetime | None
    frames_captured: int
    checked_at: datetime


class CameraQualityRead(BaseModel):
    score: float
    is_black: bool
    is_frozen: bool
    is_blurry: bool
    is_obstructed: bool
    is_timestamp_inconsistent: bool
    flags: list[str]


class CameraTestConnectionResult(BaseModel):
    camera_id: int
    ok: bool
    message: str
    latency_ms: float | None
    frame_width: int | None
    frame_height: int | None
    captured_at: datetime | None
    quality: CameraQualityRead | None


class CameraCalibrationCreate(BaseModel):
    reference_points: dict[str, Any] | None = None
    homography_matrix: list[Any] | None = None
    meters_per_pixel: float | None = Field(default=None, gt=0)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    notes: str | None = Field(default=None, max_length=2000)
    set_active: bool = True

    @field_validator("homography_matrix")
    @classmethod
    def _check_homography(cls, value: list[Any] | None) -> list[Any] | None:
        if value is None:
            return value
        rows_ok = all(isinstance(row, list | tuple) and len(row) == 3 for row in value)
        if len(value) != 3 or not rows_ok:
            raise ValueError("homography_matrix must be a 3x3 row-major matrix")
        return value


class CameraCalibrationRead(BaseModel):
    id: int
    camera_id: int
    version: int
    reference_points: dict[str, Any] | None
    homography_matrix: list[Any] | None
    meters_per_pixel: float | None
    is_active: bool
    valid_from: datetime | None
    valid_to: datetime | None
    notes: str | None
    created_at: datetime
