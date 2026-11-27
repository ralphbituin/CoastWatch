"""Schemas for camera management and calibration profiles (Phase 3)."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.services.vision.roi import RoiPolygonError, normalize_roi_polygon

CameraType = Literal["rtsp", "usb", "upload", "video", "youtube_live"]


class RoiPolygon(BaseModel):
    """One simple monitoring ring in normalized frame coordinates."""

    coordinate_space: Literal["normalized"] = "normalized"
    points: list[tuple[float, float]] = Field(min_length=3, max_length=200)
    frame_width: int | None = Field(default=None, ge=1, le=16384)
    frame_height: int | None = Field(default=None, ge=1, le=16384)

    @model_validator(mode="after")
    def _validate_geometry(self) -> RoiPolygon:
        try:
            normalized = normalize_roi_polygon(self.model_dump())
        except RoiPolygonError as exc:
            raise ValueError(str(exc)) from exc
        if normalized is None:  # pragma: no cover - fields are required
            raise ValueError("roi_polygon is required")
        self.points = [(float(x), float(y)) for x, y in normalized["points"]]
        return self


class RoiPolygonSave(BaseModel):
    """Append a new calibration version carrying this outline (or clear it)."""

    roi_polygon: RoiPolygon | None = None
    notes: str | None = Field(default=None, max_length=2000)


class CameraCreate(BaseModel):
    code: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=200)
    monitoring_station_id: int
    camera_type: CameraType
    # RTSP transport. Never logged with credentials.
    rtsp_url: str | None = Field(default=None, max_length=500)
    # USB device index (OpenCV VideoCapture index).
    device_index: int | None = Field(default=None, ge=0)
    # Virtual video camera: video file name relative to DATA_DIR/videos/.
    video_path: str | None = Field(default=None, max_length=500)
    # YouTube Live (public/authorized streams only). Query strings are never
    # returned by the API and never logged.
    youtube_url: str | None = Field(default=None, max_length=500)
    stream_quality: str | None = Field(default=None, max_length=32)
    stream_format: str | None = Field(default=None, max_length=32)
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
        if self.camera_type == "upload" and (
            self.rtsp_url or self.device_index is not None or self.video_path
        ):
            raise ValueError("upload cameras take neither rtsp_url, device_index nor video_path")
        if self.camera_type == "video":
            if not self.video_path or not self.video_path.strip():
                raise ValueError("video_path is required for camera_type='video'")
            if self.rtsp_url or self.device_index is not None:
                raise ValueError("video cameras take neither rtsp_url nor device_index")
            if self.youtube_url:
                raise ValueError("video cameras do not take youtube_url")
        if self.camera_type == "youtube_live":
            if not self.youtube_url or not self.youtube_url.strip():
                raise ValueError("youtube_url is required for camera_type='youtube_live'")
            if self.rtsp_url or self.device_index is not None or self.video_path:
                raise ValueError(
                    "youtube_live cameras take neither rtsp_url, device_index nor video_path"
                )
            from app.services.camera.youtube import validate_youtube_url

            try:
                self.youtube_url = validate_youtube_url(self.youtube_url)
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
        elif self.youtube_url:
            raise ValueError("youtube_url is only valid for camera_type='youtube_live'")
        return self


class CameraUpdate(BaseModel):
    """Partial camera update. Omitted fields are left unchanged.

    Source secrets (``rtsp_url``, ``youtube_url``) are never returned by the API,
    so clients omit them when they only edit display/metadata fields. The route
    validates the merged transport contract before writing.
    """

    code: str | None = Field(default=None, min_length=1, max_length=64)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    monitoring_station_id: int | None = None
    camera_type: CameraType | None = None
    rtsp_url: str | None = Field(default=None, max_length=500)
    device_index: int | None = Field(default=None, ge=0)
    video_path: str | None = Field(default=None, max_length=500)
    youtube_url: str | None = Field(default=None, max_length=500)
    stream_quality: str | None = Field(default=None, max_length=32)
    stream_format: str | None = Field(default=None, max_length=32)
    resolution_width: int | None = Field(default=None, ge=1, le=16384)
    resolution_height: int | None = Field(default=None, ge=1, le=16384)
    fps: float | None = Field(default=None, gt=0, le=1000)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    orientation_deg: float | None = Field(default=None, ge=0.0, lt=360.0)
    monitoring_region: dict[str, Any] | None = None
    enabled: bool | None = None

    @field_validator("code", "name")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return value
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value


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
    video_path: str | None
    youtube_url_redacted: str | None = None
    stream_quality: str | None = None
    stream_format: str | None = None
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
    # --- Phase 7 live capture status (optional; absent for non-live clients) ---
    is_live: bool | None = None
    stream_state: str | None = None
    stream_session_id: str | None = None
    stream_recording_id: str | None = None
    frames_persisted: int | None = None
    frames_rejected: int | None = None
    reconnect_count: int | None = None
    last_error: str | None = None
    last_rolling_classification: str | None = None
    last_event_at: datetime | None = None
    resolved_quality: str | None = None
    resolved_format: str | None = None
    experimental_notice: str = (
        "Experimental monitoring prototype — not a validated tsunami detector. "
        "Follow the official national tsunami warning authority."
    )


class CameraStreamStatusRead(BaseModel):
    camera_id: int
    state: str
    session_id: str | None = None
    recording_id: str | None = None
    started_at: datetime | None = None
    last_frame_at: datetime | None = None
    frames_captured: int = 0
    frames_persisted: int = 0
    frames_rejected: int = 0
    consecutive_failures: int = 0
    reconnect_count: int = 0
    last_error: str | None = None
    last_rolling_classification: str | None = None
    last_event_at: datetime | None = None
    sample_interval_seconds: float = 30.0
    is_live: bool | None = None
    experimental_notice: str = (
        "Experimental in-process live capture. YouTube Live timestamps are "
        "wall-clock receipt times, not per-frame capture times."
    )


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
    is_live: bool | None = None
    resolved_format: str | None = None
    resolved_quality: str | None = None
    source_url_redacted: str | None = None


class CameraCalibrationCreate(BaseModel):
    reference_points: dict[str, Any] | None = None
    homography_matrix: list[Any] | None = None
    meters_per_pixel: float | None = Field(default=None, gt=0)
    roi_polygon: RoiPolygon | None = None
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
    roi_polygon: RoiPolygon | None = None
    is_active: bool
    valid_from: datetime | None
    valid_to: datetime | None
    notes: str | None
    created_at: datetime
