"""Predictor + Anomaly Test Lab API (Phase 7).

Persistence is **off by default**: a lab run never creates production
observations/events unless ``persist=true`` and a camera is selected. Even then
it uses the Phase 7 recording service and its idempotency keys.
"""

from __future__ import annotations

import json
from dataclasses import fields as dataclass_fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import get_db, pagination_params
from app.core.config import get_settings
from app.core.exceptions import NotFoundError, ValidationError
from app.models.camera import Camera
from app.models.video import VideoRecording
from app.schemas.lab import (
    LabRowsResponse,
    LabRunRead,
    LabStreamRunRequest,
)
from app.schemas.videos import RollingBaselineOptions
from app.services.camera import manager as camera_manager
from app.services.camera.video import resolve_video_path
from app.services.camera.youtube import YouTubeLiveProvider, validate_youtube_url
from app.services.lab import runner
from app.services.lab.registry import LabRun, get_lab_registry
from app.services.video.timestamps import ClockAnchor, anchor_dicts
from app.services.vision.roi import load_active_roi

router = APIRouter(tags=["lab"])

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
_ROLLING_FIELDS = {item.name for item in dataclass_fields(runner.RollingBaselineConfig)}


def _run_read(run: LabRun) -> LabRunRead:
    return LabRunRead(**run.snapshot())


def _rolling_from_json(text: str | None) -> runner.RollingBaselineConfig:
    if not text:
        config = runner.RollingBaselineConfig()
    else:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid rolling config JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValidationError("rolling config must be a JSON object")
        unknown = set(data) - _ROLLING_FIELDS
        if unknown:
            raise ValidationError(
                f"unknown rolling config fields: {sorted(unknown)}",
                details={"allowed": sorted(_ROLLING_FIELDS)},
            )
        config = runner.RollingBaselineConfig(**data)
    try:
        config.validate()
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return config


def _rolling_from_options(options: RollingBaselineOptions) -> runner.RollingBaselineConfig:
    config = runner.RollingBaselineConfig(**options.model_dump())
    try:
        config.validate()
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return config


def _camera_or_404(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id)
    if camera is None:
        raise NotFoundError(f"Camera not found: {camera_id}")
    return camera


@router.get("/runs", response_model=list[LabRunRead], summary="List lab runs")
def list_lab_runs() -> list[LabRunRead]:
    registry = get_lab_registry()
    registry.cleanup_expired()
    return [_run_read(run) for run in registry.list()]


@router.post(
    "/runs/images",
    response_model=LabRunRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run the predictor + rolling detector over batch still images",
)
async def create_image_run(
    images: list[UploadFile] = File(...),
    manifest: UploadFile | None = File(default=None),
    manifest_text: str | None = Form(default=None),
    start_at: str | None = Form(default=None),
    interval_seconds: float | None = Form(default=None),
    timezone: str = Form(default="Asia/Bangkok"),
    rolling_json: str | None = Form(default=None),
    persist: bool = Form(default=False),
    camera_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> LabRunRead:
    settings = get_settings()
    if not images:
        raise ValidationError("at least one image is required")
    if len(images) > settings.lab_max_images:
        raise ValidationError(f"too many images ({len(images)}; max {settings.lab_max_images})")
    if persist and camera_id is None:
        raise ValidationError("persist=true requires a camera_id")
    roi_polygon = None
    if camera_id is not None:
        roi_polygon = load_active_roi(db, _camera_or_404(db, camera_id))

    payloads: list[tuple[str, bytes]] = []
    total_bytes = 0
    for upload in images:
        name = Path(upload.filename or "image").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            raise ValidationError(f"unsupported image type {suffix!r} ({name})")
        content = await upload.read()
        if len(content) > settings.lab_max_image_bytes:
            raise ValidationError(
                f"{name} is too large ({len(content)} bytes; max {settings.lab_max_image_bytes})"
            )
        total_bytes += len(content)
        payloads.append((name, content))
    if total_bytes > settings.lab_max_total_upload_bytes:
        raise ValidationError(
            f"total upload is too large ({total_bytes} bytes; "
            f"max {settings.lab_max_total_upload_bytes})"
        )

    text = manifest_text
    if manifest is not None:
        raw = await manifest.read()
        if len(raw) > runner.MAX_MANIFEST_BYTES:
            raise ValidationError("timestamp manifest is too large")
        text = raw.decode("utf-8", errors="replace")

    try:
        timestamps, source = runner.resolve_image_timestamps(
            [name for name, _ in payloads],
            manifest_text=text,
            start_at=start_at,
            interval_seconds=interval_seconds,
            timezone=timezone,
        )
    except ValueError as exc:
        raise ValidationError(str(exc), details={"reason": "timestamp_source_required"}) from exc
    config = _rolling_from_json(rolling_json)

    registry = get_lab_registry()
    try:
        run = registry.create("images", timezone=timezone, persist=persist, camera_id=camera_id)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    run.update(
        source={
            "kind": "images",
            "file_count": len(payloads),
            "timestamp_source": source,
            "start_at": start_at,
            "interval_seconds": interval_seconds,
            "files": [name for name, _ in payloads],
        }
    )
    runner_arguments: dict[str, Any] = {
        "images": payloads,
        "timestamps": timestamps,
        "timestamp_source": source,
        "rolling": config,
        "interval_seconds": interval_seconds
        or ((timestamps[1] - timestamps[0]).total_seconds() if len(timestamps) > 1 else 30.0),
        "persist": persist,
        "camera_id": camera_id,
        "roi_polygon": roi_polygon,
    }
    registry.submit(run, lambda job: runner.run_images(job, **runner_arguments))
    return _run_read(run)


@router.post(
    "/runs/video",
    response_model=LabRunRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run the predictor + rolling detector over a video recording or upload",
)
async def create_video_run(
    recording_id: str | None = Form(default=None),
    upload: UploadFile | None = File(default=None),
    video_path: str | None = Form(default=None),
    sample_interval_seconds: float = Form(default=30.0),
    start_elapsed: float = Form(default=0.0),
    end_elapsed: float | None = Form(default=None),
    max_samples: int = Form(default=12),
    start_at: str | None = Form(default=None),
    clock_anchors: str | None = Form(default=None),
    timezone: str = Form(default="Asia/Bangkok"),
    rolling_json: str | None = Form(default=None),
    persist: bool = Form(default=False),
    camera_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
) -> LabRunRead:
    settings = get_settings()
    if max_samples < 1 or max_samples > settings.lab_max_video_samples:
        raise ValidationError(f"max_samples must be between 1 and {settings.lab_max_video_samples}")
    if sample_interval_seconds < 1.0:
        raise ValidationError("sample_interval_seconds must be >= 1")
    if persist and camera_id is None:
        raise ValidationError("persist=true requires a camera_id")
    roi_polygon = None
    if camera_id is not None:
        roi_polygon = load_active_roi(db, _camera_or_404(db, camera_id))

    anchors: list[ClockAnchor] = []
    if clock_anchors:
        try:
            raw_anchors = json.loads(clock_anchors)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"invalid clock_anchors JSON: {exc}") from exc
        if not isinstance(raw_anchors, list):
            raise ValidationError("clock_anchors must be a JSON list")
        for item in raw_anchors:
            if (
                not isinstance(item, dict)
                or "elapsed_seconds" not in item
                or "timestamp" not in item
            ):
                raise ValidationError("each clock anchor needs elapsed_seconds and timestamp")
            anchors.append(
                ClockAnchor(
                    elapsed_seconds=float(item["elapsed_seconds"]),
                    timestamp=str(item["timestamp"]),
                    source=str(item.get("source", "visible_clock")),
                    confidence=float(item.get("confidence", 0.9)),
                    verification=item.get("verification"),
                )
            )
    elif start_at:
        anchors = [
            ClockAnchor(
                elapsed_seconds=0.0,
                timestamp=start_at,
                source="operator_declared",
                confidence=0.8,
                verification="operator-declared archive start",
            )
        ]
    if not anchors:
        raise ValidationError(
            "archive video needs verified clock anchors or an operator start time; "
            "processing time is never used as capture time",
            details={"reason": "timestamp_source_required"},
        )

    source_label: str
    if upload is not None:
        name = Path(upload.filename or "upload.mp4").name
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_VIDEO_EXTENSIONS:
            raise ValidationError(f"unsupported video type {suffix!r}")
        content = await upload.read()
        if len(content) > settings.lab_max_video_upload_bytes:
            raise ValidationError(
                f"video is too large ({len(content)} bytes; "
                f"max {settings.lab_max_video_upload_bytes})"
            )
        video_bytes: bytes | None = content
        resolved_path: Path | None = None
        source_label = name
    elif recording_id:
        recording = db.scalar(
            select(VideoRecording).where(VideoRecording.recording_id == recording_id).limit(1)
        )
        if recording is None:
            raise NotFoundError(f"Video recording not found: {recording_id}")
        resolved_path = Path(recording.file_path)
        if not resolved_path.is_file():
            raise ValidationError(f"recording file is no longer available: {recording.file_path}")
        video_bytes = None
        source_label = recording_id
    elif video_path:
        try:
            resolved_path = resolve_video_path(settings.data_dir, video_path)
        except Exception as exc:
            raise ValidationError(str(exc)) from exc
        if not resolved_path.is_file():
            raise ValidationError(f"video not found under DATA_DIR/videos: {video_path}")
        video_bytes = None
        source_label = video_path
    else:
        raise ValidationError("provide a recording_id, video_path or uploaded video file")

    config = _rolling_from_json(rolling_json)
    registry = get_lab_registry()
    try:
        run = registry.create("video", timezone=timezone, persist=persist, camera_id=camera_id)
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    if video_bytes is not None:
        if run.temp_dir is None:  # pragma: no cover - created by registry
            raise ValidationError("lab temp directory unavailable")
        target = run.temp_dir / f"upload{Path(source_label).suffix.lower() or '.mp4'}"
        target.write_bytes(video_bytes)
        resolved_path = target
    assert resolved_path is not None
    run.update(
        source={
            "kind": "video",
            "recording_id": recording_id,
            "file_name": source_label,
            "anchors": anchor_dicts(anchors),
            "utc_offset_note": (
                "timestamps are UTC in the database; display uses the deployment timezone"
            ),
        }
    )
    runner_arguments: dict[str, Any] = {
        "video_path": resolved_path,
        "anchors": anchors,
        "rolling": config,
        "sample_interval_seconds": sample_interval_seconds,
        "start_elapsed": start_elapsed,
        "end_elapsed": end_elapsed,
        "max_samples": max_samples,
        "persist": persist,
        "camera_id": camera_id,
        "roi_polygon": roi_polygon,
    }
    registry.submit(run, lambda job: runner.run_video(job, **runner_arguments))
    return _run_read(run)


@router.post(
    "/runs/stream",
    response_model=LabRunRead,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run the predictor + stateful rolling detector over a bounded live stream",
)
def create_stream_run(payload: LabStreamRunRequest, db: Session = Depends(get_db)) -> LabRunRead:
    settings = get_settings()
    if payload.max_samples > settings.lab_max_stream_samples:
        raise ValidationError(f"max_samples must be <= {settings.lab_max_stream_samples}")
    if payload.max_duration_seconds > settings.lab_max_stream_seconds:
        raise ValidationError(f"max_duration_seconds must be <= {settings.lab_max_stream_seconds}")
    camera_id = payload.camera_id
    roi_polygon = None
    if camera_id is not None:
        camera = _camera_or_404(db, camera_id)
        roi_polygon = load_active_roi(db, camera)
        if camera.camera_type not in {"rtsp", "usb", "video", "youtube_live"}:
            raise ValidationError(
                "camera has no live stream",
                details={"camera_type": camera.camera_type},
            )
        provider = camera_manager.build_provider(camera)
    else:
        provider = YouTubeLiveProvider(
            validate_youtube_url(payload.youtube_url or ""),
            quality="best",
        )
    config = _rolling_from_options(payload.rolling)
    registry = get_lab_registry()
    try:
        run = registry.create(
            "stream",
            timezone=payload.timezone,
            persist=payload.persist,
            camera_id=camera_id,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    run.update(
        source={
            "kind": "stream",
            "camera_id": camera_id,
            "youtube": payload.youtube_url is not None,
            "timestamp_note": (
                "live rows use wall-clock UTC at frame receipt (confidence <= 0.6); "
                "YouTube provides no true per-frame capture timestamp"
            ),
        }
    )
    runner_arguments: dict[str, Any] = {
        "provider": provider,
        "rolling": config,
        "sample_interval_seconds": payload.sample_interval_seconds,
        "max_samples": payload.max_samples,
        "max_duration_seconds": payload.max_duration_seconds,
        "persist": payload.persist,
        "camera_id": camera_id,
        "roi_polygon": roi_polygon,
    }
    registry.submit(run, lambda job: runner.run_stream(job, **runner_arguments))
    return _run_read(run)


@router.get("/runs/{run_id}", response_model=LabRunRead, summary="Lab run status")
def get_lab_run(run_id: str) -> LabRunRead:
    run = get_lab_registry().get(run_id)
    if run is None:
        raise NotFoundError(f"Lab run not found: {run_id}")
    return _run_read(run)


@router.get(
    "/runs/{run_id}/rows",
    response_model=LabRowsResponse,
    summary="Paginated per-timestamp rows",
)
def get_lab_rows(
    run_id: str,
    pagination: tuple[int, int] = Depends(pagination_params),
) -> LabRowsResponse:
    run = get_lab_registry().get(run_id)
    if run is None:
        raise NotFoundError(f"Lab run not found: {run_id}")
    limit, offset = pagination
    with run.lock:
        rows = list(run.rows[offset : offset + limit])
        total = len(run.rows)
    return LabRowsResponse(run_id=run_id, total=total, limit=limit, offset=offset, items=rows)


@router.get(
    "/runs/{run_id}/events",
    summary="Rolling events produced by a lab run",
)
def get_lab_events(run_id: str) -> dict:
    run = get_lab_registry().get(run_id)
    if run is None:
        raise NotFoundError(f"Lab run not found: {run_id}")
    with run.lock:
        return {"run_id": run_id, "events": list(run.events)}


@router.get(
    "/runs/{run_id}/artifacts/{name}",
    summary="Download a bounded CSV/report/image artifact",
)
def get_lab_artifact(run_id: str, name: str) -> FileResponse:
    run = get_lab_registry().get(run_id)
    if run is None:
        raise NotFoundError(f"Lab run not found: {run_id}")
    path_value = run.artifacts.get(name)
    if path_value is None:
        raise NotFoundError(f"Artifact not found: {name}")
    path = Path(path_value).resolve()
    if run.temp_dir is None or not path.is_relative_to(run.temp_dir.resolve()):
        raise ValidationError("artifact path escapes the run directory")
    if not path.is_file():
        raise NotFoundError(f"Artifact no longer available: {name}")
    return FileResponse(path)


@router.delete("/runs/{run_id}", summary="Cancel and delete a lab run and its artifacts")
def delete_lab_run(run_id: str) -> dict:
    if not get_lab_registry().delete(run_id):
        raise NotFoundError(f"Lab run not found: {run_id}")
    return {"run_id": run_id, "deleted": True}


@router.get("/status", summary="Lab capability/limits summary")
def lab_status() -> dict:
    settings = get_settings()
    registry = get_lab_registry()
    return {
        "persistence_default": False,
        "active_runs": registry.active_count(),
        "limits": {
            "max_concurrent_runs": settings.lab_max_concurrent_runs,
            "max_images": settings.lab_max_images,
            "max_image_bytes": settings.lab_max_image_bytes,
            "max_total_upload_bytes": settings.lab_max_total_upload_bytes,
            "max_video_upload_bytes": settings.lab_max_video_upload_bytes,
            "max_video_samples": settings.lab_max_video_samples,
            "max_stream_samples": settings.lab_max_stream_samples,
            "max_stream_seconds": settings.lab_max_stream_seconds,
            "run_ttl_seconds": settings.lab_run_ttl_seconds,
        },
        "timestamp_sources": {
            "images": ["manifest", "operator_declared"],
            "video": ["derived (verified visible-clock anchors)", "operator_declared"],
            "stream": ["stream_wall_clock (confidence <= 0.6)"],
        },
        "experimental_notice": (
            "Experimental rolling-baseline drawdown detector — not validated for "
            "tsunami detection. Follow the official national tsunami warning authority."
        ),
        "checked_at": datetime.now(UTC).isoformat(),
    }
