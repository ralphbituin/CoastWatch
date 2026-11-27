"""Shoreline routes (observations, calibrations). Implemented in Phase 4.

Until a validated segmentation model exists, waterlines come from
**operator-labelled masks** (`POST /shorelines/observations`) or, once a
model is trained and provisioned, from `POST /shorelines/observe`. Nothing
here fabricates a waterline: unavailable segmentation reports
``DATA_UNRELIABLE`` with an explicit reason.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.api.deps import get_db, pagination_params
from app.core.exceptions import (
    DataUnavailableError,
    NotFoundError,
    ValidationError,
)
from app.core.logging import get_logger
from app.models.camera import Camera, CameraCalibration
from app.models.shoreline import ShorelineBaseline, ShorelineObservation
from app.models.station import MonitoringStation
from app.schemas.shorelines import (
    ShorelineBaselineRead,
    ShorelineBaselineStatus,
    ShorelineCalibrationRead,
    ShorelineModuleStatus,
    ShorelineObservationDetail,
    ShorelineObservationSummary,
    ShorelineObserveRequest,
    ShorelineObserveResult,
)
from app.services.pipeline import CoastWatchPipeline, record_observation
from app.services.vision.calibration import (
    CalibrationProfile,
)
from app.services.vision.segmentation import model_status
from app.services.vision.shoreline import (
    ShorelineExtractionError,
    extract_shoreline,
)

logger = get_logger(__name__)

router = APIRouter(tags=["shorelines"])

MAX_MASK_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


def _get_camera(db: Session, camera_id: int) -> Camera:
    camera = db.get(Camera, camera_id, options=[joinedload(Camera.station)])
    if camera is None:
        raise NotFoundError(f"Camera not found: {camera_id}")
    return camera


def _observation_summary(
    row: ShorelineObservation,
    camera_code: str | None = None,
    station_code: str | None = None,
) -> ShorelineObservationSummary:
    return ShorelineObservationSummary(
        camera_id=row.camera_id,
        camera_code=camera_code,
        observed_at=row.observed_at,
        monitoring_station_id=row.monitoring_station_id,
        station_code=station_code,
        calibration_id=row.calibration_id,
        calibrated_position_m=row.calibrated_position_m,
        segmentation_confidence=row.segmentation_confidence,
        frame_quality_score=row.frame_quality_score,
        frame_quality_flags=list(row.frame_quality_flags or []),
        created_at=row.created_at,
        recording_id=row.recording_id,
        frame_index=row.frame_index,
        extraction_status=row.extraction_status,
        extraction_reason=row.extraction_reason,
        shoreline_position_px=row.shoreline_position_px,
        signed_position_px=row.signed_position_px,
        position_axis=row.position_axis,
        shoreline_axis=row.shoreline_axis,
        measured_coverage=row.measured_coverage,
        segment_count=row.segment_count,
        inference_time_ms=row.inference_time_ms,
        horizon_y_px=row.horizon_y_px,
        capture_timestamp_source=row.capture_timestamp_source,
        capture_timestamp_confidence=row.capture_timestamp_confidence,
        elapsed_seconds=row.elapsed_seconds,
        stream_session_id=row.stream_session_id,
        ingest_latency_ms=row.ingest_latency_ms,
    )


def _observation_detail(
    row: ShorelineObservation,
    camera_code: str | None = None,
    station_code: str | None = None,
) -> ShorelineObservationDetail:
    return ShorelineObservationDetail(
        camera_id=row.camera_id,
        camera_code=camera_code,
        observed_at=row.observed_at,
        monitoring_station_id=row.monitoring_station_id,
        station_code=station_code,
        calibration_id=row.calibration_id,
        shoreline_geometry=row.shoreline_geometry,
        shoreline_segments=row.shoreline_segments,
        shoreline_axis=row.shoreline_axis,
        interpolated_ranges=row.interpolated_ranges,
        measured_coverage=row.measured_coverage,
        calibrated_position_m=row.calibrated_position_m,
        segmentation_confidence=row.segmentation_confidence,
        frame_quality_score=row.frame_quality_score,
        frame_quality_flags=list(row.frame_quality_flags or []),
        source_frame_path=row.source_frame_path,
        created_at=row.created_at,
        recording_id=row.recording_id,
        frame_index=row.frame_index,
        extraction_status=row.extraction_status,
        extraction_reason=row.extraction_reason,
        shoreline_position_px=row.shoreline_position_px,
        signed_position_px=row.signed_position_px,
        position_axis=row.position_axis,
        position_sign=row.position_sign,
        segment_count=row.segment_count,
        segment_measurements=row.segment_measurements,
        class_fractions=row.class_fractions,
        inference_time_ms=row.inference_time_ms,
        capture_timestamp_source=row.capture_timestamp_source,
        capture_timestamp_confidence=row.capture_timestamp_confidence,
        horizon_y_px=row.horizon_y_px,
        elapsed_seconds=row.elapsed_seconds,
        stream_session_id=row.stream_session_id,
        ingest_latency_ms=row.ingest_latency_ms,
    )


@router.get("", response_model=ShorelineModuleStatus, summary="Shorelines module status")
def shorelines_status() -> ShorelineModuleStatus:
    status = model_status()
    available = bool(status.get("available"))
    config_valid = bool(status.get("config_valid"))
    if not available:
        model_note = (
            "No segmentation model is provisioned; waterlines require operator-labelled masks."
        )
    elif not config_valid:
        model_note = (
            "Model artifact present but its config.json is unusable: " f"{status.get('reason')}"
        )
    else:
        model_note = (
            "SegFormer MiT-B2 loaded from MODEL_PATH/shoreline-segmentation "
            "(classes Beach=0, Ocean=1, Sky=2; water = Ocean only)."
        )
    notes = [model_note, "Baselines land in Phase 5; anomalies in Phase 6."]
    for limitation in status.get("limitations", []):
        notes.append(f"Model limitation: {limitation}")
    return ShorelineModuleStatus(
        segmentation_model_available=available,
        implemented_endpoints=[
            "GET /shorelines/observations",
            "GET /shorelines/observations/latest",
            "POST /shorelines/observations (operator-labelled mask)",
            "POST /shorelines/observe",
            "GET /shorelines/calibrations",
        ],
        notes=notes,
        architecture=status.get("architecture"),
        encoder=status.get("encoder"),
        model_version=status.get("model_version"),
        class_names=status.get("class_names"),
        class_ids=status.get("class_ids"),
        water_classes=status.get("water_classes"),
        ignore_index=status.get("ignore_index"),
        checkpoint_id=status.get("checkpoint_id"),
        checkpoint_sha256=status.get("checkpoint_sha256"),
        providers=list(status.get("providers", [])),
        precision=status.get("precision", {}) or {},
        limitations=list(status.get("limitations", [])),
    )


@router.get(
    "/observations",
    response_model=list[ShorelineObservationSummary],
    summary="List shoreline observations",
)
def list_observations(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    station_id: int | None = None,
    pagination: tuple[int, int] = Depends(pagination_params),
) -> list[ShorelineObservationSummary]:
    limit, offset = pagination
    stmt = (
        select(ShorelineObservation, Camera.code, MonitoringStation.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .join(MonitoringStation, MonitoringStation.id == ShorelineObservation.monitoring_station_id)
        .order_by(ShorelineObservation.observed_at.desc())
        .limit(limit)
        .offset(offset)
    )
    if camera_id is not None:
        stmt = stmt.where(ShorelineObservation.camera_id == camera_id)
    if station_id is not None:
        stmt = stmt.where(ShorelineObservation.monitoring_station_id == station_id)
    return [
        _observation_summary(row, camera_code, station_code)
        for row, camera_code, station_code in db.execute(stmt)
    ]


@router.get(
    "/observations/latest",
    response_model=ShorelineObservationDetail,
    summary="Latest shoreline observation (optionally per camera or station)",
)
def latest_observation(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    station_id: int | None = None,
) -> ShorelineObservationDetail:
    stmt = (
        select(ShorelineObservation, Camera.code, MonitoringStation.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .join(MonitoringStation, MonitoringStation.id == ShorelineObservation.monitoring_station_id)
        .order_by(ShorelineObservation.observed_at.desc())
        .limit(1)
    )
    if camera_id is not None:
        stmt = stmt.where(ShorelineObservation.camera_id == camera_id)
    if station_id is not None:
        stmt = stmt.where(ShorelineObservation.monitoring_station_id == station_id)
    row = db.execute(stmt).first()
    if row is None:
        raise DataUnavailableError(
            "No shoreline observations recorded yet",
            details={"camera_id": camera_id, "station_id": station_id},
        )
    observation, camera_code, station_code = row
    return _observation_detail(observation, camera_code, station_code)


@router.get(
    "/observations/detail",
    response_model=ShorelineObservationDetail,
    summary="One shoreline observation by camera and timestamp",
)
def observation_detail(
    camera_id: int,
    observed_at: datetime,
    db: Session = Depends(get_db),
) -> ShorelineObservationDetail:
    row = db.execute(
        select(ShorelineObservation, Camera.code, MonitoringStation.code)
        .join(Camera, Camera.id == ShorelineObservation.camera_id)
        .join(
            MonitoringStation,
            MonitoringStation.id == ShorelineObservation.monitoring_station_id,
        )
        .where(
            ShorelineObservation.camera_id == camera_id,
            ShorelineObservation.observed_at == observed_at,
        )
        .limit(1)
    ).first()
    if row is None:
        raise NotFoundError(
            f"Shoreline observation not found for camera {camera_id} at {observed_at}"
        )
    observation, camera_code, station_code = row
    return _observation_detail(observation, camera_code, station_code)


@router.post(
    "/observations",
    response_model=ShorelineObservationDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Record an observation from an operator-labelled water mask",
)
async def create_observation_from_mask(
    camera_id: int = Form(...),
    seaward_axis: str | None = Form(default=None),
    observed_at: str | None = Form(default=None),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> ShorelineObservationDetail:
    camera = _get_camera(db, camera_id)
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg"}:
        raise ValidationError(f"Mask must be PNG/JPEG, got {suffix!r}")
    content = await file.read()
    if len(content) > MAX_MASK_UPLOAD_BYTES:
        raise ValidationError(
            f"Mask too large ({len(content)} bytes; max {MAX_MASK_UPLOAD_BYTES}).",
            details={"max_bytes": MAX_MASK_UPLOAD_BYTES},
        )
    mask = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValidationError("Mask file could not be decoded as an image")

    axis: tuple[float, float] | None = None
    if seaward_axis:
        try:
            parts = [float(p) for p in seaward_axis.split(",")]
            if len(parts) != 2:
                raise ValueError
            axis = (parts[0], parts[1])
        except ValueError as exc:
            raise ValidationError("seaward_axis must be 'dx,dy' (e.g. '0.94,0.34')") from exc

    parsed_observed_at: datetime | None = None
    if observed_at:
        try:
            parsed_observed_at = datetime.fromisoformat(observed_at)
        except ValueError as exc:
            raise ValidationError(
                "observed_at must be ISO 8601 (e.g. 2026-11-19T06:30:00+00:00)"
            ) from exc
    if parsed_observed_at is None:
        raise ValidationError(
            "operator-labelled masks require an explicit observed_at; processing "
            "time is not used as a capture time",
            details={"reason": "observed_at_required"},
        )
    if parsed_observed_at.tzinfo is None:
        raise ValidationError(
            "observed_at must include a UTC offset (e.g. 2026-11-19T06:30:00+00:00)"
        )

    calibration = None
    if camera.active_calibration_id:
        row = db.get(CameraCalibration, camera.active_calibration_id)
        if row is not None and row.homography_matrix is not None:
            calibration = CalibrationProfile(
                camera_id=camera.id,
                version=row.version,
                homography=row.homography_matrix,
                meters_per_pixel=row.meters_per_pixel,
            )
    try:
        result = extract_shoreline(mask, calibration=calibration, seaward_axis=axis)
    except ShorelineExtractionError as exc:
        raise ValidationError(str(exc), details={"camera_id": camera.id}) from exc

    if axis is not None and calibration is None:
        # seaward_axis is only meaningful with a calibrated camera; surface that.
        result.calibrated_position_m = None

    try:
        observation = record_observation(
            db,
            camera,
            observed_at=parsed_observed_at,
            result=result,
            calibration_id=camera.active_calibration_id,
            quality_score=None,
            quality_flags=[],
            source_frame_path=None,
            timestamp_source="supplied",
            timestamp_confidence=1.0,
        )
    except ValueError as exc:
        raise ValidationError(str(exc), details={"reason": "out_of_order"}) from exc
    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("failed to persist operator-mask observation for camera %s", camera.id)
        raise ValidationError(f"could not persist observation: {exc}") from exc
    db.refresh(observation)
    station_code = camera.station.code if camera.station else None
    return _observation_detail(observation, camera.code, station_code)


@router.post(
    "/observe",
    response_model=ShorelineObserveResult,
    summary="Run one observation cycle (camera → vision) for a camera",
)
def observe(
    payload: ShorelineObserveRequest, db: Session = Depends(get_db)
) -> ShorelineObserveResult:
    pipeline = CoastWatchPipeline(db)
    result = pipeline.process_once(station_id=payload.station_id, camera_id=payload.camera_id)
    return ShorelineObserveResult(
        station_id=result.station_id,
        camera_id=result.camera_id,
        observed_at=result.observed_at,
        actual_shoreline_m=result.actual_shoreline_m,
        expected_shoreline_m=result.expected_shoreline_m,
        residual_m=result.residual_m,
        retreat_velocity_m_s=result.retreat_velocity_m_s,
        anomaly_score=result.anomaly_score,
        state=result.state,  # type: ignore[arg-type]
        details=result.details,
    )


@router.get(
    "/calibrations",
    response_model=list[ShorelineCalibrationRead],
    summary="List calibration profiles across cameras (read-only)",
)
def list_calibrations(db: Session = Depends(get_db)) -> list[ShorelineCalibrationRead]:
    rows = db.execute(
        select(CameraCalibration, Camera.code)
        .join(Camera, Camera.id == CameraCalibration.camera_id)
        .order_by(Camera.code, CameraCalibration.version.desc())
    ).all()
    return [
        ShorelineCalibrationRead(
            id=cal.id,
            camera_id=cal.camera_id,
            camera_code=code,
            version=cal.version,
            meters_per_pixel=cal.meters_per_pixel,
            is_active=cal.is_active,
            valid_from=cal.valid_from,
            valid_to=cal.valid_to,
            notes=cal.notes,
            created_at=cal.created_at,
        )
        for cal, code in rows
    ]


@router.get(
    "/baselines",
    response_model=ShorelineBaselineStatus,
    summary="Provisional tide-conditioned baselines (explicit unavailable state)",
)
def list_baselines(
    db: Session = Depends(get_db),
    camera_id: int | None = None,
    station_id: int | None = None,
) -> ShorelineBaselineStatus:
    stmt = select(ShorelineBaseline).order_by(ShorelineBaseline.created_at.desc())
    if camera_id is not None:
        stmt = stmt.where(ShorelineBaseline.camera_id == camera_id)
    if station_id is not None:
        stmt = stmt.where(ShorelineBaseline.monitoring_station_id == station_id)
    rows = list(db.scalars(stmt.limit(50)).all())
    if not rows:
        return ShorelineBaselineStatus(
            available=False,
            status="unavailable",
            detector_mode="rolling_baseline",
            reason=(
                "No baseline has been fitted for this selection. A tide-conditioned "
                "baseline requires verified, geographically appropriate tide data and "
                "a documented minimum of paired observations. No rolling-median "
                "baseline has been computed yet either."
            ),
            baselines=[],
        )
    latest = rows[0]
    tide_available = any(
        bool(row.is_active) and (row.status or "") in {"provisional", "available"} for row in rows
    )
    rolling = next((row for row in rows if row.detector_mode == "rolling_baseline"), None)
    if tide_available:
        status = latest.status or "provisional"
        detector_mode = latest.detector_mode or "tide_conditioned"
        parameters = latest.model_params
        reason = latest.reason
    elif rolling is not None:
        status = "rolling_unverified"
        detector_mode = "rolling_baseline"
        parameters = (
            rolling.model_params.get("config")
            if isinstance(rolling.model_params, dict)
            else rolling.model_params
        )
        reason = rolling.reason or (
            "No verified tide-conditioned baseline is active. The rolling median is "
            "tide-independent and cannot establish an absolute expected shoreline "
            "or a NORMAL state."
        )
    else:
        status = latest.status or "unavailable"
        detector_mode = latest.detector_mode or "unavailable"
        parameters = latest.model_params
        reason = latest.reason

    return ShorelineBaselineStatus(
        available=tide_available,
        status=status,
        reason=reason,
        detector_mode=detector_mode,
        parameters=parameters if isinstance(parameters, dict) else None,
        baselines=[
            ShorelineBaselineRead(
                id=row.id,
                monitoring_station_id=row.monitoring_station_id,
                camera_id=row.camera_id,
                tide_station_id=row.tide_station_id,
                recording_id=row.recording_id,
                name=row.name,
                model_type=row.model_type,
                status=row.status,
                provisional=row.provisional,
                reason=row.reason,
                sample_count=row.sample_count,
                is_active=row.is_active,
                position_axis=row.position_axis,
                position_sign=row.position_sign,
                position_unit=row.position_unit,
                detector_mode=row.detector_mode,
                r_squared=row.r_squared,
                model_params=row.model_params,
                prediction_interval=row.prediction_interval,
                valid_from=row.valid_from,
                valid_to=row.valid_to,
                notes=row.notes,
                created_at=row.created_at,
            )
            for row in rows
        ],
    )
