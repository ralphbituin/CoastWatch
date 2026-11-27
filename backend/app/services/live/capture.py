"""In-process live capture service (Phase 7).

Design choice: **one supervised daemon thread per enabled live camera**, managed
by a single process-local :class:`LiveCaptureManager`. The repository has no
external worker framework and uses synchronous SQLAlchemy/OpenCV, so adding
Celery/Redis would be a second architecture. FastAPI's lifespan starts/stops the
manager; explicit stream endpoints start/stop one camera.

Each worker:

* samples on a monotonic schedule (default 30 s) and drops missed slots instead
  of queueing frames (backpressure);
* opens a **fresh database session per iteration** and never holds a transaction
  during inference;
* runs one inference at a time per camera;
* persists one canonical observation per sample through the recording service;
* feeds one stateful :class:`RollingBaselineDetector` exactly once per stored
  row, warm-started from stored history on start/reconnect;
* reconnects with bounded exponential backoff and records the failure state.

Unit tests inject a fake provider and the runner never opens a real stream.
"""

from __future__ import annotations

import contextlib
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.camera import Camera
from app.models.stream import (
    STREAM_FAILED,
    STREAM_RECONNECTING,
    STREAM_RUNNING,
    STREAM_STARTING,
    STREAM_STOPPED,
    StreamSession,
)
from app.services.anomaly.rolling_baseline import (
    RollingBaselineConfig,
    RollingBaselineDetector,
)
from app.services.camera import manager as camera_manager
from app.services.camera.base import CameraError, CameraProvider
from app.services.camera.youtube import redact_youtube_url
from app.services.live.preview_cache import PreviewCacheEntry, get_preview_cache
from app.services.recording.service import (
    load_rolling_history,
    observation_to_rolling,
    persist_rolling_row,
    write_observation,
)
from app.services.vision.roi import apply_roi_to_class_mask, load_active_roi
from app.services.vision.segmentation import (
    InvalidFrameError,
    ShorelineSegmenter,
)

logger = get_logger(__name__)

LIVE_CAMERA_TYPES = {"rtsp", "usb", "video", "youtube_live"}
MAX_BACKOFF_SECONDS = 300.0


def _session_id(camera_id: int) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"stream-{camera_id}-{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(slots=True)
class LiveStatus:
    camera_id: int
    state: str = STREAM_STOPPED
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "state": self.state,
            "session_id": self.session_id,
            "recording_id": self.recording_id,
            "started_at": None if self.started_at is None else self.started_at.isoformat(),
            "last_frame_at": (
                None if self.last_frame_at is None else self.last_frame_at.isoformat()
            ),
            "frames_captured": self.frames_captured,
            "frames_persisted": self.frames_persisted,
            "frames_rejected": self.frames_rejected,
            "consecutive_failures": self.consecutive_failures,
            "reconnect_count": self.reconnect_count,
            "last_error": self.last_error,
            "last_rolling_classification": self.last_rolling_classification,
            "last_event_at": (
                None if self.last_event_at is None else self.last_event_at.isoformat()
            ),
            "sample_interval_seconds": self.sample_interval_seconds,
            "is_live": self.is_live,
        }


@dataclass(slots=True)
class _WorkerContext:
    provider: CameraProvider
    detector: RollingBaselineDetector
    rolling_config: RollingBaselineConfig
    status: LiveStatus
    frame_counter: int = 0
    previous_signed_position: float | None = None
    previous_orientation: str | None = None
    # Resolved once at worker start. Restarting the stream picks up a new outline.
    roi_polygon: dict[str, Any] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


class LiveCaptureWorker:
    """Supervises one camera's sampled capture loop."""

    def __init__(
        self,
        camera_id: int,
        *,
        session_factory: Callable[[], Any],
        provider_factory: Callable[[Camera], CameraProvider] | None = None,
        segmenter_factory: Callable[[], Any] | None = None,
        sample_interval_seconds: float = 30.0,
    ) -> None:
        self.camera_id = camera_id
        self._session_factory = session_factory
        self._provider_factory = provider_factory or camera_manager.build_provider
        self._segmenter_factory = segmenter_factory or ShorelineSegmenter
        self._interval = max(1.0, float(sample_interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._status = LiveStatus(camera_id=camera_id, sample_interval_seconds=self._interval)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------
    @property
    def status(self) -> LiveStatus:
        with self._lock:
            from dataclasses import replace

            return replace(self._status)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._set_status(state=STREAM_STARTING, last_error=None)
        self._thread = threading.Thread(
            target=self._run,
            name=f"coastwatch-live-{self.camera_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
        self._thread = None

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def _run(self) -> None:
        provider: CameraProvider | None = None
        try:
            provider, context = self._initialise()
            self._set_status(state=STREAM_RUNNING)
            next_tick = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                if now < next_tick:
                    self._stop.wait(next_tick - now)
                    continue
                overdue = now - next_tick
                if overdue > self._interval:
                    # Backpressure: drop missed slots instead of queueing.
                    next_tick = now
                self._sample_once(context)
                next_tick += self._interval
        except Exception as exc:  # pragma: no cover - startup/defensive
            logger.exception("live capture worker %s failed", self.camera_id)
            self._set_status(state=STREAM_FAILED, last_error=str(exc))
            self._record_session_error(str(exc), state=STREAM_FAILED)
        finally:
            if provider is not None:
                try:
                    provider.disconnect()
                except Exception:  # pragma: no cover - best effort
                    logger.warning("live worker %s: disconnect failed", self.camera_id)
            self._finish_session()

    def _initialise(self) -> tuple[CameraProvider, _WorkerContext]:
        with self._session_factory() as session:
            camera = session.get(Camera, self.camera_id)
            if camera is None:
                raise CameraError(f"camera {self.camera_id} no longer exists")
            if camera.camera_type not in LIVE_CAMERA_TYPES:
                raise CameraError(
                    f"camera {camera.code} type {camera.camera_type!r} has no live stream"
                )
            session_id = _session_id(camera.id)
            source_redacted = (
                redact_youtube_url(camera.youtube_url)
                if camera.camera_type == "youtube_live"
                else camera_manager.redact_rtsp_url(camera.rtsp_url)
            )
            row = StreamSession(
                session_id=session_id,
                recording_id=session_id,
                camera_id=camera.id,
                started_at=datetime.now(UTC),
                status=STREAM_STARTING,
                source_url_redacted=source_redacted,
                sample_interval_seconds=self._interval,
                metadata_json={
                    "camera_type": camera.camera_type,
                    "stream_quality": camera.stream_quality,
                    "stream_format": camera.stream_format,
                },
            )
            session.add(row)
            session.commit()

            config = RollingBaselineConfig(
                nominal_interval_seconds=self._interval,
                frame_height=(
                    float(camera.resolution_height) if camera.resolution_height else None
                ),
            )
            detector = RollingBaselineDetector(config)
            # Warm start across the most recent window for this camera, so a
            # restart is not blind for 20 minutes. The detector gates decide
            # which historical rows are usable.
            history = load_rolling_history(
                session,
                camera.id,
                window_seconds=config.window_seconds,
            )
            detector.warm_start([observation_to_rolling(item) for item in history])
            roi_polygon = load_active_roi(session, camera)
            # Never serve a previous session's cached frame after a restart.
            get_preview_cache().clear(camera.id)
            provider = self._provider_factory(camera)
            self._set_status(
                state=STREAM_STARTING,
                session_id=session_id,
                recording_id=session_id,
                started_at=row.started_at,
                is_live=(camera.camera_type == "youtube_live"),
            )
            provider.connect()
            logger.info(
                "live capture %s started session=%s warm_start_rows=%s "
                "frame_height=%s (camera record fallback; replaced by the first "
                "decoded frame)",
                camera.code,
                session_id,
                len(history),
                config.frame_height,
            )
            return provider, _WorkerContext(
                provider=provider,
                detector=detector,
                rolling_config=config,
                status=self.status,
                roi_polygon=roi_polygon,
            )

    def _sample_once(self, context: _WorkerContext) -> None:
        if self._stop.is_set():
            return
        try:
            frame = context.provider.read_frame()
        except CameraError as exc:
            self._handle_failure(context, str(exc))
            return
        self._sync_frame_height(context, frame)
        received = time.perf_counter()
        self._set_status(last_frame_at=frame.captured_at)

        segmenter = self._segmenter_factory()
        result = None
        position = None
        segmentation: Any = None
        extraction_status = "unavailable"
        extraction_reason: str | None = None
        horizon_y: float | None = None
        class_fractions: dict[str, Any] | None = None
        inference_ms: float | None = None
        try:
            segmentation = segmenter.predict(frame.image)
            inference_ms = getattr(segmentation, "inference_ms", None)
            class_fractions = {
                str(name): row.get("fraction")
                for name, row in (getattr(segmentation, "class_statistics", {}) or {}).items()
            }
            class_mask = getattr(segmentation, "class_mask", None)
            if class_mask is None:
                raise InvalidFrameError("segmenter returned no native class mask")
            from app.services.video.pipeline import _horizon_y

            # The operator outline is applied to a copy before any use of the
            # class mask; the stored/segmented mask itself is never modified.
            monitored_mask = apply_roi_to_class_mask(class_mask, context.roi_polygon)
            horizon_y = _horizon_y(monitored_mask)
            from app.services.vision.shoreline import extract_shoreline_from_classes

            result = extract_shoreline_from_classes(
                monitored_mask, confidence=segmentation.confidence
            )
            if not result.shoreline_available:
                extraction_reason = result.reason or "shoreline_unavailable"
            else:
                from app.services.vision.position import measure_shoreline_position

                continuity_ok = self._context_orientation(context) == result.orientation_used
                position = measure_shoreline_position(
                    result.segments,
                    result.orientation_used,
                    previous_signed_position=(
                        context.previous_signed_position if continuity_ok else None
                    ),
                    frame_shape=frame.image.shape[:2],
                    primary_region_fraction=0.5,
                )
                if position.available:
                    extraction_status = "available"
                    with context.lock:
                        context.previous_signed_position = position.signed_position_px
                        context.previous_orientation = result.orientation_used
                else:
                    extraction_reason = position.reason or "no_usable_segment"
        except InvalidFrameError as exc:
            extraction_reason = f"invalid_frame: {exc}"
        except Exception as exc:
            extraction_reason = f"segmentation_failed: {exc}"
            logger.warning("live capture %s: inference failed: %s", self.camera_id, exc)

        elapsed_started = self.status.started_at or frame.captured_at
        elapsed = max(0.0, (frame.captured_at - elapsed_started).total_seconds())
        with context.lock:
            context.frame_counter += 1
            frame_index = context.frame_counter

        with self._session_factory() as session:
            camera = session.get(Camera, self.camera_id)
            if camera is None:
                return
            ingest_latency_ms = round((time.perf_counter() - received) * 1000.0, 3)
            outcome = write_observation(
                session,
                camera,
                observed_at=frame.captured_at,
                result=result if (result is not None and result.shoreline_available) else None,
                quality_score=None,
                quality_flags=[],
                source_frame_path=None,
                recording_id=context.status.recording_id,
                frame_index=frame_index,
                elapsed_seconds=round(elapsed, 3),
                stream_session_id=context.status.session_id,
                ingest_latency_ms=ingest_latency_ms,
                timestamp_source=(
                    "provider_virtual_clock"
                    if camera.camera_type == "video"
                    else "stream_wall_clock"
                ),
                timestamp_confidence=0.3 if camera.camera_type == "video" else 0.6,
                extraction_status=extraction_status,
                extraction_reason=extraction_reason,
                shoreline_geometry=getattr(result, "points", None) if result else None,
                shoreline_segments=getattr(result, "segments", None) if result else None,
                segmentation_confidence=(
                    getattr(segmentation, "confidence", None) if segmentation else None
                ),
                shoreline_position_px=(position.position_px if position is not None else None),
                signed_position_px=(position.signed_position_px if position is not None else None),
                position_axis=position.position_axis if position is not None else None,
                position_sign=position.position_sign if position is not None else None,
                segment_count=position.segment_count if position is not None else 0,
                segment_measurements=(position.segment_dicts() if position is not None else None),
                class_fractions=class_fractions,
                inference_time_ms=inference_ms,
                horizon_y_px=horizon_y,
                frame_shape=frame.image.shape[:2],
            )
            if outcome.status not in {"persisted", "updated"}:
                try:
                    session.commit()
                except Exception:  # pragma: no cover - defensive
                    session.rollback()
                self._increment_rejected()
                self._set_status(last_error=outcome.reason)
                return
            try:
                session.commit()
            except Exception as exc:
                session.rollback()
                self._increment_rejected()
                self._set_status(last_error=f"persistence_failed: {exc}")
                logger.exception("live capture %s: observation persistence failed", self.camera_id)
                return
            self._increment_persisted()
            self._set_status(last_frame_at=frame.captured_at)
            if segmentation is not None:
                get_preview_cache().put(
                    PreviewCacheEntry(
                        camera_id=self.camera_id,
                        frame=frame,
                        segmentation=segmentation,
                        shoreline=result,
                        captured_at=frame.captured_at,
                        observation_id=(
                            f"{context.status.recording_id or self.camera_id}:{frame_index}"
                        ),
                        recording_id=context.status.recording_id,
                        frame_index=frame_index,
                        inference_ms=inference_ms,
                        ingest_latency_ms=ingest_latency_ms,
                        stored_monotonic=time.monotonic(),
                        monitoring_outline_present=context.roi_polygon is not None,
                    )
                )

            # Advance the stateful detector only after the row is stored.
            if outcome.observation is None:  # pragma: no cover - persisted rows carry one
                return
            try:
                rolling_row = context.detector.update(observation_to_rolling(outcome.observation))
                persist_rolling_row(
                    session,
                    camera,
                    rolling_row,
                    recording_id=context.status.recording_id or "",
                    frame_index=frame_index,
                )
                self._update_session_counters(session, context, rolling_row)
                session.commit()
                self._set_status(
                    last_rolling_classification=rolling_row.classification,
                    last_error=None,
                )
                if rolling_row.is_event:
                    self._set_status(
                        last_event_at=rolling_row.detection_time or rolling_row.timestamp
                    )
            except Exception as exc:
                session.rollback()
                logger.exception("live capture %s: rolling persistence failed", self.camera_id)
                self._set_status(last_error=f"DATA_UNRELIABLE: rolling persistence failed: {exc}")
                self._rebuild_detector(context)
        # Successful sample resets failure counters.
        self._set_status(consecutive_failures=0, state=STREAM_RUNNING)

    def _sync_frame_height(self, context: _WorkerContext, frame: Any) -> None:
        """Keep the detector's pixel floors tied to the decoded frame size."""
        height = float(frame.image.shape[0])
        current = context.rolling_config.frame_height
        if current is None:
            # No camera-record fallback: adopt the first real frame without
            # treating it as a change (there is no prior pixel scale).
            context.rolling_config.frame_height = height
            logger.info(
                "live capture %s: effective frame_height=%.0f px " "(from first decoded frame)",
                self.camera_id,
                height,
            )
            return
        if current == height:
            return
        config = replace(context.rolling_config, frame_height=height)
        context.rolling_config = config
        detector = RollingBaselineDetector(config)
        detector.reset_for_resolution_change(frame.captured_at)
        context.detector = detector
        with context.lock:
            context.previous_signed_position = None
        logger.info(
            "live capture %s: decoded frame height changed %.0f -> %.0f px; "
            "detector rebuilt (drawdown floor %.1f px)",
            self.camera_id,
            current,
            height,
            config.drawdown_floor_px,
        )

    def _context_orientation(self, context: _WorkerContext) -> str | None:
        with context.lock:
            return context.previous_orientation

    def _rebuild_detector(self, context: _WorkerContext) -> None:
        """Recover in-memory state from persisted rows after a write failure."""
        with self._session_factory() as session:
            history = load_rolling_history(
                session,
                self.camera_id,
                window_seconds=context.rolling_config.window_seconds,
            )
            detector = RollingBaselineDetector(context.rolling_config)
            detector.warm_start([observation_to_rolling(item) for item in history])
            context.detector = detector

    def _handle_failure(self, context: _WorkerContext, message: str) -> None:
        failures = self.status.consecutive_failures + 1
        reconnects = self.status.reconnect_count + 1
        self._set_status(
            state=STREAM_RECONNECTING,
            consecutive_failures=failures,
            reconnect_count=reconnects,
            last_error=message,
        )
        self._record_session_error(message, state=STREAM_RECONNECTING)
        with contextlib.suppress(Exception):
            context.provider.disconnect()
        delay = min(self._interval * (2 ** min(failures, 5)), MAX_BACKOFF_SECONDS)
        self._stop.wait(max(1.0, delay))
        if self._stop.is_set():
            return
        try:
            # Re-resolve/reconnect lazily on the next read (YouTube URLs expire).
            context.provider.connect()
            self._rebuild_detector(context)
            self._set_status(state=STREAM_RUNNING, last_error=None)
        except Exception as exc:
            self._set_status(last_error=f"{message}; reconnect failed: {exc}")

    # ------------------------------------------------------------------
    # Status / persistence helpers
    # ------------------------------------------------------------------
    def _set_status(self, **values: Any) -> None:
        with self._lock:
            for key, value in values.items():
                if hasattr(self._status, key):
                    setattr(self._status, key, value)

    def _increment_persisted(self) -> None:
        with self._lock:
            self._status.frames_captured += 1
            self._status.frames_persisted += 1

    def _increment_rejected(self) -> None:
        with self._lock:
            self._status.frames_captured += 1
            self._status.frames_rejected += 1

    def _update_session_counters(
        self, session: Any, context: _WorkerContext, rolling_row: Any
    ) -> None:
        row = (
            session.query(StreamSession)
            .filter_by(session_id=context.status.session_id)
            .one_or_none()
        )
        if row is None:
            return
        status = self.status
        row.status = STREAM_RUNNING
        row.last_frame_at = status.last_frame_at
        row.frames_captured = status.frames_captured
        row.frames_persisted = status.frames_persisted
        row.frames_rejected = status.frames_rejected
        row.reconnect_count = status.reconnect_count
        row.last_rolling_classification = rolling_row.classification
        row.last_error = None
        if rolling_row.is_event and rolling_row.detection_time is not None:
            row.last_event_at = rolling_row.detection_time

    def _record_session_error(self, message: str, *, state: str) -> None:
        try:
            with self._session_factory() as session:
                session_id = self.status.session_id
                if not session_id:
                    return
                row = session.query(StreamSession).filter_by(session_id=session_id).one_or_none()
                if row is None:
                    return
                row.status = state
                row.last_error = message
                row.frames_captured = self.status.frames_captured
                row.frames_persisted = self.status.frames_persisted
                row.frames_rejected = self.status.frames_rejected
                row.reconnect_count = self.status.reconnect_count
                session.commit()
        except Exception:  # pragma: no cover - status must never crash capture
            logger.warning("live capture %s: could not record session error", self.camera_id)

    def _finish_session(self) -> None:
        final_state = STREAM_FAILED if self.status.state == STREAM_FAILED else STREAM_STOPPED
        try:
            with self._session_factory() as session:
                session_id = self.status.session_id
                if not session_id:
                    return
                row = session.query(StreamSession).filter_by(session_id=session_id).one_or_none()
                if row is None:
                    return
                row.status = final_state
                row.ended_at = datetime.now(UTC)
                row.frames_captured = self.status.frames_captured
                row.frames_persisted = self.status.frames_persisted
                row.frames_rejected = self.status.frames_rejected
                row.reconnect_count = self.status.reconnect_count
                session.commit()
        except Exception:  # pragma: no cover - shutdown best effort
            logger.warning("live capture %s: could not finalise session", self.camera_id)
        get_preview_cache().clear(self.camera_id)
        self._set_status(state=final_state)


class LiveCaptureManager:
    """Process-local registry of per-camera capture workers."""

    def __init__(self) -> None:
        self._workers: dict[int, LiveCaptureWorker] = {}
        self._last: dict[int, LiveStatus] = {}
        self._lock = threading.Lock()
        self._session_factory: Callable[[], Any] | None = None
        self._provider_factory: Callable[[Camera], CameraProvider] | None = None
        self._segmenter_factory: Callable[[], Any] | None = None
        self._interval = 30.0

    def configure(
        self,
        *,
        session_factory: Callable[[], Any] | None = None,
        provider_factory: Callable[[Camera], CameraProvider] | None = None,
        segmenter_factory: Callable[[], Any] | None = None,
        sample_interval_seconds: float | None = None,
    ) -> None:
        if session_factory is not None:
            self._session_factory = session_factory
        if provider_factory is not None:
            self._provider_factory = provider_factory
        if segmenter_factory is not None:
            self._segmenter_factory = segmenter_factory
        if sample_interval_seconds is not None:
            self._interval = max(1.0, float(sample_interval_seconds))

    def _session_factory_resolved(self) -> Callable[[], Any]:
        if self._session_factory is not None:
            return self._session_factory
        from app.database.session import get_session_factory

        return get_session_factory()

    def start_camera(self, camera_id: int) -> LiveStatus:
        with self._lock:
            worker = self._workers.get(camera_id)
            if worker is not None and worker.alive:
                return worker.status
            worker = LiveCaptureWorker(
                camera_id,
                session_factory=self._session_factory_resolved(),
                provider_factory=self._provider_factory,
                segmenter_factory=self._segmenter_factory,
                sample_interval_seconds=self._interval,
            )
            self._workers[camera_id] = worker
        worker.start()
        return worker.status

    def stop_camera(self, camera_id: int, timeout: float = 10.0) -> LiveStatus:
        with self._lock:
            worker = self._workers.pop(camera_id, None)
        if worker is not None:
            worker.stop(timeout=timeout)
            self._last[camera_id] = worker.status
            return worker.status
        return self._last.get(camera_id, LiveStatus(camera_id=camera_id, state=STREAM_STOPPED))

    def status(self, camera_id: int) -> LiveStatus:
        with self._lock:
            worker = self._workers.get(camera_id)
        if worker is not None:
            snapshot = worker.status
            self._last[camera_id] = snapshot
            return snapshot
        return self._last.get(camera_id, LiveStatus(camera_id=camera_id, state=STREAM_STOPPED))

    def running_camera_ids(self) -> list[int]:
        with self._lock:
            return [camera_id for camera_id, worker in self._workers.items() if worker.alive]

    def start(self) -> None:
        """Lifespan startup: resume autostart cameras if configured."""
        settings = get_settings()
        self._interval = float(settings.live_capture_interval_seconds)
        if not settings.live_capture_autostart:
            logger.info("live capture manager started (autostart disabled)")
            return
        try:
            factory = self._session_factory_resolved()
            with factory() as session:
                from sqlalchemy import select

                cameras = session.scalars(
                    select(Camera).where(
                        Camera.enabled.is_(True),
                        Camera.camera_type.in_(LIVE_CAMERA_TYPES),
                    )
                ).all()
                ids = [camera.id for camera in cameras]
        except Exception as exc:  # pragma: no cover - startup DB may be absent
            logger.warning("live capture autostart skipped: %s", exc)
            return
        for camera_id in ids:
            try:
                self.start_camera(camera_id)
            except Exception:  # pragma: no cover - one bad camera must not block
                logger.exception("live capture autostart failed for camera %s", camera_id)

    def stop(self) -> None:
        """Lifespan shutdown: stop every worker and disconnect providers."""
        with self._lock:
            worker_ids = list(self._workers)
        for camera_id in worker_ids:
            try:
                self.stop_camera(camera_id)
            except Exception:  # pragma: no cover - best effort
                logger.exception("live capture stop failed for camera %s", camera_id)
        camera_manager.drop_all_providers()


_manager: LiveCaptureManager | None = None
_manager_lock = threading.Lock()


def get_live_manager() -> LiveCaptureManager:
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = LiveCaptureManager()
        return _manager
