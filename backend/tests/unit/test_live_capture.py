"""Phase 7 live capture worker tests with an injected provider (no network)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.models.anomaly import AnomalyEvent
from app.models.shoreline import ShorelineObservation
from app.models.stream import StreamSession
from app.services.camera.base import (
    CameraMetadata,
    CameraProvider,
    CameraStatus,
    Frame,
)
from app.services.live import capture as capture_module
from app.services.live.capture import LiveCaptureWorker
from app.services.live.preview_cache import get_preview_cache


@pytest.fixture(autouse=True)
def _clear_preview_cache() -> None:
    get_preview_cache().clear_all()
    yield
    get_preview_cache().clear_all()


class FakeProvider(CameraProvider):
    def __init__(self, *, base: datetime | None = None, heights: list[int] | None = None) -> None:
        self.base = base or datetime.now(UTC)
        self.reads = 0
        self.connected = 0
        self.heights = heights or []

    def connect(self) -> None:
        self.connected += 1

    def disconnect(self) -> None:
        return None

    def read_frame(self) -> Frame:
        self.reads += 1
        moment = self.base + timedelta(seconds=30 * self.reads)
        height = self.heights[min(self.reads - 1, len(self.heights) - 1)] if self.heights else 60
        return Frame(
            image=np.zeros((height, 80, 3), dtype=np.uint8),
            captured_at=moment,
            metadata=CameraMetadata(
                camera_id=7,
                camera_type="youtube_live",
                width=80,
                height=height,
                extra={"is_live": True},
            ),
            sequence=self.reads,
        )

    def get_status(self) -> CameraStatus:
        return CameraStatus(True, True, "ok")

    def get_metadata(self) -> CameraMetadata:
        return CameraMetadata(camera_id=7, camera_type="youtube_live")


class FakeSegmentation:
    def __init__(self, shape: tuple[int, int] = (60, 80)) -> None:
        self.class_mask = self._mask(shape)
        self.confidence = 0.95
        self.inference_ms = 4.2
        self.class_statistics = {
            "Beach": {"fraction": 0.3},
            "Ocean": {"fraction": 0.5},
            "Sky": {"fraction": 0.2},
        }

    @staticmethod
    def _mask(shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[: max(1, height // 5), :] = 2  # Sky
        mask[max(1, height // 5) : max(2, (height * 2) // 3), :] = 1  # Ocean
        mask[max(2, (height * 2) // 3) :, :] = 0  # Beach
        return mask


class FakeSegmenter:
    def predict(self, image) -> FakeSegmentation:
        return FakeSegmentation(image.shape[:2])


def _camera() -> SimpleNamespace:
    return SimpleNamespace(
        id=7,
        code="YT-TEST",
        monitoring_station_id=3,
        camera_type="youtube_live",
        youtube_url="https://youtu.be/abc123",
        stream_quality="best",
        stream_format=None,
        resolution_height=60,
    )


class _Query:
    def __init__(self, rows: dict[str, StreamSession]) -> None:
        self.rows = rows
        self.key: str | None = None

    def filter_by(self, **kwargs: Any) -> _Query:
        self.key = kwargs.get("session_id")
        return self

    def one_or_none(self) -> StreamSession | None:
        if self.key and self.key in self.rows:
            return self.rows[self.key]
        return None


class FakeSession:
    """A tiny session double: no SQL, but the same add/commit surface."""

    def __init__(self, shared: dict[str, Any]) -> None:
        self.shared = shared

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get(self, _model, _identifier):  # noqa: ANN001
        return _camera()

    def scalar(self, _stmt):  # noqa: ANN001
        return None

    def query(self, _model):  # noqa: ANN001
        return _Query(self.shared["sessions"])

    def add(self, obj: object) -> None:
        self.shared["added"].append(obj)
        if isinstance(obj, StreamSession):
            self.shared["sessions"][obj.session_id] = obj

    def commit(self) -> None:
        self.shared["commits"] += 1

    def rollback(self) -> None:
        return None


def _worker(
    provider: FakeProvider | None = None,
) -> tuple[LiveCaptureWorker, dict[str, Any], FakeProvider]:
    shared: dict[str, Any] = {"added": [], "commits": 0, "sessions": {}}
    provider = provider or FakeProvider()

    def session_factory() -> FakeSession:
        return FakeSession(shared)

    worker = LiveCaptureWorker(
        7,
        session_factory=session_factory,
        provider_factory=lambda _camera: provider,
        segmenter_factory=FakeSegmenter,
        sample_interval_seconds=30.0,
    )
    return worker, shared, provider


def test_live_sample_uses_wall_clock_and_records_latency() -> None:
    worker, shared, provider = _worker()
    _provider, context = worker._initialise()
    worker._sample_once(context)

    observations = [o for o in shared["added"] if isinstance(o, ShorelineObservation)]
    assert len(observations) == 1
    row = observations[0]
    assert row.capture_timestamp_source == "stream_wall_clock"
    assert row.capture_timestamp_confidence == 0.6
    assert row.ingest_latency_ms is not None and row.ingest_latency_ms >= 0
    assert row.stream_session_id is not None
    assert row.recording_id == row.stream_session_id
    assert row.elapsed_seconds is not None
    assert row.signed_position_px is not None
    assert row.frame_index == 1
    assert provider.reads == 1

    events = [e for e in shared["added"] if isinstance(e, AnomalyEvent)]
    assert len(events) == 1
    assert events[0].detector_mode == "rolling_baseline"
    assert events[0].recording_id == row.recording_id
    assert worker.status.frames_persisted == 1
    assert worker.status.last_rolling_classification is not None


def test_live_detector_is_fed_once_per_persisted_row() -> None:
    worker, shared, _provider = _worker()
    _provider2, context = worker._initialise()
    worker._sample_once(context)
    worker._sample_once(context)

    observations = [o for o in shared["added"] if isinstance(o, ShorelineObservation)]
    events = [e for e in shared["added"] if isinstance(e, AnomalyEvent)]
    assert len(observations) == 2
    assert len(events) == 2
    assert context.detector._rows == 2  # noqa: SLF001 - contract test
    assert [o.frame_index for o in observations] == [1, 2]
    assert observations[0].observed_at < observations[1].observed_at


def test_decoded_height_drives_floor_and_change_resets_detector() -> None:
    # Camera record says 60 px; the live frames arrive at 60 then 540 px.
    provider = FakeProvider(heights=[60, 540])
    worker, shared, _provider = _worker(provider)
    _provider2, context = worker._initialise()
    assert context.rolling_config.frame_height == 60.0

    worker._sample_once(context)
    assert context.rolling_config.frame_height == 60.0

    worker._sample_once(context)
    assert context.rolling_config.frame_height == 540.0
    assert context.rolling_config.drawdown_floor_px == pytest.approx(75.0)
    assert context.detector.config.frame_height == 540.0

    events = [e for e in shared["added"] if isinstance(e, AnomalyEvent)]
    assert len(events) == 2
    assert events[-1].classification == "DATA_UNRELIABLE"
    assert "resolution_change" in (events[-1].reason or "")


def test_session_identity_is_generated_once_not_per_frame() -> None:
    worker, shared, _provider = _worker()
    _provider2, context = worker._initialise()
    worker._sample_once(context)
    worker._sample_once(context)
    observations = [o for o in shared["added"] if isinstance(o, ShorelineObservation)]
    sessions = [o for o in shared["added"] if isinstance(o, StreamSession)]
    assert len(sessions) == 1
    assert len({o.recording_id for o in observations}) == 1
    assert observations[0].stream_session_id == sessions[0].session_id


def test_worker_builds_one_segmenter_and_runs_one_inference_per_sample() -> None:
    shared: dict[str, Any] = {"added": [], "commits": 0, "sessions": {}}
    provider = FakeProvider()
    factory_calls: list[int] = []

    class CountingSegmenter(FakeSegmenter):
        def __init__(self) -> None:
            self.predictions = 0

        def predict(self, image):  # noqa: ANN001
            self.predictions += 1
            return super().predict(image)

    segmenter = CountingSegmenter()

    def segmenter_factory() -> CountingSegmenter:
        factory_calls.append(1)
        return segmenter

    worker = LiveCaptureWorker(
        7,
        session_factory=lambda: FakeSession(shared),
        provider_factory=lambda _camera: provider,
        segmenter_factory=segmenter_factory,
        sample_interval_seconds=30.0,
    )
    _provider, context = worker._initialise()
    assert len(factory_calls) == 1

    worker._sample_once(context)
    worker._sample_once(context)

    assert len(factory_calls) == 1
    assert segmenter.predictions == 2
    assert provider.reads == 2


def test_worker_publishes_preview_before_persistence_and_after_commit() -> None:
    worker, _shared, _provider = _worker()
    _provider2, context = worker._initialise()
    worker._sample_once(context)

    entry = get_preview_cache().get(7, max_age_seconds=3600.0)
    assert entry is not None
    assert entry.persisted is True
    assert entry.observation_id is not None
    assert entry.frame_index == 1
    assert entry.recording_id == context.status.recording_id


def test_worker_still_publishes_preview_when_persistence_rejects(monkeypatch) -> None:
    worker, _shared, _provider = _worker()
    _provider2, context = worker._initialise()
    monkeypatch.setattr(
        capture_module,
        "write_observation",
        lambda *_args, **_kwargs: SimpleNamespace(
            status="rejected", reason="quality_rejected", observation=None
        ),
    )

    worker._sample_once(context)

    entry = get_preview_cache().get(7, max_age_seconds=3600.0)
    assert entry is not None
    assert entry.persisted is False
    assert entry.observation_id is None
    assert worker.status.frames_rejected == 1


def test_worker_counts_skipped_slots_without_catching_up() -> None:
    worker, _shared, _provider = _worker()
    _provider2, context = worker._initialise()

    # A sample that overran by 0.2 s missed the due slot; 1.5 s missed two.
    assert LiveCaptureWorker._missed_slots(0.2, 1.0) == 1
    assert LiveCaptureWorker._missed_slots(1.5, 1.0) == 2
    assert LiveCaptureWorker._missed_slots(0.0, 1.0) == 0

    worker._add_skipped(2)
    assert worker.status.samples_skipped == 2
