"""Phase 7 recording-service contract tests (no database)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from app.models.anomaly import AnomalyEvent
from app.models.shoreline import ShorelineObservation
from app.services.anomaly.rolling_baseline import (
    RollingBaselineConfig,
    RollingBaselineDetector,
    RollingObservation,
)
from app.services.recording.service import (
    STATUS_DUPLICATE,
    STATUS_OUT_OF_ORDER,
    STATUS_PERSISTED,
    STATUS_UPDATED,
    observation_to_rolling,
    persist_rolling_row,
    write_observation,
)

T0 = datetime(2026, 11, 25, 6, 0, tzinfo=UTC)
CAMERA = SimpleNamespace(id=7, monitoring_station_id=3)


class FakeSession:
    """Minimal scalar/add surface used by the writer."""

    def __init__(self, results: list[object | None] | None = None) -> None:
        self.results = list(results or [])
        self.added: list[object] = []

    def scalar(self, _stmt):  # noqa: ANN001 - SQLAlchemy statement
        return self.results.pop(0) if self.results else None

    def add(self, obj: object) -> None:
        self.added.append(obj)


def _write(session: FakeSession, *, frame_index: int = 1, observed_at: datetime = T0):
    return write_observation(
        session,
        CAMERA,
        observed_at=observed_at,
        frame_index=frame_index,
        recording_id="rec-1",
        elapsed_seconds=0.0,
        timestamp_source="derived",
        timestamp_confidence=0.9,
        extraction_status="available",
        signed_position_px=100.0,
        shoreline_position_px=100.0,
        segment_count=1,
    )


def test_new_measurement_persists_exactly_one_row() -> None:
    session = FakeSession([None, None, None])
    outcome = _write(session)
    assert outcome.status == STATUS_PERSISTED
    assert len(session.added) == 1
    row = session.added[0]
    assert isinstance(row, ShorelineObservation)
    assert row.observed_at == T0
    assert row.recording_id == "rec-1"
    assert row.stream_session_id is None


def test_exact_duplicate_timestamp_is_dropped_and_counted() -> None:
    existing = ShorelineObservation(
        camera_id=CAMERA.id,
        observed_at=T0,
        monitoring_station_id=CAMERA.monitoring_station_id,
        recording_id="rec-1",
        frame_index=1,
    )
    session = FakeSession([None, existing])
    outcome = _write(session, frame_index=2)
    assert outcome.status == STATUS_DUPLICATE
    assert outcome.reason == "duplicate_timestamp"
    assert session.added == []


def test_reprocessing_same_frame_updates_in_place() -> None:
    existing = ShorelineObservation(
        camera_id=CAMERA.id,
        observed_at=T0,
        monitoring_station_id=CAMERA.monitoring_station_id,
        recording_id="rec-1",
        frame_index=1,
    )
    session = FakeSession([existing])
    outcome = _write(session, frame_index=1, observed_at=T0 + timedelta(seconds=1))
    assert outcome.status == STATUS_UPDATED
    assert outcome.observation is existing
    assert session.added == []
    assert existing.signed_position_px == 100.0
    assert existing.capture_timestamp_source == "derived"


def test_out_of_order_timestamp_is_quarantined_not_reordered() -> None:
    session = FakeSession([None, None, T0])
    outcome = _write(session, observed_at=T0)
    assert outcome.status == STATUS_OUT_OF_ORDER
    assert session.added == []
    assert "not after" in (outcome.reason or "")


def test_observation_to_rolling_uses_signed_seaward_position() -> None:
    row = ShorelineObservation(
        camera_id=CAMERA.id,
        observed_at=T0,
        monitoring_station_id=CAMERA.monitoring_station_id,
        signed_position_px=123.0,
        extraction_status="available",
        capture_timestamp_confidence=0.9,
        segmentation_confidence=0.8,
        frame_index=4,
        horizon_y_px=50.0,
    )
    rolling = observation_to_rolling(row)
    assert isinstance(rolling, RollingObservation)
    assert rolling.position_px == 123.0
    assert rolling.timestamp == T0
    assert rolling.timestamp_confidence == 0.9
    assert rolling.shoreline_available is True


def test_rolling_row_persists_rolling_fields_and_evidence() -> None:
    config = RollingBaselineConfig(
        window_seconds=240.0,
        k_seconds=60.0,
        frame_height=1080.0,
        nominal_interval_seconds=30.0,
    )
    detector = RollingBaselineDetector(config)
    result = None
    for index in range(4):
        result = detector.update(
            RollingObservation(
                timestamp=T0 + timedelta(seconds=30 * index),
                position_px=1000.0 + index,
                frame_index=index,
                confidence=0.95,
                timestamp_confidence=0.9,
            )
        )
    assert result is not None
    session = FakeSession([None])
    event = persist_rolling_row(
        session, CAMERA, result, recording_id="rec-1", frame_index=result.frame_index
    )
    assert isinstance(event, AnomalyEvent)
    assert event.detector_mode == "rolling_baseline"
    assert event.baseline_status == "rolling_unverified"
    assert event.tide_adjusted is False
    assert event.classification == result.classification
    factors = event.contributing_factors or {}
    assert "rolling_baseline_px" in factors
    assert factors["units"].startswith("pixels")
    assert session.added == [event]
