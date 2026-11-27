"""Integration: rolling event idempotency key.

Skipped unless ``COASTWATCH_INTEGRATION=1`` and a migrated PostgreSQL exists.
"""

from __future__ import annotations

import os
import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.session import get_session_factory
from app.models.anomaly import AnomalyEvent
from app.models.country import Country
from app.models.station import MonitoringStation

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)


def test_rolling_event_partial_unique_key_is_idempotent() -> None:
    session = get_session_factory()()
    suffix = uuid.uuid4().hex[:8]
    station = None
    try:
        country = session.scalar(select(Country).limit(1))
        assert country is not None, "a country row is required"
        station = MonitoringStation(
            country_id=country.id,
            code=f"TEST-ROLL-{suffix}",
            name=f"Rolling test station {suffix}",
            latitude=7.8433,
            longitude=98.2939,
            timezone="Asia/Bangkok",
        )
        session.add(station)
        session.flush()
        now = datetime.now(UTC)
        for _ in range(2):
            session.add(
                AnomalyEvent(
                    detected_at=now,
                    monitoring_station_id=station.id,
                    state="ANOMALY_DETECTED",
                    classification="POSSIBLE_DRAWDOWN",
                    baseline_status="rolling_unverified",
                    recording_id="rec-rolling-test",
                    detector_mode="rolling_baseline",
                    onset_frame_index=42,
                    frame_index=42,
                    tide_adjusted=False,
                    tide_data_available=False,
                    tide_source="unavailable",
                    reason="synthetic integration key",
                )
            )
        with pytest.raises(IntegrityError):
            session.flush()
        session.rollback()
    finally:
        if station is not None:
            session.execute(
                AnomalyEvent.__table__.delete().where(
                    AnomalyEvent.recording_id == "rec-rolling-test"
                )
            )
            session.execute(
                MonitoringStation.__table__.delete().where(MonitoringStation.code == station.code)
            )
            session.commit()
        session.close()
