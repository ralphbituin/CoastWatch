"""Integration test: migrations apply cleanly and create hypertables.

Skipped unless ``COASTWATCH_INTEGRATION=1``.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text

from app.database.session import get_engine, get_session_factory

pytestmark = pytest.mark.skipif(
    os.getenv("COASTWATCH_INTEGRATION") != "1",
    reason="set COASTWATCH_INTEGRATION=1 (and run a DB) to enable",
)


def test_hypertables_exist() -> None:
    session = get_session_factory()()
    try:
        rows = (
            session.execute(
                text(
                    "SELECT hypertable_name FROM timescaledb_information.hypertables "
                    "ORDER BY hypertable_name"
                )
            )
            .scalars()
            .all()
        )
    finally:
        session.close()
    expected = {
        "tide_predictions",
        "shoreline_observations",
        "system_health_logs",
        "anomaly_events",
        "environmental_observations",
    }
    assert expected.issubset(set(rows))


def test_engine_connects() -> None:
    with get_engine().connect() as conn:
        assert conn.execute(text("SELECT 1")).scalar_one() == 1
