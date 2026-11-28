"""DEMO MODE: offline seaward-drawdown injection on recorded runs.

This package is feature-gated by ``DEMO_INJECTION_ENABLED``. It only reads
recorded CSV tables and replays them through the production rolling-baseline
detector. It never imports the live worker, database models or camera code and
never writes to a database or a recording directory.
"""

from __future__ import annotations

SIM_PREFIX = "SIM-"
BANNER_TEXT = "SIMULATION - not real data"

__all__ = ["BANNER_TEXT", "SIM_PREFIX"]
