"""Predictor + anomaly test lab (Phase 7)."""

from app.services.lab.registry import (
    LabRegistry,
    LabRun,
    get_lab_registry,
    write_artifacts,
)

__all__ = ["LabRegistry", "LabRun", "get_lab_registry", "write_artifacts"]
