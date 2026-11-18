"""SQLAlchemy ORM models.

Importing this package registers every table on ``Base.metadata``. TimescaleDB
hypertables (``tide_predictions``, ``shoreline_observations``,
``system_health_logs``, ``anomaly_events``, ``environmental_observations``) use
composite primary keys that include their time column, which TimescaleDB
requires for partitioning.
"""

from app.models.anomaly import AnomalyEvent

# Re-export geometry helper for convenience in later phases.
from app.models.camera import Camera, CameraCalibration
from app.models.country import Country
from app.models.environmental import EnvironmentalObservation
from app.models.shoreline import ShorelineBaseline, ShorelineObservation
from app.models.station import MonitoringStation
from app.models.system import SystemHealthLog
from app.models.tide import (
    DatasetImport,
    TideDataset,
    TidePrediction,
    TideStation,
)

__all__ = [
    "AnomalyEvent",
    "Camera",
    "CameraCalibration",
    "Country",
    "DatasetImport",
    "EnvironmentalObservation",
    "MonitoringStation",
    "ShorelineBaseline",
    "ShorelineObservation",
    "SystemHealthLog",
    "TideDataset",
    "TidePrediction",
    "TideStation",
]
