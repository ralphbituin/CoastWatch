"""In-memory lab run job store (Phase 7).

Runs are bounded, process-local jobs. Rows and small CSV/JSON artifacts live on
disk under ``DATA_DIR/lab_runs/<run_id>``; image/video blobs are never stored in
PostgreSQL. A run is deleted explicitly or after ``lab_run_ttl_seconds``.

This store is intentionally not a second production writer: persistence is off
by default and, when enabled, goes through the same recording service.
"""

from __future__ import annotations

import csv
import json
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"
ACTIVE_STATUSES = {STATUS_PENDING, STATUS_RUNNING}


@dataclass
class LabRun:
    run_id: str
    kind: str
    status: str = STATUS_PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    timezone: str = "Asia/Bangkok"
    source: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    temp_dir: Path | None = None
    persist: bool = False
    camera_id: int | None = None
    recording_id: str | None = None
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def update(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)
            self.updated_at = datetime.now(UTC)

    def set_progress(self, **values: Any) -> None:
        with self.lock:
            self.progress.update(values)
            self.updated_at = datetime.now(UTC)

    def append_row(self, row: dict[str, Any]) -> None:
        with self.lock:
            self.rows.append(row)

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "run_id": self.run_id,
                "kind": self.kind,
                "status": self.status,
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "timezone": self.timezone,
                "source": dict(self.source),
                "progress": dict(self.progress),
                "row_count": len(self.rows),
                "event_count": len(self.events),
                "errors": list(self.errors),
                "warnings": list(self.warnings),
                "artifacts": sorted(self.artifacts),
                "persist": self.persist,
                "camera_id": self.camera_id,
                "recording_id": self.recording_id,
            }


class LabRegistry:
    def __init__(self) -> None:
        self._runs: dict[str, LabRun] = {}
        self._lock = threading.RLock()
        self._executor: ThreadPoolExecutor | None = None
        self._executor_size = 0

    def _root(self) -> Path:
        root = Path(get_settings().data_dir) / "lab_runs"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _executor_for(self, size: int) -> ThreadPoolExecutor:
        with self._lock:
            if self._executor is None or self._executor_size != size:
                if self._executor is not None:
                    self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = ThreadPoolExecutor(
                    max_workers=size, thread_name_prefix="coastwatch-lab"
                )
                self._executor_size = size
            return self._executor

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for run in self._runs.values() if run.status in ACTIVE_STATUSES)

    def create(self, kind: str, *, timezone: str, persist: bool, camera_id: int | None) -> LabRun:
        settings = get_settings()
        if self.active_count() >= int(settings.lab_max_concurrent_runs):
            raise ValueError(
                f"too many active lab runs (max {settings.lab_max_concurrent_runs})"
            )
        self.cleanup_expired()
        run_id = f"lab-{uuid.uuid4().hex[:16]}"
        temp_dir = self._root() / run_id
        temp_dir.mkdir(parents=True, exist_ok=True)
        run = LabRun(
            run_id=run_id,
            kind=kind,
            timezone=timezone,
            persist=persist,
            camera_id=camera_id,
            temp_dir=temp_dir,
            source={"kind": kind},
        )
        with self._lock:
            self._runs[run_id] = run
        return run

    def get(self, run_id: str) -> LabRun | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self) -> list[LabRun]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda run: run.created_at, reverse=True)

    def submit(self, run: LabRun, target: Callable[[LabRun], None]) -> None:
        executor = self._executor_for(int(get_settings().lab_max_concurrent_runs))
        executor.submit(target, run)

    def delete(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.pop(run_id, None)
        if run is None:
            return False
        run.cancel_event.set()
        if run.temp_dir is not None:
            shutil.rmtree(run.temp_dir, ignore_errors=True)
        return True

    def cleanup_expired(self) -> int:
        ttl = float(get_settings().lab_run_ttl_seconds)
        now = time.time()
        expired: list[str] = []
        with self._lock:
            for run_id, run in list(self._runs.items()):
                if run.status in ACTIVE_STATUSES:
                    continue
                if now - run.updated_at.timestamp() > ttl:
                    expired.append(run_id)
        for run_id in expired:
            self.delete(run_id)
        return len(expired)

    def shutdown(self) -> None:
        with self._lock:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
            run_ids = list(self._runs)
        for run_id in run_ids:
            self.delete(run_id)


def write_artifacts(run: LabRun, *, report: dict[str, Any]) -> None:
    """Write bounded CSV/JSON artifacts (never raw image/video blobs)."""
    if run.temp_dir is None:
        return
    run.temp_dir.mkdir(parents=True, exist_ok=True)
    rows_path = run.temp_dir / "rows.csv"
    fieldnames = [
        "observed_at",
        "observed_at_local",
        "elapsed_seconds",
        "frame_index",
        "timestamp_source",
        "timestamp_confidence",
        "shoreline_position_px",
        "signed_position_px",
        "extraction_status",
        "extraction_reason",
        "segmentation_confidence",
        "rolling_baseline_px",
        "rolling_scale_px",
        "residual_px",
        "robust_z",
        "cumulative_drawdown_px",
        "classification",
        "quality_status",
        "event_max_classification",
    ]
    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in run.rows:
            writer.writerow(row)
    report_path = run.temp_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with run.lock:
        run.artifacts["rows.csv"] = str(rows_path)
        run.artifacts["report.json"] = str(report_path)


_registry: LabRegistry | None = None
_registry_lock = threading.Lock()


def get_lab_registry() -> LabRegistry:
    global _registry
    with _registry_lock:
        if _registry is None:
            _registry = LabRegistry()
        return _registry
