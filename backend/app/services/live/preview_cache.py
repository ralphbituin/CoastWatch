"""In-process cache of the live worker's most recent scored frame.

The live worker and the preview endpoint run in the same process. Serving the
worker's already-computed frame/mask/shoreline avoids a second ONNX run and a
second shoreline extraction for every dashboard poll. The cache is deliberately
small: one entry per camera, replaced after each persisted observation and
cleared when the worker stops.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class PreviewCacheEntry:
    camera_id: int
    frame: Any
    segmentation: Any
    shoreline: Any
    captured_at: datetime
    # ShorelineObservation has a composite (camera_id, observed_at) key, so the
    # stable reference is the recording/session plus frame index.
    observation_id: str | None
    recording_id: str | None
    frame_index: int | None
    inference_ms: float | None
    ingest_latency_ms: float | None
    stored_monotonic: float
    monitoring_outline_present: bool = False
    # False when the worker sampled/segmented the frame but persistence rejected
    # it. The preview may show it; it must not imply the observation was stored.
    persisted: bool = True

    def age_seconds(self, now: float | None = None) -> float:
        reference = time.monotonic() if now is None else now
        return max(0.0, reference - self.stored_monotonic)

    def age_ms(self, now: float | None = None) -> float:
        return self.age_seconds(now) * 1000.0

    @property
    def captured_age_ms(self) -> float:
        return max(0.0, (datetime.now(UTC) - self.captured_at).total_seconds() * 1000.0)


class PreviewCache:
    """Thread-safe one-entry-per-camera cache for live previews."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[int, PreviewCacheEntry] = {}

    def put(self, entry: PreviewCacheEntry) -> None:
        with self._lock:
            self._entries[entry.camera_id] = entry

    def get(self, camera_id: int, *, max_age_seconds: float) -> PreviewCacheEntry | None:
        with self._lock:
            entry = self._entries.get(camera_id)
            if entry is None:
                return None
            now = time.monotonic()
            if now - entry.stored_monotonic > max(0.0, float(max_age_seconds)):
                return None
            return entry

    def clear(self, camera_id: int) -> None:
        with self._lock:
            self._entries.pop(camera_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._entries.clear()


_cache = PreviewCache()


def get_preview_cache() -> PreviewCache:
    return _cache
