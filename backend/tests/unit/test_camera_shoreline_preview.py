"""The live shoreline preview runs inference without persisting observations."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.api.routes import cameras
from app.core.exceptions import PreviewBusyError, ValidationError
from app.services.live.preview_cache import PreviewCacheEntry, get_preview_cache


@pytest.fixture(autouse=True)
def _clear_preview_cache() -> None:
    get_preview_cache().clear_all()
    yield
    get_preview_cache().clear_all()


def _class_mask(height: int = 32, width: int = 48) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[: height // 4] = 2  # Sky
    mask[height // 4 : height // 2] = 1  # Ocean
    # Beach below
    return mask


def test_shoreline_preview_returns_three_class_overlay_without_persisting(monkeypatch) -> None:
    camera = SimpleNamespace(id=7, code="CAM-7", enabled=True)
    frame_image = np.full((32, 48, 3), 100, dtype=np.uint8)
    frame = SimpleNamespace(
        image=frame_image,
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    class_mask = _class_mask()

    class FakeSegmenter:
        config = {
            "color_map": {
                "Beach": [214, 181, 137],
                "Ocean": [32, 112, 190],
                "Sky": [135, 206, 235],
            }
        }

        def predict(self, _image):
            return SimpleNamespace(
                mask=np.where(class_mask == 1, 255, 0).astype(np.uint8),
                class_mask=class_mask,
                classes={0: "Beach", 1: "Ocean", 2: "Sky"},
                confidence=0.91,
                inference_ms=4.5,
                architecture="SegFormer",
                encoder="MiT-B2",
                checkpoint_sha256="A" * 64,
            )

    encoded_images = []
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras.manager, "capture_frame", lambda _camera: (frame, None))
    monkeypatch.setattr(
        cameras.manager,
        "encode_jpeg",
        lambda image: encoded_images.append(image.copy()) or b"jpeg",
    )
    monkeypatch.setattr(cameras, "ShorelineSegmenter", FakeSegmenter)

    response = cameras.get_camera_shoreline_preview(7, mask=True, db=object())

    assert response.media_type == "image/jpeg"
    assert response.body == b"jpeg"
    assert response.headers["x-shoreline-polyline"]
    assert response.headers["x-segmentation-confidence"] == "0.91"
    assert response.headers["x-inference-ms"] == "4.5"
    assert response.headers["x-model-architecture"] == "SegFormer"
    assert response.headers["x-model-encoder"] == "MiT-B2"
    assert response.headers["x-shoreline-available"] == "1"
    assert int(response.headers["x-shoreline-segments"]) >= 1
    assert "Server-Timing" in response.headers["access-control-expose-headers"]
    phases = {
        part.split(";", 1)[0]: float(part.split("dur=", 1)[1])
        for part in response.headers["server-timing"].split(", ")
    }
    assert {
        "capture",
        "capture_age",
        "predict",
        "onnx",
        "extract",
        "overlay",
        "encode",
        "response_age",
        "total",
    } <= phases.keys()
    assert all(value >= 0 for value in phases.values())
    assert len(encoded_images) == 1
    # Class tint and the shoreline stroke are burned into the image sent to the browser.
    assert not np.array_equal(encoded_images[0], frame_image)
    assert tuple(encoded_images[0][5, 5]) != tuple(frame_image[5, 5])

    # The default (mask off) is the raw frame: the browser draws the polyline.
    response = cameras.get_camera_shoreline_preview(7, db=object())
    assert response.body == b"jpeg"
    assert len(encoded_images) == 2
    assert np.array_equal(encoded_images[1], frame_image)
    assert response.headers["x-shoreline-polyline"]
    assert "X-Shoreline-Polyline" in response.headers["access-control-expose-headers"]


def test_shoreline_preview_reports_no_shoreline_without_fabricating(monkeypatch) -> None:
    camera = SimpleNamespace(id=8, code="CAM-8", enabled=True)
    frame_image = np.full((32, 48, 3), 100, dtype=np.uint8)
    frame = SimpleNamespace(image=frame_image, captured_at=datetime(2026, 1, 1, tzinfo=UTC))
    # Sky-only / no Beach-Ocean contact.
    class_mask = np.full((32, 48), 2, dtype=np.uint8)
    encoded_images = []

    class FakeSegmenter:
        config: dict = {}

        def predict(self, _image):
            return SimpleNamespace(
                mask=np.zeros((32, 48), dtype=np.uint8),
                class_mask=class_mask,
                classes={0: "Beach", 1: "Ocean", 2: "Sky"},
                confidence=0.5,
                inference_ms=1.0,
                architecture="SegFormer",
                encoder="MiT-B2",
                checkpoint_sha256="B" * 64,
            )

    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras.manager, "capture_frame", lambda _camera: (frame, None))
    monkeypatch.setattr(
        cameras.manager,
        "encode_jpeg",
        lambda image: encoded_images.append(image.copy()) or b"jpeg",
    )
    monkeypatch.setattr(cameras, "ShorelineSegmenter", FakeSegmenter)

    response = cameras.get_camera_shoreline_preview(8, db=object())
    assert response.headers["x-shoreline-available"] == "0"
    assert int(response.headers["x-shoreline-segments"]) == 0
    assert response.headers["x-shoreline-reason"]
    # The class overlay is still real model output; no line is drawn.
    assert len(encoded_images) == 1


def test_shoreline_preview_applies_active_monitoring_outline(monkeypatch) -> None:
    """A false Ocean/Beach patch outside the outline cannot produce a shoreline."""
    camera = SimpleNamespace(id=9, code="CAM-9", enabled=True, active_calibration_id=3)
    frame_image = np.full((32, 48, 3), 100, dtype=np.uint8)
    frame = SimpleNamespace(image=frame_image, captured_at=datetime(2026, 1, 1, tzinfo=UTC))
    class_mask = _class_mask()
    # False distant island on the right third.
    class_mask[6:10, 34:46] = 1
    class_mask[10:12, 34:46] = 0
    outline = {
        "coordinate_space": "normalized",
        "points": [[0.0, 0.0], [0.6, 0.0], [0.6, 1.0], [0.0, 1.0]],
        "frame_width": 48,
        "frame_height": 32,
    }

    class FakeSegmenter:
        config: dict = {}

        def predict(self, _image):
            return SimpleNamespace(
                mask=np.where(class_mask == 1, 255, 0).astype(np.uint8),
                class_mask=class_mask,
                classes={0: "Beach", 1: "Ocean", 2: "Sky"},
                confidence=0.9,
                inference_ms=2.0,
                architecture="SegFormer",
                encoder="MiT-B2",
                checkpoint_sha256="C" * 64,
            )

    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras.manager, "capture_frame", lambda _camera: (frame, None))
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"jpeg")
    monkeypatch.setattr(cameras, "ShorelineSegmenter", FakeSegmenter)
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: outline)

    response = cameras.get_camera_shoreline_preview(9, db=object())

    assert response.headers["x-monitoring-outline"] == "1"
    assert response.headers["x-shoreline-available"] == "1"
    assert "X-Monitoring-Outline" in response.headers["access-control-expose-headers"]


def test_shoreline_preview_reports_no_outline_header(monkeypatch) -> None:
    camera = SimpleNamespace(id=10, code="CAM-10", enabled=True, active_calibration_id=None)
    frame = SimpleNamespace(
        image=np.full((32, 48, 3), 100, dtype=np.uint8),
        captured_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    class_mask = _class_mask()

    class FakeSegmenter:
        config: dict = {}

        def predict(self, _image):
            return SimpleNamespace(
                mask=np.zeros((32, 48), dtype=np.uint8),
                class_mask=class_mask,
                classes={0: "Beach", 1: "Ocean", 2: "Sky"},
                confidence=0.9,
                inference_ms=1.0,
                architecture="SegFormer",
                encoder="MiT-B2",
                checkpoint_sha256="D" * 64,
            )

    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras.manager, "capture_frame", lambda _camera: (frame, None))
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"jpeg")
    monkeypatch.setattr(cameras, "ShorelineSegmenter", FakeSegmenter)

    response = cameras.get_camera_shoreline_preview(10, db=object())
    assert response.headers["x-monitoring-outline"] == "0"


def test_shoreline_preview_rejects_disabled_camera(monkeypatch) -> None:
    camera = SimpleNamespace(id=7, code="CAM-7", enabled=False)
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)

    with pytest.raises(ValidationError, match="disabled"):
        cameras.get_camera_shoreline_preview(7, db=object())


def _cached_entry(camera_id: int = 7, **overrides) -> PreviewCacheEntry:
    frame_image = np.full((32, 48, 3), 100, dtype=np.uint8)
    class_mask = _class_mask()
    segmentation = SimpleNamespace(
        class_mask=class_mask,
        classes={0: "Beach", 1: "Ocean", 2: "Sky"},
        confidence=0.91,
        inference_ms=4.5,
        architecture="SegFormer",
        encoder="MiT-B2",
        checkpoint_sha256="A" * 64,
    )
    shoreline = SimpleNamespace(
        segments=[[[10.0, 16.0], [20.0, 16.0]]],
        points=[[10.0, 16.0], [20.0, 16.0]],
        shoreline_available=True,
        shoreline_axis="horizontal",
        interpolated_ranges=[],
        measured_coverage=0.5,
        reason=None,
    )
    values = {
        "camera_id": camera_id,
        "frame": SimpleNamespace(image=frame_image, captured_at=datetime(2026, 1, 1, tzinfo=UTC)),
        "segmentation": segmentation,
        "shoreline": shoreline,
        "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
        "observation_id": "rec-1:4",
        "recording_id": "rec-1",
        "frame_index": 4,
        "inference_ms": 4.5,
        "ingest_latency_ms": 10.0,
        "stored_monotonic": time.monotonic(),
        "monitoring_outline_present": False,
    }
    values.update(overrides)
    return PreviewCacheEntry(**values)


def test_preview_serves_fresh_worker_cache_without_capture(monkeypatch) -> None:
    camera = SimpleNamespace(id=7, code="CAM-7", enabled=True)
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: None)
    monkeypatch.setattr(
        cameras.manager,
        "capture_frame",
        lambda _camera: (_ for _ in ()).throw(AssertionError("cache must avoid capture")),
    )
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"cached-jpeg")
    get_preview_cache().put(_cached_entry(7))

    response = cameras.get_camera_shoreline_preview(7, mask=False, db=object())

    assert response.body == b"cached-jpeg"
    assert response.headers["x-preview-source"] == "cache"
    assert response.headers["x-preview-observation-id"] == "rec-1:4"
    assert float(response.headers["x-preview-age-ms"]) >= 0
    assert "X-Preview-Source" in response.headers["access-control-expose-headers"]


def test_preview_ignores_stale_cache(monkeypatch) -> None:
    camera = SimpleNamespace(id=7, code="CAM-7", enabled=True)
    frame_image = np.full((32, 48, 3), 100, dtype=np.uint8)
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: None)
    monkeypatch.setattr(
        cameras.manager,
        "capture_frame",
        lambda _camera: (
            SimpleNamespace(image=frame_image, captured_at=datetime.now(UTC)),
            None,
        ),
    )
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"on-demand")
    monkeypatch.setattr(cameras, "ShorelineSegmenter", _CachedFakeSegmenter)
    get_preview_cache().put(_cached_entry(7, stored_monotonic=time.monotonic() - 3600.0))

    response = cameras.get_camera_shoreline_preview(7, mask=False, db=object())

    assert response.body == b"on-demand"
    assert response.headers["x-preview-source"] == "on-demand"


def test_preview_single_flight_returns_429_when_no_cached_result(monkeypatch) -> None:
    camera = SimpleNamespace(id=7, code="CAM-7", enabled=True)
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: None)

    assert cameras._preview_inference_slot.acquire(blocking=False)  # noqa: SLF001
    try:
        with pytest.raises(PreviewBusyError) as raised:
            cameras.get_camera_shoreline_preview(7, mask=False, db=object())
    finally:
        cameras._preview_inference_slot.release()  # noqa: SLF001
    assert raised.value.status_code == 429
    assert raised.value.headers["Retry-After"] == "1"


def test_preview_concurrent_requests_never_run_two_inferences(monkeypatch) -> None:
    camera = SimpleNamespace(id=7, code="CAM-7", enabled=True)
    frame_image = np.full((32, 48, 3), 100, dtype=np.uint8)
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: None)
    monkeypatch.setattr(
        cameras.manager,
        "capture_frame",
        lambda _camera: (
            SimpleNamespace(image=frame_image, captured_at=datetime.now(UTC)),
            None,
        ),
    )
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"jpeg")
    monkeypatch.setattr(cameras, "ShorelineSegmenter", _SlowFakeSegmenter)
    _SlowFakeSegmenter.active = 0
    _SlowFakeSegmenter.max_active = 0

    def one_request(_index: int) -> str:
        try:
            response = cameras.get_camera_shoreline_preview(7, mask=False, db=object())
            return response.headers["x-preview-source"]
        except PreviewBusyError:
            return "busy"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(one_request, range(4)))

    assert _SlowFakeSegmenter.max_active == 1
    assert results.count("on-demand") == 1
    assert results.count("busy") == 3


class _CachedFakeSegmenter:
    config: dict = {}

    def predict(self, _image):
        return SimpleNamespace(
            class_mask=_class_mask(),
            classes={0: "Beach", 1: "Ocean", 2: "Sky"},
            confidence=0.9,
            inference_ms=1.0,
            architecture="SegFormer",
            encoder="MiT-B2",
            checkpoint_sha256="B" * 64,
        )


class _SlowFakeSegmenter:
    config: dict = {}
    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self) -> None:
        pass

    def predict(self, _image):
        with _SlowFakeSegmenter.lock:
            _SlowFakeSegmenter.active += 1
            _SlowFakeSegmenter.max_active = max(
                _SlowFakeSegmenter.max_active, _SlowFakeSegmenter.active
            )
        try:
            time.sleep(0.3)
            return SimpleNamespace(
                class_mask=_class_mask(),
                classes={0: "Beach", 1: "Ocean", 2: "Sky"},
                confidence=0.9,
                inference_ms=100.0,
                architecture="SegFormer",
                encoder="MiT-B2",
                checkpoint_sha256="C" * 64,
            )
        finally:
            with _SlowFakeSegmenter.lock:
                _SlowFakeSegmenter.active -= 1
