"""The live shoreline preview is cache-only; the worker is the sole producer."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np
import pytest

from app.api.routes import cameras
from app.core.exceptions import PreviewNotReadyError, ValidationError
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


def _camera(camera_id: int = 7, *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(id=camera_id, code=f"CAM-{camera_id}", enabled=enabled)


def _segmentation() -> SimpleNamespace:
    return SimpleNamespace(
        class_mask=_class_mask(),
        classes={0: "Beach", 1: "Ocean", 2: "Sky"},
        confidence=0.91,
        inference_ms=4.5,
        architecture="SegFormer",
        encoder="MiT-B2",
        checkpoint_sha256="A" * 64,
    )


def _shoreline() -> SimpleNamespace:
    return SimpleNamespace(
        segments=[[[10.0, 16.0], [20.0, 16.0]]],
        points=[[10.0, 16.0], [20.0, 16.0]],
        shoreline_available=True,
        shoreline_axis="horizontal",
        interpolated_ranges=[],
        measured_coverage=0.5,
        reason=None,
    )


def _cached_entry(camera_id: int = 7, **overrides) -> PreviewCacheEntry:
    values = {
        "camera_id": camera_id,
        "frame": SimpleNamespace(
            image=np.full((32, 48, 3), 100, dtype=np.uint8),
            captured_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
        "segmentation": _segmentation(),
        "shoreline": _shoreline(),
        "captured_at": datetime(2026, 1, 1, tzinfo=UTC),
        "observation_id": "rec-1:4",
        "recording_id": "rec-1",
        "frame_index": 4,
        "inference_ms": 4.5,
        "ingest_latency_ms": 10.0,
        "stored_monotonic": time.monotonic(),
        "monitoring_outline_present": False,
        "persisted": True,
    }
    values.update(overrides)
    return PreviewCacheEntry(**values)


def _patch_common(monkeypatch, camera, *, running: bool = True) -> None:
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: None)
    monkeypatch.setattr(
        cameras,
        "get_live_manager",
        lambda: SimpleNamespace(running_camera_ids=lambda: [camera.id] if running else []),
    )
    monkeypatch.setattr(
        cameras.manager,
        "capture_frame",
        lambda _camera: (_ for _ in ()).throw(AssertionError("preview must never capture")),
    )


def test_preview_renders_worker_cache_without_capture(monkeypatch) -> None:
    camera = _camera(7)
    _patch_common(monkeypatch, camera)
    encoded_images = []
    monkeypatch.setattr(
        cameras.manager,
        "encode_jpeg",
        lambda image: encoded_images.append(image.copy()) or b"jpeg",
    )
    entry = _cached_entry(7)
    get_preview_cache().put(entry)

    response = cameras.get_camera_shoreline_preview(7, mask=True, db=object())

    assert response.media_type == "image/jpeg"
    assert response.body == b"jpeg"
    assert response.headers["x-preview-source"] == "worker"
    assert response.headers["x-preview-observation-id"] == "rec-1:4"
    assert response.headers["x-preview-persisted"] == "1"
    assert response.headers["x-preview-stale"] == "0"
    assert float(response.headers["x-preview-age-ms"]) >= 0
    assert response.headers["x-preview-sample-interval-ms"]
    assert response.headers["x-shoreline-polyline"]
    assert response.headers["x-segmentation-confidence"] == "0.91"
    assert response.headers["x-inference-ms"] == "4.5"
    assert response.headers["x-model-architecture"] == "SegFormer"
    assert response.headers["x-model-encoder"] == "MiT-B2"
    assert response.headers["x-shoreline-available"] == "1"
    assert int(response.headers["x-shoreline-segments"]) >= 1
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
    assert phases["predict"] == 0.0
    assert phases["capture"] >= 0.0
    assert len(encoded_images) == 1
    # Class tint and the shoreline stroke are burned into the image.
    assert not np.array_equal(encoded_images[0], entry.frame.image)
    assert "X-Preview-Persisted" in response.headers["access-control-expose-headers"]
    assert "X-Preview-Stale" in response.headers["access-control-expose-headers"]

    # The default (mask off) is the raw frame: the browser draws the polyline.
    response = cameras.get_camera_shoreline_preview(7, db=object())
    assert response.body == b"jpeg"
    assert len(encoded_images) == 2
    assert np.array_equal(encoded_images[1], entry.frame.image)


def test_preview_cache_miss_returns_503_without_capture(monkeypatch) -> None:
    camera = _camera(7)
    _patch_common(monkeypatch, camera, running=True)

    with pytest.raises(PreviewNotReadyError) as raised:
        cameras.get_camera_shoreline_preview(7, mask=False, db=object())

    assert raised.value.status_code == 503
    assert raised.value.code == "preview_not_ready"
    assert raised.value.headers["Retry-After"]
    assert raised.value.details["camera_id"] == 7


def test_preview_serves_stale_worker_cache_when_worker_stopped(monkeypatch) -> None:
    camera = _camera(7)
    _patch_common(monkeypatch, camera, running=False)
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"jpeg")
    get_preview_cache().put(_cached_entry(7, stored_monotonic=time.monotonic() - 3600.0))

    response = cameras.get_camera_shoreline_preview(7, mask=False, db=object())

    assert response.body == b"jpeg"
    assert response.headers["x-preview-source"] == "worker"
    assert response.headers["x-preview-stale"] == "1"


def test_preview_does_not_serve_stale_cache_while_worker_running(monkeypatch) -> None:
    camera = _camera(7)
    _patch_common(monkeypatch, camera, running=True)
    get_preview_cache().put(_cached_entry(7, stored_monotonic=time.monotonic() - 3600.0))

    with pytest.raises(PreviewNotReadyError):
        cameras.get_camera_shoreline_preview(7, mask=False, db=object())


def test_preview_marks_outline_mismatch_without_dropping_cache(monkeypatch) -> None:
    camera = _camera(9)
    _patch_common(monkeypatch, camera)
    outline = {
        "coordinate_space": "normalized",
        "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        "frame_width": 48,
        "frame_height": 32,
    }
    monkeypatch.setattr(cameras, "load_active_roi", lambda _db, _camera: outline)
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"jpeg")
    # Worker produced this entry without a monitoring outline.
    get_preview_cache().put(_cached_entry(9, monitoring_outline_present=False))

    response = cameras.get_camera_shoreline_preview(9, mask=False, db=object())

    assert response.headers["x-monitoring-outline"] == "1"
    assert response.headers["x-monitoring-outline-matches"] == "0"
    assert response.headers["x-preview-source"] == "worker"


def test_preview_reports_when_frame_was_not_persisted(monkeypatch) -> None:
    camera = _camera(7)
    _patch_common(monkeypatch, camera)
    monkeypatch.setattr(cameras.manager, "encode_jpeg", lambda _image: b"jpeg")
    get_preview_cache().put(
        _cached_entry(7, observation_id=None, persisted=False, frame_index=None)
    )

    response = cameras.get_camera_shoreline_preview(7, mask=False, db=object())

    assert response.headers["x-preview-persisted"] == "0"
    assert "x-preview-observation-id" not in response.headers


def test_shoreline_preview_rejects_disabled_camera(monkeypatch) -> None:
    camera = _camera(7, enabled=False)
    monkeypatch.setattr(cameras, "_get_camera", lambda _db, _camera_id: camera)

    with pytest.raises(ValidationError, match="disabled"):
        cameras.get_camera_shoreline_preview(7, db=object())
