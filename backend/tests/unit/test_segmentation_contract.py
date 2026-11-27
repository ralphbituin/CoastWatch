"""Segmentation contract tests (no database).

Ten checks that pin the seam between the active three-class SegFormer artifact
and CoastWatch:

1. no artifact  -> ``find_model_artifact`` is ``None``
2. no artifact  -> ``load()`` raises ``ModelUnavailableError`` with guidance
3. artifact without ``config.json`` -> loud, specific failure
4. the ONNX session is loaded once and shared across instances
5. BGR frames are converted to RGB (channel order is not guessed)
6. the returned ``mask`` is an Ocean-only binary water mask, not the class map
7. class-aware extraction consumes the class mask and returns the Ocean/Beach
   boundary, not the Sky/Ocean horizon
8. class statistics are consistent and confidence stays a probability
9. a class-count / class-order mismatch between config and the contract fails
10. ``model_status`` never raises and reports config validity + model identity
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.core.exceptions import ModelUnavailableError
from app.services.vision import segmentation
from app.services.vision.shoreline import extract_shoreline_from_classes
from tests.onnx_fixture import frame_bgr, write_artifact


def _use_model_path(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    settings = type("S", (), {"model_path": str(path)})()
    monkeypatch.setattr(segmentation, "get_settings", lambda: settings)
    segmentation._load_session.cache_clear()


@pytest.fixture(autouse=True)
def _clear_session_cache():
    segmentation._load_session.cache_clear()
    yield
    segmentation._load_session.cache_clear()


def test_1_missing_artifact_is_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_model_path(monkeypatch, tmp_path)
    assert segmentation.find_model_artifact() is None


def test_2_missing_artifact_raises_with_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_model_path(monkeypatch, tmp_path)
    with pytest.raises(ModelUnavailableError) as error:
        segmentation.ShorelineSegmenter().load()
    message = str(error.value)
    assert "shoreline-segmentation/model.onnx" in message
    assert "observations" in message


def test_3_missing_config_is_specific(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = write_artifact(tmp_path / "shoreline-segmentation")
    (artifact / "config.json").unlink()
    _use_model_path(monkeypatch, tmp_path)
    with pytest.raises(ModelUnavailableError, match="no config.json"):
        segmentation.ShorelineSegmenter().load()


def test_4_session_is_loaded_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_artifact(tmp_path / "shoreline-segmentation")
    _use_model_path(monkeypatch, tmp_path)
    first = segmentation.ShorelineSegmenter()
    second = segmentation.ShorelineSegmenter()
    first.load()
    second.load()
    assert first._session is second._session
    assert segmentation._load_session.cache_info().hits == 1


def test_5_bgr_is_converted_to_rgb() -> None:
    bgr = np.array([[[10, 20, 30]]], dtype=np.uint8)
    rgb = segmentation.ShorelineSegmenter._to_rgb(bgr)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_6_mask_is_ocean_only_binary_mask(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_artifact(tmp_path / "shoreline-segmentation")
    _use_model_path(monkeypatch, tmp_path)
    result = segmentation.ShorelineSegmenter().predict(frame_bgr(120, 80))

    assert result.mask.shape == (80, 120)
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.mask).tolist()) == {0, 255}
    # Fixture: Sky 0-20, Ocean 20-40, Beach 40-80. Only Ocean is water.
    assert result.mask[:19].max() == 0  # Sky is not water
    assert result.mask[22:38].min() == 255  # Ocean is water
    assert result.mask[42:].max() == 0  # Beach is not water

    assert result.class_mask is not None
    assert set(np.unique(result.class_mask).tolist()) == {0, 1, 2}
    assert result.class_mask.max() == 2  # never 255


def test_7_class_aware_extractor_finds_ocean_beach_not_sky_ocean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_artifact(tmp_path / "shoreline-segmentation")
    _use_model_path(monkeypatch, tmp_path)
    result = segmentation.ShorelineSegmenter().predict(frame_bgr(120, 80))

    shoreline = extract_shoreline_from_classes(
        result.class_mask, min_ocean_component_px=10, min_beach_component_px=10
    )
    assert shoreline.shoreline_available is True
    ys = np.array([point[1] for point in shoreline.points])
    # Ocean/Beach boundary is at row ~40; the Sky/Ocean horizon is at ~20.
    assert 36 <= float(ys.mean()) <= 44
    assert float(ys.mean()) > 30
    assert len(shoreline.segments) >= 1
    assert shoreline.ocean_pixels and shoreline.ocean_pixels > 0
    assert shoreline.beach_pixels and shoreline.beach_pixels > 0


def test_8_statistics_and_confidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_artifact(tmp_path / "shoreline-segmentation")
    _use_model_path(monkeypatch, tmp_path)
    result = segmentation.ShorelineSegmenter().predict(
        frame_bgr(120, 80), include_probabilities=True
    )

    fractions = [stats["fraction"] for stats in result.class_statistics.values()]
    assert sum(fractions) == pytest.approx(1.0, abs=1e-5)
    assert result.class_statistics["Sky"]["pixels"] > 0
    assert result.class_statistics["Ocean"]["pixels"] > 0
    assert result.class_statistics["Beach"]["pixels"] > 0
    assert 0.0 < result.confidence <= 1.0
    assert result.inference_ms is not None and result.inference_ms >= 0.0
    assert result.classes == {0: "Beach", 1: "Ocean", 2: "Sky"}
    assert result.architecture == "SegFormer"
    assert result.encoder == "MiT-B2"
    assert result.checkpoint_sha256 == "0" * 64
    assert result.probabilities is not None
    assert result.probabilities.shape == (80, 120, 3)
    assert np.allclose(result.probabilities.sum(axis=2), 1.0, atol=1e-5)


def test_9_class_contract_mismatch_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_artifact(
        tmp_path / "shoreline-segmentation",
        num_classes=5,
        class_names=["unknown", "land", "water", "foam", "other"],
    )
    _use_model_path(monkeypatch, tmp_path)
    with pytest.raises(ModelUnavailableError, match="num_classes=3"):
        segmentation.ShorelineSegmenter().load()


def test_10_model_status_is_non_raising(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_model_path(monkeypatch, tmp_path)
    missing = segmentation.model_status()
    assert missing == {"available": False, "reason": "artifact_missing", "config_valid": False}

    write_artifact(tmp_path / "shoreline-segmentation")
    present = segmentation.model_status()
    assert present["available"] is True
    assert present["config_valid"] is True
    assert present["class_names"] == ["Beach", "Ocean", "Sky"]
    assert present["class_ids"] == {"Beach": 0, "Ocean": 1, "Sky": 2}
    assert present["water_classes"] == [1]
    assert present["architecture"] == "SegFormer"
    assert present["encoder"] == "MiT-B2"

    (tmp_path / "shoreline-segmentation" / "config.json").unlink()
    invalid = segmentation.model_status()
    assert invalid["available"] is True
    assert invalid["config_valid"] is False
