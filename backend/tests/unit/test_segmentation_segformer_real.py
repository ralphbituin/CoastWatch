"""Integration-in-depth: the real SegFormer ONNX artifact on real frames.

These tests require the exported artifact at
``ml/models/shoreline-segmentation/model.onnx`` and a real frame from the
SegFormer dataset. They are skipped (not faked) when either is absent.

They prove the active artifact's contract: Beach/Ocean/Sky ids, Ocean-only
binary water, native-resolution outputs, no 255 predictions, and model identity.
They do not re-derive accuracy.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.vision.segmentation import (
    ShorelineSegmenter,
    find_model_artifact,
    load_model_config,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_IMAGES = REPO_ROOT / "ml" / "datasets" / "coastwatch_segformer" / "images"
EXPECTED_SHA256 = "F598031A1A5891C64EDA8F8D58BF56E8398528BDF457C9336D5741594C0C8A20"


def _artifact() -> Path:
    artifact = find_model_artifact()
    if artifact is None:
        pytest.skip("real SegFormer artifact is not provisioned")
    return artifact


def _real_frame() -> np.ndarray:
    if not DATASET_IMAGES.is_dir():
        pytest.skip("SegFormer dataset images are not available")
    candidates = (
        sorted(DATASET_IMAGES.glob("*.jpg"))[:1] or sorted(DATASET_IMAGES.glob("*.png"))[:1]
    )
    if not candidates:
        pytest.skip("no real frame available")
    frame = cv2.imread(str(candidates[0]), cv2.IMREAD_COLOR)
    if frame is None:
        pytest.skip(f"could not decode {candidates[0]}")
    return frame


def test_real_artifact_config_is_three_class_segformer() -> None:
    artifact = _artifact()
    config = load_model_config(artifact)
    assert config["architecture"] == "SegFormer"
    assert config["encoder"] == "MiT-B2"
    assert config["num_classes"] == 3
    assert config["class_names"] == ["Beach", "Ocean", "Sky"]
    assert config["class_ids"] == {"Beach": 0, "Ocean": 1, "Sky": 2}
    assert config["water_classes"] == [1]
    assert config["ignore_index"] == 255
    assert config["input_size"] == [512, 512]
    assert config["normalization"]["mean"] == pytest.approx([0.485, 0.456, 0.406])
    assert config["normalization"]["std"] == pytest.approx([0.229, 0.224, 0.225])
    assert config["preprocessing"]["resize"] == "bilinear"
    assert config["preprocessing"]["opencv_interpolation"] == "INTER_LINEAR"
    assert config["checkpoint"]["sha256"] == EXPECTED_SHA256
    assert config["checkpoint"]["epoch"] == 8
    assert config["checkpoint"]["val_mean_iou"] == pytest.approx(0.958624, abs=5e-5)
    assert config["checkpoint"]["path"].endswith("ml/checkpoints/segformer_b2_multisite/best.pt")
    assert config["artifact"]["input_name"] == "input"
    assert config["artifact"]["output_name"] == "logits"
    assert "CUDAExecutionProvider" in config["providers"]
    assert "CPUExecutionProvider" in config["providers"]
    assert config["limitations"]
    # The automatic-label / site-held-out caveat must be explicit, not assumed.
    assert any(
        "SAM3" in item or "site-held-out" in item or "not human ground truth" in item
        for item in config["limitations"]
    )


def test_real_artifact_predicts_native_three_class_ocean_only() -> None:
    artifact = _artifact()
    config = json.loads((artifact.parent / "config.json").read_text(encoding="utf-8"))
    frame = _real_frame()
    height, width = frame.shape[:2]

    result = ShorelineSegmenter().predict(frame, include_probabilities=True)

    assert result.class_mask.shape == (height, width)
    assert result.mask.shape == (height, width)
    assert result.class_mask.dtype == np.uint8
    assert result.mask.dtype == np.uint8
    assert set(np.unique(result.class_mask).tolist()) <= {0, 1, 2}
    assert 255 not in set(np.unique(result.class_mask).tolist())
    assert set(np.unique(result.mask).tolist()) <= {0, 255}

    # Binary water is Ocean only, never Sky or Beach.
    ocean = (result.class_mask == 1).astype(np.uint8) * 255
    assert np.array_equal(result.mask, ocean)
    assert np.array_equal(result.mask == 255, result.class_mask == 1)

    # Probabilities are native-sized and sum to one per pixel.
    assert result.probabilities is not None
    assert result.probabilities.shape == (height, width, 3)
    assert np.allclose(result.probabilities.sum(axis=2), 1.0, atol=1e-5)

    assert result.classes == {0: "Beach", 1: "Ocean", 2: "Sky"}
    assert result.architecture == "SegFormer"
    assert result.encoder == "MiT-B2"
    assert result.checkpoint_sha256 == EXPECTED_SHA256
    assert result.checkpoint_id and result.checkpoint_id.endswith(
        "ml/checkpoints/segformer_b2_multisite/best.pt"
    )
    assert result.model_version == config["version"]
    assert result.providers
    assert result.inference_ms is not None and result.inference_ms > 0
    assert set(result.class_statistics) == {"Beach", "Ocean", "Sky"}


def test_status_endpoint_reports_segformer_metadata_without_database() -> None:
    from app.api.routes.shorelines import shorelines_status

    _artifact()
    status = shorelines_status()
    assert status.segmentation_model_available is True
    assert status.architecture == "SegFormer"
    assert status.encoder == "MiT-B2"
    assert status.class_names == ["Beach", "Ocean", "Sky"]
    assert status.class_ids == {"Beach": 0, "Ocean": 1, "Sky": 2}
    assert status.water_classes == [1]
    assert status.checkpoint_sha256 == EXPECTED_SHA256
    assert status.limitations
    assert any("SegFormer MiT-B2" in note for note in status.notes)


def test_real_artifact_rejects_legacy_four_class_config(tmp_path: Path) -> None:
    from app.core.exceptions import ModelUnavailableError
    from app.services.vision.segmentation import _validate_three_class_contract

    legacy_path = REPO_ROOT / "ml" / "models" / "archive" / "unet_resnet34_4class" / "config.json"
    if not legacy_path.is_file():
        pytest.skip("legacy archive is not present on this deployment")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    with pytest.raises(ModelUnavailableError, match="num_classes=3"):
        _validate_three_class_contract(legacy, tmp_path / "config.json")


# ---------------------------------------------------------------------------
# Real-frame shoreline extraction regressions
# ---------------------------------------------------------------------------


def _segmenter_or_skip() -> ShorelineSegmenter:
    _artifact()
    return ShorelineSegmenter()


def _labelled_frame(stem_fragment: str) -> np.ndarray:
    if not DATASET_IMAGES.is_dir():
        pytest.skip("SegFormer dataset images are not available")
    matches = sorted(DATASET_IMAGES.glob(f"*{stem_fragment}*.jpg"))
    if not matches:
        pytest.skip(f"labelled frame {stem_fragment!r} is not available")
    frame = cv2.imread(str(matches[0]), cv2.IMREAD_COLOR)
    if frame is None:
        pytest.skip(f"could not decode {matches[0]}")
    return frame


def test_real_frame_segments_are_ordered_bounded_and_legacy_compatible() -> None:
    from app.services.vision.shoreline import extract_shoreline_from_classes

    frame = _labelled_frame("03c456f2")
    class_mask = _segmenter_or_skip().predict(frame).class_mask
    result = extract_shoreline_from_classes(class_mask)

    assert result.shoreline_available is True, result.reason
    assert len(result.segments) == 1, "single-line contract requires exactly one segment"
    height, width = class_mask.shape
    points = np.asarray(result.points, dtype=np.float64)
    assert np.allclose(points, np.asarray(result.segments[0], dtype=np.float64))
    assert points[:, 0].min() >= 0 and points[:, 0].max() < width
    assert points[:, 1].min() >= 0 and points[:, 1].max() < height
    assert result.shoreline_axis in {"horizontal", "vertical"}
    assert result.measured_coverage is not None


def test_real_video_frame_with_foreground_occluders_is_one_segment() -> None:
    from app.services.vision.shoreline import extract_shoreline_from_classes

    video = REPO_ROOT / "data" / "videos" / "testvideo.mp4"
    if not video.is_file():
        pytest.skip("testvideo.mp4 is not available")
    capture = cv2.VideoCapture(str(video))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        pytest.skip("could not decode testvideo.mp4")
    class_mask = _segmenter_or_skip().predict(frame).class_mask
    result = extract_shoreline_from_classes(class_mask)
    assert result.shoreline_available is True, result.reason
    assert result.orientation_used == "land_below"
    # The palm trunk at the left and the umbrella at the right must not break
    # the near-shore waterline at this frame.
    assert len(result.segments) == 1
    assert len(result.segments[0]) >= 0.5 * class_mask.shape[1]


def test_real_difficult_frame_has_no_tiny_midwater_fragments() -> None:
    from app.services.vision.shoreline import extract_shoreline_from_classes

    frame = _labelled_frame("5ae846f5")
    class_mask = _segmenter_or_skip().predict(frame).class_mask
    result = extract_shoreline_from_classes(class_mask)

    assert result.shoreline_available is True, result.reason
    assert len(result.segments) == 1
    scan_length = (
        class_mask.shape[0] if result.shoreline_axis == "horizontal" else class_mask.shape[1]
    )
    min_segment = max(4, int(round(scan_length * 0.015)))
    assert len(result.segments[0]) >= min_segment


def test_real_oblique_coastsnap_frame_uses_majority_scan_axis() -> None:
    """Karon oblique view: the single-line selector owns the final axis choice.

    The old raw-count heuristic chose ``land_right``; the contract now chooses
    the axis with the best compact/open anchor, so either axis is acceptable as
    long as the result is one continuous line.
    """
    from app.services.vision.shoreline import extract_shoreline_from_classes

    frame = _labelled_frame("5ae846f5")
    class_mask = _segmenter_or_skip().predict(frame).class_mask
    result = extract_shoreline_from_classes(class_mask)
    assert result.shoreline_available is True, result.reason
    assert result.shoreline_axis in {"horizontal", "vertical"}
    assert len(result.segments) == 1


def test_real_night_oblique_frame_reaches_mask_supported_extent() -> None:
    """Night frame whose line used to stop partway down the visible coast.

    The class mask supports Beach/Ocean contact from the distant point
    (row ~335) to the bottom margin row. The pre-fix inference chose
    ``land_below`` and, together with the over-broad edge skip, started the line
    at row ~622 and ended it at row ~1018.
    """
    from app.services.vision.shoreline import extract_shoreline_from_classes

    frame_path = (
        REPO_ROOT
        / "ml"
        / "datasets"
        / "image_sam3"
        / "thailand - koh samui left pan"
        / "2026-11-25_00-19-04.png"
    )
    if not frame_path.is_file():
        pytest.skip(f"night frame {frame_path.name} is not available")
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        pytest.skip(f"could not decode {frame_path}")
    class_mask = _segmenter_or_skip().predict(frame).class_mask
    result = extract_shoreline_from_classes(class_mask)

    assert result.shoreline_available is True, result.reason
    assert result.shoreline_axis == "vertical"
    assert len(result.segments) == 1
    ys = np.asarray([point[1] for point in result.points], dtype=np.float64)
    assert ys.min() <= 360, "line must reach the distant Ocean/Beach contact"
    assert ys.max() >= 1000, "line must cover the mask-supported bottom extent"
