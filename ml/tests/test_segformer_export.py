"""Real-checkpoint export/parity tests for the active SegFormer MiT-B2 model.

These tests use the actual ``best.pt`` checkpoint and the real exported ONNX
artifact plus real held-out frames. They are skipped (never faked) when the
local weights are absent.
"""

from __future__ import annotations

import json

import pytest

from ml.src.dataset import build_eval_transform, load_config
from ml.src.inference.export_onnx import (
    _load_segformer_checkpoint,
    _sha256,
    _verify_segformer_onnx,
)
from ml.src.utils import project_root

CHECKPOINT = (
    project_root() / "ml" / "checkpoints" / "segformer_b2_multisite" / "best.pt"
)
CONFIG = project_root() / "ml" / "configs" / "segformer_b2_multisite.yaml"
IMAGES = project_root() / "ml" / "datasets" / "coastwatch_segformer_multisite" / "images"
SPLIT = (
    project_root()
    / "ml"
    / "datasets"
    / "coastwatch_segformer_multisite"
    / "splits"
    / "test.txt"
)
ARTIFACT = project_root() / "ml" / "models" / "shoreline-segmentation"


def _artifact_config() -> dict:
    return json.loads((ARTIFACT / "config.json").read_text(encoding="utf-8"))


def _require_real_inputs() -> None:
    if not CHECKPOINT.is_file() or not IMAGES.is_dir():
        pytest.skip("SegFormer checkpoint/dataset not available")
    if not (ARTIFACT / "model.onnx").is_file():
        pytest.skip("exported SegFormer ONNX artifact not available")


def test_checkpoint_identity_and_no_random_initialisation() -> None:
    _require_real_inputs()
    artifact_cfg = _artifact_config()
    assert _sha256(CHECKPOINT) == artifact_cfg["checkpoint"]["sha256"]
    assert CHECKPOINT.stat().st_size == artifact_cfg["checkpoint"]["size_bytes"]

    config = load_config(CONFIG)
    model, payload, identity = _load_segformer_checkpoint(
        CHECKPOINT, config, allow_any_checkpoint=True
    )
    assert payload["epoch"] == artifact_cfg["checkpoint"]["epoch"]
    assert payload["best_val_iou"] == pytest.approx(
        artifact_cfg["checkpoint"]["val_mean_iou"], abs=5e-5
    )
    assert payload["num_labels"] == 3
    assert payload["class_names"] == ["Beach", "Ocean", "Sky"]
    assert payload["ignore_index"] == 255
    assert identity["sha256"] == artifact_cfg["checkpoint"]["sha256"]
    assert identity["tensors_verified_equal"] == len(payload["model_state"])
    # The wrapper emits 1/4-resolution logits, as the ONNX contract states.
    import torch

    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 512, 512))
    assert tuple(logits.shape) == (1, 3, 128, 128)


def test_model_module_loads_trained_weights_without_random_init() -> None:
    _require_real_inputs()
    import torch

    from ml.src.model import load_checkpoint

    config = load_config(CONFIG)
    model, payload = load_checkpoint(CHECKPOINT, config=config, device="cpu")
    state = payload["model_state"]
    loaded = model.state_dict()
    assert len(state) == len(loaded) == 380
    assert all(torch.equal(state[key], loaded[key]) for key in state)


def test_exported_artifact_metadata() -> None:
    _require_real_inputs()
    config = json.loads((ARTIFACT / "config.json").read_text(encoding="utf-8"))
    assert config["architecture"] == "SegFormer"
    assert config["encoder"] == "MiT-B2"
    assert config["num_classes"] == 3
    assert config["class_names"] == ["Beach", "Ocean", "Sky"]
    assert config["class_ids"] == {"Beach": 0, "Ocean": 1, "Sky": 2}
    assert config["ignore_index"] == 255
    assert config["water_classes"] == [1]
    assert config["input_size"] == [512, 512]
    assert config["preprocessing"]["resize"] == "bilinear"
    assert config["preprocessing"]["opencv_interpolation"] == "INTER_LINEAR"
    assert config["checkpoint"]["sha256"] == _sha256(CHECKPOINT)
    assert config["checkpoint"]["epoch"] == _artifact_config()["checkpoint"]["epoch"]
    assert config["version"] == _artifact_config()["version"]
    assert config["artifact"]["input_name"] == "input"
    assert config["artifact"]["output_name"] == "logits"
    assert config["artifact"]["output_stride"] == 4
    assert config["verification"]["passed"] is True


def test_onnx_matches_pytorch_native_on_twenty_real_frames() -> None:
    _require_real_inputs()
    config = load_config(CONFIG)
    model, _payload, _identity = _load_segformer_checkpoint(
        CHECKPOINT, config, allow_any_checkpoint=True
    )
    names = [
        line.strip()
        for line in SPLIT.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    frames = [IMAGES / name for name in names[:20] if (IMAGES / name).is_file()]
    assert len(frames) >= 20

    verification = _verify_segformer_onnx(
        model,
        ARTIFACT / "model.onnx",
        frames,
        build_eval_transform(config),
        device="cpu",
    )
    assert verification["passed"] is True, verification
    assert verification["frame_count"] == 20
    assert verification["max_abs_diff"] < 1e-3
    assert verification["quarter_label_agreement"] >= 0.9999
    assert verification["native_label_agreement"] >= 0.9999
    assert verification["native_mismatch_pixels"] <= 10
    assert verification["predict_mask_agreement"] >= 0.9999
