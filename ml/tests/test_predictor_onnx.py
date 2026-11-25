"""End-to-end ONNX artifact contract shared by ml/ and the CoastWatch backend.

A tiny real ONNX graph is exported here so the test exercises the same code
path as a deployed model: config.json + model.onnx -> mask, water mask,
probabilities, class statistics and shoreline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")

from ml.src.inference.predictor import SegmentationPredictor, load_config  # noqa: E402


class _EdgeModel(torch.nn.Module):
    """Water where the red channel is bright, land otherwise (top half red)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=True)
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.bias.zero_()
            self.conv.weight[1, 0, 1, 1] = -1.0  # land
            self.conv.weight[2, 0, 1, 1] = 1.0  # water

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _write_artifact(directory: Path, *, size: int = 64) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    model = _EdgeModel().eval()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, size, size),
        str(directory / "model.onnx"),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
        opset_version=17,
    )
    config = {
        "name": "test-segmenter",
        "architecture": "tiny",
        "num_classes": 4,
        "class_names": ["unknown", "land", "water", "foam"],
        "class_ids": {"unknown": 0, "land": 1, "water": 2, "foam": 3},
        "input_size": [size, size],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "water_classes": [2, 3],
        "color_map": {
            "unknown": [40, 40, 40],
            "land": [210, 180, 140],
            "water": [30, 110, 190],
            "foam": [245, 245, 245],
        },
        "providers": ["CPUExecutionProvider"],
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def _split_frame(size: int = 64) -> np.ndarray:
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[: size // 2] = (255, 0, 0)  # red top -> water
    frame[size // 2 :] = (0, 0, 255)  # blue bottom -> land
    return frame


def test_config_requires_documented_keys(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    directory.mkdir()
    (directory / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(directory)
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nowhere")


def test_predictor_returns_the_documented_contract(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    predictor = SegmentationPredictor(artifact, providers=["CPUExecutionProvider"])
    prediction = predictor.predict(_split_frame())

    assert prediction.mask.shape == (64, 64)
    assert prediction.mask.dtype == np.uint8
    assert set(np.unique(prediction.mask).tolist()) <= {0, 1, 2, 3}
    assert prediction.water_mask.dtype == np.uint8
    assert set(np.unique(prediction.water_mask).tolist()) <= {0, 255}

    # Top half should be water, bottom half land.
    assert prediction.class_statistics["water"]["fraction"] == pytest.approx(0.5, abs=0.02)
    assert prediction.class_statistics["land"]["fraction"] == pytest.approx(0.5, abs=0.02)
    assert prediction.class_statistics["foam"]["pixels"] == 0

    assert prediction.probabilities is not None
    assert prediction.probabilities.shape == (4, 64, 64)
    np.testing.assert_allclose(prediction.probabilities.sum(axis=0), np.ones((64, 64)), atol=1e-5)

    assert prediction.shoreline is not None
    assert prediction.shoreline["available"] is True
    ys = [point[1] for point in prediction.shoreline["points"]]
    assert float(np.mean(ys)) == pytest.approx(32, abs=2.5)
    assert len(ys) == 64

    assert prediction.inference_ms >= 0.0
    assert prediction.overlay is not None
    assert prediction.overlay.shape == (64, 64, 3)

    summary = prediction.to_summary()
    assert "class_statistics" in summary
    assert "mask" not in summary  # arrays are never pushed through JSON


def test_predictor_restores_original_resolution(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact", size=32)
    predictor = SegmentationPredictor(artifact, providers=["CPUExecutionProvider"])
    frame = _split_frame(size=96)
    prediction = predictor.predict(frame)
    assert prediction.mask.shape == (96, 96)
    assert prediction.water_mask.shape == (96, 96)
    assert prediction.class_statistics["water"]["fraction"] == pytest.approx(0.5, abs=0.02)


def test_predictor_accepts_an_inference_size_override(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact", size=64)
    predictor = SegmentationPredictor(
        artifact, providers=["CPUExecutionProvider"], input_size=(32, 32)
    )
    prediction = predictor.predict(_split_frame(size=64))
    assert prediction.mask.shape == (64, 64)
    assert prediction.water_mask.shape == (64, 64)


def test_foam_is_flagged_as_unvalidated(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    predictor = SegmentationPredictor(artifact, providers=["CPUExecutionProvider"])
    prediction = predictor.predict(_split_frame())
    assert any("foam" in warning.lower() for warning in prediction.warnings)


def test_missing_onnx_is_a_loud_error(tmp_path: Path) -> None:
    artifact = _write_artifact(tmp_path / "artifact")
    (artifact / "model.onnx").unlink()
    with pytest.raises(FileNotFoundError, match="model.onnx"):
        SegmentationPredictor(artifact)
