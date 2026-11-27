"""Tiny three-class ONNX segmentation artifact used by the backend tests.

The graph is deliberately trivial but real: it multiplies the input by zero (so
the input is genuinely consumed and the input shape is dynamic), reduces it to a
scalar, and broadcasts a constant logit map:

* top quarter  -> Sky (2)
* next quarter -> Ocean (1)
* bottom half  -> Beach (0)

That gives a deterministic Ocean/Beach waterline at the half-height row and a
Sky/Ocean horizon above it, so the class-aware extractor can be asserted to use
the former and never the latter.

It is a *fixture*, not a trained model — tests must not claim accuracy for it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

CLASS_NAMES = ["Beach", "Ocean", "Sky"]
CLASS_IDS = {"Beach": 0, "Ocean": 1, "Sky": 2}


def write_artifact(
    directory: Path,
    *,
    size: int = 16,
    num_classes: int = 3,
    class_names: list[str] | None = None,
    architecture: str = "SegFormer",
    input_size: list[int] | None = None,
) -> Path:
    """Write ``model.onnx`` + ``config.json`` into ``directory``."""
    class_names = class_names or CLASS_NAMES
    directory.mkdir(parents=True, exist_ok=True)

    logits = np.zeros((1, num_classes, size, size), dtype=np.float32)
    if num_classes >= 3:
        logits[:, 2, : size // 4] = 8.0  # Sky above
        logits[:, 1, size // 4 : size // 2] = 8.0  # Ocean
        logits[:, 0, size // 2 :] = 8.0  # Beach below

    zero = helper.make_tensor("zero", TensorProto.FLOAT, [1], [0.0])
    constants = helper.make_tensor(
        "constants", TensorProto.FLOAT, list(logits.shape), logits.ravel().tolist()
    )
    axes = helper.make_tensor("axes", TensorProto.INT64, [3], [1, 2, 3])

    nodes = [
        # Consume the real input, then reduce it to a scalar and broadcast a
        # constant map. Spatial axes stay dynamic on the input.
        helper.make_node("Mul", ["input", "zero"], ["zeros"]),
        helper.make_node("ReduceSum", ["zeros", "axes"], ["scalar"], keepdims=1),
        helper.make_node("Mul", ["scalar", "zero"], ["scalar_zero"]),
        helper.make_node("Add", ["scalar_zero", "constants"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "fixture_segmenter",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, "height", "width"])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, num_classes, size, size])],
        initializer=[zero, constants, axes],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="coastwatch-tests",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, directory / "model.onnx")

    config = {
        "name": "coastwatch-test-segmenter",
        "version": "0.0.0-test",
        "architecture": architecture,
        "encoder": "MiT-B2",
        "num_classes": num_classes,
        "class_names": class_names,
        "class_ids": dict(CLASS_IDS),
        "ignore_index": 255,
        "water_classes": [1],
        "input_size": input_size or [512, 512],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "preprocessing": {"resize": "bilinear", "opencv_interpolation": "INTER_LINEAR"},
        "color_map": {
            "Beach": [214, 181, 137],
            "Ocean": [32, 112, 190],
            "Sky": [135, 206, 235],
        },
        "artifact": {
            "onnx": "model.onnx",
            "opset": 17,
            "input_name": "input",
            "output_name": "logits",
            "output_stride": 4,
        },
        "providers": ["CPUExecutionProvider"],
        "checkpoint": {
            "path": "fixture/best.pt",
            "sha256": "0" * 64,
            "epoch": 0,
            "val_mean_iou": 0.0,
        },
        "limitations": ["fixture model, not trained"],
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return directory


def frame_bgr(width: int = 120, height: int = 80, water_rows: int | None = None) -> np.ndarray:
    """Synthetic BGR frame: water (blue) above, sand (tan) below."""
    water_rows = height // 2 if water_rows is None else water_rows
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:water_rows] = (190, 110, 40)  # BGR blue-ish water
    frame[water_rows:] = (140, 180, 210)  # BGR tan sand
    return frame
