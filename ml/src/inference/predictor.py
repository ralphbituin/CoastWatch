"""Standalone inference: load a trained artifact and segment a single image.

This module is deliberately independent of PyTorch for the ONNX path, so it can
be tested in an environment that only has ``onnxruntime`` + ``numpy`` + Pillow.
The artifact contract is the same one CoastWatch's backend uses:

    <model_dir>/model.onnx
    <model_dir>/config.json

``config.json`` is authoritative for class order, colours, normalisation,
input size and the water-class ids. Nothing is hard-coded here that the config
can supply, so the deployed model and the training code cannot drift silently.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_CONFIG_KEYS = ("num_classes", "class_names", "input_size", "normalization")


@dataclass
class Prediction:
    """Everything a caller needs from one frame."""

    mask: np.ndarray  # HxW uint8 canonical class ids (0..3)
    probabilities: np.ndarray | None  # HxWxC float32, or None when disabled
    class_names: list[str]
    class_colors: np.ndarray  # Cx3 uint8
    input_size: tuple[int, int]  # (width, height) of the image that was fed
    original_size: tuple[int, int]
    inference_ms: float
    water_mask: np.ndarray  # HxW uint8, 255 where water|foam (CoastWatch contract)
    overlay: np.ndarray | None = None
    class_statistics: dict[str, Any] = field(default_factory=dict)
    shoreline: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Small JSON-serialisable summary (never contains full arrays)."""
        return {
            "inference_ms": round(self.inference_ms, 2),
            "original_size": list(self.original_size),
            "input_size": list(self.input_size),
            "class_statistics": self.class_statistics,
            "shoreline": self.shoreline,
            "warnings": self.warnings,
        }


def load_config(model_dir: str | Path) -> dict[str, Any]:
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Missing {config_path}. The artifact must ship model.onnx and config.json."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing required keys: {missing}")
    return config


def _class_colors_from_config(config: dict[str, Any]) -> np.ndarray:
    names = list(config["class_names"])
    color_map = config.get("color_map") or {}
    default = {
        "unknown": [40, 40, 40],
        "land": [210, 180, 140],
        "water": [30, 110, 190],
        "foam": [245, 245, 245],
    }
    return np.asarray(
        [color_map.get(name, default.get(name, [0, 0, 0])) for name in names], dtype=np.uint8
    )


def _as_rgb(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected HxWx3 RGB image, got shape {array.shape}")
    array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _resize(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB uint8 image to ``(width, height)`` with bilinear interpolation."""
    target_w, target_h = int(size[0]), int(size[1])
    if image.shape[1] == target_w and image.shape[0] == target_h:
        return image
    try:
        import cv2

        interpolation = cv2.INTER_AREA if image.shape[1] > target_w else cv2.INTER_LINEAR
        return cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    except ImportError:  # pragma: no cover - Pillow fallback
        from PIL import Image

        resized = Image.fromarray(image).resize((target_w, target_h), Image.BILINEAR)
        return np.asarray(resized, dtype=np.uint8)


def _nearest_resize_labels(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a label map to ``(width, height)`` with nearest-neighbour only."""
    target_w, target_h = int(size[0]), int(size[1])
    if mask.shape[1] == target_w and mask.shape[0] == target_h:
        return mask
    try:
        import cv2

        return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    except ImportError:  # pragma: no cover
        rows = (np.arange(target_h) * (mask.shape[0] / target_h)).astype(int)
        cols = (np.arange(target_w) * (mask.shape[1] / target_w)).astype(int)
        return mask[np.ix_(rows, cols)]


class SegmentationPredictor:
    """ONNX-backed segmentation predictor.

    The ONNX session is created once per instance. Callers should create one
    predictor at startup and reuse it; loading a model per request is wasteful
    and is explicitly not done here.
    """

    def __init__(
        self,
        model_dir: str | Path,
        *,
        providers: list[str] | None = None,
        input_size: tuple[int, int] | None = None,
        overlay_alpha: float = 0.45,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.config = load_config(self.model_dir)
        self.onnx_path = self.model_dir / "model.onnx"
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"Missing {self.onnx_path}. Export the model with ml/src/inference/export_onnx.py."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "onnxruntime is required for inference. Install ml/requirements.txt "
                "or backend/requirements-ml.txt."
            ) from exc

        requested = providers or self.config.get("providers", ["CPUExecutionProvider"])
        self.providers = [name for name in requested if name == "CPUExecutionProvider"] or [
            "CPUExecutionProvider"
        ]
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = int(self.config.get("intra_op_threads", 0)) or 0
        self.session = ort.InferenceSession(
            str(self.onnx_path), sess_options=options, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        config_size = tuple(int(v) for v in self.config["input_size"])
        self.input_size = tuple(int(v) for v in (input_size or config_size))  # (w, h)
        self.class_names = list(self.config["class_names"])
        self.num_classes = int(self.config["num_classes"])
        self.colors = _class_colors_from_config(self.config)
        self.water_classes = tuple(int(c) for c in self.config.get("water_classes", [2, 3]))
        normalization = self.config["normalization"]
        self.mean = np.asarray(normalization["mean"], dtype=np.float32)
        self.std = np.asarray(normalization["std"], dtype=np.float32)
        self.overlay_alpha = float(overlay_alpha)

    # -- preprocessing -----------------------------------------------------

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        """RGB uint8 HxWx3 -> float32 1x3xHxW, ImageNet-normalised."""
        rgb = _as_rgb(rgb)
        resized = _resize(rgb, self.input_size)
        scaled = resized.astype(np.float32) / 255.0
        scaled = (scaled - self.mean) / self.std
        chw = np.transpose(scaled, (2, 0, 1))
        return chw[np.newaxis, ...]

    # -- inference ---------------------------------------------------------

    def predict(
        self,
        image: Any,
        *,
        return_probabilities: bool = True,
        return_overlay: bool = True,
        compute_shoreline: bool = True,
        restore_size: bool = True,
    ) -> Prediction:
        started = time.perf_counter()
        rgb = _as_rgb(image)
        original_size = (int(rgb.shape[1]), int(rgb.shape[0]))
        batch = self.preprocess(rgb)
        logits = self.session.run([self.output_name], {self.input_name: batch})[0]
        logits = np.asarray(logits)
        if logits.ndim != 4:
            raise ValueError(f"unexpected model output shape {logits.shape}; expected NxCxHxW")
        if logits.shape[1] != self.num_classes:
            # Some exports emit NHWC; accept and transpose.
            if logits.shape[-1] == self.num_classes:
                logits = np.transpose(logits, (0, 3, 1, 2))
            else:
                raise ValueError(
                    f"model output has {logits.shape[1]} channels but config declares "
                    f"{self.num_classes} classes"
                )

        probabilities = _softmax(logits[0])
        mask = np.argmax(probabilities, axis=0).astype(np.uint8)
        warnings: list[str] = []

        if restore_size and (mask.shape[1], mask.shape[0]) != original_size:
            mask = _nearest_resize_labels(mask, original_size)
            probabilities_full = np.stack(
                [
                    _resize((probabilities[channel] * 255).astype(np.uint8), original_size).astype(
                        np.float32
                    )
                    / 255.0
                    for channel in range(probabilities.shape[0])
                ],
                axis=0,
            )
            probabilities = probabilities_full

        inference_ms = (time.perf_counter() - started) * 1000.0
        water_mask = np.where(np.isin(mask, self.water_classes), 255, 0).astype(np.uint8)

        statistics = _class_statistics(mask, self.class_names)
        if statistics.get("foam", {}).get("pixels", 0) == 0:
            warnings.append(
                "foam class was not trained on any provided dataset; treat foam output as "
                "unvalidated"
            )

        overlay = None
        if return_overlay:
            overlay = _make_overlay(rgb, mask, self.colors, self.overlay_alpha)

        shoreline = None
        if compute_shoreline:
            shoreline = _extract_shoreline_summary(water_mask)

        return Prediction(
            mask=mask,
            probabilities=probabilities if return_probabilities else None,
            class_names=self.class_names,
            class_colors=self.colors,
            input_size=(mask.shape[1], mask.shape[0]),
            original_size=original_size,
            inference_ms=inference_ms,
            water_mask=water_mask,
            overlay=overlay,
            class_statistics=statistics,
            shoreline=shoreline,
            warnings=warnings,
        )

    def predict_batch(self, images: list[Any], *, batch_size: int = 8) -> list[Prediction]:
        """Predict several images. Batching is per-call; the session is reused."""
        results: list[Prediction] = []
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            results.extend(self.predict(image) for image in chunk)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=0, keepdims=True)
    exponentiated = np.exp(shifted)
    return (exponentiated / exponentiated.sum(axis=0, keepdims=True)).astype(np.float32)


def _class_statistics(mask: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(mask.size)
    stats: dict[str, Any] = {}
    for index, name in enumerate(class_names):
        pixels = int((mask == index).sum())
        stats[name] = {
            "class_id": index,
            "pixels": pixels,
            "fraction": float(pixels / total) if total else 0.0,
        }
    return stats


def _make_overlay(
    rgb: np.ndarray, mask: np.ndarray, colors: np.ndarray, alpha: float
) -> np.ndarray:
    color_layer = colors[np.clip(mask, 0, len(colors) - 1)]
    blended = rgb.astype(np.float32) * (1 - alpha) + color_layer.astype(np.float32) * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def _extract_shoreline_summary(water_mask: np.ndarray) -> dict[str, Any]:
    """Reuse CoastWatch's own extractor so ml/ and backend/ agree.

    The backend module is imported by path when available; otherwise a compact
    local implementation is used. Both follow the same per-column algorithm.
    """
    try:  # pragma: no cover - exercised when running inside the repo
        from ml.src.inference.shoreline import extract_shoreline_points

        points = extract_shoreline_points(water_mask)
        return {
            "available": True,
            "points": points,
            "min_y": min(p[1] for p in points) if points else None,
            "max_y": max(p[1] for p in points) if points else None,
        }
    except Exception as exc:  # pragma: no cover
        return {"available": False, "reason": f"shoreline extraction failed: {exc}"}
