"""Semantic segmentation inference for the active SegFormer MiT-B2 artifact.

Loads the ONNX artifact described in ``docs/MODEL_DOCUMENTATION.md``::

    <MODEL_PATH>/shoreline-segmentation/model.onnx
    <MODEL_PATH>/shoreline-segmentation/config.json

and turns a camera frame into:

* the **canonical three-class mask** (0 Beach, 1 Ocean, 2 Sky) at native size;
* a **binary water mask** (``mask``, 0/255) equal to Ocean only — Sky and Beach
  are never water. This is the contract consumed by the operator-mask path in
  :func:`app.services.vision.shoreline.extract_shoreline`;
* optional native-resolution probabilities and per-class statistics;
* model identity (architecture, checkpoint hash, version) and inference time.

Pipeline order is deliberate: **logits are upsampled to the native frame first**,
then softmax and argmax are computed. Resizing an argmax class map (the legacy
behaviour) changes class boundaries and is not equivalent.

Nothing here fabricates a mask. When the artifact is missing, unreadable, or
onnxruntime is unavailable, :class:`ModelUnavailableError` is raised. A malformed
frame raises :class:`InvalidFrameError`; neither path returns a placeholder.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.core.config import get_settings
from app.core.exceptions import ModelUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)

CONFIG_FILENAME = "config.json"
MODEL_FILENAME = "model.onnx"
REQUIRED_CONFIG_KEYS = (
    "class_names",
    "class_ids",
    "num_classes",
    "input_size",
    "normalization",
    "water_classes",
)
REQUIRED_CLASS_NAMES = ("Beach", "Ocean", "Sky")
REQUIRED_CLASS_IDS = {"Beach": 0, "Ocean": 1, "Sky": 2}
OCEAN_CLASS_ID = 1
IGNORE_INDEX = 255
MAX_CLASS_ID = 2
DEFAULT_INPUT_SIZE = (512, 512)
# Kept for backwards compatibility with callers that import the old constant.
WATER_CLASSES = (OCEAN_CLASS_ID,)


class InvalidFrameError(ValueError):
    """Raised when a frame cannot be decoded into a usable RGB array."""


@dataclass(slots=True)
class SegmentationResult:
    """Segmentation output for one frame, at the frame's native resolution."""

    mask: Any  # 2-D uint8 binary Ocean mask (0/255), native size
    confidence: float
    classes: dict[int, str]
    class_mask: Any = None  # 2-D uint8 class ids 0/1/2, native size
    probabilities: Any = None  # optional HxWx3 float32 at native size
    class_statistics: dict[str, Any] = field(default_factory=dict)
    inference_ms: float | None = None
    architecture: str | None = None
    encoder: str | None = None
    checkpoint_id: str | None = None
    checkpoint_sha256: str | None = None
    model_version: str | None = None
    providers: tuple[str, ...] = ()
    water_classes: tuple[int, ...] = (OCEAN_CLASS_ID,)


def find_model_artifact(model_path: str | Path | None = None) -> Path | None:
    """Return the path to a usable ONNX artifact, or ``None``.

    Looks for ``<MODEL_PATH>/shoreline-segmentation/model.onnx``. Never
    downloads, substitutes or invents an artifact.
    """
    base = Path(model_path) if model_path is not None else Path(get_settings().model_path)
    artifact = base / "shoreline-segmentation" / MODEL_FILENAME
    return artifact if artifact.is_file() else None


def load_model_config(artifact: Path) -> dict[str, Any]:
    config_path = artifact.parent / CONFIG_FILENAME
    if not config_path.is_file():
        raise ModelUnavailableError(
            f"Segmentation artifact at {artifact} has no {CONFIG_FILENAME}. "
            "The artifact is unusable without its class map and preprocessing config."
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailableError(f"Cannot read {config_path}: {exc}") from exc
    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in config]
    if missing:
        raise ModelUnavailableError(f"{config_path} is missing required keys: {missing}")
    _validate_three_class_contract(config, config_path)
    return config


def _validate_three_class_contract(config: dict[str, Any], config_path: Path) -> None:
    """Reject any legacy four-class / mis-ordered artifact.

    This is intentionally strict: the backend and the shoreline extractor are
    built around Beach=0, Ocean=1, Sky=2. Accepting a 4-class artifact here is
    how a silent class-id regression would reach production.
    """
    class_names = list(config.get("class_names") or [])
    class_ids = config.get("class_ids") or {}
    num_classes = int(config.get("num_classes", -1))
    water_classes = [int(value) for value in config.get("water_classes", [])]
    input_size = [int(value) for value in config.get("input_size", [])]
    if num_classes != 3 or class_names != list(REQUIRED_CLASS_NAMES):
        raise ModelUnavailableError(
            f"{config_path}: expected num_classes=3 and class_names={list(REQUIRED_CLASS_NAMES)}, "
            f"got num_classes={num_classes}, class_names={class_names}. "
            "The active artifact must be the SegFormer MiT-B2 three-class model."
        )
    if class_ids != REQUIRED_CLASS_IDS:
        raise ModelUnavailableError(
            f"{config_path}: class_ids must be {REQUIRED_CLASS_IDS}, got {class_ids}"
        )
    if water_classes != [OCEAN_CLASS_ID]:
        raise ModelUnavailableError(
            f"{config_path}: water_classes must be [1] (Ocean only), got {water_classes}"
        )
    if int(config.get("ignore_index", -1)) != IGNORE_INDEX:
        raise ModelUnavailableError(
            f"{config_path}: ignore_index must be {IGNORE_INDEX}, "
            f"got {config.get('ignore_index')!r}"
        )
    if input_size != list(DEFAULT_INPUT_SIZE):
        raise ModelUnavailableError(
            f"{config_path}: input_size must be {list(DEFAULT_INPUT_SIZE)}, got {input_size}"
        )
    architecture = str(config.get("architecture", "")).lower()
    if architecture not in {"segformer", "segformerb2"}:
        raise ModelUnavailableError(
            f"{config_path}: expected architecture='SegFormer', got {config.get('architecture')!r}"
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=0, keepdims=True)
    exponentiated = np.exp(shifted)
    return (exponentiated / exponentiated.sum(axis=0, keepdims=True)).astype(np.float32)


def _bilinear_align_corners_false(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize of ``(C, H, W)`` matching ``align_corners=False``.

    Separable implementation (height then width) so a 4K frame does not allocate
    four full-resolution index planes. This is the exact convention used by
    ``torch.nn.functional.interpolate(..., mode="bilinear", align_corners=False)``
    and by ``ml/scripts/infer.py:predict_mask``.
    """
    channels, in_h, in_w = chw.shape
    if (in_h, in_w) == (out_h, out_w):
        return np.asarray(chw, dtype=np.float32)

    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (in_h / out_h) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    wy = (ys - y0).astype(np.float32)[:, None]
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    y0 = np.clip(y0, 0, in_h - 1)
    vertical = chw[:, y0, :] * (1.0 - wy) + chw[:, y1, :] * wy

    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (in_w / out_w) - 0.5
    x0 = np.floor(xs).astype(np.int64)
    wx = (xs - x0).astype(np.float32)[None, :]
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    x0 = np.clip(x0, 0, in_w - 1)
    return (vertical[:, :, x0] * (1.0 - wx) + vertical[:, :, x1] * wx).astype(
        np.float32, copy=False
    )


CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"
CPU_PROVIDERS: tuple[str, ...] = (CPU_EXECUTION_PROVIDER,)

# ONNX Runtime is CPU-only in this deployment. A process-wide semaphore caps
# concurrent runs so multiple camera workers cannot oversubscribe the cores.
_inference_gate_lock = threading.Lock()
_inference_gate: threading.BoundedSemaphore | None = None
_inference_gate_limit: int | None = None


def _inference_gate_for(limit: int) -> threading.BoundedSemaphore:
    global _inference_gate, _inference_gate_limit
    configured = max(1, int(limit))
    with _inference_gate_lock:
        if _inference_gate is None or _inference_gate_limit != configured:
            _inference_gate = threading.BoundedSemaphore(configured)
            _inference_gate_limit = configured
        return _inference_gate


def _max_concurrent_inferences() -> int:
    return max(1, int(getattr(get_settings(), "onnx_max_concurrent_inferences", 1) or 1))


@lru_cache(maxsize=8)
def _load_session(artifact_str: str, intra_op_threads: int) -> Any:
    """Create and cache the CPU-only ONNX Runtime session (one per artifact)."""
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise ModelUnavailableError(
            "onnxruntime is not installed, so the segmentation artifact cannot be used. "
            "Install backend/requirements-ml.txt."
        ) from exc

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # One inference at a time per process: one inter-op thread and sequential
    # execution avoid thread-pool fights with the capture/HTTP threads.
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_cpu_mem_arena = True
    if intra_op_threads > 0:
        options.intra_op_num_threads = intra_op_threads
    try:
        session = ort.InferenceSession(
            artifact_str, sess_options=options, providers=list(CPU_PROVIDERS)
        )
    except Exception as exc:  # pragma: no cover - corrupt artifact
        raise ModelUnavailableError(f"Failed to load {artifact_str}: {exc}") from exc
    logger.info(
        "loaded SegFormer segmentation model %s with providers %s "
        "(intra_op_threads=%s, inter_op_threads=1, sequential, mem_arena=True)",
        artifact_str,
        session.get_providers(),
        options.intra_op_num_threads,
    )
    return session


class ShorelineSegmenter:
    """Loads and runs the validated three-class SegFormer segmentation model."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        input_size: tuple[int, int] | None = None,
        overlay_alpha: float = 0.45,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else None
        self._input_size_override = input_size
        self.overlay_alpha = overlay_alpha
        self._session: Any | None = None
        self._config: dict[str, Any] | None = None
        self._input_name = ""
        self._output_name = ""
        self._providers: tuple[str, ...] = ()
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load config + session once. Raises ``ModelUnavailableError``."""
        if self._session is not None:
            return
        with self._lock:
            if self._session is not None:  # double-checked locking
                return
            artifact = find_model_artifact(self.model_path)
            if artifact is None:
                raise ModelUnavailableError(
                    "No segmentation model artifact found. Provision the SegFormer MiT-B2 "
                    "artifact under MODEL_PATH/shoreline-segmentation/model.onnx (see "
                    "docs/MODEL_DOCUMENTATION.md), or supply operator-labelled masks via "
                    "POST /api/v1/shorelines/observations."
                )
            config = load_model_config(artifact)
            configured_threads = int(getattr(get_settings(), "onnx_intra_op_threads", 0) or 0)
            if configured_threads <= 0:
                configured_threads = int(config.get("intra_op_threads", 0) or 0)
            session = _load_session(str(artifact), configured_threads)

            self._config = config
            self._session = session
            self._providers = tuple(session.get_providers())
            self._input_name = session.get_inputs()[0].name
            self._output_name = session.get_outputs()[0].name

    @property
    def config(self) -> dict[str, Any]:
        self.load()
        assert self._config is not None
        return self._config

    @property
    def class_names(self) -> list[str]:
        return list(self.config["class_names"])

    @property
    def input_size(self) -> tuple[int, int]:
        """Inference size as ``(width, height)``."""
        config_size = tuple(int(v) for v in self.config["input_size"])
        chosen = self._input_size_override or config_size
        return (int(chosen[0]), int(chosen[1]))

    @property
    def metadata(self) -> dict[str, Any]:
        config = self.config
        checkpoint = config.get("checkpoint", {})
        return {
            "architecture": config.get("architecture"),
            "encoder": config.get("encoder"),
            "model_version": config.get("version"),
            "checkpoint_id": checkpoint.get("path") or checkpoint.get("id"),
            "checkpoint_sha256": checkpoint.get("sha256"),
            "providers": list(self._providers),
        }

    # -- inference ---------------------------------------------------------

    def predict(self, frame: Any, *, include_probabilities: bool = False) -> SegmentationResult:
        """Segment one BGR frame (OpenCV convention, as produced by cameras)."""
        self.load()
        if self._session is None or self._config is None:  # pragma: no cover - defensive
            raise ModelUnavailableError("segmentation model failed to load")

        rgb = self._to_rgb(frame)
        height, width = rgb.shape[:2]
        target_width, target_height = self.input_size
        import cv2

        resized = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        normalization = self._config["normalization"]
        mean = np.asarray(normalization["mean"], dtype=np.float32)
        std = np.asarray(normalization["std"], dtype=np.float32)
        scaled = resized.astype(np.float32) / 255.0
        scaled = (scaled - mean) / std
        batch = np.transpose(scaled, (2, 0, 1))[np.newaxis, ...]

        gate = _inference_gate_for(_max_concurrent_inferences())
        with gate:
            started = time.perf_counter()
            outputs, providers_used = self._run(batch)
            inference_ms = (time.perf_counter() - started) * 1000.0

        logits = np.asarray(outputs[0])
        if logits.ndim != 4:
            raise ModelUnavailableError(
                f"unexpected model output shape {logits.shape}; expected NxCxHxW"
            )
        num_classes = int(self._config["num_classes"])
        if logits.shape[1] != num_classes:
            if logits.shape[-1] == num_classes:  # NHWC export
                logits = np.transpose(logits, (0, 3, 1, 2))
            else:
                raise ModelUnavailableError(
                    f"model outputs {logits.shape[1]} channels but config declares "
                    f"{num_classes} classes"
                )

        # Upsample logits to the original frame, then softmax/argmax. Never
        # resize the argmax map: that changes class boundaries.
        logits = _bilinear_align_corners_false(logits[0], height, width)
        probabilities = _softmax(logits)
        class_mask = probabilities.argmax(axis=0).astype(np.uint8)
        confidence = float(probabilities.max(axis=0).mean())

        if class_mask.shape != (height, width):
            raise ModelUnavailableError(
                f"class mask shape {class_mask.shape} does not match frame {(height, width)}"
            )
        if int(class_mask.max(initial=0)) > MAX_CLASS_ID:
            raise ModelUnavailableError(
                f"class mask contains id {int(class_mask.max())}; expected only 0/1/2"
            )

        water_classes = tuple(int(c) for c in self._config.get("water_classes", [1]))
        water_mask = np.where(
            np.isin(class_mask, water_classes), np.uint8(255), np.uint8(0)
        ).astype(np.uint8)
        statistics = _class_statistics(class_mask, self._config["class_names"])
        checkpoint = self._config.get("checkpoint", {})

        return SegmentationResult(
            mask=water_mask,
            confidence=confidence,
            classes=dict(enumerate(self._config["class_names"])),
            class_mask=class_mask,
            probabilities=(
                np.transpose(probabilities, (1, 2, 0)) if include_probabilities else None
            ),
            class_statistics=statistics,
            inference_ms=round(inference_ms, 2),
            architecture=self._config.get("architecture"),
            encoder=self._config.get("encoder"),
            checkpoint_id=checkpoint.get("path") or checkpoint.get("id"),
            checkpoint_sha256=checkpoint.get("sha256"),
            model_version=self._config.get("version"),
            providers=tuple(providers_used),
            water_classes=water_classes,
        )

    def _run(self, batch: np.ndarray) -> tuple[list[Any], tuple[str, ...]]:
        """Run the CPU session once; any failure is surfaced, never faked."""
        assert self._session is not None
        try:
            outputs = self._session.run([self._output_name], {self._input_name: batch})
        except Exception as exc:
            raise ModelUnavailableError(f"segmentation inference failed: {exc}") from exc
        return outputs, tuple(self._session.get_providers())

    @staticmethod
    def _to_rgb(frame: Any) -> np.ndarray:
        if frame is None:
            raise InvalidFrameError("frame is None; cannot segment")
        array = np.asarray(frame)
        if array.size == 0:
            raise InvalidFrameError(f"frame is empty (shape {array.shape})")
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[2] < 3:
            raise InvalidFrameError(
                f"unsupported frame shape {array.shape}; expected HxWx3 BGR uint8"
            )
        # Cameras deliver BGR; training and the exported graph use RGB.
        if array.shape[2] == 3:
            rgb = array[:, :, 2::-1]
        else:
            import cv2

            rgb = cv2.cvtColor(array, cv2.COLOR_BGRA2RGB)
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.nan_to_num(rgb), 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb)


def _class_statistics(class_mask: np.ndarray, class_names: list[str]) -> dict[str, Any]:
    total = int(class_mask.size) or 1
    statistics: dict[str, Any] = {}
    for index, name in enumerate(class_names):
        pixels = int((class_mask == index).sum())
        statistics[name] = {
            "class_id": index,
            "pixels": pixels,
            "fraction": round(pixels / total, 6),
        }
    return statistics


def model_status(model_path: str | Path | None = None) -> dict[str, Any]:
    """Non-raising status summary for diagnostics and the /shorelines endpoint."""
    artifact = find_model_artifact(model_path)
    if artifact is None:
        return {"available": False, "reason": "artifact_missing", "config_valid": False}
    try:
        config = load_model_config(artifact)
        checkpoint = config.get("checkpoint", {})
        return {
            "available": True,
            "path": str(artifact),
            "config_valid": True,
            "architecture": config.get("architecture"),
            "encoder": config.get("encoder"),
            "model_version": config.get("version"),
            "input_size": config.get("input_size"),
            "class_names": config.get("class_names"),
            "class_ids": config.get("class_ids"),
            "water_classes": config.get("water_classes"),
            "ignore_index": config.get("ignore_index"),
            "checkpoint_id": checkpoint.get("path") or checkpoint.get("id"),
            "checkpoint_sha256": checkpoint.get("sha256"),
            "providers": config.get("providers", []),
            "precision": config.get("precision", {}),
            "limitations": config.get("limitations", []),
        }
    except ModelUnavailableError as exc:
        return {
            "available": True,
            "path": str(artifact),
            "config_valid": False,
            "reason": str(exc),
        }
