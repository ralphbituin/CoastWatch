"""Checkpoint save/load with the metadata needed for standalone inference."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ml.src.models.factory import build_model
from ml.src.utils import ensure_dir, get_logger, load_yaml, resolve_path, save_json

logger = get_logger("training.checkpoint")


def checkpoint_dir(config: dict[str, Any], architecture: str) -> Path:
    return ensure_dir(resolve_path(config["paths"]["checkpoints"]) / architecture)


def checkpoint_path(
    config: dict[str, Any], architecture: str, stage: str, kind: str = "best"
) -> Path:
    return checkpoint_dir(config, architecture) / f"{stage}_{kind}.pth"


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    architecture: str,
    encoder: str,
    stage: str,
    epoch: int,
    best_metric: float | None,
    config: dict[str, Any],
    class_names: list[str],
    metrics: dict[str, Any] | None = None,
    optimizer: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required to save checkpoints") from exc
    path = resolve_path(path)
    ensure_dir(path.parent)
    payload = {
        "format_version": 1,
        "architecture": architecture,
        "encoder": encoder,
        "num_classes": int(config["data"]["num_classes"]),
        "ignore_index": int(config["data"]["ignore_index"]),
        "class_names": class_names,
        "input_size": [int(config["data"]["image_size"])] * 2,
        "normalization": {
            "mean": config.get("preprocessing", {}).get("imagenet_mean", [0.485, 0.456, 0.406]),
            "std": config.get("preprocessing", {}).get("imagenet_std", [0.229, 0.224, 0.225]),
        },
        "stage": stage,
        "epoch": int(epoch),
        "best_metric": best_metric,
        "metrics": metrics or {},
        "datasets": config["data"]["datasets"],
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    logger.info("saved checkpoint %s", path)
    return path


def load_torch_checkpoint(path: str | Path, *, map_location: str | Any = "cpu") -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyTorch is required to load checkpoints") from exc
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def build_model_from_checkpoint(checkpoint: dict[str, Any], *, device: str = "cpu") -> Any:
    architecture = checkpoint["architecture"]
    model = build_model(
        architecture,
        num_classes=int(checkpoint.get("num_classes", 4)),
        encoder_name=checkpoint.get("encoder"),
        encoder_weights=None,
        in_channels=int(checkpoint.get("in_channels", 3)),
    )
    state = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def copy_as_deployable(
    source: str | Path,
    destination: str | Path,
) -> Path:
    source = resolve_path(source)
    destination = resolve_path(destination)
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    logger.info("copied %s -> %s", source, destination)
    return destination


def write_model_metadata(
    destination: str | Path,
    *,
    architecture: str,
    checkpoint_metadata: dict[str, Any],
    metrics: dict[str, Any],
    limitations: list[str],
    class_config_path: str = "ml/configs/class_mapping.yaml",
    config_path: str = "ml/configs/config.yaml",
) -> Path:
    """Write the ``config.json`` consumed by CoastWatch at inference time."""
    destination = resolve_path(destination)
    ensure_dir(destination.parent)
    config = load_yaml(config_path)
    mapping = load_yaml(class_config_path)
    payload = {
        "name": "coastwatch-shoreline-segmentation",
        "version": checkpoint_metadata.get("version", "1.0.0"),
        "architecture": architecture,
        "encoder": checkpoint_metadata.get("encoder"),
        "input_size": checkpoint_metadata.get("input_size", [512, 512]),
        "num_classes": 4,
        "class_names": ["unknown", "land", "water", "foam"],
        "class_ids": {"unknown": 0, "land": 1, "water": 2, "foam": 3},
        "ignore_index": 255,
        "water_classes": [2, 3],
        "normalization": checkpoint_metadata.get(
            "normalization",
            {
                "mean": config["preprocessing"]["imagenet_mean"],
                "std": config["preprocessing"]["imagenet_std"],
            },
        ),
        "color_map": {
            name: [int(c) for c in color]
            for name, color in zip(
                ["unknown", "land", "water", "foam"],
                [[40, 40, 40], [210, 180, 140], [30, 110, 190], [245, 245, 245]],
                strict=True,
            )
        },
        "training": {
            "datasets": config["data"]["datasets"],
            "stage": checkpoint_metadata.get("stage"),
            "epoch": checkpoint_metadata.get("epoch"),
            "seed": config["project"]["seed"],
            "class_mapping": mapping["datasets"],
        },
        "metrics": metrics,
        "limitations": limitations,
        "created_by": "ml/src/inference/export_onnx.py",
    }
    return _write_json(destination, payload)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    save_json(path, payload)
    return path
