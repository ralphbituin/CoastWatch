"""Shared utilities for the CoastWatch beach segmentation pipeline."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOGGER_NAME = "coastwatch.ml"


def project_root() -> Path:
    """Return the repository root (the directory containing ``ml/``)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "ml").is_dir() and (parent / "backend").is_dir():
            return parent
    # Fallback: ml/src/utils.py -> ml/src -> ml -> repo root
    return here.parents[3]


def resolve_path(path: str | Path, *, base: Path | None = None) -> Path:
    """Resolve a config path relative to the repository root when needed."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = base or project_root()
    return (root / candidate).resolve()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = resolve_path(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}")
    return data


def save_json(path: str | Path, payload: Any, *, indent: int = 2) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, default=_json_default)
        handle.write("\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def ensure_dir(path: str | Path) -> Path:
    path = resolve_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int, *, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch is optional for inspection-only workflows
        pass


def resolve_device(device: str | None = "auto") -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_logger(name: str | None = None) -> logging.Logger:
    logger = logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def count_parameters(model: Any) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
