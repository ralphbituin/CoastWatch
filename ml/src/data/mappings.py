"""Dataset-specific label mappings into the unified CoastWatch schema.

The authoritative mapping lives in ``ml/configs/class_mapping.yaml``. This module
only *applies* it, so the mapping can be audited without reading Python code.

Rules enforced here:
* Unknown source values fail loudly instead of being coerced into land/water.
* Ignored source classes become 255 (the loss ignore index), never class 0,
  because 0 is a real predicted class ("unknown") and must not be trained on
  fabricated labels.
* Mappings are vectorised through a 256-entry lookup table.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.src.utils import load_yaml, resolve_path

IGNORE_INDEX = 255
_UNDEFINED = 254  # sentinel inside the LUT; never a valid target


@dataclass(frozen=True)
class ClassMapping:
    """A resolved source -> target lookup for one dataset."""

    dataset: str
    display_name: str
    lookup: np.ndarray  # uint8[256], undefined values hold _UNDEFINED
    defined_values: tuple[int, ...]
    supervised_classes: tuple[int, ...]
    raw_mapping: dict[str, Any]
    kind: str

    def apply(self, mask: np.ndarray) -> np.ndarray:
        """Map a raw label image to the canonical mask (with 255 ignore)."""
        array = np.asarray(mask)
        if array.ndim != 2:
            raise ValueError(f"{self.dataset}: mask must be 2-D, got shape {array.shape}")
        if not np.issubdtype(array.dtype, np.integer):
            rounded = np.rint(array)
            if not np.allclose(array, rounded):
                raise ValueError(f"{self.dataset}: non-integer mask values")
            array = rounded.astype(np.int64)
        if array.min() < 0 or array.max() > 255:
            raise ValueError(
                f"{self.dataset}: mask values outside 0..255 "
                f"(min={array.min()}, max={array.max()})"
            )
        mapped = self.lookup[array.astype(np.uint8)]
        undefined = mapped == _UNDEFINED
        if undefined.any():
            values = sorted(int(v) for v in np.unique(array[undefined]))
            raise ValueError(
                f"{self.dataset}: unmapped source mask values {values}. "
                f"Documented values: {sorted(self.defined_values)}. "
                "Add an explicit mapping decision to ml/configs/class_mapping.yaml."
            )
        return mapped.copy()


def load_class_config(path: str | Path = "ml/configs/class_mapping.yaml") -> dict[str, Any]:
    config = load_yaml(path)
    if "schema" not in config or "datasets" not in config:
        raise ValueError(f"{path} is missing schema/datasets sections")
    return config


def build_mapping(dataset: str, config: dict[str, Any] | None = None) -> ClassMapping:
    config = config or load_class_config()
    if dataset not in config["datasets"]:
        raise KeyError(f"Unknown dataset {dataset!r}; known: {sorted(config['datasets'])}")
    entry = config["datasets"][dataset]
    raw = entry.get("mask_values")
    if not raw:
        raise ValueError(f"{dataset}: no mask_values mapping defined")

    lookup = np.full(256, _UNDEFINED, dtype=np.uint8)
    defined: list[int] = []
    for raw_value, spec in raw.items():
        value = int(raw_value)
        if not 0 <= value <= 255:
            raise ValueError(f"{dataset}: source value {value} outside 0..255")
        target = int(spec["target"]) if isinstance(spec, dict) else int(spec)
        if target not in (0, 1, 2, 3, IGNORE_INDEX):
            raise ValueError(f"{dataset}: target {target} is not a canonical class")
        lookup[value] = target
        defined.append(value)

    supervised = tuple(int(c) for c in entry.get("supervised_classes", [1, 2]))
    return ClassMapping(
        dataset=dataset,
        display_name=entry.get("display_name", dataset),
        lookup=lookup,
        defined_values=tuple(sorted(defined)),
        supervised_classes=supervised,
        raw_mapping=raw,
        kind=entry.get("kind", "rgb"),
    )


def build_all_mappings(config: dict[str, Any] | None = None) -> dict[str, ClassMapping]:
    config = config or load_class_config()
    return {
        name: build_mapping(name, config)
        for name, entry in config["datasets"].items()
        if entry.get("kind") != "webcam" and entry.get("mask_values")
    }


def class_names() -> list[str]:
    return ["unknown", "land", "water", "foam"]


def class_colors() -> np.ndarray:
    """RGB colours for visualisation (uint8)."""
    return np.array(
        [
            [40, 40, 40],  # unknown
            [210, 180, 140],  # land
            [30, 110, 190],  # water
            [245, 245, 245],  # foam
        ],
        dtype=np.uint8,
    )


def mapping_table_summary(mapping: ClassMapping) -> list[dict[str, Any]]:
    """Human-readable mapping rows for reports."""
    rows: list[dict[str, Any]] = []
    for raw_value, spec in sorted(mapping.raw_mapping.items(), key=lambda kv: int(kv[0])):
        if isinstance(spec, dict):
            target = int(spec["target"])
            name = spec.get("name", "")
            reason = spec.get("reason", "")
        else:
            target = int(spec)
            name, reason = "", ""
        rows.append(
            {
                "source_value": int(raw_value),
                "source_name": name,
                "target": "ignore" if target == IGNORE_INDEX else class_names()[target],
                "target_id": target,
                "reason": reason,
            }
        )
    return rows


def canonical_mask_path_for(dataset: str, raw_mask_path: str | Path) -> Path:
    """Return the expected location of a pre-converted canonical mask."""
    return (
        resolve_path("ml/manifests/canonical_masks") / dataset / f"{Path(raw_mask_path).stem}.npy"
    )
