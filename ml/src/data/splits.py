"""Dataset split generation and CSV manifests.

Splits are generated once and then treated as immutable inputs to training, so
every run uses exactly the same files. Policies (documented in
``ml/reports/dataset_inspection.md``):

* **India** — the distributed ``training_set/validation_set/testing_set``
  folders are preserved. No regrouping metadata exists.
* **SCLabels** — chronological split *within each site* (70/15/15), using the
  metadata capture date. This keeps every site represented while preventing
  later dates from leaking into earlier training data. It is a temporal
  generalisation test, not an unseen-site test.
* **Webcam** — the provided ``splits/*.txt`` files are used for qualitative
  inference only (no masks, no metrics).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.src.data.mappings import load_class_config
from ml.src.utils import ensure_dir, get_logger, load_yaml, project_root, resolve_path

logger = get_logger("data.splits")

MANIFEST_COLUMNS = [
    "dataset",
    "split",
    "group",
    "image_path",
    "mask_path",
    "width",
    "height",
    "timestamp",
    "source_classes",
]


@dataclass(slots=True)
class DatasetRecord:
    dataset: str
    split: str
    group: str
    image_path: str
    mask_path: str
    width: int
    height: int
    timestamp: str = ""
    source_classes: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_root()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        # Multispectral TIFFs are not readable by Pillow. Their dimensions are
        # read lazily by the dataset; use the mask size, which is paired.
        return 0, 0


def _source_values(mask_path: Path) -> str:
    try:
        with Image.open(mask_path) as image:
            values = np.unique(np.array(image))
        return ",".join(str(int(v)) for v in values)
    except Exception:
        return ""


def _split_bucket(index: int, total: int) -> str:
    """70/15/15 bucket for an already ordered sequence."""
    if total <= 0:
        return "train"
    position = index / total
    if position < 0.70:
        return "train"
    if position < 0.85:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# India
# ---------------------------------------------------------------------------


def india_records(config: dict[str, Any]) -> list[DatasetRecord]:
    entry = load_class_config()["datasets"]["india_sandy_beach"]
    root = resolve_path(entry["root"])
    records: list[DatasetRecord] = []
    for split_dir, split in [
        ("training_set", "train"),
        ("validation_set", "val"),
        ("testing_set", "test"),
    ]:
        images = sorted((root / split_dir / "images").glob("*.tif"))
        masks = sorted((root / split_dir / "masks").glob("*.tif"))
        by_stem = {p.stem: p for p in masks}
        missing = [p.name for p in images if p.stem not in by_stem]
        if missing:
            raise RuntimeError(f"India {split}: {len(missing)} images have no mask: {missing[:5]}")
        extras = sorted(set(by_stem) - {p.stem for p in images})
        if extras:
            logger.warning("India %s: %d masks have no image: %s", split, len(extras), extras[:5])
        for image_path in images:
            mask_path = by_stem[image_path.stem]
            width, height = _image_size(mask_path)
            records.append(
                DatasetRecord(
                    dataset="india_sandy_beach",
                    split=split,
                    group=f"india/{split_dir}",
                    image_path=_rel(image_path),
                    mask_path=_rel(mask_path),
                    width=width,
                    height=height,
                    source_classes=_source_values(mask_path),
                )
            )
    return records


# ---------------------------------------------------------------------------
# SCLabels
# ---------------------------------------------------------------------------


def sclabels_records(config: dict[str, Any]) -> list[DatasetRecord]:
    entry = load_class_config()["datasets"]["sclabels"]
    root = resolve_path(entry["root"])
    metadata = json.loads((root / entry["metadata"]).read_text(encoding="utf-8"))
    if not isinstance(metadata, list):
        raise ValueError("SCLabels metadata.json must be a list")

    by_site: dict[str, list[dict[str, Any]]] = {}
    for item in metadata:
        site = item["image"]["site"]["CSname"]
        by_site.setdefault(site, []).append(item)

    records: list[DatasetRecord] = []
    for site, items in sorted(by_site.items()):
        items.sort(key=lambda row: (row["image"]["date"]["date"], row["image"]["filename"]))
        total = len(items)
        for index, item in enumerate(items):
            split = _split_bucket(index, total)
            image_name = item["image"]["filename"]
            mask_name = item["image"]["mask"]["filename"]
            image_path = root / entry["image_dir"] / image_name
            mask_path = root / entry["mask_dir"] / mask_name
            if not image_path.is_file() or not mask_path.is_file():
                raise FileNotFoundError(f"SCLabels pair missing on disk: {image_path}, {mask_path}")
            width, height = _image_size(image_path)
            records.append(
                DatasetRecord(
                    dataset="sclabels",
                    split=split,
                    group=f"sclabels/{site}",
                    image_path=_rel(image_path),
                    mask_path=_rel(mask_path),
                    width=width,
                    height=height,
                    timestamp=item["image"]["date"]["date"],
                    source_classes=_source_values(mask_path),
                )
            )
    return records


# ---------------------------------------------------------------------------
# Webcam (unlabelled)
# ---------------------------------------------------------------------------


def webcam_records(config: dict[str, Any]) -> list[DatasetRecord]:
    entry = load_class_config()["datasets"].get("webcam_unlabelled")
    if not entry:
        return []
    root = resolve_path(entry["root"])
    splits_dir = root / "splits"
    records: list[DatasetRecord] = []
    for split in ("train", "val", "test"):
        list_path = splits_dir / f"{split}.txt"
        if not list_path.is_file():
            continue
        for name in list_path.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if not name:
                continue
            image_path = root / name
            if not image_path.is_file():
                continue
            width, height = _image_size(image_path)
            records.append(
                DatasetRecord(
                    dataset="webcam_unlabelled",
                    split=split,
                    group="webcam/karon",
                    image_path=_rel(image_path),
                    mask_path="",
                    width=width,
                    height=height,
                    source_classes="",
                )
            )
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_splits(
    config: dict[str, Any] | None = None,
    *,
    output_dir: str | Path | None = None,
    write: bool = True,
) -> dict[str, list[DatasetRecord]]:
    config = config or load_yaml("ml/configs/config.yaml")
    groups = {
        "india_sandy_beach": india_records(config),
        "sclabels": sclabels_records(config),
    }
    if write:
        destination = resolve_path(output_dir or config["paths"]["manifests"])
        ensure_dir(destination)
        for dataset, records in groups.items():
            for split in ("train", "val", "test"):
                path = destination / f"{dataset}_{split}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
                    writer.writeheader()
                    for record in records:
                        if record.split == split:
                            writer.writerow(record.to_row())
                logger.info("%s: wrote %s", dataset, path.relative_to(project_root()))
        # Webcam list is informational only.
        webcam = webcam_records(config)
        if webcam:
            path = destination / "webcam_unlabelled.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
                writer.writeheader()
                for record in webcam:
                    writer.writerow(record.to_row())
    return groups


def load_manifest(path: str | Path) -> list[DatasetRecord]:
    path = resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}. Run ml.src.data.splits first.")
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            records.append(
                DatasetRecord(
                    dataset=row["dataset"],
                    split=row["split"],
                    group=row["group"],
                    image_path=row["image_path"],
                    mask_path=row["mask_path"],
                    width=int(row["width"] or 0),
                    height=int(row["height"] or 0),
                    timestamp=row.get("timestamp", ""),
                    source_classes=row.get("source_classes", ""),
                )
            )
    return records


def load_all_manifests(
    config: dict[str, Any] | None = None, split: str = "train"
) -> dict[str, list[DatasetRecord]]:
    config = config or load_yaml("ml/configs/config.yaml")
    root = resolve_path(config["paths"]["manifests"])
    result: dict[str, list[DatasetRecord]] = {}
    for dataset in config["data"]["datasets"]:
        path = root / f"{dataset}_{split}.csv"
        if path.is_file():
            result[dataset] = load_manifest(path)
    return result


def manifest_summary(records: Iterable[DatasetRecord]) -> dict[str, Any]:
    rows = list(records)
    groups = {r.group for r in rows}
    return {
        "count": len(rows),
        "groups": len(groups),
        "splits": {
            split: sum(r.split == split for r in rows) for split in ("train", "val", "test")
        },
    }
