"""Audit CoastWatch datasets before any conversion or training.

This script is intentionally read-only. It never modifies a source image, mask
or annotation file. It writes reports and previews only under
``ml/outputs/audit/``.

Usage:
    python -m ml.scripts.audit_datasets
    python -m ml.scripts.audit_datasets --config ml/configs/segformer_b2.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml.src.dataset import (
    CLASS_COLORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    VALID_CLASS_IDS,
    load_config,
    project_root,
    resolve_path,
)
from ml.src.visualization import preview_grid

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class PairRecord:
    key: str
    image_path: Path
    mask_path: Path | None
    width: int | None = None
    height: int | None = None
    image_mode: str | None = None
    image_format: str | None = None
    image_bit_depth: int | None = None
    image_channels: int | None = None
    mask_mode: str | None = None
    mask_bit_depth: int | None = None
    mask_size: tuple[int, int] | None = None
    raw_values: dict[int, int] = field(default_factory=dict)
    mapped_values: dict[int, int] = field(default_factory=dict)
    invalid_values: dict[int, int] = field(default_factory=dict)
    dimension_match: bool | None = None
    image_corrupt: str | None = None
    mask_corrupt: str | None = None
    empty_mask: bool | None = None
    exact_sha256: str | None = None
    phash: np.ndarray | None = None
    dhash: np.ndarray | None = None
    alpha_min: int | None = None
    alpha_max: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "image": str(self.image_path),
            "mask": str(self.mask_path) if self.mask_path else None,
            "width": self.width,
            "height": self.height,
            "image_mode": self.image_mode,
            "image_format": self.image_format,
            "image_bit_depth": self.image_bit_depth,
            "image_channels": self.image_channels,
            "mask_mode": self.mask_mode,
            "mask_bit_depth": self.mask_bit_depth,
            "mask_size": list(self.mask_size) if self.mask_size else None,
            "raw_values": {str(k): int(v) for k, v in sorted(self.raw_values.items())},
            "mapped_values": {str(k): int(v) for k, v in sorted(self.mapped_values.items())},
            "invalid_values": {str(k): int(v) for k, v in sorted(self.invalid_values.items())},
            "dimension_match": self.dimension_match,
            "image_corrupt": self.image_corrupt,
            "mask_corrupt": self.mask_corrupt,
            "empty_mask": self.empty_mask,
            "exact_sha256": self.exact_sha256,
            "error": self.error,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bit_depth(mode: str, array: np.ndarray) -> int:
    if array.dtype == np.uint16:
        return 16
    if array.dtype in (np.float32, np.float64):
        return 32
    if mode in ("I;16", "I;16B", "I;16L"):
        return 16
    return int(array.dtype.itemsize * 8) if array.dtype.itemsize else 8


def _perceptual_hashes(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    try:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None, None
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(small))
        low = dct[:8, :8]
        phash = (low > np.median(low[1:])).flatten()
        tiny = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
        dhash = (tiny[:, 1:] > tiny[:, :-1]).flatten()
        return phash, dhash
    except Exception:
        return None, None


def _relative_key(path: Path, root: Path) -> str:
    return str(path.relative_to(root).with_suffix("")).replace("\\", "/")


def _collect_images(root: Path, exts: Iterable[str]) -> dict[str, Path]:
    exts = {e.lower() for e in exts}
    if not root.exists():
        return {}
    result: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            key = _relative_key(path, root)
            if key in result:
                raise ValueError(f"duplicate image key {key!r} under {root}")
            result[key] = path
    return result


def _polygon_analysis(
    pair: PairRecord,
    polygon_path: Path,
    expected_mapping: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Rasterize JSON polygons and compare their labels to the PNG mask.

    Returns per-pair statistics or None when the pair has no polygon JSON.
    """
    if not polygon_path.exists():
        return None
    try:
        with polygon_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {"error": f"cannot read polygon JSON: {exc}"}

    mask = np.asarray(Image.open(pair.mask_path)) if pair.mask_path else None
    if mask is None or mask.ndim != 2:
        return {"error": "mask is not a 2-D class-id image"}
    h, w = mask.shape
    if int(payload.get("imageHeight", h)) != h or int(payload.get("imageWidth", w)) != w:
        return {
            "error": "polygon JSON dimensions differ from mask",
            "polygon_size": [payload.get("imageWidth"), payload.get("imageHeight")],
            "mask_size": [w, h],
        }

    label_masks: dict[str, np.ndarray] = {}
    shape_counts: Counter[str] = Counter()
    exact_duplicate_polygons = 0
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        shape_counts[label] += 1
        points = shape.get("points") or []
        if len(points) < 3:
            continue
        canvas = Image.new("L", (w, h), 0)
        ImageDraw.Draw(canvas).polygon([tuple(float(v) for v in p) for p in points], fill=1)
        component = np.asarray(canvas, dtype=bool)
        if label in label_masks and np.array_equal(label_masks[label], component):
            exact_duplicate_polygons += 1
        label_masks[label] = label_masks.get(label, np.zeros((h, w), dtype=bool)) | component

    per_label: dict[str, Any] = {}
    expected_by_name = {
        str(spec.get("name", "")).lower(): int(source)
        for source, spec in expected_mapping.items()
        if str(spec.get("name", "")).lower() in {"beach", "ocean", "sky"}
    }
    mapping_confident = True
    for label, region in label_masks.items():
        values, counts = np.unique(mask[region], return_counts=True)
        distribution = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
        total = int(sum(distribution.values()))
        dominant = max(distribution, key=distribution.get) if distribution else None
        expected = expected_by_name.get(label)
        purity = float(distribution.get(expected, 0) / total) if total and expected is not None else None
        if expected is None or (purity is not None and purity < 0.5):
            mapping_confident = False
        per_label[label] = {
            "polygon_count": int(shape_counts[label]),
            "pixels": total,
            "mask_value_distribution": distribution,
            "dominant_mask_value": dominant,
            "expected_mask_value": expected,
            "expected_purity": purity,
        }

    overlap: dict[str, int] = {}
    labels = sorted(label_masks)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            count = int((label_masks[left] & label_masks[right]).sum())
            if count:
                overlap[f"{left}+{right}"] = count

    union = (
        np.logical_or.reduce(list(label_masks.values()))
        if label_masks
        else np.zeros((h, w), dtype=bool)
    )
    uncovered = ~union
    uncovered_values, uncovered_counts = np.unique(mask[uncovered], return_counts=True)
    return {
        "shape_counts": dict(shape_counts),
        "exact_duplicate_polygons": exact_duplicate_polygons,
        "per_label": per_label,
        "inter_class_overlap_pixels": overlap,
        "mapping_confident": mapping_confident,
        "polygon_union_pixels": int(union.sum()),
        "coverage_fraction": float(union.mean()),
        "uncovered_pixels": int(uncovered.sum()),
        "uncovered_value_counts": {
            int(v): int(c) for v, c in zip(uncovered_values.tolist(), uncovered_counts.tolist())
        },
    }


def _near_duplicate_groups(
    records: list[PairRecord],
    *,
    phash_distance: int,
    dhash_distance: int,
) -> list[dict[str, Any]]:
    usable = [r for r in records if r.phash is not None and r.dhash is not None]
    if not usable:
        return []
    names = [r.key for r in usable]
    ph = np.stack([r.phash for r in usable]).astype(np.int16)
    dh = np.stack([r.dhash for r in usable]).astype(np.int16)
    ph_sum = ph.sum(axis=1)
    dh_sum = dh.sum(axis=1)
    ph_dist = ph_sum[:, None] + ph_sum[None, :] - 2 * (ph @ ph.T)
    dh_dist = dh_sum[:, None] + dh_sum[None, :] - 2 * (dh @ dh.T)
    close = (ph_dist <= int(phash_distance)) | (dh_dist <= int(dhash_distance))
    same_sha = np.array(
        [[a.exact_sha256 == b.exact_sha256 for b in usable] for a in usable], dtype=bool
    )
    close |= same_sha
    parent = list(range(len(usable)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    rows, cols = np.nonzero(np.triu(close, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        union(i, j)
    groups: defaultdict[int, list[str]] = defaultdict(list)
    for index, name in enumerate(names):
        groups[find(index)].append(name)
    result = []
    for members in groups.values():
        if len(members) > 1:
            result.append(
                {
                    "size": len(members),
                    "members": sorted(members),
                    "reason": "exact SHA-256 or perceptual-hash proximity",
                }
            )
    return sorted(result, key=lambda item: (-item["size"], item["members"][0]))


def _exact_duplicate_groups(records: list[PairRecord]) -> list[dict[str, Any]]:
    grouped: defaultdict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.exact_sha256:
            grouped[record.exact_sha256].append(record.key)
    return [
        {"sha256": digest, "size": len(members), "members": sorted(members)}
        for digest, members in grouped.items()
        if len(members) > 1
    ]


def _class_counts(records: Iterable[PairRecord]) -> dict[str, int]:
    counts: Counter[int] = Counter()
    for record in records:
        counts.update(record.mapped_values)
    return {
        "Beach": int(counts.get(0, 0)),
        "Ocean": int(counts.get(1, 0)),
        "Sky": int(counts.get(2, 0)),
        "ignore": int(counts.get(IGNORE_INDEX, 0)),
    }


def _image_preview(
    records: list[PairRecord], output_path: Path, *, seed: int = 42, count: int = 6
) -> str | None:
    if not records:
        return None
    rng = random.Random(seed)
    sample = rng.sample(records, min(count, len(records)))
    tiles = []
    for record in sample:
        image = np.asarray(Image.open(record.image_path).convert("RGB"))
        image = cv2.resize(image, (420, 236), interpolation=cv2.INTER_AREA)
        canvas = Image.new("RGB", (420, 262), (24, 24, 24))
        canvas.paste(Image.fromarray(image), (0, 26))
        ImageDraw.Draw(canvas).text((6, 6), record.key, fill=(240, 240, 240))
        tiles.append(canvas)
    width = 420 * min(3, len(tiles))
    rows = (len(tiles) + 2) // 3
    sheet = Image.new("RGB", (width, rows * 262), (18, 18, 18))
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % 3) * 420, (index // 3) * 262))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _pair_preview(
    records: list[PairRecord], output_path: Path, *, seed: int = 42, count: int = 9
) -> str | None:
    usable = [r for r in records if r.mask_path is not None and not r.image_corrupt]
    if not usable:
        return None
    rng = random.Random(seed)
    sample = rng.sample(usable, min(count, len(usable)))
    samples = []
    for record in sample:
        image = np.asarray(Image.open(record.image_path).convert("RGB"))
        mask = np.asarray(Image.open(record.mask_path))
        if image.shape[:2] != mask.shape[:2]:
            continue
        samples.append((record.key, image, mask))
    if not samples:
        return None
    return str(preview_grid(samples, output_path, columns=3))


def _inspect_dataset(
    name: str,
    entry: dict[str, Any],
    config: dict[str, Any],
    *,
    audit_dir: Path,
    seed: int,
) -> dict[str, Any]:
    root = resolve_path(entry.get("root", config["paths"]["datasets_root"]))
    image_dir = root / entry["image_dir"]
    mask_dir = root / entry["mask_dir"]
    image_exts = entry.get("image_exts", IMAGE_EXTS)
    mask_exts = entry.get("mask_exts", [".png"])
    source_mapping = {
        int(key): value for key, value in entry.get("annotation", {}).get("source_values", {}).items()
    }
    polygon_dir_raw = entry.get("polygon_dir")
    polygon_dir = root / polygon_dir_raw if polygon_dir_raw else None
    label_dir_raw = entry.get("label_dir")
    label_dir = root / label_dir_raw if label_dir_raw else None

    images = _collect_images(image_dir, image_exts)
    masks = _collect_images(mask_dir, mask_exts)
    keys = sorted(set(images) | set(masks))
    records: list[PairRecord] = []
    for key in keys:
        record = PairRecord(key=key, image_path=images.get(key, image_dir / f"{key}.missing"),
                            mask_path=masks.get(key))
        if key not in images:
            record.error = "missing image"
            records.append(record)
            continue
        # Inspect the image even when its mask is absent. A blocked dataset still
        # needs an honest format/size/duplicate report.
        if key not in masks:
            record.error = "missing mask"
        try:
            record.exact_sha256 = _sha256(record.image_path)
            record.phash, record.dhash = _perceptual_hashes(record.image_path)
        except Exception as exc:
            record.error = f"hash error: {exc}"
        try:
            with Image.open(record.image_path) as image:
                image.verify()
            with Image.open(record.image_path) as image:
                array = np.asarray(image.convert("RGB"))
                record.width, record.height = image.size
                record.image_mode = image.mode
                record.image_format = image.format
                record.image_bit_depth = _bit_depth(image.mode, np.asarray(image))
                record.image_channels = len(image.getbands())
                if "A" in image.getbands():
                    alpha = np.asarray(image.getchannel("A"))
                    record.alpha_min = int(alpha.min())
                    record.alpha_max = int(alpha.max())
        except Exception as exc:
            record.image_corrupt = str(exc)
        if record.mask_path is not None:
            try:
                with Image.open(record.mask_path) as mask:
                    mask.verify()
                with Image.open(record.mask_path) as mask:
                    mask_array = np.asarray(mask)
                    record.mask_mode = mask.mode
                    record.mask_bit_depth = _bit_depth(mask.mode, mask_array)
                    record.mask_size = mask.size
                    if mask_array.ndim != 2:
                        record.mask_corrupt = f"mask is not single-channel (shape {mask_array.shape})"
                    else:
                        values, counts = np.unique(mask_array, return_counts=True)
                        record.raw_values = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
                        record.mapped_values = {}
                        for value, count in record.raw_values.items():
                            spec = source_mapping.get(value)
                            if spec is None:
                                record.invalid_values[value] = count
                            else:
                                target = int(spec.get("target", IGNORE_INDEX))
                                record.mapped_values[target] = record.mapped_values.get(target, 0) + count
                        record.empty_mask = len([v for v in values.tolist() if v != IGNORE_INDEX]) == 0
            except Exception as exc:
                record.mask_corrupt = str(exc)
        if (
            record.width is not None
            and record.height is not None
            and record.mask_size is not None
        ):
            record.dimension_match = (record.width, record.height) == record.mask_size
        else:
            record.dimension_match = None
        records.append(record)

    # Polygon/mask semantic verification, only for pairs that have both.
    polygon_stats: dict[str, Any] = {}
    polygon_records = 0
    mapping_votes: defaultdict[str, Counter[int]] = defaultdict(Counter)
    inter_overlap: Counter[str] = Counter()
    coverage_values: list[float] = []
    uncovered_total = 0
    uncovered_by_class: Counter[int] = Counter()
    if polygon_dir and source_mapping:
        for record in records:
            if record.mask_path is None or record.image_corrupt or record.mask_corrupt:
                continue
            polygon_path = polygon_dir / f"{record.key}.json"
            stats = _polygon_analysis(record, polygon_path, source_mapping)
            if stats is None:
                continue
            if stats.get("error"):
                polygon_stats[record.key] = stats
                continue
            polygon_records += 1
            for label, info in stats.get("per_label", {}).items():
                mapping_votes[label].update(info["mask_value_distribution"])
            inter_overlap.update(stats.get("inter_class_overlap_pixels", {}))
            if stats.get("coverage_fraction") is not None:
                coverage_values.append(float(stats["coverage_fraction"]))
            uncovered_total += int(stats.get("uncovered_pixels", 0))
            for value, count in stats.get("uncovered_value_counts", {}).items():
                uncovered_by_class[int(value)] += int(count)
            polygon_stats[record.key] = {
                "mapping_confident": stats.get("mapping_confident"),
                "inter_class_overlap_pixels": stats.get("inter_class_overlap_pixels", {}),
                "exact_duplicate_polygons": stats.get("exact_duplicate_polygons", 0),
                "coverage_fraction": stats.get("coverage_fraction"),
                "uncovered_pixels": stats.get("uncovered_pixels", 0),
            }

    label_verification = {
        label: {
            "mask_value_distribution": dict(sorted(counter.items())),
            "dominant_mask_value": counter.most_common(1)[0][0] if counter else None,
            "dominant_purity": (
                counter.most_common(1)[0][1] / sum(counter.values()) if counter else None
            ),
        }
        for label, counter in sorted(mapping_votes.items())
    }
    expected_label_to_source = {
        str(spec.get("name", "")).lower(): source
        for source, spec in source_mapping.items()
        if str(spec.get("name", "")).lower() in {"beach", "ocean", "sky"}
    }
    mapping_verified = True
    mapping_errors: list[str] = []
    for label, expected_source in expected_label_to_source.items():
        info = label_verification.get(label)
        if not info or info["dominant_mask_value"] is None:
            mapping_errors.append(f"no polygon evidence for label {label!r}")
            mapping_verified = False
        elif int(info["dominant_mask_value"]) != int(expected_source):
            mapping_errors.append(
                f"{label}: expected mask value {expected_source}, "
                f"dominant value is {info['dominant_mask_value']}"
            )
            mapping_verified = False

    exact_dupes = _exact_duplicate_groups(records)
    near_dupes = _near_duplicate_groups(
        records,
        phash_distance=int(config["data"]["split"].get("near_duplicate_phash_distance", 6)),
        dhash_distance=int(config["data"]["split"].get("near_duplicate_dhash_distance", 4)),
    )
    paired = [r for r in records if r.mask_path is not None and not r.error]
    valid_pairs = [
        r
        for r in paired
        if not r.image_corrupt
        and not r.mask_corrupt
        and r.dimension_match
        and not r.invalid_values
    ]
    raw_mask_counts: Counter[int] = Counter()
    for record in paired:
        for value, count in record.raw_values.items():
            raw_mask_counts[int(value)] += int(count)
    class_counts = _class_counts(valid_pairs)
    total_pixels = sum(class_counts.values())
    class_fractions = {
        key: (value / total_pixels if total_pixels else None) for key, value in class_counts.items()
    }
    alpha_min = min((r.alpha_min for r in records if r.alpha_min is not None), default=None)
    alpha_max = max((r.alpha_max for r in records if r.alpha_max is not None), default=None)
    images_with_alpha = sum(1 for r in records if r.alpha_min is not None)
    non_opaque = sum(
        1
        for r in records
        if r.alpha_min is not None and (r.alpha_min < 255 or r.alpha_max < 255)
    )

    preview_dir = audit_dir / "previews"
    pair_preview = _pair_preview(paired, preview_dir / f"{name}_pairs.png", seed=seed)
    image_preview = _image_preview(records, preview_dir / f"{name}_images.png", seed=seed)

    documented_conflict = None
    if label_dir and (label_dir / "classes.json").exists():
        try:
            documented = json.loads((label_dir / "classes.json").read_text(encoding="utf-8"))
            documented_classes = documented.get("classes", {})
            documented_names = sorted(
                str(spec.get("name", "")).lower()
                for spec in documented_classes.values()
                if isinstance(spec, dict)
            )
            actual_labels = sorted(label_verification)
            actual_values = sorted({int(v) for r in paired for v in r.raw_values})
            documented_encoding = str(documented.get("description", ""))
            conflict = (
                documented_names != actual_labels
                or any(int(key) not in actual_values for key in documented_classes)
                or "rgb" in documented_encoding.lower()
            )
            documented_conflict = {
                "path": str(label_dir / "classes.json"),
                "documented_classes": documented_classes,
                "documented_names": documented_names,
                "actual_polygon_labels": actual_labels,
                "actual_mask_modes": sorted({r.mask_mode for r in paired if r.mask_mode}),
                "actual_raw_values": actual_values,
                "verdict": (
                    "CONFLICT: classes.json describes a different class schema/encoding "
                    "than the delivered PNG rasters. The verified polygon labels and actual "
                    "pixel values are authoritative; classes.json is stale and is not used "
                    "for conversion."
                    if conflict
                    else "consistent"
                ),
            }
        except Exception as exc:
            documented_conflict = {"path": str(label_dir / "classes.json"), "error": str(exc)}

    blocking: list[str] = []
    if not records:
        blocking.append("no image/mask files found")
    if not masks:
        blocking.append("mask directory is empty or missing")
    if any(r.error == "missing mask" for r in records):
        blocking.append(f"{sum(r.error == 'missing mask' for r in records)} images have no mask")
    if any(r.error == "missing image" for r in records):
        blocking.append(f"{sum(r.error == 'missing image' for r in records)} masks have no image")
    if any(r.image_corrupt for r in records):
        blocking.append(f"{sum(bool(r.image_corrupt) for r in records)} corrupt images")
    if any(r.mask_corrupt for r in records):
        blocking.append(f"{sum(bool(r.mask_corrupt) for r in records)} corrupt masks")
    if any(r.dimension_match is False for r in records):
        blocking.append(f"{sum(r.dimension_match is False for r in records)} dimension mismatches")
    if any(r.invalid_values for r in records):
        blocking.append("unmapped/invalid mask class ids")
    if source_mapping and not mapping_verified:
        blocking.append("source-to-target class mapping not verified by polygons")
    if not valid_pairs:
        blocking.append("no valid image/mask pairs")

    usable = not blocking
    return {
        "name": name,
        "display_name": entry.get("display_name", name),
        "root": str(root),
        "image_dir": str(image_dir),
        "mask_dir": str(mask_dir),
        "image_files": len(images),
        "mask_files": len(masks),
        "paired_files": len(paired),
        "valid_pairs": len(valid_pairs),
        "missing_masks": int(sum(r.error == "missing mask" for r in records)),
        "missing_images": int(sum(r.error == "missing image" for r in records)),
        "corrupt_images": int(sum(bool(r.image_corrupt) for r in records)),
        "corrupt_masks": int(sum(bool(r.mask_corrupt) for r in records)),
        "dimension_mismatches": int(sum(r.dimension_match is False for r in records)),
        "empty_masks": int(sum(bool(r.empty_mask) for r in records)),
        "invalid_class_id_images": int(sum(bool(r.invalid_values) for r in records)),
        "image_formats": dict(Counter(r.image_format for r in records if r.image_format)),
        "image_modes": dict(Counter(r.image_mode for r in records if r.image_mode)),
        "mask_modes": dict(Counter(r.mask_mode for r in paired if r.mask_mode)),
        "image_bit_depths": sorted({r.image_bit_depth for r in records if r.image_bit_depth}),
        "mask_bit_depths": sorted({r.mask_bit_depth for r in paired if r.mask_bit_depth}),
        "image_sizes": {
            f"{w}x{h}": count
            for (w, h), count in Counter(
                (r.width, r.height) for r in records if r.width and r.height
            ).most_common()
        },
        "images_with_alpha_channel": images_with_alpha,
        "non_opaque_images": non_opaque,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "raw_mask_values": {str(k): int(v) for k, v in sorted(raw_mask_counts.items())},
        "mapped_class_pixel_counts": class_counts,
        "mapped_class_fraction": class_fractions,
        "annotation_format": entry.get("annotation", {}).get("format"),
        "source_values": {str(k): v for k, v in sorted(source_mapping.items())},
        "mapping_verified": mapping_verified if source_mapping else None,
        "mapping_errors": mapping_errors,
        "polygon_records": polygon_records,
        "polygon_label_verification": label_verification,
        "polygon_inter_class_overlap_pixels": dict(inter_overlap),
        "polygon_union_coverage": {
            "min": min(coverage_values) if coverage_values else None,
            "median": float(np.median(coverage_values)) if coverage_values else None,
            "mean": float(np.mean(coverage_values)) if coverage_values else None,
            "images_below_80pct": int(sum(1 for value in coverage_values if value < 0.8)),
        },
        "uncovered_pixels_total": int(uncovered_total),
        "uncovered_pixels_by_mask_value": {
            str(value): int(count) for value, count in sorted(uncovered_by_class.items())
        },
        "unlabeled_policy": entry.get("annotation", {}).get("unlabeled_policy", "keep"),
        "documentation_conflict": documented_conflict,
        "exact_duplicate_groups": exact_dupes,
        "near_duplicate_groups": near_dupes,
        "blocking_issues": blocking,
        "usable": usable,
        "preview_pairs": pair_preview,
        "preview_images": image_preview,
        "pairs": [r.to_dict() for r in records],
    }


def _markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# CoastWatch SegFormer Dataset Audit\n")
    lines.append(f"Generated: `{report['generated_at']}`\n")
    status = report["verdict"]["status"]
    lines.append(f"## Verdict: **{status}**\n")
    lines.append(report["verdict"]["summary"] + "\n")
    lines.append("### Dataset usage\n")
    lines.append("| dataset | usable | images | masks | valid pairs | blocked reason |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for name, dataset in report["datasets"].items():
        reason = "; ".join(dataset["blocking_issues"]) or "—"
        lines.append(
            f"| `{name}` | {'yes' if dataset['usable'] else 'NO'} | "
            f"{dataset['image_files']} | {dataset['mask_files']} | "
            f"{dataset['valid_pairs']} | {reason} |"
        )
    lines.append("")
    for name, dataset in report["datasets"].items():
        lines.append(f"## {dataset['display_name']} (`{name}`)\n")
        lines.append(f"- Root: `{dataset['root']}`")
        lines.append(f"- Images: `{dataset['image_dir']}`")
        lines.append(f"- Masks: `{dataset['mask_dir']}`")
        lines.append(f"- Image formats: `{dataset['image_formats']}`")
        lines.append(f"- Image modes: `{dataset['image_modes']}`")
        lines.append(f"- Mask modes: `{dataset['mask_modes']}`")
        lines.append(f"- Image bit depths: `{dataset['image_bit_depths']}`")
        lines.append(f"- Mask bit depths: `{dataset['mask_bit_depths']}`")
        size_items = list(dataset["image_sizes"].items())
        size_preview = ", ".join(f"{key} ({value})" for key, value in size_items[:6])
        if len(size_items) > 6:
            size_preview += f", … {len(size_items) - 6} more"
        lines.append(f"- Image sizes: {size_preview or '—'}")
        lines.append(
            f"- Alpha channels: `{dataset['images_with_alpha_channel']}` "
            f"(non-opaque: `{dataset['non_opaque_images']}`, "
            f"range `{dataset['alpha_min']}`..`{dataset['alpha_max']}`)"
        )
        lines.append(f"- Raw mask values: `{dataset['raw_mask_values']}`")
        lines.append(f"- Mapped class counts: `{dataset['mapped_class_pixel_counts']}`")
        if dataset["mapped_class_fraction"]:
            fractions = ", ".join(
                f"{key}={value:.4%}"
                for key, value in dataset["mapped_class_fraction"].items()
                if value is not None
            )
            lines.append(f"- Mapped class fractions: {fractions}")
        lines.append(f"- Mapping verified from polygon JSON: `{dataset['mapping_verified']}`")
        if dataset["mapping_errors"]:
            lines.append(f"- Mapping errors: `{dataset['mapping_errors']}`")
        lines.append(f"- Polygon records checked: `{dataset['polygon_records']}`")
        if dataset["polygon_inter_class_overlap_pixels"]:
            lines.append(
                f"- Inter-class polygon overlap: `{dataset['polygon_inter_class_overlap_pixels']}` "
                "(the delivered raster resolves overlaps; this is an annotation ambiguity warning)"
            )
        coverage = dataset.get("polygon_union_coverage") or {}
        if coverage.get("median") is not None:
            lines.append(
                f"- Polygon union coverage of the image: median `{coverage['median']:.1%}`, "
                f"min `{coverage['min']:.1%}`, mean `{coverage['mean']:.1%}`; "
                f"images below 80%: `{coverage['images_below_80pct']}`"
            )
            lines.append(
                f"- Pixels outside every polygon: `{dataset['uncovered_pixels_total']:,}` "
                f"with mask values `{dataset['uncovered_pixels_by_mask_value']}` "
                f"(unlabeled policy: `{dataset['unlabeled_policy']}`)"
            )
        if dataset["documentation_conflict"]:
            lines.append(
                f"- **Documentation conflict**: `{dataset['documentation_conflict'].get('verdict')}` "
                f"({dataset['documentation_conflict'].get('path')})"
            )
        lines.append(
            f"- Exact duplicate groups: `{len(dataset['exact_duplicate_groups'])}`; "
            f"near-duplicate groups: `{len(dataset['near_duplicate_groups'])}`"
        )
        if dataset["preview_pairs"]:
            lines.append(f"\n![{name} pairs]({Path(dataset['preview_pairs']).name})")
        elif dataset["preview_images"]:
            lines.append(f"\n![{name} images (no masks)]({Path(dataset['preview_images']).name})")
        lines.append("")
    lines.append("## Verified class conversion (usable dataset)\n")
    lines.append("| source value | source meaning | target id | target class |")
    lines.append("|---:|---|---:|---|")
    for source, spec in report["verified_mapping"].items():
        target = int(spec["target"])
        target_name = "ignore" if target == IGNORE_INDEX else CLASS_NAMES[target]
        lines.append(
            f"| {source} | {spec.get('name', '')} | {target} | {target_name} |"
        )
    lines.append("")
    lines.append("## Known limitations\n")
    for note in report["verdict"]["limitations"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Full records\n")
    lines.append(
        "Per-pair records (dimensions, modes, raw value counts, hashes) are in "
        "`dataset_audit.json`.\n"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--audit-dir", default=None)
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Return non-zero if any configured dataset is unusable.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    audit_dir = resolve_path(args.audit_dir or config["paths"]["audit_dir"])
    audit_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("project", {}).get("seed", 42))

    dataset_reports: dict[str, Any] = {}
    for name, entry in config["data"]["datasets"].items():
        if not entry.get("enabled", True):
            continue
        print(f"[audit] inspecting {name} ...", flush=True)
        dataset_reports[name] = _inspect_dataset(name, entry, config, audit_dir=audit_dir, seed=seed)

    usable = [name for name, report in dataset_reports.items() if report["usable"]]
    blocked = [name for name, report in dataset_reports.items() if not report["usable"]]
    primary = usable[0] if usable else None

    limitations: list[str] = []
    for name in blocked:
        dataset = dataset_reports[name]
        limitations.append(
            f"`{name}` is blocked: {'; '.join(dataset['blocking_issues'])}. "
            "It is excluded rather than labelled synthetically."
        )
    if primary:
        dataset = dataset_reports[primary]
        limitations.append(
            f"Only {dataset['valid_pairs']} labeled pairs are usable. This is a small "
            "dataset; validation/test metrics will have wide confidence intervals."
        )
        limitations.append(
            "No session/camera metadata is available for the UUID/videoframe images. "
            "Leakage protection is limited to exact duplicates, perceptual near-duplicates, "
            "and keeping timestamp-named frames together."
        )
        limitations.append(
            "No foam/whitewater class exists in the source annotations; wave foam will be "
            "learned as Ocean or Beach, not as a separate class."
        )
        if dataset["polygon_inter_class_overlap_pixels"]:
            limitations.append(
                "Beach/Ocean polygon overlap exists in the JSON annotations. The delivered "
                "single-label PNG resolves it; no overlapping pixels remain in training masks."
            )
        uncovered = int(dataset.get("uncovered_pixels_total") or 0)
        uncovered_values = dataset.get("uncovered_pixels_by_mask_value", {})
        uncovered_sky = int(uncovered_values.get("0", 0))
        total_sky = int(dataset.get("raw_mask_values", {}).get("0", 0))
        if uncovered and total_sky and uncovered_sky / max(uncovered, 1) > 0.9:
            limitations.append(
                f"**Unlabeled background is encoded as 0 (Sky).** {uncovered:,} pixels "
                f"({uncovered_sky / max(total_sky, 1):.1%} of all source Sky pixels) lie "
                "outside every annotation polygon, i.e. the raster background defaults to "
                "the Sky id. This includes wet sand / swash-zone gaps between the Beach "
                "and Ocean polygons. With `unlabeled_policy: ignore_outside_polygons` "
                "these pixels are converted to ignore 255 rather than trained as Sky."
            )
        if dataset.get("unlabeled_policy") not in (None, "keep"):
            limitations.append(
                f"Label policy in effect: `{dataset['unlabeled_policy']}`. Pixels outside "
                "the verified polygon union are excluded from loss and metrics."
            )
        limitations.append(
            "Nighttime, low-light, cloud, sun-glare, wet-sand, vegetation, rock and tide "
            "coverage is not annotated as metadata. Condition breakdowns in evaluation are "
            "image-statistics heuristics, not ground-truth meteorological labels."
        )

    status = "PASS" if usable and not blocked else ("PASS_WITH_BLOCKED_DATASET" if usable else "FAIL")
    summary = (
        f"Usable datasets: {usable or 'none'}. Blocked datasets: {blocked or 'none'}. "
        "Training may proceed only on the usable datasets listed in `training_dataset`."
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root()),
        "config_path": str(resolve_path(args.config)),
        "seed": seed,
        "class_schema": {
            "names": list(CLASS_NAMES),
            "ids": {name: index for index, name in enumerate(CLASS_NAMES)},
            "colors": {name: list(CLASS_COLORS[index]) for index, name in enumerate(CLASS_NAMES)},
            "ignore_index": IGNORE_INDEX,
        },
        "verified_mapping": (
            dataset_reports[primary]["source_values"] if primary else {}
        ),
        "datasets": {name: {k: v for k, v in report.items() if k != "pairs"} for name, report in dataset_reports.items()},
        "verdict": {
            "status": status,
            "summary": summary,
            "usable_datasets": usable,
            "blocked_datasets": blocked,
            "training_dataset": primary,
            "limitations": limitations,
        },
    }

    json_path = audit_dir / "dataset_audit.json"
    md_path = audit_dir / "dataset_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")

    # Keep the full per-pair records out of the summary Markdown but available
    # for prepare_dataset.py and human inspection.
    full_path = audit_dir / "dataset_audit_full.json"
    full = dict(report)
    full["datasets"] = dataset_reports
    full_path.write_text(json.dumps(full, indent=2), encoding="utf-8")

    print(f"[audit] status: {status}")
    print(f"[audit] usable: {usable}; blocked: {blocked}")
    print(f"[audit] wrote {json_path}")
    print(f"[audit] wrote {md_path}")
    if args.require_all and blocked:
        return 2
    return 0 if usable else 1


if __name__ == "__main__":
    raise SystemExit(main())
