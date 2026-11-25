"""Build the unified three-class CoastWatch dataset.

Reads the audit JSON (which must show a usable dataset), copies source images
byte-for-byte, converts class-ID masks to the target schema, detects duplicate /
near-duplicate groups, and writes a manifest. Source datasets are never modified.

Run the audit first:
    python -m ml.scripts.audit_datasets
    python -m ml.scripts.prepare_dataset
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from ml.scripts.audit_datasets import _perceptual_hashes, _sha256
from ml.src.dataset import (
    CLASS_COLORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    VALID_CLASS_IDS,
    load_config,
    resolve_path,
    unified_paths,
)
from ml.src.visualization import overlay_mask, preview_grid


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _reset_generated_dirs(root: Path) -> None:
    for name in ("images", "masks", "splits", "previews"):
        target = root / name
        if target.exists():
            if not _is_within(target, root):
                raise RuntimeError(f"refusing to remove {target}: outside unified root")
            shutil.rmtree(target)
    for name in ("manifest.csv", "manifest.json", "class_mapping.json", "audit_report.json"):
        target = root / name
        if target.exists():
            target.unlink()


def _normalise_id(prefix: str, key: str, sha256: str) -> str:
    raw = f"{prefix}_{key.replace('/', '__')}"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    safe = re.sub(r"_+", "_", safe).strip("._-")
    if not safe:
        safe = f"{prefix}_{sha256[:12]}"
    return safe


def _build_lut(mapping: dict[int, dict[str, Any]]) -> np.ndarray:
    # 254 is a private "undefined" sentinel; it must never appear in output.
    lut = np.full(256, 254, dtype=np.uint8)
    for source, spec in mapping.items():
        target = int(spec.get("target", IGNORE_INDEX))
        if target not in (*VALID_CLASS_IDS, IGNORE_INDEX):
            raise ValueError(f"source value {source} maps to invalid target {target}")
        lut[int(source)] = target
    return lut


def _convert_mask(
    source: Path,
    destination: Path,
    lut: np.ndarray,
    *,
    valid_region: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    with Image.open(source) as image:
        if image.mode not in ("L", "P", "I", "I;16"):
            raise ValueError(
                f"{source}: mask mode {image.mode} is not a single-channel class-id image. "
                "RGB masks require a verified colour-to-class mapping."
            )
        array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"{source}: mask must be 2-D, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{source}: non-integer mask dtype {array.dtype}")
        array = array.astype(np.uint8)
    if valid_region is not None:
        if valid_region.shape != array.shape:
            raise ValueError(
                f"{source}: polygon coverage {valid_region.shape} differs from mask {array.shape}"
            )
        # Pixels outside every annotation polygon are unlabeled background.
        # Exclude them instead of letting the raster's 0 (Sky) default supervise.
        array = array.copy()
        array[~valid_region] = IGNORE_INDEX
    mapped = lut[array]
    undefined = sorted(int(v) for v in np.unique(array[mapped == 254]))
    if undefined:
        raise ValueError(
            f"{source}: unmapped source class ids {undefined}. "
            "Add an explicit verified mapping to ml/configs/segformer_b2.yaml first."
        )
    output = mapped.astype(np.uint8)
    output_values = sorted(int(v) for v in np.unique(output))
    invalid = set(output_values) - set(VALID_CLASS_IDS) - {IGNORE_INDEX}
    if invalid:
        raise ValueError(f"{source}: converted mask has invalid target ids {sorted(invalid)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output, mode="L").save(destination, format="PNG")
    counts = {
        **{str(class_id): int((output == class_id).sum()) for class_id in VALID_CLASS_IDS},
        "ignore": int((output == IGNORE_INDEX).sum()),
    }
    return output, counts


def _near_duplicate_groups(
    entries: list[dict[str, Any]],
    *,
    phash_distance: int,
    dhash_distance: int,
) -> list[list[str]]:
    """Union-find groups using SHA equality and perceptual-hash proximity."""
    if not entries:
        return []
    ph = np.stack([entry["phash"] for entry in entries]).astype(np.int16)
    dh = np.stack([entry["dhash"] for entry in entries]).astype(np.int16)
    ph_sum = ph.sum(axis=1)
    dh_sum = dh.sum(axis=1)
    ph_dist = ph_sum[:, None] + ph_sum[None, :] - 2 * (ph @ ph.T)
    dh_dist = dh_sum[:, None] + dh_sum[None, :] - 2 * (dh @ dh.T)
    close = (ph_dist <= phash_distance) | (dh_dist <= dhash_distance)
    sha = [entry["sha256"] for entry in entries]
    same_sha = np.array([[a == b for b in sha] for a in sha], dtype=bool)
    close |= same_sha
    parent = list(range(len(entries)))

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
    for index, entry in enumerate(entries):
        groups[find(index)].append(entry["key"])
    return sorted((sorted(members) for members in groups.values()), key=lambda m: m[0])


def _merge_timestamp_sequence(groups: list[list[str]]) -> list[list[str]]:
    """Keep all timestamp-named frames in one group when metadata is absent."""
    timestamp_keys = [m for group in groups for m in group if re.fullmatch(r"\d+", m)]
    if len(timestamp_keys) < 2:
        return groups
    timestamp_set = set(timestamp_keys)
    merged: list[list[str]] = []
    combined: list[str] = []
    for group in groups:
        if timestamp_set.intersection(group):
            combined.extend(group)
        else:
            merged.append(group)
    merged.append(sorted(combined))
    return sorted(merged, key=lambda m: m[0])


def _polygon_union(
    polygon_path: Path, width: int, height: int
) -> np.ndarray:
    """Rasterize the union of all annotation polygons.

    Uses the same polygon points that were verified against the delivered PNG
    in the audit. Pixels outside this union are unlabeled background.
    """
    payload = json.loads(polygon_path.read_text(encoding="utf-8"))
    declared_w = int(payload.get("imageWidth", width))
    declared_h = int(payload.get("imageHeight", height))
    if (declared_w, declared_h) != (width, height):
        raise ValueError(
            f"{polygon_path}: polygon size {(declared_w, declared_h)} differs from "
            f"mask size {(width, height)}"
        )
    union = np.zeros((height, width), dtype=bool)
    for shape in payload.get("shapes", []):
        points = shape.get("points") or []
        if len(points) < 3:
            continue
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(
            [(float(point[0]), float(point[1])) for point in points], fill=1
        )
        union |= np.asarray(canvas, dtype=bool)
    return union


def _mean_luma(path: Path) -> float:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return float("nan")
    return float(image.mean())


def _load_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"audit report not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--audit", default=None)
    parser.add_argument("--force", action="store_true", help="replace previously generated output")
    parser.add_argument("--dataset", default=None, help="usable audit dataset name (default: first)")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = unified_paths(config)
    audit_path = resolve_path(args.audit or (Path(config["paths"]["audit_dir"]) / "dataset_audit_full.json"))
    audit = _load_audit(audit_path)
    datasets = audit.get("datasets", {})
    usable = [name for name, item in datasets.items() if item.get("usable")]
    if not usable:
        raise RuntimeError("audit has no usable dataset; refusing to prepare anything")
    dataset_name = args.dataset or usable[0]
    if dataset_name not in datasets or not datasets[dataset_name].get("usable"):
        raise RuntimeError(
            f"dataset {dataset_name!r} is not usable; usable: {usable}. "
            "Fix the audit or pass an explicit usable --dataset."
        )
    dataset = datasets[dataset_name]
    source_mapping = {
        int(key): value for key, value in dataset.get("source_values", {}).items()
    }
    if not source_mapping:
        raise RuntimeError(f"{dataset_name}: no source_values mapping in audit")
    if not dataset.get("mapping_verified"):
        raise RuntimeError(f"{dataset_name}: audit did not verify the class mapping")
    lut = _build_lut(source_mapping)
    entry = config["data"]["datasets"][dataset_name]
    unlabeled_policy = str(
        dataset.get("unlabeled_policy") or entry.get("annotation", {}).get("unlabeled_policy", "keep")
    )
    if unlabeled_policy not in ("keep", "ignore_outside_polygons"):
        raise ValueError(
            f"unsupported unlabeled_policy {unlabeled_policy!r}; "
            "expected 'keep' or 'ignore_outside_polygons'"
        )
    polygon_dir: Path | None = None
    if unlabeled_policy == "ignore_outside_polygons":
        polygon_dir_raw = entry.get("polygon_dir")
        if not polygon_dir_raw:
            raise RuntimeError(
                "ignore_outside_polygons requires annotation.polygon_dir in the config"
            )
        polygon_dir = resolve_path(Path(dataset.get("root") or entry.get("root"))) / polygon_dir_raw
        if not polygon_dir.exists():
            raise FileNotFoundError(polygon_dir)

    root = paths["root"]
    if root.exists() and any(root.iterdir()) and not args.force:
        raise RuntimeError(
            f"{root} already exists and is not empty; pass --force to regenerate generated files"
        )
    if args.force and root.exists():
        _reset_generated_dirs(root)
    for directory in ("images", "masks", "splits", "previews"):
        (root / directory).mkdir(parents=True, exist_ok=True)

    records = [
        record
        for record in dataset.get("pairs", [])
        if record.get("mask") and not record.get("error") and not record.get("image_corrupt")
        and not record.get("mask_corrupt") and record.get("dimension_match") is not False
        and not record.get("invalid_values")
    ]
    if not records:
        raise RuntimeError(f"{dataset_name}: no valid pairs in audit")

    exact_groups = dataset.get("exact_duplicate_groups", [])
    exclude_keys: dict[str, str] = {}
    for group in exact_groups:
        members = sorted(group.get("members", []))
        for duplicate in members[1:]:
            exclude_keys[duplicate] = (
                f"exact duplicate of {members[0]} (SHA-256 {group.get('sha256', '')[:12]})"
            )

    included = [record for record in records if record["key"] not in exclude_keys]
    entries: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for record in included:
        key = record["key"]
        source_image = Path(record["image"])
        source_mask = Path(record["mask"])
        sha = record.get("exact_sha256") or _sha256(source_image)
        unique_id = _normalise_id(dataset_name, key, sha)
        if unique_id in used_ids:
            unique_id = f"{unique_id}_{sha[:8]}"
        used_ids.add(unique_id)
        extension = source_image.suffix.lower()
        if extension not in (".jpg", ".jpeg", ".png"):
            # Training loader normalises to RGB, so any Pillow-readable format is
            # acceptable, but a stable extension avoids ambiguity.
            extension = ".png"
        image_out = root / "images" / f"{unique_id}{extension}"
        mask_out = root / "masks" / f"{unique_id}.png"
        shutil.copy2(source_image, image_out)
        valid_region = None
        if polygon_dir is not None:
            polygon_path = polygon_dir / f"{key}.json"
            if not polygon_path.exists():
                raise FileNotFoundError(
                    f"{key}: polygon annotation missing at {polygon_path}; "
                    "refusing to guess which pixels are labelled"
                )
            valid_region = _polygon_union(
                polygon_path, int(record["width"]), int(record["height"])
            )
        converted, counts = _convert_mask(
            source_mask, mask_out, lut, valid_region=valid_region
        )
        if converted.shape[:2] != (record["height"], record["width"]):
            raise RuntimeError(f"{key}: converted mask dimensions changed unexpectedly")
        phash, dhash = _perceptual_hashes(source_image)
        if phash is None or dhash is None:
            raise RuntimeError(f"{key}: cannot compute perceptual hash")
        entries.append(
            {
                "id": unique_id,
                "key": key,
                "image": image_out.name,
                "mask": mask_out.name,
                "source_dataset": dataset_name,
                "source_image": str(source_image),
                "source_mask": str(source_mask),
                "width": int(record["width"]),
                "height": int(record["height"]),
                "sha256": sha,
                "phash": phash,
                "dhash": dhash,
                "mean_luma": _mean_luma(image_out),
                "beach_pixels": counts["0"],
                "ocean_pixels": counts["1"],
                "sky_pixels": counts["2"],
                "ignore_pixels": counts["ignore"],
                "image_mode": record.get("image_mode"),
                "mask_mode": "L",
                "group_id": "",
                "group_size": 0,
                "group_method": "",
            }
        )

    # Group exact/near duplicates and the timestamp sequence. Every group is
    # kept entirely in one split later, so adjacent frames cannot leak.
    distance_phash = int(config["data"]["split"].get("near_duplicate_phash_distance", 6))
    distance_dhash = int(config["data"]["split"].get("near_duplicate_dhash_distance", 4))
    groups = _near_duplicate_groups(
        entries, phash_distance=distance_phash, dhash_distance=distance_dhash
    )
    groups = _merge_timestamp_sequence(groups)
    key_to_entry = {entry["key"]: entry for entry in entries}
    for group_index, members in enumerate(groups, start=1):
        group_id = f"g{group_index:04d}"
        for key in members:
            entry = key_to_entry[key]
            entry["group_id"] = group_id
            entry["group_size"] = len(members)
            if len(members) > 1:
                has_timestamp = any(re.fullmatch(r"\d+", member) for member in members)
                entry["group_method"] = (
                    "timestamp_sequence+near_duplicate" if has_timestamp else "near_duplicate_cluster"
                )
            else:
                entry["group_method"] = "singleton"

    # Write manifest.
    fieldnames = [
        "id", "image", "mask", "source_dataset", "source_image", "source_mask",
        "width", "height", "sha256", "group_id", "group_size", "group_method",
        "beach_pixels", "ocean_pixels", "sky_pixels", "ignore_pixels", "mean_luma",
        "split",
    ]
    with paths["manifest"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for entry in sorted(entries, key=lambda item: item["id"]):
            row = {key: entry.get(key, "") for key in fieldnames}
            row["split"] = ""
            writer.writerow(row)

    manifest_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset_name,
        "source_root": dataset.get("root"),
        "source_image_dir": dataset.get("image_dir"),
        "source_mask_dir": dataset.get("mask_dir"),
        "num_source_records": len(records),
        "num_included": len(entries),
        "num_excluded_duplicates": len(exclude_keys),
        "excluded": [
            {"source_key": key, "reason": reason} for key, reason in sorted(exclude_keys.items())
        ],
        "verified_source_mapping": {str(k): v for k, v in sorted(source_mapping.items())},
        "unlabeled_policy": unlabeled_policy,
        "ignored_unlabeled_pixels": int(
            sum(entry_item["ignore_pixels"] for entry_item in entries)
        ),
        "target_schema": {
            "class_names": list(CLASS_NAMES),
            "class_ids": {name: index for index, name in enumerate(CLASS_NAMES)},
            "ignore_index": IGNORE_INDEX,
        },
        "groups": [
            {"group_id": f"g{index:04d}", "members": members}
            for index, members in enumerate(groups, start=1)
        ],
        "records": [
            {key: value for key, value in entry.items() if key not in ("phash", "dhash")}
            for entry in sorted(entries, key=lambda item: item["id"])
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")
    (root / "class_mapping.json").write_text(
        json.dumps(
            {
                "target_schema": manifest_json["target_schema"],
                "verified_source_mapping": manifest_json["verified_source_mapping"],
                "documentation_conflict": dataset.get("documentation_conflict"),
                "polygon_verification": dataset.get("polygon_label_verification"),
                "polygon_inter_class_overlap_pixels": dataset.get(
                    "polygon_inter_class_overlap_pixels"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Copy the audit summary into the unified dataset. Keep the original audit
    # output too; this is a traceability copy, not a replacement.
    summary = _load_audit(resolve_path(Path(config["paths"]["audit_dir"]) / "dataset_audit.json"))
    summary["conversion"] = {
        "dataset": dataset_name,
        "included": len(entries),
        "excluded_exact_duplicates": len(exclude_keys),
        "unlabeled_policy": unlabeled_policy,
        "ignored_unlabeled_pixels": manifest_json["ignored_unlabeled_pixels"],
        "source_mapping": manifest_json["verified_source_mapping"],
        "target_schema": manifest_json["target_schema"],
    }
    (root / "audit_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Previews: audit copy + a fresh sample from the converted dataset.
    preview_dir = paths["previews"]
    audit_preview = dataset.get("preview_pairs")
    if audit_preview and Path(audit_preview).exists():
        shutil.copy2(audit_preview, preview_dir / "audit_pairs.png")
    rng = random.Random(int(config.get("project", {}).get("seed", 42)))
    sample = rng.sample(entries, min(12, len(entries)))
    samples = []
    for entry in sample:
        image = np.asarray(Image.open(root / "images" / entry["image"]).convert("RGB"))
        mask = np.asarray(Image.open(root / "masks" / entry["mask"]))
        samples.append((entry["id"], image, mask))
    preview_grid(samples, preview_dir / "prepared_pairs.png", columns=3)

    print(f"[prepare] dataset: {dataset_name}")
    print(f"[prepare] included: {len(entries)}  excluded exact duplicates: {len(exclude_keys)}")
    print(f"[prepare] unlabeled policy: {unlabeled_policy}")
    if polygon_dir is not None:
        print(
            f"[prepare] ignore pixels from outside polygons: "
            f"{manifest_json['ignored_unlabeled_pixels']:,}"
        )
    print(f"[prepare] groups: {len(groups)}")
    print(f"[prepare] wrote {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
