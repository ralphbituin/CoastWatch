"""Build a combined multi-site dataset from the original SAM3 pairs and the
new per-site SAM3 collections, then write leakage-safe splits.

Source layouts handled without modifying either source:

1. Original flat dataset (already converted by ``prepare_dataset.py``)::

       coastwatch_segformer/images/<id>.(jpg|png)
       coastwatch_segformer/masks/<id>.png
       coastwatch_segformer/manifest.csv   # keeps its original train/val/test

2. New per-site layout added under ``image_sam3``::

       image_sam3/<site>/<timestamp>.png       # image
       image_sam3/<site>/<timestamp>.json      # polygons
       image_sam3/<site>/mask/<timestamp>.png  # source class ids 0=Sky 1=Ocean 2=Beach

New masks use the verified source order and the same background-as-Sky issue as
the original data: pixels outside the polygon union are converted to ignore 255.

Usage:
    python -m ml.scripts.prepare_multisite
    python -m ml.scripts.prepare_multisite --force
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from ml.scripts.audit_datasets import _perceptual_hashes, _sha256
from ml.scripts.prepare_dataset import (
    _build_lut,
    _convert_mask,
    _is_within,
    _mean_luma,
    _normalise_id,
    _polygon_union,
    _reset_generated_dirs,
)
from ml.src.dataset import (
    CLASS_COLORS,
    CLASS_NAMES,
    IGNORE_INDEX,
    load_config,
    resolve_path,
    unified_paths,
)
from ml.src.visualization import preview_grid

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
SKIP_DIRS = {"mask", "labels", "splits", "_preview"}
MANIFEST_FIELDS = [
    "id", "image", "mask", "source_dataset", "source_image", "source_mask",
    "width", "height", "sha256", "group_id", "group_size", "group_method",
    "beach_pixels", "ocean_pixels", "sky_pixels", "ignore_pixels", "mean_luma",
    "split", "site", "source_layout",
]


def _slug(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return value or "site"


def _site_dirs(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and path.name.lower() not in SKIP_DIRS
    )


def _site_files(site_dir: Path) -> list[tuple[Path, Path, Path]]:
    pairs: list[tuple[Path, Path, Path]] = []
    for image_path in sorted(site_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = image_path.stem
        mask_path = site_dir / "mask" / f"{stem}.png"
        polygon_path = site_dir / f"{stem}.json"
        pairs.append((image_path, mask_path, polygon_path))
    return pairs


def _rasterize_label(mask: np.ndarray, polygons: list[list[list[float]]]) -> np.ndarray:
    height, width = mask.shape
    union = np.zeros((height, width), dtype=bool)
    for points in polygons:
        if len(points) < 3:
            continue
        canvas = Image.new("L", (width, height), 0)
        ImageDraw.Draw(canvas).polygon(
            [(float(point[0]), float(point[1])) for point in points], fill=1
        )
        union |= np.asarray(canvas, dtype=bool)
    return union


def _verify_site_mapping(
    site: str,
    pairs: list[tuple[Path, Path, Path]],
    expected: dict[str, int],
) -> dict[str, Any]:
    """Verify the source-value mapping from the site's first polygon JSON."""
    if not pairs:
        raise RuntimeError(f"{site}: no pairs")
    _, mask_path, polygon_path = pairs[0]
    if not polygon_path.exists():
        raise FileNotFoundError(f"{site}: missing polygon {polygon_path}")
    source = np.asarray(Image.open(mask_path))
    payload = json.loads(polygon_path.read_text(encoding="utf-8"))
    labels: defaultdict[str, list[list[list[float]]]] = defaultdict(list)
    for shape in payload.get("shapes", []):
        label = str(shape.get("label", "")).strip().lower()
        points = shape.get("points") or []
        if label and len(points) >= 3:
            labels[label].append(points)
    evidence: dict[str, Any] = {}
    for label, polygons in labels.items():
        if label not in expected:
            continue
        region = _rasterize_label(source, polygons)
        values, counts = np.unique(source[region], return_counts=True)
        distribution = {int(v): int(c) for v, c in zip(values.tolist(), counts.tolist())}
        dominant = max(distribution, key=distribution.get) if distribution else None
        evidence[label] = {
            "expected_source_value": expected[label],
            "dominant_mask_value": dominant,
            "distribution": distribution,
        }
        if dominant != expected[label]:
            raise RuntimeError(
                f"{site}: polygon label {label!r} has dominant mask value {dominant}, "
                f"expected {expected[label]}. Refusing to convert with an unverified mapping."
            )
    required = {"sky", "ocean", "beach"}
    missing = required - set(evidence)
    if missing:
        raise RuntimeError(f"{site}: polygon labels missing verification evidence for {sorted(missing)}")
    return evidence


def _load_old_rows(old_root: Path) -> list[dict[str, str]]:
    manifest = old_root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"base manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{manifest} is empty")
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/segformer_b2_multisite.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = unified_paths(config)
    root = paths["root"]
    old_root = resolve_path(config["paths"]["base_unified_dataset"])
    sites_root = resolve_path(config["paths"]["sites_root"])
    source_mapping = {
        int(key): value
        for key, value in config["data"]["datasets"]["sam3"]["annotation"][
            "source_values"
        ].items()
    }
    lut = _build_lut(source_mapping)
    expected_by_name = {
        str(spec.get("name", "")).lower(): int(source)
        for source, spec in source_mapping.items()
        if str(spec.get("name", "")).lower() in {"beach", "ocean", "sky"}
    }
    if set(expected_by_name) != {"beach", "ocean", "sky"}:
        raise RuntimeError("config source_values must define beach, ocean and sky")

    if root.exists() and any(root.iterdir()) and not args.force:
        raise RuntimeError(f"{root} exists and is not empty; pass --force to regenerate")
    if args.force and root.exists():
        _reset_generated_dirs(root)
    for name in ("images", "masks", "splits", "previews"):
        (root / name).mkdir(parents=True, exist_ok=True)

    # ---- original dataset (already converted, preserve its split) ----------
    old_rows = _load_old_rows(old_root)
    old_shas: set[str] = set()
    rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for row in old_rows:
        image_path = old_root / "images" / row["image"]
        mask_path = old_root / "masks" / row["mask"]
        if not image_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"base pair missing: {image_path} / {mask_path}")
        unique_id = row["id"]
        if unique_id in used_ids:
            raise RuntimeError(f"duplicate base id {unique_id}")
        used_ids.add(unique_id)
        shutil.copy2(image_path, root / "images" / row["image"])
        shutil.copy2(mask_path, root / "masks" / row["mask"])
        old_shas.add(row["sha256"])
        rows.append(
            {
                "id": unique_id,
                "image": row["image"],
                "mask": row["mask"],
                "source_dataset": row.get("source_dataset", "sam3"),
                "source_image": row.get("source_image", ""),
                "source_mask": row.get("source_mask", ""),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "sha256": row["sha256"],
                "group_id": f"old:{row['group_id']}",
                "group_size": int(row["group_size"]),
                "group_method": row.get("group_method", ""),
                "beach_pixels": int(row["beach_pixels"]),
                "ocean_pixels": int(row["ocean_pixels"]),
                "sky_pixels": int(row["sky_pixels"]),
                "ignore_pixels": int(row["ignore_pixels"]),
                "mean_luma": float(row["mean_luma"]),
                "split": row["split"],
                "site": "Karon Beach (original)",
                "source_layout": "flat",
            }
        )

    # ---- new per-site collections -----------------------------------------
    site_split_cfg = config["data"]["split"].get("site_split", {})
    split_for_site: dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        for site in site_split_cfg.get(split_name, []):
            split_for_site[site] = split_name
    all_sites = _site_dirs(sites_root)
    unknown_sites = [site.name for site in all_sites if site.name not in split_for_site]
    for name in unknown_sites:
        split_for_site[name] = "train"

    site_reports: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    mapping_evidence: dict[str, Any] = {}
    coverage_values: list[float] = []
    for site_dir in all_sites:
        site = site_dir.name
        slug = _slug(site)
        pairs = _site_files(site_dir)
        if not pairs:
            continue
        evidence = _verify_site_mapping(site, pairs, expected_by_name)
        mapping_evidence[site] = evidence
        group_id = f"site:{slug}"
        site_counts = Counter()
        site_ignore = 0
        site_coverage: list[float] = []
        included = 0
        for image_path, mask_path, polygon_path in pairs:
            if not mask_path.exists():
                raise FileNotFoundError(f"{site}: missing mask {mask_path}")
            sha = _sha256(image_path)
            if sha in old_shas:
                excluded.append(
                    {"source_key": f"{site}/{image_path.stem}", "reason": "exact duplicate of base image"}
                )
                continue
            with Image.open(image_path) as image:
                width, height = image.size
            with Image.open(mask_path) as mask_image:
                if mask_image.size != (width, height):
                    raise ValueError(
                        f"{site}/{image_path.name}: image {width}x{height} and "
                        f"mask {mask_image.size} differ"
                    )
                source_mask = np.asarray(mask_image)
            values = set(int(v) for v in np.unique(source_mask))
            invalid = values - set(source_mapping)
            if invalid:
                raise ValueError(f"{site}/{image_path.name}: invalid source ids {sorted(invalid)}")
            valid_region = _polygon_union(polygon_path, width, height)
            site_coverage.append(float(valid_region.mean()))
            coverage_values.append(float(valid_region.mean()))
            unique_id = _normalise_id(f"site_{slug}", image_path.stem, sha)
            if unique_id in used_ids:
                unique_id = f"{unique_id}_{sha[:8]}"
            used_ids.add(unique_id)
            image_out = root / "images" / f"{unique_id}{image_path.suffix.lower()}"
            mask_out = root / "masks" / f"{unique_id}.png"
            shutil.copy2(image_path, image_out)
            _, counts = _convert_mask(
                mask_path, mask_out, lut, valid_region=valid_region
            )
            phash, dhash = _perceptual_hashes(image_out)
            if phash is None:
                raise RuntimeError(f"{site}/{image_path.name}: cannot compute perceptual hash")
            rows.append(
                {
                    "id": unique_id,
                    "image": image_out.name,
                    "mask": mask_out.name,
                    "source_dataset": f"sam3:{site}",
                    "source_image": str(image_path),
                    "source_mask": str(mask_path),
                    "width": width,
                    "height": height,
                    "sha256": sha,
                    "group_id": group_id,
                    "group_size": 0,  # filled after duplicate filtering
                    "group_method": "site_sequence",
                    "beach_pixels": counts["0"],
                    "ocean_pixels": counts["1"],
                    "sky_pixels": counts["2"],
                    "ignore_pixels": counts["ignore"],
                    "mean_luma": _mean_luma(image_out),
                    "split": split_for_site[site],
                    "site": site,
                    "source_layout": "site_mask_dir",
                }
            )
            site_counts.update(
                {
                    "Beach": int(counts["0"]),
                    "Ocean": int(counts["1"]),
                    "Sky": int(counts["2"]),
                    "ignore": int(counts["ignore"]),
                }
            )
            site_ignore += int(counts["ignore"])
            included += 1
        site_reports.append(
            {
                "site": site,
                "split": split_for_site[site],
                "source_pairs": len(pairs),
                "included": included,
                "excluded_duplicates": len(pairs) - included,
                "class_pixels": dict(site_counts),
                "ignore_pixels": int(site_ignore),
                "polygon_coverage_median": float(np.median(site_coverage)),
                "polygon_coverage_min": float(np.min(site_coverage)),
                "polygon_coverage_mean": float(np.mean(site_coverage)),
                "mapping_evidence": evidence,
            }
        )
        size = included
        for row in rows:
            if row["group_id"] == group_id:
                row["group_size"] = size

    if not rows:
        raise RuntimeError("no usable pairs")

    # Sort groups are assigned explicitly; verify every row has a split.
    for row in rows:
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"{row['id']}: invalid split {row['split']!r}")

    # ---- write manifest and splits ----------------------------------------
    with paths["manifest"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["id"]):
            writer.writerow(row)

    by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "val", "test")}
    for row in rows:
        by_split[row["split"]].append(row)
    for split_name, members in by_split.items():
        names = sorted(row["image"] for row in members)
        with (paths["splits"] / f"{split_name}.txt").open("w", encoding="utf-8") as handle:
            handle.write(
                f"# {split_name}: {len(names)} images; sites: "
                f"{', '.join(sorted({row['site'] for row in members}))}\n"
            )
            for name in names:
                handle.write(name + "\n")

    class_totals = {
        "Beach": int(sum(row["beach_pixels"] for row in rows)),
        "Ocean": int(sum(row["ocean_pixels"] for row in rows)),
        "Sky": int(sum(row["sky_pixels"] for row in rows)),
        "ignore": int(sum(row["ignore_pixels"] for row in rows)),
    }
    supervised = sum(class_totals[name] for name in CLASS_NAMES)
    split_reports = {
        split_name: {
            "images": len(members),
            "sites": sorted({row["site"] for row in members}),
            "class_pixels": {
                name: int(sum(row[{"Beach": "beach_pixels", "Ocean": "ocean_pixels", "Sky": "sky_pixels"}[name]] for row in members))
                for name in CLASS_NAMES
            },
        }
        for split_name, members in by_split.items()
    }
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_images": len(rows),
        "total_groups": len({row["group_id"] for row in rows}),
        "old_images": len(old_rows),
        "new_site_images": len(rows) - len(old_rows),
        "class_pixels": class_totals,
        "class_fraction": {
            name: (class_totals[name] / supervised if supervised else None)
            for name in CLASS_NAMES
        },
        "splits": split_reports,
        "sites": site_reports,
        "mapping_verified": True,
        "mapping_evidence": mapping_evidence,
        "excluded_duplicates": excluded,
        "polygon_coverage": {
            "median": float(np.median(coverage_values)) if coverage_values else None,
            "min": float(np.min(coverage_values)) if coverage_values else None,
            "mean": float(np.mean(coverage_values)) if coverage_values else None,
        },
        "policy": {
            "source_mapping": {str(k): v for k, v in sorted(source_mapping.items())},
            "unlabeled": "pixels outside the polygon union -> ignore 255",
            "old_splits": "preserved from the base manifest",
            "new_splits": "assigned per site; whole site kept in one split",
        },
        "warnings": (
            [f"site {name!r} had no explicit split and was assigned to train" for name in unknown_sites]
            + [
                "New site masks are automatic SAM3 annotations and have not been "
                "manually reviewed frame by frame; site-held-out test metrics are "
                "the only quality evidence."
            ]
        ),
    }
    (root / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (paths["splits"] / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (paths["splits"] / "README.md").write_text(
        "\n".join(
            [
                "# Combined multi-site split",
                "",
                f"- Images: {len(rows)}; groups: {summary['total_groups']}",
                f"- Train: {split_reports['train']['images']} images "
                f"({', '.join(split_reports['train']['sites'])})",
                f"- Val: {split_reports['val']['images']} images "
                f"({', '.join(split_reports['val']['sites'])})",
                f"- Test: {split_reports['test']['images']} images "
                f"({', '.join(split_reports['test']['sites'])})",
                "",
                "The original Karon dataset keeps its train/val/test split.",
                "Every new site is kept entirely in one split; sequences never leak.",
                "Pixels outside polygon annotations are ignore 255.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    # ---- preview ----------------------------------------------------------
    rng = np.random.default_rng(int(config.get("project", {}).get("seed", 42)))
    samples = []
    one_per_site: dict[str, dict[str, Any]] = {}
    for row in rows:
        one_per_site.setdefault(row["site"], row)
    for row in one_per_site.values():
        image = np.asarray(Image.open(root / "images" / row["image"]).convert("RGB"))
        mask = np.asarray(Image.open(root / "masks" / row["mask"]))
        samples.append((row["id"], image, mask))
    if samples:
        preview_grid(samples, paths["previews"] / "multisite_pairs.png", columns=3)

    print(f"[multisite] total images: {len(rows)}")
    print(f"[multisite] split counts: {{'train': {split_reports['train']['images']}, "
          f"'val': {split_reports['val']['images']}, 'test': {split_reports['test']['images']}}}")
    print(f"[multisite] class pixels: {class_totals}")
    print(f"[multisite] excluded duplicates: {len(excluded)}")
    print(f"[multisite] wrote {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
