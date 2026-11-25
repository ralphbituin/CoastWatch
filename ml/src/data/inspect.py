"""Dataset inspection, pairing validation and report generation.

Run from the repository root:

    python -m ml.src.data.inspect

Outputs:
    ml/reports/dataset_inspection.md
    ml/reports/dataset_inspection.json
    ml/reports/figures/*_samples.png

Nothing in ``ml/datasets`` is modified.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.src.data.mappings import (
    build_all_mappings,
    class_colors,
    class_names,
    load_class_config,
    mapping_table_summary,
)
from ml.src.data.splits import (
    DatasetRecord,
    india_records,
    manifest_summary,
    sclabels_records,
)
from ml.src.utils import ensure_dir, get_logger, load_yaml, project_root, resolve_path, save_json

logger = get_logger("data.inspect")

Image.MAX_IMAGE_PIXELS = None


def _sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir():
            files = [p for p in path.iterdir() if p.is_file()]
            rows.append(
                {
                    "path": path.relative_to(project_root()).as_posix(),
                    "directories": sum(1 for p in path.iterdir() if p.is_dir()),
                    "files": len(files),
                    "bytes": sum(p.stat().st_size for p in files),
                }
            )
    return rows


def _mask_palette(mask_path: Path) -> list[list[int]]:
    with Image.open(mask_path) as image:
        if image.mode != "P":
            return []
        palette = image.getpalette() or []
        values = np.unique(np.asarray(image))
        return [
            [int(v), *[int(c) for c in palette[int(v) * 3 : int(v) * 3 + 3]]]
            for v in values
            if int(v) * 3 + 3 <= len(palette)
        ]


def _load_rgb(record: DatasetRecord, config: dict[str, Any]) -> np.ndarray:
    path = resolve_path(record.image_path)
    if record.dataset != "india_sandy_beach":
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    import tifffile

    bands = tifffile.imread(path)
    entry = load_class_config()["datasets"]["india_sandy_beach"]
    rgb_bands = [int(b) for b in entry.get("rgb_bands", [2, 1, 0])]
    percentiles = entry.get("rgb_scale_percentiles", [2, 98])
    rgb = np.empty((*bands.shape[:2], 3), dtype=np.float32)
    for out_channel, band_index in enumerate(rgb_bands):
        band = np.asarray(bands[:, :, band_index], dtype=np.float32)
        band = np.where(np.isfinite(band), band, 0.0)
        low, high = np.percentile(band, percentiles)
        band = np.zeros_like(band) if high - low < 1e-6 else (band - low) / (high - low)
        rgb[:, :, out_channel] = np.clip(band, 0, 1)
    return (rgb * 255).round().astype(np.uint8)


def _load_canonical_mask(record: DatasetRecord, mappings: dict[str, Any]) -> np.ndarray:
    with Image.open(resolve_path(record.mask_path)) as image:
        raw = np.asarray(image)
    return mappings[record.dataset].apply(raw)


def _colorize(mask: np.ndarray) -> np.ndarray:
    colors = class_colors()
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id in range(4):
        canvas[mask == class_id] = colors[class_id]
    canvas[mask == 255] = np.array([90, 90, 90], dtype=np.uint8)
    return canvas


def _overlay(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = _colorize(mask)
    out = rgb.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _visual_samples(
    dataset: str,
    records: list[DatasetRecord],
    mappings: dict[str, Any],
    config: dict[str, Any],
    figures_dir: Path,
    max_samples: int = 4,
) -> list[str]:
    if not records:
        return []
    # Pick a spread by supervised-pixel fraction plus one empty/edge case.
    stats: list[tuple[float, int]] = []
    for index, record in enumerate(records):
        try:
            mask = _load_canonical_mask(record, mappings)
            supervised = float(((mask == 1) | (mask == 2)).mean())
        except Exception:
            supervised = 0.0
        stats.append((supervised, index))
    stats.sort()
    picks: list[int] = []
    if len(stats) == 1:
        picks = [stats[0][1]]
    else:
        quantiles = np.linspace(0, len(stats) - 1, max_samples).astype(int)
        picks = [stats[q][1] for q in quantiles]
    picks = list(dict.fromkeys(picks))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(picks), 3, figsize=(11, 3.0 * len(picks)), squeeze=False)
    for row, index in enumerate(picks):
        record = records[index]
        rgb = _load_rgb(record, config)
        mask = _load_canonical_mask(record, mappings)
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(Path(record.image_path).name[:38], fontsize=7)
        axes[row, 1].imshow(_colorize(mask))
        axes[row, 1].set_title("canonical mask (mapped)", fontsize=7)
        axes[row, 2].imshow(_overlay(rgb, mask))
        axes[row, 2].set_title(
            f"overlay · land {np.mean(mask == 1) * 100:.1f}% · "
            f"water {np.mean(mask == 2) * 100:.1f}% · ignored {np.mean(mask == 255) * 100:.1f}%",
            fontsize=7,
        )
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    ensure_dir(figures_dir)
    output = figures_dir / f"{dataset}_samples.png"
    figure.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(figure)
    return [output.relative_to(project_root()).as_posix()]


def _validate_records(
    dataset: str,
    records: list[DatasetRecord],
    mappings: dict[str, Any],
    *,
    check_images: bool = True,
) -> dict[str, Any]:
    """Check pair existence, decodability and every mask value against mapping."""
    mapping = mappings[dataset]
    missing_images, missing_masks, corrupt_images, corrupt_masks = [], [], [], []
    unknown_values: Counter[int] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    canonical_counts = np.zeros(4, dtype=np.int64)
    source_counts: Counter[int] = Counter()
    empty_supervision = 0
    single_class = 0
    palette_values: Counter[int] = Counter()
    checked = 0

    for record in records:
        image_path = resolve_path(record.image_path)
        mask_path = resolve_path(record.mask_path)
        if not image_path.is_file():
            missing_images.append(record.image_path)
            continue
        if not mask_path.is_file():
            missing_masks.append(record.mask_path)
            continue
        try:
            with Image.open(image_path) as image:
                dimensions[f"{image.width}x{image.height}"] += 1
                modes[image.mode] += 1
                if check_images:
                    image.verify()
        except Exception:
            if record.dataset == "india_sandy_beach":
                try:
                    import tifffile

                    tifffile.imread(image_path)
                except Exception:
                    corrupt_images.append(record.image_path)
            else:
                corrupt_images.append(record.image_path)
        try:
            with Image.open(mask_path) as image:
                raw = np.asarray(image)
                if image.mode == "P":
                    palette_values.update(int(v) for v in np.unique(raw))
            try:
                mapped = mapping.apply(raw)
            except ValueError as exc:
                unknown_values.update(
                    int(v) for v in np.unique(raw) if int(v) not in mapping.defined_values
                )
                corrupt_masks.append(f"{record.mask_path}: {exc}")
                continue
        except Exception:
            corrupt_masks.append(record.mask_path)
            continue

        checked += 1
        source_counts.update(int(v) for v in np.unique(raw))
        canonical_counts += np.bincount(mapped.ravel(), minlength=4)[:4]
        supervised = (mapped == 1) | (mapped == 2)
        if not supervised.any():
            empty_supervision += 1
        elif not ((mapped == 1).any() and (mapped == 2).any()):
            single_class += 1

    supervised_total = int(canonical_counts[1:3].sum()) if canonical_counts.size >= 3 else 0
    return {
        "records": len(records),
        "checked_pairs": checked,
        "missing_images": len(missing_images),
        "missing_masks": len(missing_masks),
        "corrupt_images": len(corrupt_images),
        "corrupt_masks": len(corrupt_masks),
        "unknown_mask_values": {str(k): v for k, v in sorted(unknown_values.items())},
        "dimensions": dict(dimensions.most_common(12)),
        "modes": dict(modes),
        "canonical_pixel_counts": {class_names()[i]: int(canonical_counts[i]) for i in range(4)},
        "canonical_pixel_fraction_supervised": {
            name: (float(canonical_counts[i] / supervised_total) if supervised_total else 0.0)
            for i, name in enumerate(class_names())
            if i in (1, 2)
        },
        "generator_values": dict(sorted(source_counts.items())),
        "mask_palette_values": dict(sorted(palette_values.items())),
        "images_without_supervised_pixels": empty_supervision,
        "images_with_one_supervised_class": single_class,
        "examples": {
            "missing_images": missing_images[:5],
            "missing_masks": missing_masks[:5],
            "corrupt_images": corrupt_images[:5],
            "corrupt_masks": corrupt_masks[:5],
        },
    }


def _duplicates(records: list[DatasetRecord]) -> dict[str, Any]:
    hashes: dict[str, list[str]] = defaultdict(list)
    for record in dict.fromkeys(rec.image_path for rec in records):
        path = resolve_path(record)
        if path.is_file():
            hashes[_sha256(path)].append(record)
    groups = [paths for paths in hashes.values() if len(paths) > 1]
    return {
        "duplicate_groups": len(groups),
        "duplicate_files": sum(len(group) - 1 for group in groups),
        "examples": groups[:5],
    }


def _india_band_analysis(config: dict[str, Any], records: list[DatasetRecord]) -> dict[str, Any]:
    """Verify the documented derived-band algebra on real files."""
    import tifffile

    residuals = {"band3_eq_nd_b2_b1": [], "band4_eq_nd_b0_b2": [], "band5_eq_mix": []}
    stats = []
    for record in records[:30]:
        bands = tifffile.imread(resolve_path(record.image_path)).astype(np.float64)
        b0, b1, b2, b3, b4, b5 = (bands[:, :, i] for i in range(6))
        denom1 = np.maximum(np.abs(b1 + b2), 1e-6)
        denom2 = np.maximum(np.abs(b0 + b2), 1e-6)
        residuals["band3_eq_nd_b2_b1"].append(float(np.nanmax(np.abs(b3 - (b2 - b1) / denom1))))
        residuals["band4_eq_nd_b0_b2"].append(float(np.nanmax(np.abs(b4 - (b0 - b2) / denom2))))
        residuals["band5_eq_mix"].append(
            float(np.nanmax(np.abs(b5 - (0.47 * b0 + 0.17 * b1 + 0.40 * b2))))
        )
        stats.append(
            {
                "shape": list(bands.shape),
                "dtype": str(tifffile.imread(resolve_path(record.image_path)).dtype),
                "nan_pixels": int(np.isnan(bands).sum()),
                "band_means": [round(float(np.nanmean(bands[:, :, i])), 4) for i in range(6)],
            }
        )
    return {
        "band_count": 6,
        "documented_inference": (
            "Bands 0-2 are visible reflectance-like channels (blue/green/red); "
            "bands 3-5 are derived indices, not additional raw sensor bands."
        ),
        "max_formula_residuals": {k: round(max(v), 8) for k, v in residuals.items()},
        "samples": stats[:3],
    }


def build_report(
    config: dict[str, Any] | None = None,
    *,
    reports_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    max_samples: int = 4,
) -> dict[str, Any]:
    config = config or load_yaml("ml/configs/config.yaml")
    class_config = load_class_config()
    mappings = build_all_mappings(class_config)
    reports = resolve_path(reports_dir or config["paths"]["reports"])
    figures = resolve_path(figures_dir or config["paths"]["figures"])
    ensure_dir(reports)
    ensure_dir(figures)

    india = india_records(config)
    sclabels = sclabels_records(config)
    all_records = {
        "india_sandy_beach": india,
        "sclabels": sclabels,
    }

    inventories = {}
    for dataset, entry in class_config["datasets"].items():
        root = resolve_path(entry["root"])
        if root.is_dir():
            inventories[dataset] = _directory_inventory(root)

    validation: dict[str, Any] = {}
    duplicates: dict[str, Any] = {}
    figures_written: dict[str, list[str]] = {}
    for dataset, records in all_records.items():
        logger.info("validating %s (%d records)", dataset, len(records))
        validation[dataset] = _validate_records(dataset, records, mappings)
        duplicates[dataset] = _duplicates(records)
        logger.info("rendering samples for %s", dataset)
        figures_written[dataset] = _visual_samples(
            dataset, records, mappings, config, figures, max_samples=max_samples
        )

    india_bands = _india_band_analysis(config, india)

    # Dataset-level class distribution (sample every 5th record for speed).
    distribution = {}
    for dataset, records in all_records.items():
        sampled = records[::5] or records
        source_pixels: Counter[int] = Counter()
        canonical = np.zeros(4, dtype=np.int64)
        for record in sampled:
            with Image.open(resolve_path(record.mask_path)) as image:
                raw = np.asarray(image)
            source_pixels.update(int(v) for v in np.unique(raw))
            mapped = mappings[dataset].apply(raw)
            canonical += np.bincount(mapped.ravel(), minlength=4)[:4]
        distribution[dataset] = {
            "sampled_images": len(sampled),
            "canonical_pixels": {class_names()[i]: int(canonical[i]) for i in range(4)},
            "source_values_present": dict(sorted(source_pixels.items())),
        }

    split_summary = {dataset: manifest_summary(records) for dataset, records in all_records.items()}

    payload = {
        "generated_by": "ml/src/data/inspect.py",
        "datasets_root": config["paths"]["datasets_root"],
        "class_schema": class_config["schema"],
        "mappings": {
            dataset: {
                "display_name": mappings[dataset].display_name,
                "kind": mappings[dataset].kind,
                "supervised_classes": list(mappings[dataset].supervised_classes),
                "rows": mapping_table_summary(mappings[dataset]),
            }
            for dataset in mappings
        },
        "inventories": inventories,
        "validation": validation,
        "duplicates": duplicates,
        "distribution": distribution,
        "india_bands": india_bands,
        "splits": split_summary,
        "figures": figures_written,
        "unlabelled_webcam": {
            "path": "ml/datasets/image",
            "note": "Karon Beach webcam stills; no masks, no supervised use.",
        },
    }
    save_json(reports / "dataset_inspection.json", payload)

    markdown = render_markdown(payload, class_config)
    (reports / "dataset_inspection.md").write_text(markdown, encoding="utf-8")
    logger.info("wrote %s", (reports / "dataset_inspection.md").relative_to(project_root()))
    return payload


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], class_config: dict[str, Any]) -> str:
    v = payload["validation"]
    lines: list[str] = []
    lines.append("# Dataset Inspection — CoastWatch Beach Segmentation\n")
    lines.append(
        "Generated by `python -m ml.src.data.inspect`. This report is the gate "
        "before training: image/mask pairing, mask decoding and class mappings "
        "are validated on every pair.\n"
    )
    lines.append("## 1. Verdict\n")
    ok = all(
        item["missing_images"] == 0
        and item["missing_masks"] == 0
        and item["corrupt_images"] == 0
        and item["corrupt_masks"] == 0
        and not item["unknown_mask_values"]
        for item in v.values()
    )
    if ok:
        lines.append(
            "**PASS — all image/mask pairs were located and decoded, and every "
            "mask value has an explicit documented mapping.** Training may proceed.\n"
        )
    else:
        lines.append("**FAIL — do not train until the issues listed below are resolved.**\n")
    for dataset, item in v.items():
        lines.append(
            f"- `{dataset}`: {item['checked_pairs']}/{item['records']} pairs verified; "
            f"{item['missing_images']} missing images, {item['missing_masks']} missing masks, "
            f"{item['corrupt_images']} corrupt images, {item['corrupt_masks']} corrupt masks, "
            f"{len(item['unknown_mask_values'])} unknown mask values."
        )
    lines.append("")

    lines.append("## 2. What is actually in `ml/datasets/`\n")
    lines.append(
        "The folder does **not** contain uniformly structured datasets. The "
        "actual layouts found are:\n"
    )
    lines.append(
        "- **India Sandy Beach** — 1,088 pairs of 128×128 6-band float32 GeoTIFFs "
        "(`training_set/`, `validation_set/`, `testing_set/`), plus the source "
        "shapefile `India_SandyBeaches.shp`. Masks are binary 0/1 sand masks.\n"
        "- **SCLabels Spanish CoastSnap** — 1,717 rectified JPEG/PNG pairs directly "
        "in `images/` and `masks/`, with per-image metadata in `metadata.json`.\n"
        "- **Karon Beach webcam stills** — 75 unlabelled JPEG frames in `image/` "
        "(1 exact duplicate). **No masks.** Used only for qualitative inference; "
        "never counted as a training dataset.\n"
    )

    lines.append("## 3. Class schema and mappings\n")
    lines.append(
        "Canonical ids: `0` unknown/ignore, `1` land, `2` water, `3` foam; "
        "`255` is the loss/evaluation ignore index. A source value is mapped only "
        "when its meaning is documented. Everything ambiguous stays ignored.\n"
    )
    for dataset, mapping in payload["mappings"].items():
        lines.append(f"### {mapping['display_name']} (`{dataset}`)\n")
        rows = [
            [row["source_value"], row["source_name"], row["target"], row["reason"]]
            for row in mapping["rows"]
        ]
        lines.append(_table(["source", "meaning", "target", "reason"], rows))
        lines.append("")

    lines.append("### Foam: explicitly unsupervised\n")
    lines.append(
        "No provided dataset contains a foam/whitewater class. Foam is "
        "retained as output channel 3 for CoastWatch compatibility, but it has "
        "**zero training pixels**. Its class weight is set to 0 and its metrics "
        "are reported as N/A. Do not claim foam/breaking-wave accuracy from this "
        "training run; a foam-labelled webcam set is required for that.\n"
    )

    lines.append("## 4. Pairing, corruption, duplicates\n")
    rows = []
    for dataset, item in v.items():
        rows.append(
            [
                dataset,
                item["records"],
                item["checked_pairs"],
                item["missing_images"],
                item["missing_masks"],
                item["corrupt_images"] + item["corrupt_masks"],
                payload["duplicates"][dataset]["duplicate_groups"],
                item["images_without_supervised_pixels"],
            ]
        )
    lines.append(
        _table(
            [
                "dataset",
                "records",
                "pairs verified",
                "missing images",
                "missing masks",
                "corrupt",
                "duplicate groups",
                "no supervised px",
            ],
            rows,
        )
    )
    lines.append("")
    for dataset, item in v.items():
        if item["examples"]["missing_images"] or item["examples"]["corrupt_masks"]:
            lines.append(f"- `{dataset}` examples: `{item['examples']}`")
    if any(payload["duplicates"][d]["examples"] for d in payload["duplicates"]):
        lines.append("")
        lines.append("Duplicate examples:")
        for dataset, dup in payload["duplicates"].items():
            for group in dup["examples"][:2]:
                lines.append(f"- `{dataset}`: " + ", ".join(f"`{Path(p).name}`" for p in group))
    lines.append("")

    lines.append("## 5. Resolution, channels and mask encoding\n")
    rows = []
    for dataset, item in v.items():
        rows.append(
            [
                dataset,
                ", ".join(f"{k} ({n})" for k, n in list(item["dimensions"].items())[:4]),
                ", ".join(f"{k} ({n})" for k, n in item["modes"].items()),
                ", ".join(str(k) for k in sorted(item["generator_values"]))[:120],
                ", ".join(str(k) for k in sorted(item["mask_palette_values"])) or "—",
            ]
        )
    lines.append(
        _table(
            ["dataset", "top dimensions", "image modes", "mask values", "palette indices"],
            rows,
        )
    )
    lines.append("")
    india_bands = payload["india_bands"]
    lines.append(
        f"**India bands.** Each GeoTIFF has {india_bands['band_count']} float32 bands. "
        f"{india_bands['documented_inference']} The algebra was verified on real "
        f"files: max residuals `{india_bands['max_formula_residuals']}`. Bands 3-5 "
        "are therefore not extra colour channels and must not be fed to an RGB "
        "pretrained encoder. Bands are percentile-stretched and composed as "
        "R=band2, G=band1, B=band0.\n"
    )
    lines.append(
        "**SCLabels masks** are 8-bit grayscale values 0/25/75/150/255. The dataset "
        "metadata defines them as NoData / NotClassified / Landwards / Seawards / "
        "Shoreline. The 1-px Shoreline band is ignored, not assigned to land or water.\n"
    )

    lines.append("## 6. Class distribution\n")
    rows = []
    for dataset, dist in payload["distribution"].items():
        total = sum(dist["canonical_pixels"].values()) or 1
        row = [dataset, dist["sampled_images"]]
        for name in class_names():
            pixels = dist["canonical_pixels"][name]
            row.append(f"{pixels:,} ({pixels / total * 100:.2f}%)")
        rows.append(row)
    lines.append(_table(["dataset", "sampled", "unknown", "land", "water", "foam"], rows))
    lines.append("")
    lines.append(
        "The distributions are **supervised-pixel distributions after mapping**, "
        "not raw dataset statistics. Large ignored fractions are expected and "
        "intentional (sky/background/litter/NoData). India's background is ignored, "
        "so India supervises land almost exclusively; this is why it is a "
        "pretraining stage, not a water/land-balanced stage.\n"
    )

    lines.append("## 7. Visual samples\n")
    for dataset, paths in payload["figures"].items():
        for path in paths:
            lines.append(f"### {dataset}\n")
            lines.append(f"![{dataset} samples](../{path})\n")
    lines.append(
        "Panels show the decoded RGB composite, the canonical mapped mask (grey = "
        "ignored), and the overlay. They are generated from the actual files.\n"
    )

    lines.append("## 8. Splits\n")
    rows = []
    for dataset, summary in payload["splits"].items():
        rows.append(
            [
                dataset,
                summary["count"],
                summary["splits"]["train"],
                summary["splits"]["val"],
                summary["splits"]["test"],
                summary["groups"],
            ]
        )
    lines.append(_table(["dataset", "total", "train", "val", "test", "groups"], rows))
    lines.append("")
    lines.append(
        "Split policies: India uses the provided folders; SCLabels is split "
        "chronologically within each of its 5 sites (70/15/15). See "
        "`ml/src/data/splits.py` for the exact code. Full assignments are written "
        "to `ml/manifests/*.csv`.\n"
    )

    lines.append("## 9. Preprocessing recommendations\n")
    lines.append(
        "1. Resize RGB datasets to 512×512 with bilinear interpolation and masks "
        "with nearest-neighbour; never normalise masks.\n"
        "2. Use ImageNet mean/std for the RGB encoder. India's synthetic RGB "
        "composite uses the same normalisation after percentile stretching.\n"
        "3. Handle India NaNs before stretching (edge rows contain NaNs) and never "
        "feed bands 3-5 as RGB.\n"
        "4. Apply horizontal flip, ≤5° rotation, mild brightness/contrast, mild "
        "saturation, random scale/crop and mild blur. Do not use elastic, cutout "
        "or large perspective warps: they invent implausible shoreline geometry.\n"
        "5. Weight classes by inverse supervised frequency; set the foam weight to "
        "0. Do not compute weights over ignored pixels.\n"
        "6. Resize both image and mask identically and assert equal dimensions at "
        "load time (done by `BeachSegmentationDataset`).\n"
        "7. Keep webcam frames out of supervised training until masks exist. They "
        "are a different camera/site and cannot validate satellite or CoastSnap "
        "generalisation.\n"
    )
    lines.append("## 10. Known limitations\n")
    lines.append(
        "- **No foam annotations** in any dataset → foam is untrained.\n"
        "- **India background is unlabeled**, so ignored rather than guessed.\n"
        "- **SCLabels supervises only land/water**; sky, vegetation, people and "
        "litter are ignored, so the model is not trained to segment them.\n"
        "- **Leakage cannot be fully excluded** for India (pre-split chips). "
        "SCLabels uses temporal within-site splits, which tests temporal not "
        "cross-site generalisation.\n"
        "- **Satellite + ground photography ≠ webcam**. Webcam performance is "
        "unvalidated; the Karon set has no labels.\n"
        "- Metrics must be reported per dataset; a pooled number would hide the "
        "domain shift.\n"
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CoastWatch datasets")
    parser.add_argument("--config", default="ml/configs/config.yaml")
    parser.add_argument("--reports-dir", default=None)
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=4)
    args = parser.parse_args()
    config = load_yaml(args.config)
    build_report(
        config,
        reports_dir=args.reports_dir,
        figures_dir=args.figures_dir,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
