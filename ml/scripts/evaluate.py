"""Evaluate the best SegFormer checkpoint on the held-out test split.

The test split is read-only here and is never used for checkpoint selection.
Metrics are computed at native image resolution after bilinear upsampling of the
1/4-resolution SegFormer logits. Shoreline geometry is extracted by the separate
``ml.src.shoreline_extraction`` module.

Usage:
    python -m ml.scripts.evaluate
    python -m ml.scripts.evaluate --checkpoint ml/checkpoints/segformer_b2/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import cKDTree

from ml.src.dataset import (
    CoastWatchSegFormerDataset,
    build_eval_transform,
    load_config,
    load_image_rgb,
    load_mask,
    make_loader,
    read_split_file,
    resolve_path,
    unified_paths,
)
from ml.src.metrics import (
    CLASS_NAMES,
    IGNORE_INDEX,
    ConfusionMatrix,
    confusion_matrix_figure,
)
from ml.src.model import load_checkpoint
from ml.src.shoreline_extraction import ShorelineResult, extract_shoreline
from ml.src.visualization import (
    CLASS_COLORS,
    colorize_mask,
    draw_shoreline,
    overlay_mask,
    panel,
    prediction_panel,
    save_panel,
)


def _condition_tags(
    image: np.ndarray, config: dict[str, Any], gt_mask: np.ndarray | None = None
) -> list[str]:
    cfg = config.get("evaluation", {}).get("conditions", {})
    rgb = image.astype(np.float32)
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mean_luma = float(luma.mean())
    glare_fraction = float((luma >= float(cfg.get("glare_luma", 245))).mean())
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mean_saturation = float(hsv[:, :, 1].mean())
    tags: list[str] = []
    if mean_luma <= float(cfg.get("night_max_luma", 50)):
        tags.append("night")
    elif mean_luma <= float(cfg.get("low_light_max_luma", 90)):
        tags.append("low_light")
    elif mean_luma >= float(cfg.get("bright_min_luma", 170)):
        tags.append("bright_sunlight")
    else:
        tags.append("normal_light")
    if glare_fraction >= float(cfg.get("glare_fraction", 0.02)):
        tags.append("strong_glare")
    if mean_saturation < 30:
        tags.append("overcast_or_low_saturation")
    elif mean_saturation > 100:
        tags.append("high_saturation")
    if gt_mask is not None:
        valid = gt_mask != IGNORE_INDEX
        if valid.any():
            for class_id, name in enumerate(CLASS_NAMES):
                fraction = float(((gt_mask == class_id) & valid).sum() / valid.sum())
                if name == "Beach" and fraction < 0.05:
                    tags.append("small_beach_visible_fraction")
                if name == "Ocean" and fraction > 0.5:
                    tags.append("ocean_dominant")
    return tags


def _shoreline_compare(
    gt_result: ShorelineResult,
    pred_result: ShorelineResult,
) -> dict[str, Any]:
    gt = {int(round(x)): float(y) for x, y in gt_result.per_column.tolist()}
    pred = {int(round(x)): float(y) for x, y in pred_result.per_column.tolist()}
    shared = sorted(set(gt) & set(pred))
    result: dict[str, Any] = {
        "gt_valid": bool(gt_result.metadata.get("available")),
        "pred_valid": bool(pred_result.metadata.get("available")),
        "shared_columns": len(shared),
        "gt_columns": int(len(gt)),
        "pred_columns": int(len(pred)),
        "mae_px": None,
        "rmse_px": None,
        "bias_px": None,
        "p95_abs_px": None,
        "chamfer_px": None,
        "chamfer_p95_px": None,
        "interface_median_px": None,
        "interface_p95_px": None,
        "gt_segments": int(gt_result.metadata.get("segments", 0)),
        "pred_segments": int(pred_result.metadata.get("segments", 0)),
    }
    if shared:
        errors = np.array([pred[x] - gt[x] for x in shared], dtype=np.float64)
        result.update(
            {
                "mae_px": float(np.mean(np.abs(errors))),
                "rmse_px": float(np.sqrt(np.mean(errors**2))),
                "bias_px": float(np.mean(errors)),
                "p95_abs_px": float(np.percentile(np.abs(errors), 95)),
            }
        )
    # Symmetric nearest-neighbour distance between the full interface pixel
    # sets. This stays meaningful when a frame has two real waterlines and the
    # single-valued per-column metric switches branches.
    if gt_result.points.size and pred_result.points.size:
        gt_points = gt_result.points.astype(np.float64)
        pred_points = pred_result.points.astype(np.float64)
        pred_to_gt = cKDTree(gt_points).query(pred_points, k=1)[0]
        gt_to_pred = cKDTree(pred_points).query(gt_points, k=1)[0]
        combined = np.concatenate([pred_to_gt, gt_to_pred])
        result["chamfer_px"] = float(
            0.5 * (pred_to_gt.mean() + gt_to_pred.mean())
        )
        result["chamfer_p95_px"] = float(np.percentile(combined, 95))
        # The mean above is sensitive to a handful of spurious far-shore or
        # rock-boundary pixels. The median is a robust summary of how close the
        # two interfaces are for the bulk of the waterline.
        result["interface_median_px"] = float(np.median(combined))
        result["interface_p95_px"] = float(np.percentile(combined, 95))
    return result


def _aggregate_metrics(
    confusion: ConfusionMatrix,
    *,
    split: str,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    payload = confusion.compute()
    payload["split"] = split
    if tags:
        payload["condition_tags"] = tags
    return payload


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# CoastWatch SegFormer Evaluation\n")
    lines.append(f"Generated: `{report['generated_at']}`\n")
    lines.append(f"- Checkpoint: `{report['checkpoint']}`")
    lines.append(f"- Split: `{report['split']}` ({report['num_images']} images)")
    lines.append(f"- Metric resolution: native image resolution after logit upsampling")
    lines.append("")
    primary = report.get("primary_metrics")
    if primary is None:
        primary = {
            "beach_iou": report["metrics"]["per_class"]["Beach"]["iou"],
            "mean_iou": report["metrics"]["mean_iou"],
            "shoreline_mae_px": {
                "mean": report.get("shoreline", {}).get("mean_mae_px"),
                "median": report.get("shoreline", {}).get("median_mae_px"),
                "p95": report.get("shoreline", {}).get("p95_mae_px"),
            },
        }
    lines.append("## Primary metrics\n")
    lines.append(
        "Beach IoU and shoreline MAE are the primary quality signals for the "
        "application. Mean IoU is retained for continuity but is dominated by the "
        "large, easy Ocean and Sky classes.\n"
    )
    lines.append("| metric | value |")
    lines.append("|---|---:|")
    lines.append(f"| Beach IoU | {_fmt(primary.get('beach_iou'))} |")
    shore_mae = primary.get("shoreline_mae_px", {})
    lines.append(f"| Shoreline MAE, mean | {_fmt(shore_mae.get('mean'))} px |")
    lines.append(f"| Shoreline MAE, median | {_fmt(shore_mae.get('median'))} px |")
    lines.append(f"| Shoreline MAE, p95 | {_fmt(shore_mae.get('p95'))} px |")
    lines.append(f"| mean IoU (dominated by easy classes) | {_fmt(primary.get('mean_iou'))} |")
    lines.append("")
    lines.append("## Overall test metrics\n")
    lines.append("| class | IoU | Dice | precision | recall | support px |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, info in report["metrics"]["per_class"].items():
        lines.append(
            f"| {name} | {_fmt(info['iou'])} | {_fmt(info['dice'])} | "
            f"{_fmt(info['precision'])} | {_fmt(info['recall'])} | "
            f"{info['support_pixels']:,} |"
        )
    lines.append(
        f"| **mean (dominated by easy classes)** | **{_fmt(report['metrics']['mean_iou'])}** | "
        f"**{_fmt(report['metrics']['mean_dice'])}** | | | "
        f"**{report['metrics']['supervised_pixels']:,}** px |"
    )
    lines.append("")
    lines.append(f"Pixel accuracy: **{_fmt(report['metrics']['pixel_accuracy'])}**")
    lines.append("")
    lines.append("## Confusion matrix\n")
    lines.append(f"![confusion matrix](confusion_matrix.png)\n")
    lines.append("## Condition coverage (image-statistics heuristics)\n")
    lines.append(
        "These tags are **not** ground-truth weather labels. They are derived from "
        "luminance, glare and saturation. A condition with zero images is not "
        "represented and must not be claimed as supported.\n"
    )
    lines.append("| tag | images | mean IoU | pixel accuracy |")
    lines.append("|---|---:|---:|---:|")
    for tag, info in report["conditions"].items():
        metrics = info["metrics"]
        lines.append(
            f"| {tag} | {info['images']} | "
            f"{_fmt(metrics['mean_iou']) if metrics else 'n/a'} | "
            f"{_fmt(metrics['pixel_accuracy']) if metrics else 'n/a'} |"
        )
    lines.append("")
    unrepresented = [tag for tag, info in report["conditions"].items() if info["images"] == 0]
    lines.append(
        "Unrepresented condition tags: "
        + (", ".join(f"`{tag}`" for tag in unrepresented) if unrepresented else "none")
        + ".\n"
    )
    lines.append("## Shoreline extraction\n")
    shoreline = report["shoreline"]
    lines.append(
        f"- Predicted shoreline available: {shoreline['pred_valid_images']}/"
        f"{report['num_images']} test images"
    )
    lines.append(
        f"- GT shoreline available: {shoreline['gt_valid_images']}/"
        f"{report['num_images']}"
    )
    lines.append(
        f"- Mean per-column MAE: {_fmt(shoreline['mean_mae_px'])} px; "
        f"median: {_fmt(shoreline['median_mae_px'])} px; "
        f"p95: {_fmt(shoreline['p95_mae_px'])} px"
    )
    lines.append(
        "- Per-column MAE compares only image columns where both the ground-truth "
        "and predicted polylines exist. Some frames contain a far shore as well as "
        "the near waterline, so the single-valued polyline can switch branches; "
        "treat this as an image-space diagnostic, not a physical accuracy claim."
    )
    lines.append(
        "- These are image-pixel distances only. No tide, bathymetry, or metric "
        "world-coordinate claim is made."
    )
    lines.append("")
    lines.append("## Prediction visualisations\n")
    if report.get("prediction_grid"):
        lines.append(f"![prediction grid]({Path(report['prediction_grid']).name})\n")
    lines.append("Per-image four-panel outputs are in `predictions/`.\n")
    lines.append("## Honest limitations\n")
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    lines.append("## Per-image worst cases (by mean IoU)\n")
    lines.append("| image | mean IoU | Beach IoU | Ocean IoU | Sky IoU | tags |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in report["worst_images"]:
        lines.append(
            f"| `{row['image']}` | {_fmt(row['mean_iou'])} | {_fmt(row['iou_Beach'])} | "
            f"{_fmt(row['iou_Ocean'])} | {_fmt(row['iou_Sky'])} | "
            f"{', '.join(row['tags'])} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    return f"{number:.4f}"


def _merge_panels(images: list[Image.Image], columns: int = 2) -> Image.Image:
    if not images:
        raise ValueError("no panels")
    width = max(image.width for image in images)
    rows = (len(images) + columns - 1) // columns
    height = sum(max(image.height for image in images[i : i + columns]) for i in range(0, len(images), columns))
    canvas = Image.new("RGB", (width * columns, height), (18, 18, 18))
    y = 0
    for row_start in range(0, len(images), columns):
        row = images[row_start : row_start + columns]
        row_height = max(image.height for image in row)
        for index, image in enumerate(row):
            canvas.paste(image, (index * width, y))
        y += row_height
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    paths = unified_paths(config)
    checkpoint = resolve_path(
        args.checkpoint or (Path(config["paths"]["checkpoints"]) / "best.pt")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir = resolve_path(
        args.output_dir or (Path(config["paths"]["outputs"]) / "evaluation")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir = output_dir / "predictions"
    shorelines_dir = output_dir / "shorelines"
    predictions_dir.mkdir(exist_ok=True)
    shorelines_dir.mkdir(exist_ok=True)

    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    names = read_split_file(paths["splits"] / f"{args.split}.txt")
    train_count = 0
    if paths["manifest"].exists():
        with paths["manifest"].open("r", encoding="utf-8", newline="") as handle:
            manifest_rows = list(csv.DictReader(handle))
        train_count = sum(1 for row in manifest_rows if row.get("split") == "train")
    if args.limit:
        names = names[: int(args.limit)]
    if not names:
        raise RuntimeError(f"split {args.split} is empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, config=config, device=device)
    model.eval()
    use_amp = bool(
        config["training"].get("precision", "fp16").lower() == "fp16"
        and device.type == "cuda"
    )
    batch_size = int(config.get("evaluation", {}).get("batch_size", 2))
    dataset = CoastWatchSegFormerDataset(
        paths["images"],
        paths["masks"],
        names,
        transform=build_eval_transform(config),
        source_dataset="sam3",
    )
    loader = make_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(config.get("evaluation", {}).get("num_workers", 2)),
        pin_memory=device.type == "cuda",
    )

    overall = ConfusionMatrix(num_classes=3, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX)
    condition_confusions: dict[str, ConfusionMatrix] = {}
    condition_images: dict[str, int] = {}
    per_image: list[dict[str, Any]] = []
    shoreline_rows: list[dict[str, Any]] = []
    panels: list[Image.Image] = []
    max_panels = int(config.get("evaluation", {}).get("visual_sample_count", 12))

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        with torch.inference_mode(), torch.amp.autocast(
            "cuda", dtype=torch.float16, enabled=use_amp
        ):
            logits = model(pixel_values)
        for index in range(len(batch["name"])):
            name = batch["name"][index]
            image_path = batch["image_path"][index]
            mask_path = batch["mask_path"][index]
            image = load_image_rgb(image_path)
            gt = load_mask(mask_path)
            height, width = image.shape[:2]
            upsampled = F.interpolate(
                logits[index : index + 1],
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            prediction = upsampled.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            if gt.shape != prediction.shape:
                raise ValueError(f"{name}: GT {gt.shape} and prediction {prediction.shape} differ")

            image_confusion = ConfusionMatrix(
                num_classes=3, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX
            )
            image_confusion.update(prediction, gt)
            overall.update(prediction, gt)
            metrics = image_confusion.compute()

            tags = _condition_tags(image, config, gt)
            for tag in tags:
                condition_confusions.setdefault(
                    tag,
                    ConfusionMatrix(num_classes=3, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX),
                ).update(prediction, gt)
                condition_images[tag] = condition_images.get(tag, 0) + 1

            shoreline_cfg = config.get("inference", {}).get("shoreline", {})
            gt_shore = extract_shoreline(
                gt,
                min_ocean_component_px=int(shoreline_cfg.get("min_ocean_component_px", 2000)),
                min_beach_component_px=int(shoreline_cfg.get("min_beach_component_px", 1000)),
                smoothing=int(shoreline_cfg.get("boundary_smoothing", 0)),
                min_valid_columns=int(shoreline_cfg.get("min_valid_columns", 8)),
            )
            pred_shore = extract_shoreline(
                prediction,
                min_ocean_component_px=int(shoreline_cfg.get("min_ocean_component_px", 2000)),
                min_beach_component_px=int(shoreline_cfg.get("min_beach_component_px", 1000)),
                smoothing=int(shoreline_cfg.get("boundary_smoothing", 0)),
                min_valid_columns=int(shoreline_cfg.get("min_valid_columns", 8)),
            )
            shore_compare = _shoreline_compare(gt_shore, pred_shore)

            row = {
                "image": name,
                "mean_iou": metrics["mean_iou"],
                "pixel_accuracy": metrics["pixel_accuracy"],
                "iou_Beach": metrics["per_class"]["Beach"]["iou"],
                "iou_Ocean": metrics["per_class"]["Ocean"]["iou"],
                "iou_Sky": metrics["per_class"]["Sky"]["iou"],
                "dice_Beach": metrics["per_class"]["Beach"]["dice"],
                "dice_Ocean": metrics["per_class"]["Ocean"]["dice"],
                "dice_Sky": metrics["per_class"]["Sky"]["dice"],
                "tags": tags,
                "predicted_shoreline": bool(pred_shore.metadata.get("available")),
                "gt_shoreline": bool(gt_shore.metadata.get("available")),
                "shoreline_mae_px": shore_compare["mae_px"],
                "shoreline_chamfer_px": shore_compare["chamfer_px"],
                "shoreline_interface_median_px": shore_compare["interface_median_px"],
                "shoreline_shared_columns": shore_compare["shared_columns"],
            }
            per_image.append(row)
            shoreline_rows.append(
                {
                    "image": name,
                    "gt": gt_shore.metadata,
                    "pred": pred_shore.metadata,
                    "comparison": shore_compare,
                    "gt_valid_columns": gt_shore.valid_columns,
                    "pred_valid_columns": pred_shore.valid_columns,
                    "gt_ocean_pixels": gt_shore.ocean_pixels,
                    "pred_ocean_pixels": pred_shore.ocean_pixels,
                }
            )

            # Per-image panel: original, GT, prediction, correct/error overlay.
            pred_panel = prediction_panel(image, gt, prediction, shoreline_points=pred_shore.per_column)
            save_panel(pred_panel, predictions_dir / f"{Path(name).stem}.png")
            if len(panels) < max_panels:
                panels.append(pred_panel)

            # Shoreline-specific panel: predicted line (red) and GT line (green).
            pred_line = draw_shoreline(image, pred_shore.per_column, color=(255, 45, 45), thickness=3)
            gt_line = draw_shoreline(image, gt_shore.per_column, color=(45, 230, 45), thickness=3)
            shore_panel = panel(
                [
                    ("Predicted shoreline (red)", pred_line),
                    ("GT shoreline (green)", gt_line),
                    ("Prediction mask", colorize_mask(prediction)),
                    ("GT mask", colorize_mask(gt)),
                ],
                max_width=1800,
            )
            save_panel(shore_panel, shorelines_dir / f"{Path(name).stem}.png")
        print(f"[evaluate] {len(per_image)}/{len(names)}", end="\r", flush=True)

    print()
    metrics = _aggregate_metrics(overall, split=args.split)
    conditions: dict[str, Any] = {}
    expected_tags = (
        "night",
        "low_light",
        "normal_light",
        "bright_sunlight",
        "strong_glare",
        "overcast_or_low_saturation",
        "high_saturation",
        "small_beach_visible_fraction",
        "ocean_dominant",
    )
    for tag in expected_tags:
        if tag in condition_confusions:
            conditions[tag] = {
                "images": condition_images[tag],
                "metrics": _aggregate_metrics(condition_confusions[tag], split=args.split, tags=[tag]),
            }
        else:
            conditions[tag] = {"images": 0, "metrics": None}

    # Shoreline aggregate.
    mae_values = [
        row["shoreline_mae_px"] for row in per_image if row["shoreline_mae_px"] is not None
    ]
    chamfer_values = [
        row["shoreline_chamfer_px"]
        for row in per_image
        if row["shoreline_chamfer_px"] is not None
    ]
    interface_values = [
        row["shoreline_interface_median_px"]
        for row in per_image
        if row["shoreline_interface_median_px"] is not None
    ]
    shoreline_summary = {
        "pred_valid_images": int(sum(1 for row in per_image if row["predicted_shoreline"])),
        "gt_valid_images": int(sum(1 for row in per_image if row["gt_shoreline"])),
        "mean_mae_px": float(np.mean(mae_values)) if mae_values else None,
        "median_mae_px": float(np.median(mae_values)) if mae_values else None,
        "p95_mae_px": float(np.percentile(mae_values, 95)) if mae_values else None,
        "mean_chamfer_px": float(np.mean(chamfer_values)) if chamfer_values else None,
        "median_chamfer_px": float(np.median(chamfer_values)) if chamfer_values else None,
        "p95_chamfer_px": float(np.percentile(chamfer_values, 95)) if chamfer_values else None,
        "mean_interface_median_px": float(np.mean(interface_values)) if interface_values else None,
        "p95_interface_median_px": (
            float(np.percentile(interface_values, 95)) if interface_values else None
        ),
        "max_interface_median_px": float(np.max(interface_values)) if interface_values else None,
        "images_with_comparison": len(mae_values),
        "images_with_chamfer": len(chamfer_values),
    }

    worst = sorted(
        per_image,
        key=lambda row: (row["mean_iou"] is None, row["mean_iou"] if row["mean_iou"] is not None else 1.0),
    )[:8]

    limitations = [
        f"Training data: {train_count} labelled images in the training split "
        "(the baseline model used 131). Training masks are automatic SAM3 "
        "annotations, so metrics measure agreement with those annotations rather "
        "than human ground truth.",
        "Dataset B (SCLabels, 1,717 images) has no masks and is absent from training.",
        f"The test split contains {len(per_image)} images; metrics have uncertainty "
        "and are not a production or cross-camera accuracy claim.",
        "Nighttime is supported only if the test split actually contains night images. "
        "Check the condition coverage table; do not infer nighttime capability from "
        "a daytime-only evaluation.",
        "Condition tags are luminance/glare/saturation heuristics, not meteorological "
        "or tide ground truth. Waves, sea foam, wet sand, vegetation, rocks and tide "
        "are not separately annotated or labelled.",
        "Shoreline error is in image pixels. No camera calibration or world-coordinate "
        "conversion was applied, so no metre/tide claim is made.",
        "Source annotations are SAM3-generated polygons with documented Beach/Ocean "
        "overlap and visible coarse boundaries; some annotation noise is inherited.",
        "The annotation polygons leave the wet-sand swash zone unlabelled in many "
        "frames. Those pixels are ignore 255, not Sky. In 6/20 test images the "
        "remaining Beach and Ocean polygons do not touch, so no ground-truth "
        "shoreline can be extracted and that image is excluded from shoreline error.",
        "Some frames contain a far shore in addition to the near waterline. The "
        "single-valued per-column shoreline can switch branches, so shoreline MAE "
        "is an image-space diagnostic reported over shared columns, not a physical "
        "accuracy claim.",
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "split": args.split,
        "num_images": len(per_image),
        "native_resolution_metrics": True,
        "class_schema": {
            "names": list(CLASS_NAMES),
            "ids": {name: index for index, name in enumerate(CLASS_NAMES)},
            "colors": {name: list(CLASS_COLORS[index]) for index, name in enumerate(CLASS_NAMES)},
            "ignore_index": IGNORE_INDEX,
        },
        "metrics": metrics,
        "primary_metrics": {
            "beach_iou": metrics["per_class"]["Beach"]["iou"],
            "mean_iou": metrics["mean_iou"],
            "mean_iou_note": "dominated by the large, easy Ocean and Sky classes",
            "shoreline_mae_px": {
                "mean": shoreline_summary["mean_mae_px"],
                "median": shoreline_summary["median_mae_px"],
                "p95": shoreline_summary["p95_mae_px"],
            },
        },
        "conditions": conditions,
        "shoreline": shoreline_summary,
        "per_image": per_image,
        "shoreline_details": shoreline_rows,
        "worst_images": worst,
        "prediction_grid": None,
        "limitations": limitations,
    }

    if panels:
        grid = _merge_panels(panels, columns=2)
        grid_path = output_dir / "prediction_grid.png"
        grid.save(grid_path)
        report["prediction_grid"] = str(grid_path)

    fig = confusion_matrix_figure(
        np.asarray(metrics["confusion_matrix"]), CLASS_NAMES, normalize=True
    )
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    (output_dir / "per_image_metrics.csv").write_text(
        _csv_text(per_image), encoding="utf-8"
    )
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _write_markdown(report, output_dir / "evaluation_report.md")
    print(f"[evaluate] Beach IoU: {_fmt(metrics['per_class']['Beach']['iou'])}")
    print(
        "[evaluate] shoreline MAE px (mean/median/p95): "
        f"{_fmt(shoreline_summary['mean_mae_px'])}/"
        f"{_fmt(shoreline_summary['median_mae_px'])}/"
        f"{_fmt(shoreline_summary['p95_mae_px'])}"
    )
    print(f"[evaluate] mean IoU: {_fmt(metrics['mean_iou'])} (dominated by easy classes)")
    print(f"[evaluate] wrote {output_dir}")
    return 0


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    keys = list(rows[0])
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: (",".join(value) if isinstance(value, list) else value) for key, value in row.items()})
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
