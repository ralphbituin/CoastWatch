"""Independent per-dataset evaluation on the held-out test splits.

Nothing here influences model selection: the test manifests are read only after
training is finished. Every dataset is evaluated separately and the report
never pools pixel counts across domains, because a pooled number would hide
satellite→photograph domain shift.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from ml.src.data.mappings import class_names, load_class_config
from ml.src.data.splits import load_manifest
from ml.src.training.checkpoint import build_model_from_checkpoint, load_torch_checkpoint
from ml.src.training.metrics import (
    ConfusionMatrix,
    boundary_f1,
    shoreline_metrics_from_masks,
)
from ml.src.training.trainer import _make_loader, build_dataset
from ml.src.utils import ensure_dir, get_logger, load_yaml, resolve_device, resolve_path, save_json

logger = get_logger("evaluation")


def _aggregate(rows: list[float]) -> float | None:
    return float(np.mean(rows)) if rows else None


def evaluate_architecture(
    config: dict[str, Any],
    architecture: str,
    checkpoint: str | Path,
    *,
    device: str = "auto",
    splits: tuple[str, ...] = ("test",),
    max_images_per_dataset: int | None = None,
    collect_visuals: bool = True,
) -> dict[str, Any]:
    import torch

    device = resolve_device(device)
    payload = load_torch_checkpoint(checkpoint, map_location=device)
    model = build_model_from_checkpoint(payload, device=device)
    model.eval()

    ignore_index = int(config["data"]["ignore_index"])
    num_classes = int(config["data"]["num_classes"])
    unsupervised = {int(c) for c in config["evaluation"].get("unsupervised_classes", [3])}
    tolerance = int(config["evaluation"].get("boundary_tolerance_px", 2))
    water_classes = tuple(int(c) for c in config["data"].get("water_classes", [2, 3]))

    results: dict[str, Any] = {
        "architecture": architecture,
        "checkpoint": str(resolve_path(checkpoint)),
        "split": list(splits),
        "device": device,
        "datasets": {},
        "unsupervised_classes": sorted(unsupervised),
    }

    figures_dir = ensure_dir(resolve_path(config["paths"]["figures"]) / "evaluation" / architecture)
    visual_index: dict[str, list[str]] = {}

    for dataset in config["data"]["datasets"]:
        for split in splits:
            manifest = resolve_path(config["paths"]["manifests"]) / f"{dataset}_{split}.csv"
            if not manifest.is_file():
                logger.warning("manifest missing, skipping: %s", manifest)
                continue
            records = load_manifest(manifest)
            if max_images_per_dataset:
                records = records[:max_images_per_dataset]
            if not records:
                continue
            logger.info("evaluating %s/%s (%d images)", dataset, split, len(records))

            dataset_obj = build_dataset(config, dataset, records, train=False)
            loader = _make_loader(
                dataset_obj,
                batch_size=int(config["evaluation"].get("batch_size", 8)),
                num_workers=int(config["evaluation"].get("num_workers", 0)),
                shuffle=False,
                pin_memory=device.startswith("cuda"),
                seed=int(config["project"]["seed"]),
            )

            confusion = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
            boundary_rows: list[dict[str, Any]] = []
            shoreline_rows: list[dict[str, Any]] = []
            inference_times: list[float] = []
            sample_predictions: list[dict[str, Any]] = []

            started = time.perf_counter()
            with torch.no_grad():
                offset = 0
                for batch in loader:
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    batch_started = time.perf_counter()
                    logits = model(images)
                    if device.startswith("cuda"):
                        torch.cuda.synchronize()
                    batch_ms = (time.perf_counter() - batch_started) * 1000.0
                    inference_times.extend([batch_ms / images.shape[0]] * images.shape[0])

                    predictions = logits.argmax(dim=1)
                    prediction_np = predictions.detach().cpu().numpy()
                    mask_np = masks.detach().cpu().numpy()
                    confusion.update(prediction_np, mask_np)

                    for local_index, (prediction, target) in enumerate(
                        zip(prediction_np, mask_np, strict=True)
                    ):
                        boundary_rows.append(
                            boundary_f1(
                                prediction,
                                target,
                                ignore_index=ignore_index,
                                tolerance_px=tolerance,
                            )
                        )
                        shoreline_rows.append(
                            shoreline_metrics_from_masks(
                                prediction,
                                target,
                                ignore_index=ignore_index,
                                tolerance_px=tolerance,
                                water_classes=water_classes,
                            )
                        )
                        if collect_visuals and len(sample_predictions) < int(
                            config["evaluation"].get("max_visual_samples_per_dataset", 5)
                        ):
                            sample_predictions.append(
                                {
                                    "record": records[offset + local_index],
                                    "prediction": prediction,
                                    "target": target,
                                }
                            )
                    offset += images.shape[0]

            metrics = confusion.compute()
            metrics["dataset"] = dataset
            metrics["split"] = split
            metrics["images"] = len(records)
            metrics["supervised_mean_iou"] = _supervised_mean_iou(metrics, unsupervised)
            metrics["boundary"] = _aggregate_boundary(boundary_rows)
            metrics["shoreline"] = _aggregate_shoreline(shoreline_rows)
            metrics["inference_ms_per_image"] = (
                float(np.mean(inference_times)) if inference_times else None
            )
            metrics["evaluation_seconds"] = round(time.perf_counter() - started, 2)
            metrics["note"] = _honesty_note(dataset, metrics, unsupervised)

            results["datasets"].setdefault(dataset, {})[split] = metrics

            if collect_visuals and sample_predictions:
                from ml.src.evaluation.visualize import render_predictions

                paths = render_predictions(
                    dataset,
                    split,
                    sample_predictions,
                    figures_dir,
                    class_colors=_class_colors(config),
                )
                visual_index.setdefault(f"{dataset}_{split}", []).extend(paths)

    results["figures"] = visual_index
    results["classes"] = class_names()
    return results


def _class_colors(config: dict[str, Any]) -> np.ndarray:
    class_config = load_class_config()
    entries = class_config["schema"]["classes"]
    return np.asarray([entries[index]["color"] for index in sorted(entries)], dtype=np.uint8)


def _supervised_mean_iou(metrics: dict[str, Any], unsupervised: set[int]) -> float | None:
    values = [
        row["iou"]
        for index, row in metrics["per_class"].items()
        if int(index) not in unsupervised and row["iou"] is not None
    ]
    return float(np.mean(values)) if values else None


def _aggregate_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for class_id in ("1", "2"):
        usable = [row[class_id] for row in rows if row.get(class_id, {}).get("f1") is not None]
        result[class_id] = {
            "images": len(usable),
            "f1": _aggregate([row["f1"] for row in usable]),
            "precision": _aggregate(
                [row["precision"] for row in usable if row["precision"] is not None]
            ),
            "recall": _aggregate([row["recall"] for row in usable if row["recall"] is not None]),
        }
    means = [row["mean_f1"] for row in rows if row.get("mean_f1") is not None]
    result["mean_f1"] = _aggregate([float(value) for value in means])
    return result


def _aggregate_shoreline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("available")]
    if not usable:
        return {"available": False, "reason": "no image had both predicted and ground-truth water"}
    return {
        "available": True,
        "images": len(usable),
        "boundary_f1": _aggregate(
            [row["boundary_f1"] for row in usable if row.get("boundary_f1") is not None]
        ),
        "boundary_precision": _aggregate(
            [
                row["boundary_precision"]
                for row in usable
                if row.get("boundary_precision") is not None
            ]
        ),
        "boundary_recall": _aggregate(
            [row["boundary_recall"] for row in usable if row.get("boundary_recall") is not None]
        ),
        "mean_boundary_distance_px": _aggregate(
            [
                row["mean_boundary_distance_px"]
                for row in usable
                if row.get("mean_boundary_distance_px") is not None
            ]
        ),
        "mean_boundary_distance_pred_to_gt_px": _aggregate(
            [
                row["mean_boundary_distance_pred_to_gt_px"]
                for row in usable
                if row.get("mean_boundary_distance_pred_to_gt_px") is not None
            ]
        ),
        "mean_vertical_displacement_px": _aggregate(
            [
                row["mean_vertical_displacement_px"]
                for row in usable
                if row.get("mean_vertical_displacement_px") is not None
            ]
        ),
    }


def _honesty_note(dataset: str, metrics: dict[str, Any], unsupervised: set[int]) -> str:
    notes = [
        "Metrics are computed on a held-out split that was not used for model selection or "
        "early stopping."
    ]
    if 3 in unsupervised:
        notes.append(
            "Foam metrics are reported as N/A: no provided dataset contains foam annotations."
        )
    if dataset == "india_sandy_beach":
        notes.append(
            "India masks supervise sand only; non-beach pixels (which may include water) are "
            "ignored, so water metrics on India reflect unlabelled background, not water "
            "ground truth."
        )
    if dataset == "sclabels":
        notes.append(
            "SCLabels ships no images in the test split beyond the provided rectified frames; "
            "the split here is chronological within each of 5 sites, so this measures temporal "
            "generalisation, not cross-site generalisation."
        )
    notes.append(
        "Boundary distances are in image pixels at the evaluation resolution; no metre-scale "
        "accuracy is claimed without camera calibration."
    )
    notes.append(
        "Shoreline metrics use the dominant water body and the land/water interface only. "
        "The primary distance is measured from annotated waterline pixels to the predicted "
        "waterline; the reverse distance is reported separately because predicted water in "
        "ignored regions (sky, vegetation) lowers precision without moving the waterline."
    )
    return " ".join(notes)


def render_markdown(results: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Evaluation — CoastWatch Beach Segmentation\n")
    lines.append(
        f"Architecture: `{results['architecture']}` · device: `{results['device']}` · "
        f"splits: {', '.join(results['split'])}\n"
    )
    lines.append(
        "Test splits are evaluated once, after training. Foam (class 3) is untrained and its "
        "metrics are omitted rather than reported as zero.\n"
    )
    lines.append(
        "`supervised mean IoU` averages every non-foam class that the model predicts or that is "
        "annotated, so predicting a class that has no annotations scores IoU 0. `mean Dice` and "
        "`pixel accuracy` follow the confusion matrix; the per-class support column shows which "
        "classes actually exist in the annotations.\n"
    )
    for dataset, splits in results["datasets"].items():
        for split, metrics in splits.items():
            lines.append(f"## {dataset} — {split}\n")
            lines.append(
                f"- images: {metrics.get('images')}\n"
                f"- supervised pixels: {metrics.get('supervised_pixels'):,}\n"
                f"- supervised mean IoU: {_fmt(metrics.get('supervised_mean_iou'))}\n"
                f"- mean Dice: {_fmt(metrics.get('mean_dice'))}\n"
                f"- pixel accuracy: {_fmt(metrics.get('pixel_accuracy'))}\n"
                f"- inference: {_fmt(metrics.get('inference_ms_per_image'))} ms/image\n"
            )
            rows = []
            for index, row in metrics["per_class"].items():
                name = class_names()[int(index)]
                if int(index) in results["unsupervised_classes"]:
                    rows.append([name, row["support"], "N/A (untrained)", "N/A", "N/A"])
                    continue
                rows.append(
                    [
                        name,
                        row["support"],
                        _fmt(row["iou"]),
                        _fmt(row["dice"]),
                        f"{_fmt(row['precision'])} / {_fmt(row['recall'])}",
                    ]
                )
            lines.append(_table(["class", "support px", "IoU", "Dice", "precision / recall"], rows))
            lines.append("")
            boundary = metrics.get("boundary", {})
            shoreline = metrics.get("shoreline", {})
            if boundary:
                lines.append(
                    f"Boundary F1 — land: {_fmt(boundary.get('1', {}).get('f1'))}, "
                    f"water: {_fmt(boundary.get('2', {}).get('f1'))}, "
                    f"mean: {_fmt(boundary.get('mean_f1'))}\n"
                )
            if shoreline.get("available"):
                lines.append(
                    f"Shoreline (land/water interface) — precision: "
                    f"{_fmt(shoreline.get('boundary_precision'))}, recall: "
                    f"{_fmt(shoreline.get('boundary_recall'))}, F1: "
                    f"{_fmt(shoreline.get('boundary_f1'))}, mean waterline error "
                    f"(annotated to predicted): "
                    f"{_fmt(shoreline.get('mean_boundary_distance_px'))} px, "
                    f"reverse error: "
                    f"{_fmt(shoreline.get('mean_boundary_distance_pred_to_gt_px'))} px, "
                    f"mean vertical displacement: "
                    f"{_fmt(shoreline.get('mean_vertical_displacement_px'))} px\n"
                )
            else:
                lines.append("Shoreline metrics: N/A (no comparable water predicted/annotated).\n")
            lines.append(f"> {metrics.get('note', '')}\n")
            for figure in results.get("figures", {}).get(f"{dataset}_{split}", []):
                lines.append(f"![{dataset} {split}](../{figure})\n")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on held-out test splits")
    parser.add_argument("checkpoint")
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--config", default="ml/configs/config.yaml")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--max-images-per-dataset", type=int, default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    results = evaluate_architecture(
        config,
        args.architecture,
        args.checkpoint,
        device=args.device,
        splits=tuple(args.splits),
        max_images_per_dataset=args.max_images_per_dataset,
    )
    reports = ensure_dir(resolve_path(config["paths"]["reports"]))
    output = (
        resolve_path(args.output)
        if args.output
        else reports / f"evaluation_{args.architecture}.json"
    )
    save_json(output, results)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(render_markdown(results), encoding="utf-8")
    logger.info("wrote %s and %s", output, markdown_path)


if __name__ == "__main__":
    main()
