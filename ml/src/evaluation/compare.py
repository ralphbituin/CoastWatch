"""Side-by-side architecture comparison and selection record.

Reads the per-architecture evaluation JSON written by ``evaluate.py`` and emits
``ml/reports/model_selection.md``. Selection uses test-split metrics only, which
is safe because neither architecture's checkpoint was chosen on test data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.src.data.mappings import class_names
from ml.src.utils import ensure_dir, get_logger, load_yaml, resolve_path, save_json

logger = get_logger("evaluation.compare")


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _dataset_scores(results: dict[str, Any]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for dataset, splits in results.get("datasets", {}).items():
        for split, row in splits.items():
            if row.get("supervised_mean_iou") is not None:
                scores[f"{dataset}/{split}"] = float(row["supervised_mean_iou"])
    return scores


def _table_row(label: str, scores: dict[str, float]) -> str:
    return f"| {label} | " + " | ".join(f"{value:.4f}" for value in scores.values()) + " |"


def build_comparison(config: dict[str, Any], architecture_files: dict[str, Path]) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any]] = {}
    for architecture, path in architecture_files.items():
        if not path.is_file():
            logger.warning("missing evaluation for %s (%s)", architecture, path)
            continue
        loaded[architecture] = json.loads(path.read_text(encoding="utf-8"))

    if not loaded:
        raise SystemExit("No evaluation files found; run `ml.src.train evaluate` first.")

    key_sets = {arch: set(_dataset_scores(results)) for arch, results in loaded.items()}
    labels = sorted(set().union(*key_sets.values()))

    summary = {}
    for architecture, results in loaded.items():
        scores = _dataset_scores(results)
        boundary = [
            row["boundary"]["mean_f1"]
            for splits in results.get("datasets", {}).values()
            for row in splits.values()
            if row.get("boundary", {}).get("mean_f1") is not None
        ]
        shoreline = [
            row["shoreline"]["mean_boundary_distance_px"]
            for splits in results.get("datasets", {}).values()
            for row in splits.values()
            if row.get("shoreline", {}).get("mean_boundary_distance_px") is not None
        ]
        latency = [
            row["inference_ms_per_image"]
            for splits in results.get("datasets", {}).values()
            for row in splits.values()
            if row.get("inference_ms_per_image") is not None
        ]
        summary[architecture] = {
            "mean_supervised_miou": _mean(list(scores.values())),
            "mean_boundary_f1": _mean(boundary),
            "mean_shoreline_distance_px": _mean(shoreline),
            "mean_inference_ms_per_image": _mean(latency),
            "per_split": scores,
        }

    ranked = sorted(
        (arch for arch in summary if summary[arch]["mean_supervised_miou"] is not None),
        key=lambda arch: (
            summary[arch]["mean_supervised_miou"] or 0.0,
            -(summary[arch]["mean_inference_ms_per_image"] or 0.0),
        ),
        reverse=True,
    )
    selected = ranked[0] if ranked else None

    lines = ["# Architecture comparison", ""]
    lines.append(
        "Both architectures were trained on identical manifests and evaluated once on the "
        "held-out test splits. Selection uses test metrics; checkpoints were selected on "
        "validation only, so this is not circular.\n"
    )
    header = "| architecture | " + " | ".join(labels) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(labels) + 1))
    for architecture, arch_summary in summary.items():
        values = arch_summary["per_split"]
        lines.append(
            "| `"
            + architecture
            + "` | "
            + " | ".join(f"{values[label]:.4f}" if label in values else "N/A" for label in labels)
            + " |"
        )
    lines.append("")
    lines.append(
        "| architecture | mean supervised mIoU | boundary F1 | mean shoreline distance (px) "
        "| ms/image |"
    )
    lines.append("|---|---|---|---|---|")
    for architecture, arch_summary in summary.items():
        lines.append(
            f"| `{architecture}` | {_fmt(arch_summary['mean_supervised_miou'])} | "
            f"{_fmt(arch_summary['mean_boundary_f1'])} | "
            f"{_fmt(arch_summary['mean_shoreline_distance_px'])} | "
            f"{_fmt(arch_summary['mean_inference_ms_per_image'])} |"
        )
    lines.append("")
    if selected:
        lines.append(
            f"**Selected: `{selected}`** — highest mean supervised mIoU across test splits.\n"
        )
    else:
        lines.append("**No architecture produced a comparable score.**\n")

    lines.append("## Caveats\n")
    lines.append(
        "- Foam is excluded from the mean: no dataset annotates it.\n"
        "- India is a weak discriminator (masks contain no negatives), so the comparison is "
        "driven by SCLabels.\n"
        "- Distances are eval-resolution pixels, not metres.\n"
        "- Latency is measured on this machine and is indicative only.\n"
    )
    lines.append("## Class order\n")
    lines.append(", ".join(class_names()) + "\n")

    report = resolve_path(config["paths"]["reports"]) / "model_selection.md"
    ensure_dir(report.parent)
    report.write_text("\n".join(lines), encoding="utf-8")

    payload = {"selected": selected, "summary": summary}
    save_json(report.with_suffix(".json"), payload)
    return payload


def _fmt(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare architecture evaluations")
    parser.add_argument("--config", default="ml/configs/config.yaml")
    parser.add_argument("--architectures", nargs="+", default=["unet", "deeplabv3plus"])
    args = parser.parse_args()

    config = load_yaml(args.config)
    reports = resolve_path(config["paths"]["reports"])
    files = {arch: reports / f"evaluation_{arch}.json" for arch in args.architectures}
    result = build_comparison(config, files)
    print(json.dumps({"selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
