"""LEGACY four-class (unknown/land/water/foam) U-Net/DeepLabV3+ pipeline.

This is not the active CoastWatch path. The maintained three-class SegFormer
pipeline (Beach/Ocean/Sky) lives in ``ml/configs/segformer_b2.yaml`` and
``ml/scripts/train.py``.

Command-line entrypoint for the whole legacy training pipeline.

Typical use::

    # 1. inspect datasets and write the report
    python -m ml.src.train inspect --config ml/configs/config.yaml

    # 2. build deterministic manifests
    python -m ml.src.train manifests

    # 3. smoke test the data path (1 epoch, small subsets)
    python -m ml.src.train train --architecture unet --smoke

    # 4. full staged training for both architectures
    python -m ml.src.train train --architecture unet
    python -m ml.src.train train --architecture deeplabv3plus

    # 5. evaluate on held-out test splits
    python -m ml.src.train evaluate --architecture unet

    # 6. export the selected model for CoastWatch
    python -m ml.src.train export --architecture unet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ml.src.utils import get_logger, load_yaml, resolve_path

logger = get_logger("cli")


def _manifests(config: dict[str, Any]) -> None:
    from ml.src.data.splits import build_splits

    groups = build_splits(config, write=True)
    for dataset, records in groups.items():
        counts = {
            split: sum(r.split == split for r in records) for split in ("train", "val", "test")
        }
        logger.info("%s: %s", dataset, counts)


def _inspect(config: dict[str, Any], max_samples: int) -> None:
    from ml.src.data.inspect import build_report

    build_report(config, max_samples=max_samples)


def _train(config_path: str, args: argparse.Namespace) -> None:
    from ml.src.training.trainer import run_all_stages

    for architecture in args.architectures:
        logger.info("=== training %s ===", architecture)
        results = run_all_stages(
            config_path,
            architecture,
            epochs=args.epochs,
            device=args.device,
            stages=args.stages,
            smoke=args.smoke,
            max_batches=args.max_batches,
            max_samples=args.max_samples,
        )
        summary = [
            {
                "stage": result.stage,
                "best_epoch": result.best_epoch,
                "best_metric": result.best_metric,
                "checkpoint": result.checkpoint,
            }
            for result in results
        ]
        print(json.dumps(summary, indent=2))


def _evaluate(config: dict[str, Any], args: argparse.Namespace) -> None:
    from ml.src.evaluation.evaluate import evaluate_architecture, render_markdown
    from ml.src.training.checkpoint import checkpoint_path
    from ml.src.utils import ensure_dir, save_json

    for architecture in args.architectures:
        checkpoint = args.checkpoint or checkpoint_path(
            config, architecture, "stage4_joint_finetune", "best"
        )
        if not Path(checkpoint).is_file():
            logger.warning("no checkpoint for %s at %s; skipping", architecture, checkpoint)
            continue
        results = evaluate_architecture(
            config,
            architecture,
            checkpoint,
            device=args.device,
            max_images_per_dataset=args.max_images_per_dataset,
            collect_visuals=not args.no_visuals,
        )
        reports = ensure_dir(resolve_path(config["paths"]["reports"]))
        json_path = reports / f"evaluation_{architecture}.json"
        save_json(json_path, results)
        markdown_path = reports / f"evaluation_{architecture}.md"
        markdown_path.write_text(render_markdown(results), encoding="utf-8")
        logger.info("wrote %s and %s", json_path, markdown_path)


def _export(config: dict[str, Any], args: argparse.Namespace) -> None:
    from ml.src.inference.export_onnx import export
    from ml.src.training.checkpoint import checkpoint_path

    architecture = args.architectures[0]
    checkpoint = args.checkpoint or checkpoint_path(
        config, architecture, "stage4_joint_finetune", "best"
    )
    metrics_path = resolve_path(config["paths"]["reports"]) / f"evaluation_{architecture}.json"
    metrics = None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    output_dir = resolve_path(
        args.output_dir or Path(config["paths"]["models"]) / "shoreline-segmentation"
    )
    result = export(checkpoint, output_dir, metrics=metrics, verify=not args.no_verify)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CoastWatch beach segmentation pipeline")
    parser.add_argument("--config", default="ml/configs/config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Validate datasets and write the report")
    inspect_parser.add_argument("--max-samples", type=int, default=4)

    subparsers.add_parser("manifests", help="Write deterministic train/val/test CSV manifests")

    train_parser = subparsers.add_parser("train", help="Run the staged training pipeline")
    train_parser.add_argument(
        "--architectures",
        nargs="+",
        default=["unet", "deeplabv3plus"],
        choices=["unet", "deeplabv3plus"],
    )
    train_parser.add_argument("--epochs", type=int, default=None)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--stages", nargs="+", default=None)
    train_parser.add_argument("--smoke", action="store_true")
    train_parser.add_argument("--max-batches", type=int, default=None)
    train_parser.add_argument("--max-samples", type=int, default=None)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on held-out test splits")
    eval_parser.add_argument(
        "--architectures",
        nargs="+",
        default=["unet", "deeplabv3plus"],
        choices=["unet", "deeplabv3plus"],
    )
    eval_parser.add_argument("--checkpoint", default=None)
    eval_parser.add_argument("--device", default="auto")
    eval_parser.add_argument("--max-images-per-dataset", type=int, default=None)
    eval_parser.add_argument("--no-visuals", action="store_true")

    export_parser = subparsers.add_parser(
        "export", help="Export a checkpoint to ONNX + config.json"
    )
    export_parser.add_argument(
        "--architectures", nargs="+", default=["unet"], choices=["unet", "deeplabv3plus"]
    )
    export_parser.add_argument("--checkpoint", default=None)
    export_parser.add_argument("--output-dir", default=None)
    export_parser.add_argument("--no-verify", action="store_true")

    compare_parser = subparsers.add_parser(
        "compare", help="Compare architecture evaluations and record the selection"
    )
    compare_parser.add_argument(
        "--architectures",
        nargs="+",
        default=["unet", "deeplabv3plus"],
        choices=["unet", "deeplabv3plus"],
    )

    args = parser.parse_args()
    config = load_yaml(args.config)

    if args.command == "inspect":
        _inspect(config, args.max_samples)
    elif args.command == "manifests":
        _manifests(config)
    elif args.command == "train":
        _train(args.config, args)
    elif args.command == "evaluate":
        _evaluate(config, args)
    elif args.command == "export":
        _export(config, args)
    elif args.command == "compare":
        _compare(config, args)


def _compare(config: dict[str, Any], args: argparse.Namespace) -> None:
    from ml.src.evaluation.compare import build_comparison

    reports = resolve_path(config["paths"]["reports"])
    result = build_comparison(
        config, {arch: reports / f"evaluation_{arch}.json" for arch in args.architectures}
    )
    print(json.dumps({"selected": result["selected"]}, indent=2))


if __name__ == "__main__":
    main()
