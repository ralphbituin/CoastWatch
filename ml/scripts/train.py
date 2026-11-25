"""Train SegFormer MiT-B2 on the unified CoastWatch dataset.

Examples:
    python -m ml.scripts.train --smoke
    python -m ml.scripts.train

The best checkpoint is selected by validation mean IoU only. The test split is
never read by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from transformers import get_polynomial_decay_schedule_with_warmup

from ml.src.dataset import (
    CoastWatchSegFormerDataset,
    build_eval_transform,
    build_train_transform,
    image_mean_std,
    load_config,
    make_loader,
    read_split_file,
    resolve_path,
    unified_paths,
)
from ml.src.metrics import (
    CLASS_NAMES,
    IGNORE_INDEX,
    ConfusionMatrix,
    inverse_frequency_weights,
)
from ml.src.model import build_model, parameter_groups, trainable_parameter_count

LOGGER = logging.getLogger("coastwatch.train")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(console)
    LOGGER.addHandler(file_handler)


def _read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _class_weight_vector(
    manifest_path: Path, train_names: set[str], config: dict[str, Any]
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    loss_cfg = config["training"].get("loss", {})
    mode = str(loss_cfg.get("class_weights", "none")).lower()
    rows = [row for row in _read_manifest_rows(manifest_path) if row["image"] in train_names]
    if not rows:
        raise RuntimeError("no manifest rows matched the train split")
    counts = np.array(
        [
            sum(int(row["beach_pixels"]) for row in rows),
            sum(int(row["ocean_pixels"]) for row in rows),
            sum(int(row["sky_pixels"]) for row in rows),
        ],
        dtype=np.float64,
    )
    if mode in ("none", "false", "off", "0"):
        return None, {"mode": "none", "class_pixels": counts.tolist()}
    if mode not in ("auto", "inverse_frequency", "inverse"):
        raise ValueError(f"unsupported class_weights mode: {mode!r}")
    cap = float(loss_cfg.get("class_weight_cap", 3.0))
    weights = inverse_frequency_weights(counts, cap=cap)
    beach_pixels = float(counts[0])
    ocean_per_beach = counts[1] / beach_pixels if beach_pixels else float("nan")
    sky_per_beach = counts[2] / beach_pixels if beach_pixels else float("nan")
    info = {
        "mode": mode,
        "class_pixels": counts.tolist(),
        "class_fraction": (counts / counts.sum()).tolist() if counts.sum() else None,
        "weights": weights.tolist(),
        "cap": cap,
        "reason": (
            "train manifest counts give "
            f"ocean:beach={ocean_per_beach:.2f}:1 and sky:beach={sky_per_beach:.2f}:1; "
            "inverse-frequency CE is used"
        ),
    }
    return torch.tensor(weights, dtype=torch.float32), info


def _make_loss(config: dict[str, Any], weights: torch.Tensor | None, device: torch.device) -> nn.Module:
    ignore_index = int(config["data"].get("ignore_index", IGNORE_INDEX))
    if weights is not None:
        weights = weights.to(device)
    # Plain cross entropy. Class weights are the only adjustment, and they are
    # audit-driven; no Dice term is added for this initial configuration.
    return nn.CrossEntropyLoss(weight=weights, ignore_index=ignore_index)


def _train_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    loss_fn: nn.Module,
    device: torch.device,
    config: dict[str, Any],
    epoch: int,
    num_epochs: int,
) -> dict[str, float]:
    model.train()
    accum = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    clip = float(config["training"].get("gradient_clip_norm", 0.0))
    use_amp = bool(config["training"].get("precision", "fp16").lower() == "fp16" and device.type == "cuda")
    log_every = int(config["training"].get("log_every", 10))
    total_loss = 0.0
    total_batches = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    pending_gradients = False
    for step, batch in enumerate(loader):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        supervised = int((labels != IGNORE_INDEX).sum()) > 0
        if supervised:
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(pixel_values, output_size=tuple(labels.shape[-2:]))
                loss = loss_fn(logits, labels)
            scaled = loss / accum
            scaler.scale(scaled).backward()
            pending_gradients = True
            total_loss += float(loss.detach())
            total_batches += 1
        # Count every batch in the accumulation window so a batch with no
        # supervised pixels still closes the window and advances the scheduler.
        if (step + 1) % accum == 0 or (step + 1) == len(loader):
            if pending_gradients:
                if clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending_gradients = False
            scheduler.step()
        if supervised and log_every and (step + 1) % log_every == 0:
            elapsed = time.perf_counter() - start
            LOGGER.info(
                "epoch %d/%d step %d/%d loss=%.4f lr=%.3e elapsed=%.0fs",
                epoch, num_epochs, step + 1, len(loader), float(loss.detach()),
                scheduler.get_last_lr()[0], elapsed,
            )
    mean_loss = total_loss / max(total_batches, 1)
    return {
        "train_loss": mean_loss,
        "train_batches": total_batches,
        "epoch_seconds": time.perf_counter() - start,
    }


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    use_amp = bool(config["training"].get("precision", "fp16").lower() == "fp16" and device.type == "cuda")
    confusion = ConfusionMatrix(num_classes=3, class_names=CLASS_NAMES, ignore_index=IGNORE_INDEX)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(pixel_values, output_size=tuple(labels.shape[-2:]))
            loss = loss_fn(logits, labels)
        total_loss += float(loss)
        batches += 1
        predictions = logits.argmax(dim=1).cpu().numpy()
        targets = labels.cpu().numpy()
        confusion.update(predictions, targets)
    metrics = confusion.compute()
    metrics["val_loss"] = total_loss / max(batches, 1)
    return metrics


def _history_row(epoch: int, train: dict[str, Any], val: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "epoch": epoch,
        "train_loss": train["train_loss"],
        "epoch_seconds": train["epoch_seconds"],
        "val_loss": val["val_loss"],
        "val_mean_iou": val["mean_iou"],
        "val_mean_dice": val["mean_dice"],
        "val_pixel_accuracy": val["pixel_accuracy"],
    }
    for name, info in val["per_class"].items():
        row[f"val_iou_{name}"] = info["iou"]
        row[f"val_dice_{name}"] = info["dice"]
    return row


def _write_history(output_dir: Path, history: list[dict[str, Any]]) -> None:
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if not history:
        return
    keys = list(history[0])
    with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    best_val_iou: float,
    config: dict[str, Any],
    class_weight_info: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_val_iou": float(best_val_iou),
        "num_labels": 3,
        "class_names": list(CLASS_NAMES),
        "ignore_index": IGNORE_INDEX,
        "pretrained": config["model"].get("pretrained"),
        "class_weights": class_weight_info,
        "config": config,
    }
    torch.save(payload, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--smoke", action="store_true", help="tiny 2-epoch run on a subset")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init-from",
        default=None,
        help="Initialise model weights from a checkpoint but start a fresh "
        "optimizer/scheduler at epoch 1 (fine-tuning). Cannot be combined with --resume.",
    )
    args = parser.parse_args(argv)
    if args.resume and args.init_from:
        parser.error("--resume and --init-from are mutually exclusive")

    config = load_config(args.config)
    paths = unified_paths(config)
    if args.smoke:
        output_dir = resolve_path(args.output_dir or "ml/outputs/segformer_b2/smoke")
        checkpoint_dir = resolve_path(args.checkpoint_dir or "ml/checkpoints/segformer_b2/smoke")
        epochs = int(args.epochs or 2)
        limit_train = int(args.limit_train or 12)
        limit_val = int(args.limit_val or 6)
    else:
        output_dir = resolve_path(args.output_dir or config["paths"]["outputs"])
        checkpoint_dir = resolve_path(args.checkpoint_dir or config["paths"]["checkpoints"])
        epochs = int(args.epochs or config["training"]["epochs"])
        limit_train = args.limit_train
        limit_val = args.limit_val
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_dir)

    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("device=%s", device)
    if device.type == "cuda":
        LOGGER.info("gpu=%s", torch.cuda.get_device_name(0))
        torch.cuda.empty_cache()

    # Splits and manifest.
    if not paths["manifest"].exists():
        raise FileNotFoundError(f"missing manifest: {paths['manifest']}; run prepare_dataset.py")
    train_names = read_split_file(paths["splits"] / "train.txt")
    val_names = read_split_file(paths["splits"] / "val.txt")
    if not train_names or not val_names:
        raise RuntimeError("train/val splits are empty")
    if args.smoke:
        rng = random.Random(seed)
        train_names = rng.sample(train_names, min(limit_train, len(train_names)))
        val_names = rng.sample(val_names, min(limit_val, len(val_names)))
    LOGGER.info("train=%d val=%d epochs=%d", len(train_names), len(val_names), epochs)

    train_dataset = CoastWatchSegFormerDataset(
        paths["images"],
        paths["masks"],
        train_names,
        transform=build_train_transform(config),
        source_dataset="sam3",
    )
    val_dataset = CoastWatchSegFormerDataset(
        paths["images"],
        paths["masks"],
        val_names,
        transform=build_eval_transform(config),
        source_dataset="sam3",
    )
    batch_size = int(config["training"].get("batch_size", 2))
    num_workers = int(config["training"].get("num_workers", 2))
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(config["training"].get("pin_memory", True)),
        seed=seed,
        persistent_workers=bool(config["training"].get("persistent_workers", False)),
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(config["training"].get("pin_memory", True)),
    )

    class_weights, class_weight_info = _class_weight_vector(
        paths["manifest"], set(train_names), config
    )
    LOGGER.info("class weights: %s", class_weight_info.get("weights", "none"))
    loss_fn = _make_loss(config, class_weights, device)

    model = build_model(config).to(device)
    if args.init_from:
        init_path = resolve_path(args.init_from)
        if not init_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_path}")
        init_payload = torch.load(init_path, map_location="cpu", weights_only=False)
        init_state = init_payload.get("model_state", init_payload)
        try:
            model.load_state_dict(init_state)
        except RuntimeError as exc:
            raise RuntimeError(
                f"init checkpoint {init_path} is incompatible with the 3-class "
                f"SegFormer architecture: {exc}"
            ) from exc
        LOGGER.info(
            "initialised weights from %s (epoch %s, val mIoU %s); "
            "starting a fresh optimizer/scheduler",
            init_path,
            init_payload.get("epoch"),
            init_payload.get("best_val_iou"),
        )
    counts = trainable_parameter_count(model)
    LOGGER.info("model params: %.2fM trainable", counts["trainable"] / 1e6)
    optimizer = torch.optim.AdamW(
        parameter_groups(model, float(config["training"].get("weight_decay", 0.01))),
        lr=float(config["training"].get("learning_rate", 6e-5)),
        betas=(0.9, 0.999),
    )
    accum = max(1, int(config["training"].get("gradient_accumulation_steps", 1)))
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler_cfg = config["training"].get("scheduler", {})
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0) or 0)
    if warmup_steps <= 0:
        warmup_steps = max(1, int(round(total_steps * float(scheduler_cfg.get("warmup_ratio", 0.05)))))
    scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(warmup_steps, total_steps),
        num_training_steps=total_steps,
        power=float(scheduler_cfg.get("power", 1.0)),
    )
    use_amp = bool(config["training"].get("precision", "fp16").lower() == "fp16" and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    LOGGER.info(
        "steps/epoch=%d total_steps=%d warmup=%d amp=%s accum=%d",
        steps_per_epoch, total_steps, warmup_steps, use_amp, accum,
    )

    training_config = {
        **config,
        "runtime": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "torch": torch.__version__,
            "train_images": len(train_names),
            "val_images": len(val_names),
            "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "class_weights": class_weight_info,
            "smoke": bool(args.smoke),
        },
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(training_config, indent=2), encoding="utf-8"
    )
    (output_dir / "class_mapping.json").write_text(
        json.dumps(
            {
                "target_schema": {
                    "class_names": list(CLASS_NAMES),
                    "class_ids": {name: index for index, name in enumerate(CLASS_NAMES)},
                    "ignore_index": IGNORE_INDEX,
                },
                "verified_source_mapping": config["data"]["datasets"]["sam3"]["annotation"][
                    "source_values"
                ],
                "documentation_conflict": config["data"]["datasets"]["sam3"].get(
                    "documentation_conflict"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    start_epoch = 1
    best_val_iou = -math.inf
    if args.resume:
        resume_path = Path(args.resume)
        payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload.get("epoch", 0)) + 1
        best_val_iou = float(payload.get("best_val_iou", -math.inf))
        LOGGER.info("resumed from %s at epoch %d", resume_path, start_epoch - 1)

    history: list[dict[str, Any]] = []
    history_path = output_dir / "history.json"
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    best_epoch = -1
    for epoch in range(start_epoch, epochs + 1):
        train_stats = _train_epoch(
            model, train_loader, optimizer, scheduler, scaler, loss_fn, device, config,
            epoch, epochs,
        )
        val_metrics = validate(model, val_loader, loss_fn, device, config)
        row = _history_row(epoch, train_stats, val_metrics)
        history.append(row)
        _write_history(output_dir, history)
        (output_dir / "metrics_val.json").write_text(
            json.dumps(val_metrics, indent=2), encoding="utf-8"
        )
        mean_iou = val_metrics.get("mean_iou")
        LOGGER.info(
            "epoch %d/%d train_loss=%.4f val_loss=%.4f val_mIoU=%s val_acc=%s",
            epoch, epochs, train_stats["train_loss"], val_metrics["val_loss"],
            f"{mean_iou:.4f}" if mean_iou is not None else "n/a",
            f"{val_metrics['pixel_accuracy']:.4f}" if val_metrics["pixel_accuracy"] is not None else "n/a",
        )
        comparable = mean_iou if mean_iou is not None else -math.inf
        if comparable > best_val_iou:
            best_val_iou = comparable
            best_epoch = epoch
            _save_checkpoint(
                checkpoint_dir / "best.pt",
                model=model, optimizer=optimizer, scheduler=scheduler,
                epoch=epoch, best_val_iou=best_val_iou,
                config=config, class_weight_info=class_weight_info,
            )
            LOGGER.info("saved new best checkpoint at epoch %d (val mIoU=%.4f)", epoch, best_val_iou)
        _save_checkpoint(
            checkpoint_dir / "last.pt",
            model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=epoch, best_val_iou=best_val_iou,
            config=config, class_weight_info=class_weight_info,
        )

    # Save a Hugging Face-format copy of the best weights for deployment.
    if (checkpoint_dir / "best.pt").exists():
        payload = torch.load(checkpoint_dir / "best.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.save_pretrained(
            checkpoint_dir / "best_hf",
            id2label={index: name for index, name in enumerate(CLASS_NAMES)},
        )
        LOGGER.info("saved best_hf to %s", checkpoint_dir / "best_hf")

    summary = {
        "best_epoch": best_epoch,
        "best_val_mean_iou": best_val_iou if np.isfinite(best_val_iou) else None,
        "epochs_completed": len(history),
        "checkpoint": str(checkpoint_dir / "best.pt"),
        "history": str(output_dir / "history.json"),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("training complete: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
