"""Staged trainer: India → SCLabels → joint fine-tuning.

Design notes
------------
* Each stage trains on one or more manifests and validates on *every* dataset
  in that stage, reporting per-dataset IoU. Model selection uses the mean of the
  per-dataset supervised mIoU, never a pooled number across domains.
* The test manifests are never read here. Selection happens on validation only.
* Foam (class 3) has no supervision anywhere, so its class weight is 0 and it is
  excluded from the selection metric.
* Mixed precision is enabled on CUDA and silently disabled elsewhere.
"""

from __future__ import annotations

import csv
import math
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ml.src.data.datasets import (
    BeachSegmentationDataset,
    MultispectralConfig,
    compute_class_weights,
    filter_supervised_records,
    segmentation_collate,
)
from ml.src.data.mappings import build_all_mappings, class_names, load_class_config
from ml.src.data.sampler import build_dataset_balanced_sampler
from ml.src.data.splits import DatasetRecord, load_manifest
from ml.src.data.transforms import build_eval_transform, build_train_transform
from ml.src.models.factory import architecture_config, build_model
from ml.src.training.checkpoint import (
    checkpoint_path,
    load_torch_checkpoint,
    save_checkpoint,
)
from ml.src.training.losses import build_loss
from ml.src.training.metrics import ConfusionMatrix, boundary_f1, shoreline_metrics_from_masks
from ml.src.utils import (
    count_parameters,
    ensure_dir,
    get_logger,
    load_yaml,
    resolve_device,
    resolve_path,
    save_json,
    set_seed,
)

logger = get_logger("training.trainer")


@dataclass
class StageResult:
    architecture: str
    stage: str
    best_epoch: int
    best_metric: float
    checkpoint: str
    history: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Schedulers
# ---------------------------------------------------------------------------


def warmup_cosine_factor(epoch: int, total: int, warmup: int, min_ratio: float = 0.01) -> float:
    """LR multiplier for ``epoch`` (0-based): linear warmup then cosine decay."""
    if warmup > 0 and epoch < warmup:
        return 0.1 + 0.9 * (epoch + 1) / warmup
    progress = (epoch - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _stage_entry(config: dict[str, Any], stage_name: str) -> dict[str, Any]:
    for entry in config["training"]["stages"]:
        if entry["name"] == stage_name:
            return entry
    known = [entry["name"] for entry in config["training"]["stages"]]
    raise KeyError(f"Unknown stage {stage_name!r}; known stages: {known}")


def _load_records(config: dict[str, Any], dataset: str, split: str) -> list[DatasetRecord]:
    path = resolve_path(config["paths"]["manifests"]) / f"{dataset}_{split}.csv"
    return load_manifest(path)


def _multispectral_config(dataset: str) -> MultispectralConfig | None:
    if dataset != "india_sandy_beach":
        return None
    class_config = load_class_config()
    return MultispectralConfig.from_config(class_config["datasets"][dataset])


def build_dataset(
    config: dict[str, Any],
    dataset_name: str,
    records: list[DatasetRecord],
    *,
    train: bool,
) -> BeachSegmentationDataset:
    mappings = build_all_mappings(load_class_config())
    mean = tuple(config["preprocessing"]["imagenet_mean"])
    std = tuple(config["preprocessing"]["imagenet_std"])
    size = int(config["data"]["image_size"])
    if train:
        transform = build_train_transform(
            size,
            config["preprocessing"].get("train_augmentation", {}),
            mean=mean,
            std=std,
        )
    else:
        transform = build_eval_transform(size, mean=mean, std=std)
    return BeachSegmentationDataset(
        records,
        mappings[dataset_name],
        transform=transform,
        multispectral=_multispectral_config(dataset_name),
    )


def _seed_worker(worker_id: int, *, seed: int) -> None:
    """Seed a DataLoader worker deterministically.

    Must be a module-level function: on Windows the loader uses ``spawn``, so
    the callable is pickled by reference and closures cannot be sent to the
    child process.
    """
    import random

    import torch

    worker_seed = seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _make_loader(
    dataset: Any,
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    sampler: Any | None = None,
    pin_memory: bool = True,
    seed: int = 42,
) -> Any:
    from functools import partial

    from torch.utils.data import DataLoader

    generator = None
    if shuffle:
        import torch

        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=len(dataset) >= 2 * batch_size,
        persistent_workers=num_workers > 0,
        worker_init_fn=partial(_seed_worker, seed=seed) if num_workers > 0 else None,
        generator=generator,
        collate_fn=segmentation_collate,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _amp_enabled(config: dict[str, Any], device: str) -> bool:
    return bool(config["training"].get("mixed_precision", True)) and device.startswith("cuda")


def _iter_batches(
    loader: Iterable[dict[str, Any]], max_batches: int | None
) -> Iterator[dict[str, Any]]:
    for index, batch in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            return
        yield batch


def evaluate_loader(
    model: Any,
    loader: Iterable[dict[str, Any]],
    *,
    device: str,
    num_classes: int,
    ignore_index: int,
    max_batches: int | None = None,
    collect_boundary: bool = False,
    boundary_tolerance_px: int = 2,
    water_classes: tuple[int, ...] = (2, 3),
) -> dict[str, Any]:
    import torch

    model.eval()
    confusion = ConfusionMatrix(num_classes=num_classes, ignore_index=ignore_index)
    boundary_rows: list[dict[str, Any]] = []
    shoreline_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in _iter_batches(loader, max_batches):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            predictions = logits.argmax(dim=1)
            prediction_np = predictions.detach().cpu().numpy()
            mask_np = masks.detach().cpu().numpy()
            confusion.update(prediction_np, mask_np)
            if collect_boundary:
                for prediction, target in zip(prediction_np, mask_np, strict=True):
                    boundary_rows.append(
                        boundary_f1(
                            prediction,
                            target,
                            ignore_index=ignore_index,
                            tolerance_px=boundary_tolerance_px,
                        )
                    )
                    shoreline_rows.append(
                        shoreline_metrics_from_masks(
                            prediction,
                            target,
                            ignore_index=ignore_index,
                            tolerance_px=boundary_tolerance_px,
                            water_classes=water_classes,
                        )
                    )
    metrics = confusion.compute()
    metrics["inference_seconds"] = round(time.perf_counter() - started, 3)
    if collect_boundary:
        metrics["boundary"] = _aggregate_boundary(boundary_rows)
        metrics["shoreline"] = _aggregate_shoreline(shoreline_rows)
    return metrics


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_boundary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for class_id in ("1", "2"):
        usable = [row[class_id] for row in rows if row.get(class_id, {}).get("f1") is not None]
        result[class_id] = {
            "images": len(usable),
            "f1": _mean_or_none([row["f1"] for row in usable]),
            "precision": _mean_or_none(
                [row["precision"] for row in usable if row["precision"] is not None]
            ),
            "recall": _mean_or_none([row["recall"] for row in usable if row["recall"] is not None]),
        }
    # boundary_f1() stores the mean under "mean_f1"
    all_f1 = [row["mean_f1"] for row in rows if row.get("mean_f1") is not None]
    result["mean_f1"] = _mean_or_none([float(v) for v in all_f1])
    return result


def _aggregate_shoreline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row.get("available")]
    if not usable:
        return {"available": False, "reason": "no image had both predicted and ground-truth water"}
    return {
        "available": True,
        "images": len(usable),
        "boundary_f1": _mean_or_none(
            [row["boundary_f1"] for row in usable if row.get("boundary_f1") is not None]
        ),
        "boundary_precision": _mean_or_none(
            [
                row["boundary_precision"]
                for row in usable
                if row.get("boundary_precision") is not None
            ]
        ),
        "boundary_recall": _mean_or_none(
            [row["boundary_recall"] for row in usable if row.get("boundary_recall") is not None]
        ),
        "mean_boundary_distance_px": _mean_or_none(
            [
                row["mean_boundary_distance_px"]
                for row in usable
                if row.get("mean_boundary_distance_px") is not None
            ]
        ),
        "mean_vertical_displacement_px": _mean_or_none(
            [
                row["mean_vertical_displacement_px"]
                for row in usable
                if row.get("mean_vertical_displacement_px") is not None
            ]
        ),
    }


def _validation_loss(
    model: Any,
    loader: Iterable[dict[str, Any]],
    *,
    device: str,
    loss_function: Any,
    max_batches: int | None = None,
) -> float | None:
    import torch

    model.eval()
    totals: list[float] = []
    with torch.no_grad():
        for batch in _iter_batches(loader, max_batches):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            _, parts = loss_function(model(images), masks)
            if parts["loss"] > 0:
                totals.append(float(parts["loss"]))
    return float(np.mean(totals)) if totals else None


def _selection_score(
    config: dict[str, Any], stage: dict[str, Any], validation: dict[str, Any]
) -> float:
    """Checkpoint selection score.

    India's masks only annotate sand (no negatives), so a model that predicts
    "land everywhere" scores a perfect IoU there. Selecting on that would make
    the metric useless, so stage 1 selects on validation loss and stage 4
    selects on the two datasets that actually annotate both classes.
    """
    metric_name = stage.get(
        "selection_metric", config["training"].get("selection_metric", "mean_dataset_iou")
    )
    chosen = stage.get("selection_datasets")
    per_dataset = validation["per_dataset"]
    if chosen:
        per_dataset = {name: data for name, data in per_dataset.items() if name in chosen}
    if metric_name == "negative_val_loss":
        losses = [
            data["val_loss"] for data in per_dataset.values() if data.get("val_loss") is not None
        ]
        return -float(np.mean(losses)) if losses else -1e9
    scores = [
        data["supervised_mean_iou"]
        for data in per_dataset.values()
        if data.get("supervised_mean_iou") is not None
    ]
    return float(np.mean(scores)) if scores else 0.0


def _supervised_mean_iou(metrics: dict[str, Any], unsupervised: set[int]) -> float | None:
    values = [
        row["iou"]
        for index, row in metrics["per_class"].items()
        if int(index) not in unsupervised and row["iou"] is not None
    ]
    return float(np.mean(values)) if values else None


def per_dataset_validation(
    model: Any,
    val_loaders: dict[str, Any],
    *,
    device: str,
    num_classes: int,
    ignore_index: int,
    unsupervised_classes: Iterable[int],
    max_batches: int | None = None,
    loss_function: Any | None = None,
) -> dict[str, Any]:
    unsupervised = {int(c) for c in unsupervised_classes}
    per_dataset: dict[str, Any] = {}
    scores: list[float] = []
    for dataset, loader in val_loaders.items():
        metrics = evaluate_loader(
            model,
            loader,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            max_batches=max_batches,
        )
        if loss_function is not None:
            metrics["val_loss"] = _validation_loss(
                model, loader, device=device, loss_function=loss_function, max_batches=max_batches
            )
        metrics["supervised_mean_iou"] = _supervised_mean_iou(metrics, unsupervised)
        per_dataset[dataset] = metrics
        if metrics["supervised_mean_iou"] is not None:
            scores.append(metrics["supervised_mean_iou"])
    return {
        "per_dataset": per_dataset,
        "mean_dataset_iou": float(np.mean(scores)) if scores else 0.0,
        "datasets_scored": len(scores),
    }


# ---------------------------------------------------------------------------
# Training stage
# ---------------------------------------------------------------------------


def _build_optimizer_and_scheduler(
    config: dict[str, Any], model: Any, stage: dict[str, Any], total_epochs: int
) -> tuple[Any, Any]:
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import LambdaLR

    base_lr = float(stage.get("learning_rate", config["training"]["learning_rate"]))
    if stage.get("dataset_balance") == "equal":
        base_lr *= float(config["training"].get("joint", {}).get("learning_rate_scale", 0.2))
    optimizer = AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    warmup = int(config["training"].get("warmup_epochs", 1))
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda epoch: warmup_cosine_factor(epoch, total_epochs, warmup)
    )
    return optimizer, scheduler


def run_stage(
    config: dict[str, Any],
    architecture: str,
    stage_name: str,
    *,
    epochs: int | None = None,
    device: str = "auto",
    max_batches: int | None = None,
    max_samples: int | None = None,
    init_checkpoint: str | Path | None = None,
    smoke: bool = False,
) -> StageResult:
    import torch

    stage = _stage_entry(config, stage_name)
    device = resolve_device(device)
    seed = int(config["project"]["seed"])
    set_seed(seed, deterministic=bool(config["project"].get("deterministic", True)))
    logger.info(
        "stage=%s architecture=%s device=%s datasets=%s",
        stage_name,
        architecture,
        device,
        stage["datasets"],
    )

    num_classes = int(config["data"]["num_classes"])
    ignore_index = int(config["data"]["ignore_index"])
    batch_size = int(config["data"]["batch_size"])
    num_workers = int(config["data"]["num_workers"])
    if device == "cpu":
        num_workers = min(num_workers, 2)
    pin_memory = bool(config["data"].get("pin_memory", True)) and device.startswith("cuda")

    train_datasets: list[BeachSegmentationDataset] = []
    val_loaders: dict[str, Any] = {}
    train_records: dict[str, list[DatasetRecord]] = {}
    dataset_sizes: list[int] = []
    mappings = build_all_mappings(load_class_config())
    for dataset_name in stage["datasets"]:
        train = _load_records(config, dataset_name, "train")
        val = _load_records(config, dataset_name, "val")
        train = filter_supervised_records(train, mappings[dataset_name], label="train")
        val = filter_supervised_records(val, mappings[dataset_name], label="val")
        if max_samples:
            train = train[:max_samples]
            val = val[: max(2, min(len(val), max_samples))]
        if not train:
            raise RuntimeError(f"{dataset_name}: empty train manifest")
        if not val:
            raise RuntimeError(f"{dataset_name}: empty val manifest")
        train_records[dataset_name] = train
        train_datasets.append(build_dataset(config, dataset_name, train, train=True))
        dataset_sizes.append(len(train))
        val_datasets = [build_dataset(config, dataset_name, val, train=False)]
        val_loaders[dataset_name] = _make_loader(
            torch.utils.data.ConcatDataset(val_datasets),
            batch_size=int(config["evaluation"].get("batch_size", batch_size)),
            num_workers=min(num_workers, 2),
            shuffle=False,
            pin_memory=False,
            seed=seed,
        )

    class_weights = compute_class_weights(
        train_records,
        mappings,
        num_classes=num_classes,
        ignore_index=ignore_index,
        cache_path=resolve_path(config["paths"]["manifests"]) / f"class_weights_{stage_name}.json",
        max_weight=float(config["training"].get("class_weight_max", 5.0)),
    )
    if smoke:
        class_weights = np.where(class_weights > 0, 1.0, 0.0).astype(np.float32)
    logger.info("stage class weights: %s", class_weights.round(3).tolist())

    sampler = None
    if stage.get("dataset_balance") == "equal" and len(train_datasets) > 1:
        joint = config["training"].get("joint", {})
        sampler = build_dataset_balanced_sampler(
            dataset_sizes,
            samples_per_epoch=int(joint.get("samples_per_epoch", 6000)),
            generator_seed=seed,
        )
        logger.info("balanced sampling: sizes=%s samples/epoch=%d", dataset_sizes, len(sampler))
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    train_loader = _make_loader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        sampler=sampler,
        pin_memory=pin_memory,
        seed=seed,
    )

    arch_cfg = architecture_config(config, architecture)
    model = build_model(
        architecture,
        num_classes=num_classes,
        encoder_name=arch_cfg["encoder"],
        encoder_weights=arch_cfg["encoder_weights"],
        in_channels=arch_cfg["in_channels"],
    )
    init_path: Path | None = None
    if init_checkpoint:
        init_path = resolve_path(init_checkpoint)
    elif stage.get("init_from"):
        candidate = checkpoint_path(config, architecture, stage["init_from"], "best")
        if candidate.is_file():
            init_path = candidate
    if init_path and init_path.is_file():
        checkpoint = load_torch_checkpoint(init_path, map_location="cpu")
        state = checkpoint["model_state_dict"]
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        model.load_state_dict(state)
        logger.info("initialised from %s", init_path)
    elif stage.get("init_from"):
        logger.warning(
            "checkpoint for stage %r not found (%s); continuing from ImageNet weights",
            stage["init_from"],
            init_path,
        )
    model.to(device)
    logger.info("trainable parameters: %s", f"{count_parameters(model):,}")

    total_epochs = int(epochs or stage.get("epochs", config["training"]["epochs_per_stage"]))
    if smoke:
        total_epochs = min(total_epochs, 1)
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model, stage, total_epochs)

    loss_function = build_loss(
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_weights=class_weights,
        ce_weight=float(config["training"].get("ce_weight", 0.5)),
        dice_weight=float(config["training"].get("dice_weight", 0.5)),
    ).to(device)

    amp = _amp_enabled(config, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    patience = int(config["training"].get("early_stopping_patience", 10))
    log_every = int(config["training"].get("log_every", 25))
    grad_clip = float(config["training"].get("grad_clip_norm", 1.0))
    unsupervised = config["evaluation"].get("unsupervised_classes", [3])

    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = -1
    epochs_without_improvement = 0
    started = time.perf_counter()
    reports_dir = ensure_dir(resolve_path(config["paths"]["reports"]) / "training" / architecture)

    for epoch in range(total_epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_ce: list[float] = []
        epoch_dice: list[float] = []
        epoch_started = time.perf_counter()
        for batch_index, batch in enumerate(_iter_batches(train_loader, max_batches)):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(images)
                loss, parts = loss_function(logits, masks)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1} batch {batch_index}: {parts}"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            epoch_losses.append(parts["loss"])
            epoch_ce.append(parts["ce"])
            epoch_dice.append(parts["dice"])
            if log_every > 0 and (batch_index + 1) % log_every == 0:
                window = slice(-min(log_every, len(epoch_losses)), None)
                logger.info(
                    "epoch %d/%d it %d loss %.4f ce %.4f dice %.4f",
                    epoch + 1,
                    total_epochs,
                    batch_index + 1,
                    float(np.mean(epoch_losses[window])),
                    float(np.mean(epoch_ce[window])),
                    float(np.mean(epoch_dice[window])),
                )
        scheduler.step()

        validation = per_dataset_validation(
            model,
            val_loaders,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
            unsupervised_classes=unsupervised,
            max_batches=max_batches,
            loss_function=loss_function,
        )
        metric = _selection_score(config, stage, validation)
        row = {
            "epoch": epoch + 1,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "train_ce": float(np.mean(epoch_ce)) if epoch_ce else None,
            "train_dice": float(np.mean(epoch_dice)) if epoch_dice else None,
            "val_mean_dataset_iou": metric,
            "val_per_dataset": {
                name: {
                    "mean_iou": data["mean_iou"],
                    "supervised_mean_iou": data["supervised_mean_iou"],
                    "mean_dice": data["mean_dice"],
                    "pixel_accuracy": data["pixel_accuracy"],
                    "val_loss": data.get("val_loss"),
                    "per_class": data["per_class"],
                }
                for name, data in validation["per_dataset"].items()
            },
            "seconds": round(time.perf_counter() - epoch_started, 2),
        }
        history.append(row)
        logger.info(
            "epoch %d/%d loss %.4f selection %.4f [%s] (%.1fs)",
            epoch + 1,
            total_epochs,
            row["train_loss"] or 0.0,
            metric,
            ", ".join(
                f"{name}={data['supervised_mean_iou']:.4f}"
                if data["supervised_mean_iou"] is not None
                else f"{name}=n/a"
                for name, data in validation["per_dataset"].items()
            ),
            row["seconds"],
        )

        improved = metric > best_metric
        if improved:
            best_metric = metric
            best_epoch = epoch + 1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        checkpoint_metrics = validation
        if improved:
            save_checkpoint(
                checkpoint_path(config, architecture, stage_name, "best"),
                model=model,
                architecture=architecture,
                encoder=arch_cfg["encoder"],
                stage=stage_name,
                epoch=epoch + 1,
                best_metric=best_metric,
                config=config,
                class_names=class_names(),
                metrics=checkpoint_metrics,
                optimizer=optimizer,
                extra={"class_weights": class_weights.tolist(), "smoke": smoke},
            )
        save_checkpoint(
            checkpoint_path(config, architecture, stage_name, "last"),
            model=model,
            architecture=architecture,
            encoder=arch_cfg["encoder"],
            stage=stage_name,
            epoch=epoch + 1,
            best_metric=best_metric,
            config=config,
            class_names=class_names(),
            metrics=checkpoint_metrics,
            optimizer=optimizer,
            extra={"class_weights": class_weights.tolist(), "smoke": smoke},
        )

        if not improved and epochs_without_improvement >= patience:
            logger.info(
                "early stopping after %d epochs without improvement (best epoch %d)",
                epochs_without_improvement,
                best_epoch,
            )
            break

    best_path = checkpoint_path(config, architecture, stage_name, "best")
    if not best_path.is_file():
        raise RuntimeError(f"stage {stage_name} produced no best checkpoint")

    save_json(
        reports_dir / f"{stage_name}_history.json",
        {
            "architecture": architecture,
            "stage": stage_name,
            "datasets": stage["datasets"],
            "device": device,
            "epochs_requested": total_epochs,
            "epochs_completed": len(history),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "selection_metric": stage.get(
                "selection_metric", config["training"].get("selection_metric", "mean_dataset_iou")
            ),
            "selection_datasets": stage.get("selection_datasets", stage["datasets"]),
            "class_weights": class_weights.tolist(),
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
            "smoke": smoke,
        },
    )
    _write_history_csv(reports_dir / f"{stage_name}_history.csv", history)
    logger.info(
        "stage %s done: best epoch %d, val mean dataset IoU %.4f",
        stage_name,
        best_epoch,
        best_metric,
    )

    return StageResult(
        architecture=architecture,
        stage=stage_name,
        best_epoch=best_epoch,
        best_metric=best_metric,
        checkpoint=str(best_path),
        history=history,
    )


def _write_history_csv(path: Path, history: list[dict[str, Any]]) -> None:
    if not history:
        return
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_ce",
        "train_dice",
        "val_mean_dataset_iou",
        "seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)


def run_all_stages(
    config_path: str = "ml/configs/config.yaml",
    architecture: str = "unet",
    *,
    epochs: int | None = None,
    device: str = "auto",
    stages: list[str] | None = None,
    smoke: bool = False,
    max_batches: int | None = None,
    max_samples: int | None = None,
) -> list[StageResult]:
    config = load_yaml(config_path)
    selected = stages or [entry["name"] for entry in config["training"]["stages"]]
    results: list[StageResult] = []
    for stage_name in selected:
        results.append(
            run_stage(
                config,
                architecture,
                stage_name,
                epochs=epochs,
                device=device,
                smoke=smoke,
                max_batches=max_batches,
                max_samples=max_samples,
            )
        )
    return results
