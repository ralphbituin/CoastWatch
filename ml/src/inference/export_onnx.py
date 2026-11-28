"""Export a trained PyTorch checkpoint to the CoastWatch ONNX artifact.

Two architectures are supported by this one exporter:

* the legacy four-class U-Net / DeepLabV3+ checkpoints (``export``), and
* the active three-class SegFormer MiT-B2 checkpoint (``export_segformer``).

Output layout (consumed by ``backend/app/services/vision/segmentation.py``)::

    <output_dir>/model.onnx
    <output_dir>/config.json
    <output_dir>/best_model.pth      (optional, kept for reproducibility)

The export is not assumed correct: it is verified by running the exported graph
with ONNX Runtime and comparing class decisions against PyTorch on the same
inputs. If the verification fails, the export fails loudly. The SegFormer path
additionally verifies the checkpoint identity, the class contract, and native
resolution parity on real frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.src.models.factory import build_model
from ml.src.training.checkpoint import load_torch_checkpoint, write_model_metadata
from ml.src.utils import ensure_dir, get_logger, load_yaml, project_root, resolve_path, save_json

logger = get_logger("inference.export")

DEFAULT_OPSET = 17

SEGFORMER_CLASS_NAMES = ("Beach", "Ocean", "Sky")
SEGFORMER_CLASS_IDS = {"Beach": 0, "Ocean": 1, "Sky": 2}
SEGFORMER_CLASS_COLORS = {
    "Beach": [214, 181, 137],
    "Ocean": [32, 112, 190],
    "Sky": [135, 206, 235],
}
# Checkpoint-identity guard. The task pins this exact file; refuse to export a
# different one silently.
EXPECTED_SEGFORMER_SHA256 = "1FDE1C73284DA9F686C2481E0CF758CD3DE23392E095F8190F36297BC133BE84"
EXPECTED_SEGFORMER_EPOCH = 18
EXPECTED_SEGFORMER_VAL_MIOU = 0.9874
SEGFORMER_LIMITATIONS = [
    "Trained on 131 image/mask pairs from a single small annotated dataset "
    "(image_sam3). The 20-image held-out test split is not a production or "
    "cross-camera accuracy claim.",
    "No nighttime or low-light test coverage; night behaviour is unvalidated.",
    "The swash zone is often unlabelled (source background 0=Sky is converted "
    "to ignore 255 outside polygons). In 6/20 test frames the annotated Beach "
    "and Ocean do not touch, so no ground-truth shoreline exists there.",
    "No separate foam/whitewater class; wave foam is learned as Ocean or Beach.",
    "Far-shore Ocean/Beach interfaces can add a second branch; shoreline output "
    "is split into ordered segments rather than smoothed across branches.",
    "No foam, tide, bathymetry or metric world-coordinate accuracy is claimed. "
    "Calibration is required for metres.",
]


def _repo_relative(path: str | Path) -> str:
    path = resolve_path(path)
    try:
        return path.relative_to(project_root()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_config_dict(config_path: str | Path) -> dict[str, Any]:
    return load_yaml(config_path)


def _checkpoint_metadata(checkpoint: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    input_size = config.get("inference", {}).get("input_size", [512, 512])
    return {
        "version": str(checkpoint.get("format_version", 1)) + ".0.0",
        "encoder": checkpoint.get("encoder"),
        "input_size": [int(input_size[0]), int(input_size[1])],
        "normalization": checkpoint.get(
            "normalization",
            {
                "mean": config["preprocessing"]["imagenet_mean"],
                "std": config["preprocessing"]["imagenet_std"],
            },
        ),
        "stage": checkpoint.get("stage"),
        "epoch": checkpoint.get("epoch"),
    }


def export(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "ml/configs/config.yaml",
    metrics: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
    opset: int = DEFAULT_OPSET,
    verify: bool = True,
    copy_checkpoint: bool = True,
) -> dict[str, Any]:
    import torch

    config = _load_config_dict(config_path)
    checkpoint = load_torch_checkpoint(checkpoint_path, map_location="cpu")
    architecture = checkpoint["architecture"]
    num_classes = int(checkpoint.get("num_classes", config["data"]["num_classes"]))

    model = build_model(
        architecture,
        num_classes=num_classes,
        encoder_name=checkpoint.get("encoder"),
        encoder_weights=None,
        in_channels=int(checkpoint.get("in_channels", 3)),
    )
    state = checkpoint["model_state_dict"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()

    input_size = config.get("inference", {}).get("input_size", [512, 512])
    height, width = int(input_size[1]), int(input_size[0])
    output_dir = ensure_dir(resolve_path(output_dir))
    onnx_path = output_dir / "model.onnx"

    dummy = torch.randn(1, int(config["model"]["in_channels"]), height, width, dtype=torch.float32)
    logger.info("exporting %s to %s (opset %d)", architecture, onnx_path, opset)
    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            # Batch *and* spatial axes are dynamic so the same artifact can run
            # at a different inference resolution without re-exporting.
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            },
        )

    verification: dict[str, Any] = {"performed": False}
    if verify:
        verification = _verify(model, onnx_path, dummy)
        logger.info("ONNX verification: %s", verification)
        if not verification.get("passed"):
            raise RuntimeError(
                "ONNX export verification failed: " f"{verification.get('reason', 'see details')}"
            )

    limitations = limitations or [
        "Foam (class 3) has no training annotations in the provided datasets and is untrained.",
        "Trained on satellite chips and ground/CoastSnap photography; webcam frames are "
        "unvalidated.",
        "No metric in this artifact is a metre-scale shoreline accuracy claim; calibration is "
        "required.",
    ]
    metadata = _checkpoint_metadata(checkpoint, config)
    config_file = write_model_metadata(
        output_dir / "config.json",
        architecture=architecture,
        checkpoint_metadata=metadata,
        metrics=metrics or checkpoint.get("metrics", {}),
        limitations=limitations,
        class_config_path="ml/configs/class_mapping.yaml",
        config_path=str(config_path),
    )

    # Extra fields the exporter knows but write_model_metadata does not.
    payload = load_config(config_file)
    payload.update(
        {
            "artifact": {
                "onnx": onnx_path.name,
                "opset": opset,
                "input_name": "input",
                "output_name": "logits",
                "dynamic_batch": True,
            },
            "providers": config.get("inference", {}).get(
                "providers", ["CPUExecutionProvider"]
            ),
            "preprocessing": {
                "resize": "bilinear",
                "resize_target": [width, height],
                "mean": payload["normalization"]["mean"],
                "std": payload["normalization"]["std"],
                "input_range": "0-255 uint8, RGB, resized then normalised",
            },
            "verification": verification,
            "checkpoint": {
                "source": str(resolve_path(checkpoint_path)),
                "stage": checkpoint.get("stage"),
                "epoch": checkpoint.get("epoch"),
                "best_metric": checkpoint.get("best_metric"),
                "class_weights": checkpoint.get("extra", {}).get("class_weights"),
            },
            "schema_source": {
                "class_mapping": "ml/configs/class_mapping.yaml",
                "config": "ml/configs/config.yaml",
            },
        }
    )
    save_json(config_file, payload)

    if copy_checkpoint:
        import shutil

        destination = output_dir / "best_model.pth"
        shutil.copy2(resolve_path(checkpoint_path), destination)
        logger.info("copied checkpoint to %s", destination)

    _write_model_card(
        output_dir / "README.md",
        architecture=architecture,
        checkpoint=checkpoint,
        metadata=payload,
        verification=verification,
        limitations=limitations,
        metrics=metrics or checkpoint.get("metrics", {}),
    )

    save_json(
        output_dir / "export_report.json",
        {
            "architecture": architecture,
            "encoder": checkpoint.get("encoder"),
            "onnx_bytes": onnx_path.stat().st_size,
            "verification": verification,
            "metrics_present": bool(metrics or checkpoint.get("metrics")),
        },
    )
    return {
        "onnx": str(onnx_path),
        "config": str(config_file),
        "verification": verification,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify(model: Any, onnx_path: Path, dummy: Any) -> dict[str, Any]:
    import onnxruntime as ort
    import torch

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        with torch.no_grad():
            reference = model(dummy).detach().cpu().numpy()
        exported = session.run(None, {input_name: dummy.numpy()})[0]
        if exported.shape != reference.shape:
            return {
                "performed": True,
                "passed": False,
                "reason": f"shape mismatch: torch {reference.shape} vs onnx {exported.shape}",
            }
        max_abs = float(np.max(np.abs(reference - exported)))
        torch_labels = reference.argmax(axis=1)
        onnx_labels = exported.argmax(axis=1)
        agreement = float((torch_labels == onnx_labels).mean())
        passed = max_abs < 1e-3 and agreement == 1.0
        return {
            "performed": True,
            "passed": bool(passed),
            "max_abs_diff": max_abs,
            "label_agreement": agreement,
            "reason": None if passed else "logits differ or class decisions disagree",
        }
    except Exception as exc:  # pragma: no cover - unexpected runtime issues
        return {"performed": True, "passed": False, "reason": f"{type(exc).__name__}: {exc}"}


def _load_segformer_checkpoint(
    checkpoint_path: str | Path,
    config: dict[str, Any],
    *,
    device: str = "cpu",
    allow_any_checkpoint: bool = False,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Load ``best.pt`` into the three-class SegFormer without random weights.

    The ADE-20k checkpoint is only the *training* initializer. At inference we
    build from the trained Hugging Face copy next to the checkpoint and then
    load every tensor from ``best.pt`` with ``strict=True``; a missing or extra
    key aborts. Finally each loaded tensor is compared with the checkpoint to
    prove that no randomly-initialised parameter survived.
    """
    import torch

    from ml.src.model import SegFormerB2

    checkpoint_path = resolve_path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SegFormer checkpoint not found: {checkpoint_path}")

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state")
    if not isinstance(state, dict) or not state:
        raise RuntimeError(f"{checkpoint_path} has no usable 'model_state' mapping")

    num_labels = int(payload.get("num_labels", config.get("data", {}).get("num_classes", 3)))
    class_names = list(payload.get("class_names", config.get("data", {}).get("class_names", [])))
    ignore_index = int(payload.get("ignore_index", config.get("data", {}).get("ignore_index", 255)))
    if num_labels != 3:
        raise RuntimeError(f"{checkpoint_path} declares num_labels={num_labels}, expected 3")
    if class_names != list(SEGFORMER_CLASS_NAMES):
        raise RuntimeError(
            f"{checkpoint_path} class_names={class_names}, expected {list(SEGFORMER_CLASS_NAMES)}"
        )
    if ignore_index != 255:
        raise RuntimeError(f"{checkpoint_path} ignore_index={ignore_index}, expected 255")

    sha256 = _sha256(checkpoint_path)
    if not allow_any_checkpoint:
        if sha256 != EXPECTED_SEGFORMER_SHA256:
            raise RuntimeError(
                f"checkpoint SHA-256 mismatch: {sha256} != expected {EXPECTED_SEGFORMER_SHA256}. "
                "Refusing to export a different checkpoint."
            )
    epoch = int(payload.get("epoch", -1))
    val_miou = float(payload.get("best_val_iou", float("nan")))
    if not allow_any_checkpoint:
        if epoch != EXPECTED_SEGFORMER_EPOCH:
            raise RuntimeError(f"checkpoint epoch={epoch}, expected {EXPECTED_SEGFORMER_EPOCH}")
        if not abs(val_miou - EXPECTED_SEGFORMER_VAL_MIOU) < 5e-5:
            raise RuntimeError(
                f"checkpoint val mIoU={val_miou}, expected {EXPECTED_SEGFORMER_VAL_MIOU}"
            )
    else:
        logger.warning(
            "exporting a candidate checkpoint without the pinned identity guard: "
            "sha256=%s epoch=%s val_mIoU=%s",
            sha256,
            epoch,
            val_miou,
        )

    hf_dir = checkpoint_path.parent / "best_hf"
    if not hf_dir.is_dir():
        raise FileNotFoundError(
            f"{hf_dir} is missing. It is the local trained initializer used to build the "
            "3-class architecture without downloading ADE-20k weights."
        )
    model = SegFormerB2(
        num_labels=num_labels,
        pretrained=str(hf_dir),
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.load_state_dict(state, strict=True)

    loaded = model.state_dict()
    mismatched = [key for key, value in state.items() if not torch.equal(value, loaded[key])]
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} checkpoint tensor(s) were not loaded exactly; "
            f"first: {mismatched[:3]}"
        )
    model.to(device)
    model.eval()

    identity = {
        "path": str(checkpoint_path),
        "sha256": sha256,
        "size_bytes": int(checkpoint_path.stat().st_size),
        "epoch": epoch,
        "val_mean_iou": val_miou,
        "num_labels": num_labels,
        "class_names": class_names,
        "ignore_index": ignore_index,
        "pretrained_initializer": str(payload.get("pretrained")),
        "tensors_verified_equal": len(state),
    }
    logger.info("verified SegFormer checkpoint %s (%s)", checkpoint_path, sha256)
    return model, payload, identity


def _bilinear_align_corners_false(chw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize matching PyTorch ``align_corners=False`` semantics."""
    channels, in_h, in_w = chw.shape
    if (in_h, in_w) == (out_h, out_w):
        return chw.astype(np.float32, copy=False)

    ys = (np.arange(out_h, dtype=np.float64) + 0.5) * (in_h / out_h) - 0.5
    xs = (np.arange(out_w, dtype=np.float64) + 0.5) * (in_w / out_w) - 0.5
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    wy = (ys - y0).astype(np.float32)[:, None]
    wx = (xs - x0).astype(np.float32)[None, :]
    y1 = y0 + 1
    x1 = x0 + 1
    y0 = np.clip(y0, 0, in_h - 1)
    y1 = np.clip(y1, 0, in_h - 1)
    x0 = np.clip(x0, 0, in_w - 1)
    x1 = np.clip(x1, 0, in_w - 1)

    result = np.empty((channels, out_h, out_w), dtype=np.float32)
    for channel in range(channels):
        plane = chw[channel]
        top = (
            plane[np.ix_(y0, x0)] * (1.0 - wx)
            + plane[np.ix_(y0, x1)] * wx
        )
        bottom = (
            plane[np.ix_(y1, x0)] * (1.0 - wx)
            + plane[np.ix_(y1, x1)] * wx
        )
        result[channel] = top * (1.0 - wy) + bottom * wy
    return result


def _verify_segformer_onnx(
    model: Any,
    onnx_path: Path,
    frame_paths: list[Path],
    transform: Any,
    *,
    device: str = "cpu",
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Compare ONNX Runtime against PyTorch at 1/4 and native resolution."""
    import onnxruntime as ort
    import torch
    import torch.nn.functional as F

    from ml.scripts.infer import predict_mask
    from ml.src.dataset import load_image_rgb

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    max_abs_diff = 0.0
    quarter_agreement_sum = 0.0
    native_agreement_sum = 0.0
    total_native_pixels = 0
    mismatched_native_pixels = 0
    per_frame: list[dict[str, Any]] = []

    for path in frame_paths:
        image = load_image_rgb(path)
        height, width = image.shape[:2]
        pixel_values = transform(image=image)["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            reference = model(pixel_values).detach().cpu().numpy()
        exported = np.asarray(session.run(None, {input_name: pixel_values.cpu().numpy()})[0])

        if reference.shape != exported.shape:
            raise RuntimeError(
                f"ONNX/PyTorch shape mismatch on {path.name}: {reference.shape} vs {exported.shape}"
            )
        frame_max = float(np.max(np.abs(reference - exported)))
        max_abs_diff = max(max_abs_diff, frame_max)
        quarter = float((reference.argmax(axis=1) == exported.argmax(axis=1)).mean())
        quarter_agreement_sum += quarter

        # Native-resolution parity, exactly as the backend does it: upsample
        # logits (never the argmax) with bilinear align_corners=False.
        reference_up = F.interpolate(
            torch.from_numpy(reference), size=(height, width), mode="bilinear", align_corners=False
        )
        reference_native = reference_up.argmax(dim=1)[0].numpy().astype(np.uint8)
        exported_native = _bilinear_align_corners_false(exported[0], height, width).argmax(axis=0)
        exported_native = exported_native.astype(np.uint8)
        native_agreement = float((reference_native == exported_native).mean())
        native_agreement_sum += native_agreement
        mismatched = int((reference_native != exported_native).sum())
        mismatched_native_pixels += mismatched
        total_native_pixels += int(reference_native.size)

        # The shared reference implementation must agree with the native
        # PyTorch path it defines; this catches accidental divergences.
        predict_native = predict_mask(
            model, image, transform, torch.device(device), use_amp=False
        )
        predict_agreement = float((predict_native == reference_native).mean())
        if predict_agreement < 1.0:
            raise RuntimeError(
                f"predict_mask disagrees with the direct native PyTorch path on {path.name}: "
                f"{predict_agreement:.6f}"
            )

        per_frame.append(
            {
                "image": path.name,
                "size": [int(width), int(height)],
                "max_abs_diff": frame_max,
                "quarter_label_agreement": quarter,
                "native_label_agreement": native_agreement,
                "native_mismatch_pixels": mismatched,
            }
        )
        logger.info(
            "parity %s: logits %.3g | 1/4 %.6f | native %.6f",
            path.name,
            frame_max,
            quarter,
            native_agreement,
        )

    count = max(1, len(per_frame))
    result = {
        "performed": True,
        "frame_count": len(per_frame),
        "tolerance_max_abs_logit_diff": tolerance,
        "max_abs_diff": max_abs_diff,
        "quarter_label_agreement": quarter_agreement_sum / count,
        "native_label_agreement": native_agreement_sum / count,
        "native_mismatch_pixels": mismatched_native_pixels,
        "native_total_pixels": total_native_pixels,
        "predict_mask_agreement": 1.0,
        "frames": per_frame,
    }
    result["passed"] = bool(
        len(per_frame) >= 20
        and max_abs_diff < tolerance
        and result["quarter_label_agreement"] >= 0.9999
        and result["native_label_agreement"] >= 0.9999
    )
    if not result["passed"]:
        result["reason"] = (
            "need >=20 frames, max|logit diff| < tolerance, and >=0.9999 argmax "
            "agreement at both 1/4 and native resolution"
        )
    return result


def export_segformer(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    *,
    config_path: str | Path = "ml/configs/segformer_b2.yaml",
    frames_dir: str | Path = "ml/datasets/coastwatch_segformer/images",
    frames: list[str] | None = None,
    verify_frames: int = 20,
    opset: int = DEFAULT_OPSET,
    verify: bool = True,
    allow_any_checkpoint: bool = False,
    version: str = "2.0.0",
    metrics: dict[str, Any] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    limitations: list[str] | None = None,
) -> dict[str, Any]:
    """Export the three-class SegFormer MiT-B2 checkpoint and verify parity.

    ``metrics`` may be an evaluation report JSON (``evaluation_report.json``) and
    ``dataset_summary`` the combined dataset summary JSON. They populate the
    artifact metadata so a fine-tuned candidate does not inherit the baseline's
    131-pair / 20-image numbers. ``allow_any_checkpoint`` is required when the
    checkpoint is not the pinned baseline.
    """
    import torch

    from ml.src.dataset import build_eval_transform

    config = load_yaml(config_path)
    model, payload, identity = _load_segformer_checkpoint(
        checkpoint_path, config, allow_any_checkpoint=allow_any_checkpoint
    )
    input_size = config.get("inference", {}).get("input_size", [512, 512])
    width, height = int(input_size[0]), int(input_size[1])
    if (width, height) != (512, 512):
        raise RuntimeError(f"SegFormer input_size must be 512x512, got {input_size}")

    output_dir = ensure_dir(resolve_path(output_dir))
    onnx_path = output_dir / "model.onnx"
    dummy = torch.zeros(1, 3, height, width, dtype=torch.float32)

    class _LogitsOnly(torch.nn.Module):
        """Single-positional-argument view required by the ONNX tracer."""

        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values: Any) -> Any:
            return self.wrapped(pixel_values=pixel_values)

    export_module = _LogitsOnly(model).eval()
    logger.info("exporting SegFormer MiT-B2 to %s (opset %d)", onnx_path, opset)
    with torch.no_grad():
        torch.onnx.export(
            export_module,
            dummy,
            str(onnx_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            },
        )

    if frames is not None:
        frame_paths = [resolve_path(item) for item in frames]
    else:
        directory = resolve_path(frames_dir)
        test_split = directory.parent / "splits" / "test.txt"
        names: list[str] = []
        if test_split.is_file():
            names = [
                line.strip()
                for line in test_split.read_text(encoding="utf-8").splitlines()
                if line.strip() and not line.startswith("#")
            ]
        if not names:
            names = sorted(path.name for path in directory.glob("*.jpg"))
        frame_paths = [directory / name for name in names[:verify_frames]]
    frame_paths = [path for path in frame_paths if path.is_file()]
    if verify and len(frame_paths) < verify_frames:
        raise RuntimeError(
            f"parity verification needs {verify_frames} real frames, found {len(frame_paths)}"
        )

    verification: dict[str, Any] = {"performed": False, "frame_count": 0}
    if verify:
        transform = build_eval_transform(config)
        # Verify on CPU so the result is independent of the GPU that happens to
        # be present; the exported graph is provider-agnostic.
        verification = _verify_segformer_onnx(
            model, onnx_path, frame_paths, transform, device="cpu"
        )
        logger.info("SegFormer ONNX verification: %s", verification)
        if not verification.get("passed"):
            raise RuntimeError(
                "SegFormer ONNX verification failed: "
                f"{verification.get('reason', 'see verification report')}"
            )

    checkpoint_path = resolve_path(checkpoint_path)
    training_block: dict[str, Any] = {
        "config": _repo_relative(config_path),
        "dataset": (
            "ml/datasets/coastwatch_segformer "
            "(from ml/datasets/image_sam3 + image_sam3_mask)"
        ),
        "source_pairs": 132,
        "exact_duplicates_excluded": 1,
        "training_pairs": 131,
        "split": {"train": 91, "val": 20, "test": 20},
        "class_ids": dict(SEGFORMER_CLASS_IDS),
        "source_mask_mapping": {
            "0": "Sky->2",
            "1": "Ocean->1",
            "2": "Beach->0",
            "255": "ignore",
        },
        "unlabelled_background": (
            "39.1M pixels outside annotation polygons converted to ignore 255"
        ),
    }
    metrics_block: dict[str, Any] = {
        "evaluation_split": "held-out test, 20 images, native resolution",
        "test_images": 20,
        "selection_metric": "validation mean IoU only",
        "checkpoint_epoch": int(identity["epoch"]),
        "validation_mean_iou": float(identity["val_mean_iou"]),
        "test": {
            "mean_iou": 0.9778,
            "mean_dice": 0.9887,
            "pixel_accuracy": 0.9894,
            "per_class": {
                "Beach": {"iou": 0.9580, "dice": 0.9785},
                "Ocean": {"iou": 0.9777, "dice": 0.9887},
                "Sky": {"iou": 0.9976, "dice": 0.9988},
            },
        },
        "caveat": (
            "20-image single-dataset results; not production or cross-camera accuracy."
        ),
    }
    limitations_list = list(SEGFORMER_LIMITATIONS)

    if dataset_summary is not None:
        split_counts = {
            name: int(dataset_summary["splits"][name]["images"])
            for name in ("train", "val", "test")
        }
        ignore_pixels = int(dataset_summary["class_pixels"]["ignore"])
        training_block = {
            "config": _repo_relative(config_path),
            "dataset": (
                "ml/datasets/coastwatch_segformer_multisite "
                "(131 original converted pairs + 400 per-site SAM3 pairs)"
            ),
            "source_pairs": int(dataset_summary["total_images"]),
            "exact_duplicates_excluded": len(
                dataset_summary.get("excluded_duplicates", [])
            ),
            "training_pairs": split_counts["train"],
            "split": split_counts,
            "class_ids": dict(SEGFORMER_CLASS_IDS),
            "source_mask_mapping": {
                "0": "Sky->2",
                "1": "Ocean->1",
                "2": "Beach->0",
                "255": "ignore",
            },
            "unlabelled_background": (
                f"{ignore_pixels:,} pixels outside annotation polygons converted "
                "to ignore 255"
            ),
            "sites": len(dataset_summary.get("sites", [])),
        }
        limitations_list = [
            (
                f"Fine-tuned on {split_counts['train']} training images "
                f"({dataset_summary['total_images']} pairs total: 131 original "
                "Karon pairs + 400 per-site SAM3 pairs from 10 sites)."
            ),
            (
                "New-site masks are automatic SAM3 annotations, not human ground "
                "truth. The 100-image test contains 20 original Karon frames and "
                "80 frames from two unseen sites (Japan Shirahama, USA Turquoise "
                "Beach); metrics measure agreement with those annotations."
            ),
            (
                "Japan night/dawn remains weak: Beach IoU is about 0.30 on several "
                "frames and shoreline mean MAE is inflated by branch-switch "
                "outliers (median 7.7 px, mean 45.1 px, p95 270.9 px)."
            ),
            (
                "Low-light mIoU improved from 0.6191 to 0.7111 on 26 test images, "
                "but no true night tag is represented; nighttime performance is "
                "still unvalidated."
            ),
            (
                "Beach precision decreased (0.9131 -> 0.8697) while recall "
                "increased (0.9385 -> 0.9822); wet sand may be over-predicted."
            ),
            (
                "No separate foam/whitewater class; wave foam is learned as Ocean "
                "or Beach. Far-shore Ocean/Beach interfaces can add a second "
                "shoreline branch."
            ),
            (
                "No tide, bathymetry or metric world-coordinate accuracy is "
                "claimed. Calibration is required for metres."
            ),
        ]

    if metrics is not None:
        test = metrics["metrics"]
        per_class = {
            name: {
                "iou": info["iou"],
                "dice": info["dice"],
                "precision": info.get("precision"),
                "recall": info.get("recall"),
                "support_pixels": info.get("support_pixels"),
            }
            for name, info in test["per_class"].items()
        }
        metrics_block = {
            "evaluation_split": (
                f"held-out test, {int(metrics['num_images'])} images, native resolution"
            ),
            "test_images": int(metrics["num_images"]),
            "selection_metric": "validation mean IoU only",
            "checkpoint_epoch": int(identity["epoch"]),
            "validation_mean_iou": float(identity["val_mean_iou"]),
            "test": {
                "mean_iou": test["mean_iou"],
                "mean_dice": test["mean_dice"],
                "pixel_accuracy": test["pixel_accuracy"],
                "per_class": per_class,
            },
            "caveat": (
                "Site-held-out test with automatic SAM3 labels; not human ground "
                "truth or a production/cross-camera accuracy claim."
            ),
        }

    if limitations is not None:
        limitations_list = list(limitations)

    config_payload = {
        "name": "coastwatch-shoreline-segmentation",
        "version": str(version),
        "architecture": "SegFormer",
        "encoder": "MiT-B2",
        "num_classes": 3,
        "class_names": list(SEGFORMER_CLASS_NAMES),
        "class_ids": dict(SEGFORMER_CLASS_IDS),
        "ignore_index": 255,
        "water_classes": [1],
        "input_size": [width, height],
        "normalization": {
            "mean": [float(v) for v in config["data"]["mean"]],
            "std": [float(v) for v in config["data"]["std"]],
        },
        "preprocessing": {
            "resize": "bilinear",
            "opencv_interpolation": "INTER_LINEAR",
            "resize_target": [width, height],
            "input_range": "0-255 uint8, RGB, resized then normalised",
            "mean": [float(v) for v in config["data"]["mean"]],
            "std": [float(v) for v in config["data"]["std"]],
            "channel_order": "RGB (backend converts camera BGR to RGB)",
        },
        "output_handling": {
            "logits_shape": "N x 3 x H/4 x W/4",
            "upsample": "bilinear logits to native frame, align_corners=False",
            "argmax": "uint8 class mask 0/1/2 at native resolution",
            "binary_water_mask": "class_mask == 1 (Ocean) -> 255, else 0",
        },
        "color_map": dict(SEGFORMER_CLASS_COLORS),
        "artifact": {
            "onnx": "model.onnx",
            "opset": int(opset),
            "input_name": "input",
            "output_name": "logits",
            "input_layout": "NCHW float32",
            "output_layout": "NCHW float32",
            "output_stride": 4,
            "dynamic_batch": True,
            "dynamic_spatial": True,
        },
        "providers": ["CPUExecutionProvider"],
        "runtime": {
            "note": "CPU-only inference; failures are explicit, never faked",
        },
        "precision": {
            "default": "FP32",
            "cpu": "FP32 CPU",
            "cuda": "not used; this deployment runs CPU-only",
            "fp16": "not exported; no FP16 parity was demonstrated",
        },
        "checkpoint": dict(identity, path=_repo_relative(checkpoint_path)),
        "training": training_block,
        "metrics": metrics_block,
        "verification": verification,
        "limitations": limitations_list,
        "created_by": "ml/src/inference/export_onnx.py:export_segformer",
    }
    save_json(output_dir / "config.json", config_payload)
    save_json(output_dir / "export_report.json", {
        "architecture": "SegFormer",
        "encoder": "MiT-B2",
        "onnx": onnx_path.name,
        "onnx_bytes": int(onnx_path.stat().st_size),
        "opset": int(opset),
        "checkpoint": identity,
        "verification": verification,
    })
    _write_segformer_model_card(output_dir / "README.md", config_payload)
    logger.info("wrote SegFormer artifact to %s", output_dir)
    return {
        "onnx": str(onnx_path),
        "config": str(output_dir / "config.json"),
        "verification": verification,
        "checkpoint": identity,
    }


def _write_segformer_model_card(path: Path, config: dict[str, Any]) -> None:
    metrics = config["metrics"]["test"]
    lines = [
        "# CoastWatch shoreline segmentation — SegFormer MiT-B2",
        "",
        "Generated by `ml/src/inference/export_onnx.py:export_segformer`. Do not "
        "hand-edit `config.json` without re-running the parity verification.",
        "",
        "## Artifact",
        "",
        f"- architecture: `{config['architecture']}`",
        f"- encoder: `{config['encoder']}`",
        f"- classes: {config['class_ids']}",
        f"- ignore index: {config['ignore_index']} (never emitted as a prediction)",
        f"- water classes (binary mask): {config['water_classes']} (Ocean only)",
        f"- input: {config['input_size']} RGB, ImageNet mean/std, INTER_LINEAR",
        "- output: 3-class logits at 1/4 resolution; logits are upsampled before argmax",
        "",
        "## Checkpoint",
        "",
        f"- path: `{config['checkpoint']['path']}`",
        f"- SHA-256: `{config['checkpoint']['sha256']}`",
        f"- epoch: {config['checkpoint']['epoch']}",
        f"- validation mean IoU: {config['checkpoint']['val_mean_iou']}",
        "",
        f"## Held-out test ({config['metrics'].get('test_images', 'unknown')} images, native resolution)",
        "",
        "| class | IoU | Dice |",
        "|---|---:|---:|",
    ]
    for name, row in metrics["per_class"].items():
        lines.append(f"| {name} | {row['iou']:.4f} | {row['dice']:.4f} |")
    lines.append(f"| **mean** | **{metrics['mean_iou']:.4f}** | **{metrics['mean_dice']:.4f}** |")
    lines.extend(
        [
            "",
            f"Pixel accuracy: **{metrics['pixel_accuracy']:.4f}**.",
            "",
            str(config["metrics"].get("caveat", "")),
            "",
            "## Verification",
            "",
            f"- performed: {config['verification'].get('performed')}",
            f"- passed: {config['verification'].get('passed')}",
            f"- frames: {config['verification'].get('frame_count')}",
            f"- max |logit diff|: {config['verification'].get('max_abs_diff')}",
            f"- 1/4 argmax agreement: {config['verification'].get('quarter_label_agreement')}",
            f"- native argmax agreement: {config['verification'].get('native_label_agreement')}",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in config["limitations"])
    lines.extend(
        [
            "",
            "## Regenerate",
            "",
            "```powershell",
            "ml/.venv/Scripts/python.exe -m ml.src.inference.export_onnx --segformer",
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_model_card(
    path: Path,
    *,
    architecture: str,
    checkpoint: dict[str, Any],
    metadata: dict[str, Any],
    verification: dict[str, Any],
    limitations: list[str],
    metrics: dict[str, Any],
) -> None:
    """Write the human-readable model card required next to the artifact."""
    lines = [
        "# CoastWatch shoreline segmentation",
        "",
        "Generated by `ml/src/inference/export_onnx.py`. Do not hand-edit the "
        "`config.json` without re-verifying the ONNX graph.",
        "",
        "## Artifact",
        "",
        f"- architecture: `{architecture}`",
        f"- encoder: `{checkpoint.get('encoder')}`",
        f"- training stage / epoch: `{checkpoint.get('stage')}` / {checkpoint.get('epoch')}",
        f"- selection metric on validation: {checkpoint.get('best_metric')}",
        f"- input size: {metadata.get('input_size')} (dynamic spatial axes)",
        f"- classes: {', '.join(metadata.get('class_names', []))}",
        f"- water classes (binary mask): {metadata.get('water_classes')}",
        "",
        "## Verification",
        "",
        f"- performed: {verification.get('performed')}",
        f"- passed: {verification.get('passed')}",
        f"- max |logit| difference vs PyTorch: {verification.get('max_abs_diff')}",
        f"- class-decision agreement: {verification.get('label_agreement')}",
        "",
    ]

    if metrics:
        lines.extend(
            [
                "## Metrics (held-out test splits)",
                "",
                "| dataset | split | supervised mIoU | pixel accuracy |",
                "|---|---|---|---|",
            ]
        )
        for dataset, splits in metrics.get("datasets", {}).items():
            for split, row in splits.items():
                lines.append(
                    f"| {dataset} | {split} | {row.get('supervised_mean_iou')} | "
                    f"{row.get('pixel_accuracy')} |"
                )
        lines.append("")
    else:
        lines.extend(["## Metrics", "", "No evaluation metrics were embedded in this export.", ""])

    lines.extend(["## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(
        [
            "",
            "## Usage",
            "",
            "```powershell",
            "python -m ml.src.inference.predict <this-directory> <image-or-folder>",
            "```",
            "",
            "The backend loads `model.onnx` + `config.json` from "
            "`MODEL_PATH/shoreline-segmentation/`; restart the API after replacing "
            "the artifact.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a checkpoint to ONNX for CoastWatch")
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Path to a checkpoint (defaults per --segformer or the legacy *_best.pth)",
    )
    parser.add_argument(
        "--segformer",
        action="store_true",
        help="Export the active three-class SegFormer MiT-B2 checkpoint",
    )
    parser.add_argument(
        "--allow-different-checkpoint",
        action="store_true",
        help="Allow a fine-tuned candidate instead of the pinned baseline checkpoint",
    )
    parser.add_argument(
        "--version",
        default="2.0.0",
        help="Artifact version written to config.json",
    )
    parser.add_argument(
        "--dataset-summary",
        default=None,
        help="dataset_summary.json from prepare_multisite (candidate metadata)",
    )
    parser.add_argument(
        "--output-dir",
        default="ml/models/shoreline-segmentation",
        help="Artifact directory (default: ml/models/shoreline-segmentation)",
    )
    parser.add_argument("--config", default=None, help="Training YAML config")
    parser.add_argument("--metrics", default=None, help="Optional metrics JSON to embed")
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument(
        "--verify-frames",
        type=int,
        default=20,
        help="Real frames used for SegFormer ONNX/PyTorch parity",
    )
    parser.add_argument(
        "--frames-dir",
        default="ml/datasets/coastwatch_segformer/images",
        help="Directory containing real frames for parity verification",
    )
    args = parser.parse_args()

    if args.segformer:
        checkpoint = args.checkpoint or "ml/checkpoints/segformer_b2/best.pt"
        metrics_payload = None
        if args.metrics:
            metrics_payload = json.loads(Path(resolve_path(args.metrics)).read_text(encoding="utf-8"))
        dataset_summary = None
        if args.dataset_summary:
            dataset_summary = json.loads(
                Path(resolve_path(args.dataset_summary)).read_text(encoding="utf-8")
            )
        result = export_segformer(
            checkpoint,
            args.output_dir,
            config_path=args.config or "ml/configs/segformer_b2.yaml",
            frames_dir=args.frames_dir,
            verify_frames=args.verify_frames,
            opset=args.opset,
            verify=not args.no_verify,
            allow_any_checkpoint=args.allow_different_checkpoint,
            version=args.version,
            metrics=metrics_payload,
            dataset_summary=dataset_summary,
        )
    else:
        if not args.checkpoint:
            parser.error("checkpoint is required unless --segformer is used")
        metrics = None
        if args.metrics:
            metrics = load_config(resolve_path(args.metrics))
        result = export(
            args.checkpoint,
            args.output_dir,
            config_path=args.config or "ml/configs/config.yaml",
            metrics=metrics,
            opset=args.opset,
            verify=not args.no_verify,
        )
    print(result)


if __name__ == "__main__":
    main()
