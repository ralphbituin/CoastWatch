"""Run the trained CoastWatch SegFormer on a video.

Produces:
  * an annotated MP4 (prediction overlay + extracted Ocean/Beach shoreline)
  * a contact sheet of sampled frames (original / overlay / shoreline)
  * a JSON summary with per-sample class fractions and shoreline statistics

Usage:
    python -m ml.scripts.infer_video --video data/videos/testvideo.mp4
    python -m ml.scripts.infer_video --video clip.mp4 --video-stride 15 --video-fps 15
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from ml.scripts.infer import predict_mask
from ml.src.dataset import (
    CLASS_NAMES,
    build_eval_transform,
    load_config,
    resolve_path,
)
from ml.src.model import load_checkpoint
from ml.src.shoreline_extraction import ShorelineResult, extract_shoreline
from ml.src.visualization import draw_shoreline, overlay_mask, panel

SHORELINE_COLOR = (255, 40, 40)


def _open_video(path: Path) -> tuple[cv2.VideoCapture, dict[str, float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or total <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("video metadata is incomplete")
    return capture, {
        "fps": fps,
        "frames": total,
        "width": width,
        "height": height,
        "duration_seconds": total / fps,
    }


def _fit_frame(bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    if bgr.shape[1] == width and bgr.shape[0] == height:
        return bgr
    return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)


def _label(
    bgr: np.ndarray, text: str, *, origin: tuple[int, int] | None = None, scale: float = 0.55
) -> None:
    height, width = bgr.shape[:2]
    if origin is None:
        origin = (10, height - 12)
    size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = origin
    cv2.rectangle(
        bgr,
        (x - 5, y - size[1] - 6),
        (x + size[0] + 5, y + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        bgr, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA
    )


def _annotate(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    shoreline: ShorelineResult,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    overlay = overlay_mask(image_rgb, mask, alpha=alpha)
    line = draw_shoreline(
        overlay,
        shoreline.per_column,
        color=SHORELINE_COLOR,
        thickness=max(1, round(image_rgb.shape[1] / 640)),
    )
    return overlay, line


def _fractions(mask: np.ndarray) -> dict[str, float]:
    total = int(mask.size)
    return {
        name: float((mask == class_id).sum() / total)
        for class_id, name in enumerate(CLASS_NAMES)
    }


def _write_video(
    path: Path,
    cap: cv2.VideoCapture,
    meta: dict[str, float],
    *,
    model: torch.nn.Module,
    transform: Any,
    device: torch.device,
    use_amp: bool,
    stride: int,
    output_fps: float,
    shoreline_cfg: dict[str, Any],
    alpha: float,
    progress_every: int,
) -> int:
    width, height = int(meta["width"]), int(meta["height"])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    source_index = 0
    written = 0
    while True:
        if not cap.grab():
            break
        if source_index % stride != 0:
            source_index += 1
            continue
        ok, bgr = cap.retrieve()
        if not ok:
            break
        bgr = _fit_frame(bgr, width, height)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = predict_mask(model, rgb, transform, device, use_amp=use_amp)
        shoreline = extract_shoreline(
            mask,
            min_ocean_component_px=int(shoreline_cfg.get("min_ocean_component_px", 2000)),
            min_beach_component_px=int(shoreline_cfg.get("min_beach_component_px", 1000)),
            smoothing=int(shoreline_cfg.get("boundary_smoothing", 0)),
            min_valid_columns=int(shoreline_cfg.get("min_valid_columns", 8)),
        )
        _, line = _annotate(rgb, mask, shoreline, alpha=alpha)
        annotated = cv2.cvtColor(line, cv2.COLOR_RGB2BGR)
        _label(
            annotated,
            f"t={source_index / meta['fps']:6.1f}s  frame {source_index}  "
            f"shore={'yes' if shoreline.metadata.get('available') else 'no'}",
        )
        writer.write(annotated)
        written += 1
        if progress_every and written % progress_every == 0:
            print(f"[video] wrote {written} annotated frames", flush=True)
        source_index += 1
    writer.release()
    return written


def _timeline(samples: list[dict[str, Any]], output_path: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    if not samples:
        return None
    times = np.array([sample["seconds"] / 60.0 for sample in samples])
    figure, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, dpi=150)
    for name in CLASS_NAMES:
        values = np.array([sample["fractions"][name] * 100.0 for sample in samples])
        axes[0].plot(times, values, label=name, linewidth=1.6)
    axes[0].set_ylabel("% of frame")
    axes[0].set_title("Predicted class fractions over the video")
    axes[0].legend(loc="best", ncol=3)
    axes[0].grid(alpha=0.25)
    mean_y = [
        float(np.mean(sample["shoreline"].per_column[:, 1]))
        if len(sample["shoreline"].per_column)
        else np.nan
        for sample in samples
    ]
    axes[1].plot(times, mean_y, color="#c0392b", marker=".", linewidth=1.4)
    axes[1].set_ylabel("mean shoreline row (px)")
    axes[1].set_xlabel("video time (minutes)")
    axes[1].set_title("Extracted shoreline position (image coordinates; larger = lower in frame)")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _contact_sheet(
    samples: list[dict[str, Any]],
    output_path: Path,
    *,
    rows: int = 6,
    max_width: int = 1800,
) -> Path:
    if not samples:
        raise ValueError("no samples for contact sheet")
    if len(samples) > rows:
        indices = np.linspace(0, len(samples) - 1, rows).round().astype(int)
        chosen = [samples[index] for index in indices]
    else:
        chosen = samples
    panels: list[Image.Image] = []
    for sample in chosen:
        stamp = f"frame {sample['frame']}  t={sample['seconds']:.1f}s"
        panels.append(
            panel(
                [
                    (f"{stamp} — original", sample["image"]),
                    ("prediction overlay", sample["overlay"]),
                    ("extracted shoreline", sample["shoreline_image"]),
                ],
                max_width=max_width,
            )
        )
    width = max(item.width for item in panels)
    height = sum(item.height for item in panels)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for item in panels:
        canvas.paste(item, (0, y))
        y += item.height
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sample-every-sec", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int, default=30)
    parser.add_argument("--video-stride", type=int, default=30, help="annotate every Nth frame")
    parser.add_argument("--video-fps", type=float, default=10.0, help="annotated output fps")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)

    video_path = resolve_path(args.video)
    config = load_config(args.config)
    checkpoint = resolve_path(
        args.checkpoint or (Path(config["paths"]["checkpoints"]) / "best.pt")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    output_dir = resolve_path(
        args.output_dir
        or (Path(config["paths"]["outputs"]) / "video_test" / video_path.stem)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cap, meta = _open_video(video_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, config=config, device=device)
    use_amp = bool(
        str(config["training"].get("precision", "fp16")).lower() == "fp16"
        and device.type == "cuda"
    )
    transform = build_eval_transform(config)
    shoreline_cfg = config.get("inference", {}).get("shoreline", {})
    alpha = float(config.get("inference", {}).get("overlay_alpha", 0.45))

    sample_stride = max(1, int(round(meta["fps"] * args.sample_every_sec)))
    sample_indices = list(range(0, int(meta["frames"]), sample_stride))[: args.max_samples]
    samples: list[dict[str, Any]] = []
    for position, frame_index in enumerate(sample_indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, bgr = cap.read()
        if not ok:
            continue
        bgr = _fit_frame(bgr, int(meta["width"]), int(meta["height"]))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mask = predict_mask(model, rgb, transform, device, use_amp=use_amp)
        shoreline = extract_shoreline(
            mask,
            min_ocean_component_px=int(shoreline_cfg.get("min_ocean_component_px", 2000)),
            min_beach_component_px=int(shoreline_cfg.get("min_beach_component_px", 1000)),
            smoothing=int(shoreline_cfg.get("boundary_smoothing", 0)),
            min_valid_columns=int(shoreline_cfg.get("min_valid_columns", 8)),
        )
        overlay, line = _annotate(rgb, mask, shoreline, alpha=alpha)
        samples.append(
            {
                "frame": int(frame_index),
                "seconds": float(frame_index / meta["fps"]),
                "image": rgb,
                "overlay": overlay,
                "shoreline_image": line,
                "mask": mask,
                "shoreline": shoreline,
                "fractions": _fractions(mask),
            }
        )
        if position % 10 == 0:
            print(f"[sample] {position}/{len(sample_indices)}", flush=True)

    # Save a few individual full-resolution outputs for inspection.
    full_dir = output_dir / "frames"
    full_dir.mkdir(exist_ok=True)
    for sample in samples[:: max(1, len(samples) // 12)][:12]:
        stem = f"frame_{sample['frame']:06d}"
        cv2.imwrite(
            str(full_dir / f"{stem}_overlay.png"),
            cv2.cvtColor(sample["overlay"], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            str(full_dir / f"{stem}_shoreline.png"),
            cv2.cvtColor(sample["shoreline_image"], cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(str(full_dir / f"{stem}_mask.png"), sample["mask"])

    contact = _contact_sheet(samples, output_dir / "contact_sheet.png")
    timeline = _timeline(samples, output_dir / "timeline.png")

    annotated_video: Path | None = None
    written = 0
    if not args.no_video:
        annotated_video = output_dir / "annotated.mp4"
        written = _write_video(
            annotated_video,
            cap,
            meta,
            model=model,
            transform=transform,
            device=device,
            use_amp=use_amp,
            stride=max(1, int(args.video_stride)),
            output_fps=float(args.video_fps),
            shoreline_cfg=shoreline_cfg,
            alpha=alpha,
            progress_every=50,
        )
    cap.release()

    fraction_rows = np.array(
        [[sample["fractions"][name] for name in CLASS_NAMES] for sample in samples]
    )
    shore_valid = [sample["shoreline"].metadata.get("available", False) for sample in samples]
    columns = [sample["shoreline"].valid_columns for sample in samples]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": str(video_path),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": payload.get("epoch"),
        "device": str(device),
        "video_metadata": meta,
        "samples": len(samples),
        "sample_every_sec": float(args.sample_every_sec),
        "class_fraction": {
            name: {
                "mean": float(fraction_rows[:, index].mean()),
                "std": float(fraction_rows[:, index].std()),
                "min": float(fraction_rows[:, index].min()),
                "max": float(fraction_rows[:, index].max()),
            }
            for index, name in enumerate(CLASS_NAMES)
        },
        "shoreline": {
            "available_samples": int(sum(bool(value) for value in shore_valid)),
            "total_samples": len(samples),
            "mean_valid_columns": float(np.mean(columns)) if columns else None,
            "min_valid_columns": int(np.min(columns)) if columns else None,
            "max_valid_columns": int(np.max(columns)) if columns else None,
        },
        "annotated_video": str(annotated_video) if annotated_video else None,
        "annotated_frames": int(written),
        "contact_sheet": str(contact),
        "timeline": str(timeline) if timeline else None,
        "per_sample": [
            {
                "frame": sample["frame"],
                "seconds": sample["seconds"],
                "class_fraction": sample["fractions"],
                "shoreline_available": bool(
                    sample["shoreline"].metadata.get("available", False)
                ),
                "shoreline_valid_columns": sample["shoreline"].valid_columns,
                "shoreline_segments": sample["shoreline"].metadata.get("segments"),
                "shoreline_mean_y": (
                    float(np.mean(sample["shoreline"].per_column[:, 1]))
                    if len(sample["shoreline"].per_column)
                    else None
                ),
                "rejected_ocean_components": sample["shoreline"].rejected_ocean_components,
                "rejected_beach_components": sample["shoreline"].rejected_beach_components,
            }
            for sample in samples
        ],
        "coordinate_system": "image pixels; no tide, bathymetry or metric world coordinates",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[video] samples: {len(samples)}")
    print(f"[video] shoreline available: {summary['shoreline']['available_samples']}/{len(samples)}")
    print(f"[video] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
