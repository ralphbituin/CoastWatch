"""Run a trained CoastWatch SegFormer checkpoint on new images.

Produces, for every input image:

* ``<stem>_mask.png``       three-class class-id mask (0=Beach 1=Ocean 2=Sky)
* ``<stem>_overlay.png``    prediction blended over the original image
* ``<stem>_shoreline.png``  original image with the extracted Ocean/Beach line
* ``<stem>_shoreline.json`` image-space shoreline points + per-column polyline
* ``<stem>_summary.json``   class pixel fractions and shoreline metadata

The model input is resized to the configured square size (images bilinear,
masks nearest), but predictions are upsampled back to each image's native
resolution before being saved or measured. No tide/bathymetry or metric world
coordinates are produced; an optional homography is applied only when supplied
by the caller.

Usage:
    python -m ml.scripts.infer --image path/to/frame.jpg
    python -m ml.scripts.infer --input-dir ml/datasets/image_sam3 --limit 5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ml.src.dataset import (
    CLASS_NAMES,
    build_eval_transform,
    load_config,
    load_image_rgb,
    resolve_path,
)
from ml.src.model import load_checkpoint
from ml.src.shoreline_extraction import extract_shoreline
from ml.src.visualization import (
    colorize_mask,
    draw_shoreline,
    overlay_mask,
    panel,
    save_panel,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_images(path: Path, limit: int | None) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    files = sorted(
        file for file in path.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTS
    )
    return files[:limit] if limit else files


def _load_homography(path: str | None) -> np.ndarray | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = payload.get("homography", payload) if isinstance(payload, dict) else payload
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError("homography must be a 3x3 matrix (or {'homography': [...]})")
    return matrix


def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    transform: Any,
    device: torch.device,
    *,
    use_amp: bool,
) -> np.ndarray:
    """Predict a class-id mask at the input image's native resolution."""
    prepared = transform(image=image)
    pixel_values = prepared["image"].unsqueeze(0).to(device)
    height, width = image.shape[:2]
    with torch.inference_mode(), torch.amp.autocast(
        "cuda", dtype=torch.float16, enabled=use_amp
    ):
        logits = model(pixel_values)
    logits = F.interpolate(
        logits, size=(height, width), mode="bilinear", align_corners=False
    )
    return logits.argmax(dim=1)[0].to(torch.uint8).cpu().numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="single image to segment")
    source.add_argument("--input-dir", help="directory of images to segment")
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--homography", default=None, help="optional 3x3 image-to-world matrix")
    parser.add_argument("--smoothing", type=int, default=None, help="per-column median filter width")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    checkpoint = resolve_path(
        args.checkpoint or (Path(config["paths"]["checkpoints"]) / "best.pt")
    )
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}. Train first with "
            "`python -m ml.scripts.train`."
        )
    output_dir = resolve_path(
        args.output_dir or (Path(config["paths"]["outputs"]) / "inference")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = _collect_images(resolve_path(args.image or args.input_dir), args.limit)
    if not inputs:
        raise RuntimeError("no input images found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_checkpoint(checkpoint, config=config, device=device)
    use_amp = bool(
        str(config["training"].get("precision", "fp16")).lower() == "fp16"
        and device.type == "cuda"
    )
    transform = build_eval_transform(config)
    homography = _load_homography(args.homography)
    shoreline_cfg = config.get("inference", {}).get("shoreline", {})
    smoothing = (
        int(args.smoothing)
        if args.smoothing is not None
        else int(shoreline_cfg.get("boundary_smoothing", 0))
    )
    alpha = float(config.get("inference", {}).get("overlay_alpha", 0.45))

    manifest: list[dict[str, Any]] = []
    for index, path in enumerate(inputs, start=1):
        image = load_image_rgb(path)
        mask = predict_mask(model, image, transform, device, use_amp=use_amp)
        height, width = mask.shape

        mask_path = output_dir / f"{path.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        overlay = overlay_mask(image, mask, alpha=alpha)
        overlay_path = output_dir / f"{path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        shoreline = extract_shoreline(
            mask,
            min_ocean_component_px=int(shoreline_cfg.get("min_ocean_component_px", 2000)),
            min_beach_component_px=int(shoreline_cfg.get("min_beach_component_px", 1000)),
            smoothing=smoothing,
            min_valid_columns=int(shoreline_cfg.get("min_valid_columns", 8)),
        )
        line = draw_shoreline(image, shoreline.per_column)
        shoreline_png = output_dir / f"{path.stem}_shoreline.png"
        save_panel(panel([("Predicted shoreline (red)", line)]),
                   shoreline_png)
        # Also save the plain annotated frame at native resolution.
        cv2.imwrite(
            str(output_dir / f"{path.stem}_shoreline_overlay.png"),
            cv2.cvtColor(line, cv2.COLOR_RGB2BGR),
        )

        payload_out = shoreline.as_dict()
        payload_out["image"] = str(path)
        payload_out["image_size"] = [int(width), int(height)]
        payload_out["coordinate_system"] = (
            "image pixels, origin top-left, x right, y down; NOT metric world coordinates"
        )
        if homography is not None:
            world = shoreline.to_world(homography)
            payload_out["world_points"] = world.tolist()
            payload_out["world_coordinate_system"] = "caller-supplied homography units"
        (output_dir / f"{path.stem}_shoreline.json").write_text(
            json.dumps(payload_out, indent=2), encoding="utf-8"
        )

        total = int(mask.size)
        summary = {
            "image": str(path),
            "image_size": [int(width), int(height)],
            "predicted_mask": mask_path.name,
            "overlay": overlay_path.name,
            "shoreline": shoreline_png.name,
            "class_fractions": {
                name: float((mask == class_id).sum() / total)
                for class_id, name in enumerate(CLASS_NAMES)
            },
            "shoreline_available": bool(shoreline.metadata.get("available")),
            "shoreline_valid_columns": shoreline.valid_columns,
            "shoreline_rejected_ocean_components": shoreline.rejected_ocean_components,
            "shoreline_rejected_beach_components": shoreline.rejected_beach_components,
            "checkpoint": str(checkpoint),
            "checkpoint_epoch": payload.get("epoch"),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        (output_dir / f"{path.stem}_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        manifest.append(summary)
        print(f"[infer] {index}/{len(inputs)} {path.name}", flush=True)

    (output_dir / "inference_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": payload.get("epoch"),
                "images": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[infer] wrote {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
