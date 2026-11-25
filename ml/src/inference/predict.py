"""Run the trained model on an image or a folder of images.

Example::

    python -m ml.src.inference.predict ml/models/shoreline-segmentation \
        ml/datasets/image/2024-01-01_120000.jpg --output ml/reports/predictions

Outputs per image (no large array is ever serialised into JSON):

    <stem>_mask.png      single-channel class ids 0..3
    <stem>_water.png     CoastWatch binary water mask (255 = water|foam)
    <stem>_overlay.png   RGB overlay
    <stem>_summary.json  class statistics, shoreline, timings
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image

from ml.src.inference.predictor import Prediction, SegmentationPredictor
from ml.src.inference.shoreline import max_step_px
from ml.src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("inference.predict")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _iter_images(path: Path, recursive: bool) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    pattern = "**/*" if recursive else "*"
    for candidate in sorted(path.glob(pattern)):
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES:
            yield candidate


def save_prediction(prediction: Prediction, stem: str, output_dir: Path) -> dict[str, str]:
    output_dir = ensure_dir(output_dir)
    mask_path = output_dir / f"{stem}_mask.png"
    water_path = output_dir / f"{stem}_water.png"
    overlay_path = output_dir / f"{stem}_overlay.jpg"
    summary_path = output_dir / f"{stem}_summary.json"

    Image.fromarray(prediction.mask, mode="L").save(mask_path)
    Image.fromarray(prediction.water_mask, mode="L").save(water_path)
    if prediction.overlay is not None:
        Image.fromarray(prediction.overlay, mode="RGB").save(overlay_path, quality=85)

    summary = prediction.to_summary()
    if prediction.shoreline and prediction.shoreline.get("available"):
        summary["shoreline"]["max_step_px"] = max_step_px(prediction.shoreline["points"])
        # Keep the JSON small: the polyline is resampled, never the full width.
        points = prediction.shoreline["points"]
        stride = max(1, len(points) // 200)
        summary["shoreline"]["points"] = points[::stride]
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    outputs = {
        "mask": str(mask_path),
        "water_mask": str(water_path),
        "summary": str(summary_path),
    }
    if prediction.overlay is not None:
        outputs["overlay"] = str(overlay_path)
    return outputs


def run(
    model_dir: str | Path,
    target: str | Path,
    *,
    output_dir: str | Path | None = None,
    input_size: tuple[int, int] | None = None,
    recursive: bool = False,
    save_probabilities: bool = False,
) -> list[dict[str, object]]:
    predictor = SegmentationPredictor(model_dir, input_size=input_size)
    logger.info(
        "loaded %s (%s, providers=%s, input=%s)",
        model_dir,
        predictor.config.get("architecture"),
        predictor.providers,
        predictor.input_size,
    )
    target_path = resolve_path(target)
    output_root = (
        resolve_path(output_dir)
        if output_dir
        else resolve_path("ml/reports/predictions") / Path(target_path).stem
    )

    results: list[dict[str, object]] = []
    for image_path in _iter_images(target_path, recursive):
        started = time.perf_counter()
        with Image.open(image_path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        prediction = predictor.predict(rgb, return_probabilities=save_probabilities)
        outputs = save_prediction(prediction, image_path.stem, output_root)
        if save_probabilities and prediction.probabilities is not None:
            probability_path = ensure_dir(output_root) / f"{image_path.stem}_probabilities.npy"
            np.save(probability_path, prediction.probabilities.astype(np.float16))
            outputs["probabilities"] = str(probability_path)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        summary = prediction.to_summary()
        summary["image"] = str(image_path)
        summary["total_ms"] = round(elapsed_ms, 2)
        results.append({"image": str(image_path), "outputs": outputs, "summary": summary})
        logger.info(
            "%s -> land %.1f%% water %.1f%% (%.0f ms)",
            image_path.name,
            prediction.class_statistics.get("land", {}).get("fraction", 0.0) * 100,
            prediction.class_statistics.get("water", {}).get("fraction", 0.0) * 100,
            prediction.inference_ms,
        )

    if not results:
        raise SystemExit(f"No images found under {target_path}")
    (ensure_dir(output_root) / "index.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment images with a CoastWatch artifact")
    parser.add_argument("model_dir", help="Directory containing model.onnx and config.json")
    parser.add_argument("target", help="Image file or directory")
    parser.add_argument("--output", default=None)
    parser.add_argument("--size", type=int, nargs=2, default=None, metavar=("W", "H"))
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args()

    run(
        args.model_dir,
        args.target,
        output_dir=args.output,
        input_size=tuple(args.size) if args.size else None,
        recursive=args.recursive,
        save_probabilities=args.save_probabilities,
    )


if __name__ == "__main__":
    main()
