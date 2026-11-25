"""Visual comparison figures: image, ground truth, prediction, overlay, errors.

Figures are written under ``ml/reports/figures/evaluation/<architecture>/`` and
referenced from the evaluation Markdown. They are produced from the actual test
predictions, not from training batches.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ml.src.data.mappings import class_colors, load_class_config
from ml.src.utils import ensure_dir, project_root, resolve_path

IGNORE_COLOR = np.array([90, 90, 90], dtype=np.uint8)


def load_rgb_image(record: Any) -> np.ndarray:
    """Load an RGB uint8 view of a manifest record (India needs band composition)."""
    path = resolve_path(record.image_path)
    if record.dataset != "india_sandy_beach":
        from PIL import Image

        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    import tifffile

    entry = load_class_config()["datasets"]["india_sandy_beach"]
    bands = tifffile.imread(path)
    rgb_bands = [int(b) for b in entry.get("rgb_bands", [2, 1, 0])]
    percentiles = entry.get("rgb_scale_percentiles", [2, 98])
    rgb = np.empty((*bands.shape[:2], 3), dtype=np.float32)
    for out_channel, band_index in enumerate(rgb_bands):
        band = np.asarray(bands[:, :, band_index], dtype=np.float32)
        band = np.where(np.isfinite(band), band, 0.0)
        low, high = np.percentile(band, percentiles)
        band = np.zeros_like(band) if high - low < 1e-6 else (band - low) / (high - low)
        rgb[:, :, out_channel] = np.clip(band, 0, 1)
    return (rgb * 255).round().astype(np.uint8)


def colorize(mask: np.ndarray, colors: np.ndarray | None = None) -> np.ndarray:
    colors = colors if colors is not None else class_colors()
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for index in range(len(colors)):
        canvas[mask == index] = colors[index]
    canvas[mask == 255] = IGNORE_COLOR
    return canvas


def _error_map(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Correct = mid grey, wrong = red, ignored GT = dark grey."""
    canvas = np.full((*prediction.shape, 3), 60, dtype=np.uint8)
    valid = target != 255
    correct = valid & (prediction == target)
    wrong = valid & (prediction != target)
    canvas[correct] = np.array([70, 160, 90], dtype=np.uint8)
    canvas[wrong] = np.array([220, 70, 60], dtype=np.uint8)
    return canvas


def render_predictions(
    dataset: str,
    split: str,
    samples: list[dict[str, Any]],
    figures_dir: Path,
    *,
    class_colors: np.ndarray | None = None,
) -> list[str]:
    if not samples:
        return []
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = class_colors if class_colors is not None else class_colors_default()
    rows = len(samples)
    figure, axes = plt.subplots(rows, 5, figsize=(18, 3.2 * rows), squeeze=False)
    headings = ["image", "ground truth", "prediction", "overlay", "errors (red = wrong)"]

    for row, sample in enumerate(samples):
        record = sample["record"]
        prediction = np.asarray(sample["prediction"])
        target = np.asarray(sample["target"])
        image = load_rgb_image(record)
        if image.shape[:2] != prediction.shape[:2]:
            from PIL import Image

            image = np.asarray(
                Image.fromarray(image).resize(
                    (prediction.shape[1], prediction.shape[0]), Image.BILINEAR
                ),
                dtype=np.uint8,
            )
        overlay = np.clip(
            image.astype(np.float32) * 0.55
            + colorize(prediction, colors).astype(np.float32) * 0.45,
            0,
            255,
        ).astype(np.uint8)

        panels = [
            image,
            colorize(target, colors),
            colorize(prediction, colors),
            overlay,
            _error_map(prediction, target),
        ]
        for column, panel in enumerate(panels):
            axes[row, column].imshow(panel)
            axes[row, column].axis("off")
            if row == 0:
                axes[row, column].set_title(headings[column], fontsize=10)
        axes[row, 0].set_ylabel(Path(record.image_path).name[:30], fontsize=7)

    figure.suptitle(f"{dataset} · {split} · predictions vs annotations", fontsize=12)
    figure.tight_layout()
    ensure_dir(figures_dir)
    output = figures_dir / f"{dataset}_{split}_predictions.png"
    figure.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(figure)
    try:
        return [output.resolve().relative_to(project_root()).as_posix()]
    except ValueError:  # pragma: no cover
        return [output.as_posix()]


def class_colors_default() -> np.ndarray:
    return class_colors()
