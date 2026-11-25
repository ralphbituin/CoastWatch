"""Visualisation helpers for masks, predictions and extracted shorelines."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.src.metrics import CLASS_NAMES, IGNORE_INDEX

CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    0: (214, 181, 137),  # Beach — sand
    1: (32, 112, 190),  # Ocean — blue
    2: (135, 206, 235),  # Sky
}
IGNORE_COLOR = (80, 80, 80)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map a class-id mask to an RGB image (uint8, HxWx3)."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    rgb = np.zeros((*array.shape, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        rgb[array == class_id] = color
    rgb[array == IGNORE_INDEX] = IGNORE_COLOR
    return rgb


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = 0.45,
    ignore_transparent: bool = True,
) -> np.ndarray:
    """Blend a colorised mask over an RGB image."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[:2] != tuple(mask.shape):
        raise ValueError(f"image {image.shape[:2]} and mask {mask.shape} differ")
    colors = colorize_mask(mask).astype(np.float32)
    base = image.astype(np.float32)
    blended = base * (1.0 - alpha) + colors * alpha
    if ignore_transparent:
        ignore = np.asarray(mask) == IGNORE_INDEX
        blended[ignore] = base[ignore]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # older Pillow
        return ImageFont.load_default()


def _with_title(rgb: np.ndarray, title: str, title_height: int = 34) -> Image.Image:
    array = np.asarray(rgb, dtype=np.uint8)
    if array.ndim == 2:
        array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
    h, w = array.shape[:2]
    canvas = Image.new("RGB", (w, h + title_height), (24, 24, 24))
    canvas.paste(Image.fromarray(array), (0, title_height))
    ImageDraw.Draw(canvas).text(
        (8, 7), title, fill=(240, 240, 240), font=_font(max(14, w // 55))
    )
    return canvas


def panel(
    items: Sequence[tuple[str, np.ndarray]],
    *,
    max_width: int = 1800,
) -> Image.Image:
    """Lay RGB panels out in a single row, scaling to a maximum width."""
    if not items:
        raise ValueError("panel requires at least one item")
    panels = [_with_title(rgb, title) for title, rgb in items]
    total_width = sum(p.width for p in panels)
    height = max(p.height for p in panels)
    canvas = Image.new("RGB", (total_width, height), (24, 24, 24))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    if canvas.width > max_width:
        scale = max_width / canvas.width
        canvas = canvas.resize(
            (max_width, max(1, int(round(canvas.height * scale)))), Image.Resampling.LANCZOS
        )
    return canvas


def prediction_panel(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    *,
    shoreline_points: np.ndarray | None = None,
) -> Image.Image:
    """Original / GT / prediction / overlay panel for one image."""
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[:2] != ground_truth.shape:
        ground_truth = cv2.resize(
            ground_truth, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    if image.shape[:2] != prediction.shape:
        prediction = cv2.resize(
            prediction, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    comparison = image.copy()
    # Green = correct, red = wrong. Ignore pixels stay original.
    valid = ground_truth != IGNORE_INDEX
    correct = (prediction == ground_truth) & valid
    wrong = (~correct) & valid
    comparison[correct] = (
        comparison[correct].astype(np.float32) * 0.55
        + np.array([0, 190, 0], dtype=np.float32) * 0.45
    ).astype(np.uint8)
    comparison[wrong] = (
        comparison[wrong].astype(np.float32) * 0.45
        + np.array([220, 30, 30], dtype=np.float32) * 0.55
    ).astype(np.uint8)
    if shoreline_points is not None and len(shoreline_points):
        draw_shoreline(comparison, shoreline_points)
    return panel(
        [
            ("Original", image),
            ("Ground truth", colorize_mask(ground_truth)),
            ("Prediction", colorize_mask(prediction)),
            ("Overlay (green=correct, red=error)", comparison),
        ]
    )


def draw_shoreline(
    rgb: np.ndarray,
    points: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 40, 40),
    thickness: int = 3,
    max_jump: float | None = None,
) -> np.ndarray:
    out = np.asarray(rgb).copy()
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return out
    if max_jump is None:
        # A per-column median shoreline can jump between two real waterlines
        # (near shore and far shore). Draw separate segments instead of
        # fabricating a vertical connector.
        max_jump = max(8.0, 0.02 * out.shape[0])
    # Per-column polyline: sort by x and draw connected segments.
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    for index in range(1, len(pts)):
        if abs(pts[index, 0] - pts[index - 1, 0]) > 1.0:
            continue
        if abs(pts[index, 1] - pts[index - 1, 1]) > max_jump:
            continue
        x0, y0 = int(round(pts[index - 1, 0])), int(round(pts[index - 1, 1]))
        x1, y1 = int(round(pts[index, 0])), int(round(pts[index, 1]))
        cv2.line(out, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
    return out


def save_panel(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def preview_grid(
    samples: Iterable[tuple[str, np.ndarray, np.ndarray]],
    output_path: str | Path,
    *,
    columns: int = 3,
    tile_size: tuple[int, int] = (420, 236),
) -> Path:
    """Audit preview: image, mask and overlay for each sample."""
    rows: list[Image.Image] = []
    tiles: list[Image.Image] = []
    for name, image, mask in samples:
        rgb = np.asarray(image, dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        mask_resized = cv2.resize(mask, tile_size, interpolation=cv2.INTER_NEAREST)
        rgb_resized = cv2.resize(rgb, tile_size, interpolation=cv2.INTER_AREA)
        tiles.append(_with_title(rgb_resized, f"{name} — image", 26))
        tiles.append(_with_title(colorize_mask(mask_resized), "class mask", 26))
        tiles.append(_with_title(overlay_mask(rgb_resized, mask_resized), "overlay", 26))
        if len(tiles) >= columns * 3:
            rows.append(_horizontal(tiles))
            tiles = []
    if tiles:
        rows.append(_horizontal(tiles))
    if not rows:
        raise ValueError("no samples supplied")
    width = max(row.width for row in rows)
    height = sum(row.height for row in rows)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
    return path


def _horizontal(images: Sequence[Image.Image]) -> Image.Image:
    width = sum(i.width for i in images)
    height = max(i.height for i in images)
    canvas = Image.new("RGB", (width, height), (18, 18, 18))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width
    return canvas


def class_legend_text() -> str:
    return " | ".join(
        f"{class_id}={name} {CLASS_COLORS[class_id]}" for class_id, name in enumerate(CLASS_NAMES)
    )
