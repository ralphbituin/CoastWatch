"""PyTorch datasets for the CoastWatch beach datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ml.src.data.mappings import IGNORE_INDEX, ClassMapping
from ml.src.data.splits import DatasetRecord
from ml.src.utils import ensure_dir, get_logger, resolve_path

logger = get_logger("data.datasets")

try:  # torch is required for training; inspection can run without it
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass(slots=True)
class MultispectralConfig:
    """How to turn the India 6-band GeoTIFF into an RGB composite.

    Bands 0-2 are visible reflectance-like channels; bands 3-5 are derived
    indices (verified algebraically in the inspection report). No band
    descriptions exist in the GeoTIFF metadata, so this is a documented
    inference and must not be presented as sensor ground truth.
    """

    rgb_bands: tuple[int, int, int] = (2, 1, 0)
    percentiles: tuple[float, float] = (2.0, 98.0)
    nan_fill: float = 0.0

    @classmethod
    def from_config(cls, entry: dict[str, Any]) -> MultispectralConfig:
        bands = tuple(int(b) for b in entry.get("rgb_bands", [2, 1, 0]))
        if len(bands) != 3:
            raise ValueError("rgb_bands must contain exactly three channel indices")
        percentiles = tuple(float(p) for p in entry.get("rgb_scale_percentiles", [2, 98]))
        return cls(
            rgb_bands=bands,  # type: ignore[arg-type]
            percentiles=percentiles,  # type: ignore[arg-type]
            nan_fill=float(entry.get("nan_fill", 0.0)),
        )


class BeachSegmentationDataset(Dataset):
    """Image/mask pairs mapped into the canonical 4-class schema.

    ``__getitem__`` returns a dict with:
      image: float32 tensor CHW (ImageNet-normalised)
      mask: int64 tensor HW (canonical ids; 255 = ignore)
      dataset, path, index, original_size
    """

    def __init__(
        self,
        records: list[DatasetRecord],
        mapping: ClassMapping,
        *,
        transform: Any,
        multispectral: MultispectralConfig | None = None,
        validate_masks: bool = True,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for BeachSegmentationDataset")
        if not records:
            raise ValueError(f"{mapping.dataset}: empty record list")
        self.records = records
        self.mapping = mapping
        self.transform = transform
        self.multispectral = multispectral
        self.validate_masks = validate_masks

    def __len__(self) -> int:
        return len(self.records)

    def load_image(self, record: DatasetRecord) -> np.ndarray:
        path = resolve_path(record.image_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if record.dataset == "india_sandy_beach":
            return self._load_multispectral(path)
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)

    def _load_multispectral(self, path: Path) -> np.ndarray:
        try:
            import tifffile
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "tifffile is required to read the India 6-band GeoTIFFs. "
                "Install ml/requirements.txt."
            ) from exc
        bands = tifffile.imread(path)
        if bands.ndim != 3 or bands.shape[2] < 3:
            raise ValueError(f"{path}: expected HxWxC multispectral array, got {bands.shape}")
        config = self.multispectral or MultispectralConfig()
        rgb = np.empty((*bands.shape[:2], 3), dtype=np.float32)
        for out_channel, band_index in enumerate(config.rgb_bands):
            band = np.asarray(bands[:, :, band_index], dtype=np.float32)
            band = np.where(np.isfinite(band), band, config.nan_fill)
            low, high = np.percentile(band, config.percentiles)
            if high - low < 1e-6:
                band = np.zeros_like(band)
            else:
                band = np.clip((band - low) / (high - low), 0.0, 1.0)
            rgb[:, :, out_channel] = band
        return (rgb * 255.0).round().astype(np.uint8)

    def load_mask(self, record: DatasetRecord) -> np.ndarray:
        path = resolve_path(record.mask_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            # Palette masks must be read as indices, never converted to RGB.
            raw = np.asarray(image)
        return self.mapping.apply(raw)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image = self.load_image(record)
        mask = self.load_mask(record)
        if mask.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"{record.dataset}: image {image.shape[:2]} and mask {mask.shape[:2]} "
                f"differ for {record.image_path}"
            )
        transformed = self.transform(image=image, mask=mask)
        image_tensor = transformed["image"]
        mask_tensor = transformed["mask"]
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.from_numpy(np.asarray(mask_tensor))
        mask_tensor = mask_tensor.long()
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "dataset": record.dataset,
            "path": record.image_path,
            "index": index,
            "original_size": (int(image.shape[1]), int(image.shape[0])),
        }


def segmentation_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tensors explicitly instead of relying on default collate.

    ``default_collate`` resizes the storage of the first tensor in a batch.
    Albumentations' ``ToTensorV2`` produces tensors backed by NumPy buffers,
    whose storage cannot be resized, which fails on Windows/macOS workers.
    ``torch.stack`` always allocates new storage.
    """
    import torch

    collated: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
    }
    for key in ("dataset", "path", "index", "original_size"):
        collated[key] = [item[key] for item in batch]
    return collated


def has_supervision(record: DatasetRecord, mapping: ClassMapping) -> bool:
    """True when the manifest's recorded mask values include a supervised class.

    This uses the ``source_classes`` column written by the manifest builder, so
    no mask has to be decoded. It is exact for the datasets here: a mask
    is supervised iff at least one of its unique values maps to land or water.
    """
    if not record.source_classes:
        return True  # unknown; let the loader decide rather than dropping data
    try:
        values = {int(part) for part in record.source_classes.split(",") if part.strip()}
    except ValueError:
        return True
    return any(mapping.lookup[value] not in (IGNORE_INDEX, 0, 3) for value in values)


def filter_supervised_records(
    records: list[DatasetRecord],
    mapping: ClassMapping,
    *,
    label: str = "",
) -> list[DatasetRecord]:
    """Drop images whose whole mask is ignored (no land/water pixels).

    Such images would contribute no gradient — and, before the loss guard,
    would have produced NaN. They are excluded and counted rather than being
    silently relabelled.
    """
    kept = [record for record in records if has_supervision(record, mapping)]
    dropped = len(records) - len(kept)
    if dropped:
        logger.info(
            "%s%s: dropped %d/%d images with no supervised land/water pixels",
            mapping.dataset,
            f"/{label}" if label else "",
            dropped,
            len(records),
        )
    return kept


def compute_class_weights(
    records_by_dataset: dict[str, list[DatasetRecord]],
    mappings: dict[str, ClassMapping],
    *,
    num_classes: int = 4,
    ignore_index: int = IGNORE_INDEX,
    cache_path: str | Path | None = None,
    max_weight: float = 5.0,
) -> np.ndarray:
    """Inverse-frequency class weights over all supervised pixels.

    Classes with zero training pixels (foam, in these datasets) receive weight
    0 and therefore contribute nothing to the cross-entropy loss. They are
    never silently assigned a fabricated frequency.
    """
    if cache_path:
        cache = resolve_path(cache_path)
        if cache.is_file():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            return np.asarray(payload["weights"], dtype=np.float32)

    counts = np.zeros(num_classes, dtype=np.int64)
    for dataset, records in records_by_dataset.items():
        mapping = mappings[dataset]
        for record in records:
            with Image.open(resolve_path(record.mask_path)) as image:
                raw = np.asarray(image)
            mask = mapping.apply(raw)
            valid = mask != ignore_index
            counts += np.bincount(mask[valid], minlength=num_classes)[:num_classes]

    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    if present.any():
        frequencies = counts[present] / counts[present].sum()
        raw_weights = 1.0 / np.maximum(frequencies, 1e-8)
        raw_weights = raw_weights / raw_weights.mean()
        raw_weights = np.clip(raw_weights, 0.0, max_weight)
        weights[present] = raw_weights
    logger.info("class pixel counts=%s weights=%s", counts.tolist(), weights.round(3).tolist())

    if cache_path:
        cache = resolve_path(cache_path)
        ensure_dir(cache.parent)
        cache.write_text(
            json.dumps(
                {
                    "counts": counts.tolist(),
                    "weights": weights.tolist(),
                    "note": "weight 0 = no supervision for this class",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return weights
