"""Unified CoastWatch SegFormer dataset and conservative augmentations.

The on-disk dataset produced by ``ml/scripts/prepare_dataset.py`` is:

    coastwatch_segformer/
      images/<id>.(jpg|png)
      masks/<id>.png          # uint8, 0=Beach 1=Ocean 2=Sky 255=ignore
      splits/{train,val,test}.txt
      manifest.csv
      previews/
      audit_report.json

Images are never resized on disk. Training/evaluation resize to 512x512 on the
fly (images bilinear, masks nearest-neighbour) and the model output is mapped
back to native resolution for final evaluation and shoreline extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import albumentations as A
import cv2
import numpy as np
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VALID_CLASS_IDS = (0, 1, 2)
IGNORE_INDEX = 255
CLASS_NAMES = ("Beach", "Ocean", "Sky")
CLASS_COLORS = {
    0: (214, 181, 137),
    1: (32, 112, 190),
    2: (135, 206, 235),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else project_root() / path


def load_config(path: str | Path = "ml/configs/segformer_b2.yaml") -> dict[str, Any]:
    config_path = resolve_path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} did not contain a YAML mapping")
    return config


def unified_paths(config: dict[str, Any]) -> dict[str, Path]:
    root = resolve_path(config["paths"]["unified_dataset"])
    return {
        "root": root,
        "images": root / "images",
        "masks": root / "masks",
        "splits": root / "splits",
        "previews": root / "previews",
        "manifest": root / "manifest.csv",
        "audit": root / "audit_report.json",
    }


def image_mean_std(config: dict[str, Any]) -> tuple[tuple[float, ...], tuple[float, ...]]:
    data = config.get("data", {})
    return tuple(data.get("mean", [0.485, 0.456, 0.406])), tuple(
        data.get("std", [0.229, 0.224, 0.225])
    )


def build_train_transform(config: dict[str, Any]) -> A.Compose:
    data = config.get("data", {})
    aug = config.get("augmentation", {})
    size = int(data.get("image_size", 512))
    mean, std = image_mean_std(config)
    transforms: list[Any] = []
    if float(aug.get("scale_crop_p", 0.5)) > 0:
        transforms.append(
            A.RandomResizedCrop(
                size=(size, size),
                scale=(float(aug.get("scale_min", 0.9)), float(aug.get("scale_max", 1.0))),
                ratio=(
                    float(aug.get("crop_ratio_min", 0.9)),
                    float(aug.get("crop_ratio_max", 1.111111)),
                ),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            )
        )
    else:
        transforms.append(
            A.Resize(
                size,
                size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            )
        )
    transforms.append(A.HorizontalFlip(p=float(aug.get("horizontal_flip_p", 0.5))))
    transforms.append(
        A.RandomBrightnessContrast(
            brightness_limit=float(aug.get("brightness_limit", 0.12)),
            contrast_limit=float(aug.get("contrast_limit", 0.12)),
            p=float(aug.get("brightness_contrast_p", 0.5)),
        )
    )
    transforms.append(
        A.HueSaturationValue(
            hue_shift_limit=int(aug.get("hue_shift_limit", 5)),
            sat_shift_limit=int(aug.get("saturation_shift_limit", 12)),
            val_shift_limit=0,
            p=float(aug.get("hue_saturation_p", 0.25)),
        )
    )
    transforms.append(
        A.GaussNoise(
            var_limit=tuple(aug.get("gaussian_noise_var_limit", [0.0, 25.0])),
            p=float(aug.get("gaussian_noise_p", 0.2)),
        )
    )
    transforms.append(
        A.GaussianBlur(
            blur_limit=tuple(aug.get("gaussian_blur_limit", [3, 3])),
            p=float(aug.get("gaussian_blur_p", 0.15)),
        )
    )
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_eval_transform(config: dict[str, Any]) -> A.Compose:
    data = config.get("data", {})
    size = int(data.get("image_size", 512))
    mean, std = image_mean_std(config)
    return A.Compose(
        [
            A.Resize(
                size,
                size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def load_image_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def load_mask(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in ("L", "P", "I", "I;16"):
            raise ValueError(
                f"{path}: expected single-channel class-id mask, got mode {image.mode}. "
                "RGB masks are not accepted without a verified colour mapping."
            )
        mask = np.asarray(image)
    if mask.ndim != 2:
        raise ValueError(f"{path}: expected 2-D mask, got shape {mask.shape}")
    if mask.dtype != np.uint8:
        if not np.issubdtype(mask.dtype, np.integer):
            raise ValueError(f"{path}: mask dtype {mask.dtype} is not integer")
        mask = mask.astype(np.uint8)
    values = set(int(v) for v in np.unique(mask))
    invalid = values - set(VALID_CLASS_IDS) - {IGNORE_INDEX}
    if invalid:
        raise ValueError(f"{path}: invalid class ids {sorted(invalid)}")
    return mask


def read_split_file(path: str | Path) -> list[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"split file not found: {path}")
    entries: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(line)
    return entries


def mask_path_for(images_dir: Path, masks_dir: Path, image_name: str) -> Path:
    return masks_dir / f"{Path(image_name).stem}.png"


class CoastWatchSegFormerDataset(Dataset):
    """Image/mask pair dataset with shared geometric augmentation."""

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        names: Sequence[str],
        *,
        transform: A.Compose | None = None,
        source_dataset: str = "sam3",
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.names = list(names)
        self.transform = transform
        self.source_dataset = source_dataset
        if not self.names:
            raise ValueError("dataset is empty")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, Any]:
        name = self.names[index]
        image_path = self.image_dir / name
        mask_path = mask_path_for(self.image_dir, self.mask_dir, name)
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        image = load_image_rgb(image_path)
        mask = load_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"{name}: image {image.shape[:2]} and mask {mask.shape} dimensions differ"
            )
        original_size = image.shape[:2]
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image_tensor = augmented["image"]
            mask_tensor = augmented["mask"]
        else:
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask_tensor = torch.from_numpy(mask)
        if not isinstance(mask_tensor, torch.Tensor):
            mask_tensor = torch.as_tensor(mask_tensor)
        mask_tensor = mask_tensor.long()
        values = set(int(v) for v in torch.unique(mask_tensor).tolist())
        invalid = values - set(VALID_CLASS_IDS) - {IGNORE_INDEX}
        if invalid:
            raise ValueError(f"{name}: augmentation produced invalid labels {sorted(invalid)}")
        return {
            "pixel_values": image_tensor.float(),
            "labels": mask_tensor,
            "name": name,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "original_size": tuple(int(v) for v in original_size),
            "source_dataset": self.source_dataset,
        }


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = True,
    seed: int | None = None,
    persistent_workers: bool = False,
) -> DataLoader:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=max(0, int(num_workers)),
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=bool(persistent_workers and num_workers > 0),
        generator=generator,
    )


def validate_names(images_dir: Path, masks_dir: Path, names: Iterable[str]) -> None:
    for name in names:
        if not (images_dir / name).exists():
            raise FileNotFoundError(images_dir / name)
        mask = mask_path_for(images_dir, masks_dir, name)
        if not mask.exists():
            raise FileNotFoundError(mask)
