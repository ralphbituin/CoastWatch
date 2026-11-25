"""Dataset-specific preprocessing and augmentation.

Images and masks are resized with different interpolations (bilinear vs
nearest) and augmented together, so alignment is preserved. Masks are never
normalised as RGB.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
import cv2
import numpy as np

# Albumentations' ToTensorV2 converts masks to long tensors via torch.
try:  # pragma: no cover - import guard for inspection-only installs
    from albumentations.pytorch import ToTensorV2
except Exception:  # pragma: no cover
    ToTensorV2 = None  # type: ignore[assignment]


def build_train_transform(
    image_size: int,
    aug: dict[str, Any] | None = None,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    aug = aug or {}
    transforms: list[Any] = []
    # A fixed output size is mandatory: the DataLoader stacks tensors, so every
    # sample must have identical spatial dimensions. ``RandomResizedCrop``
    # therefore runs with p=1 and a *mild* scale/ratio range instead of being
    # toggled on and off (which would produce mixed sizes).
    if aug.get("scale_crop_p", 0.5) > 0:
        transforms.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(aug.get("scale_min", 0.85), aug.get("scale_max", 1.0)),
                ratio=(0.9, 1.1111),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=1.0,
            )
        )
    else:
        transforms.append(
            A.Resize(
                image_size,
                image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            )
        )
    if aug.get("horizontal_flip_p", 0.5) > 0:
        transforms.append(A.HorizontalFlip(p=float(aug.get("horizontal_flip_p", 0.5))))
    rotate = float(aug.get("rotate_limit_deg", 5))
    if rotate > 0 and aug.get("rotate_p", 0.5) > 0:
        transforms.append(
            A.Rotate(
                limit=rotate,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=float(aug.get("rotate_p", 0.5)),
            )
        )
    if aug.get("brightness_contrast_p", 0.5) > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=float(aug.get("brightness_limit", 0.15)),
                contrast_limit=float(aug.get("contrast_limit", 0.15)),
                p=float(aug.get("brightness_contrast_p", 0.5)),
            )
        )
    if aug.get("saturation_p", 0.3) > 0:
        transforms.append(
            A.HueSaturationValue(
                hue_shift_limit=0,
                sat_shift_limit=int(float(aug.get("saturation_limit", 0.15)) * 100),
                val_shift_limit=0,
                p=float(aug.get("saturation_p", 0.3)),
            )
        )
    if aug.get("blur_p", 0.15) > 0:
        transforms.append(
            A.GaussianBlur(
                blur_limit=(3, int(aug.get("blur_limit", 3))),
                p=float(aug.get("blur_p", 0.15)),
            )
        )
    transforms.extend(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )
    return A.Compose(transforms)


def build_eval_transform(
    image_size: int,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> Any:
    return A.Compose(
        [
            A.Resize(
                image_size,
                image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0),
            ToTensorV2(),
        ]
    )


def normalize_rgb(
    rgb: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalise an HxWx3 uint8 RGB image to float32 CHW (inference)."""
    array = rgb.astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.transpose(array, (2, 0, 1))


def denormalize_rgb(
    chw: np.ndarray,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    array = np.transpose(chw, (1, 2, 0)) * np.asarray(std, dtype=np.float32) + np.asarray(
        mean, dtype=np.float32
    )
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)
