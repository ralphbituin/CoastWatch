"""Augmentation must never invent label values or change the output size."""

from __future__ import annotations

import numpy as np

from ml.src.data.transforms import build_eval_transform, build_train_transform


def _synthetic() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, size=(200, 320, 3), dtype=np.uint8)
    mask = np.zeros((200, 320), dtype=np.uint8)
    mask[100:] = 2
    mask[80:100] = 1
    return image, mask


def test_train_transform_outputs_a_fixed_size() -> None:
    image, mask = _synthetic()
    transform = build_train_transform(512, {"scale_crop_p": 0.0})
    out = transform(image=image, mask=mask)
    assert tuple(out["image"].shape) == (3, 512, 512)
    assert tuple(out["mask"].shape) == (512, 512)


def test_train_transform_keeps_labels_discrete() -> None:
    image, mask = _synthetic()
    for _ in range(5):
        transform = build_train_transform(128, {})
        out = transform(image=image, mask=mask)
        values = set(np.unique(out["mask"].numpy()).tolist())
        assert values <= {0, 1, 2}, values


def test_eval_transform_is_deterministic() -> None:
    image, mask = _synthetic()
    transform = build_eval_transform(256)
    first = transform(image=image, mask=mask)
    second = transform(image=image, mask=mask)
    assert np.array_equal(first["image"].numpy(), second["image"].numpy())
    assert np.array_equal(first["mask"].numpy(), second["mask"].numpy())


def test_normalised_image_is_float32() -> None:
    image, mask = _synthetic()
    out = build_eval_transform(64)(image=image, mask=mask)
    assert out["image"].dtype.is_floating_point
    assert out["mask"].dtype.is_floating_point is False
