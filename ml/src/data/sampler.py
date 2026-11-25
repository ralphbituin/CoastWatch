"""Dataset-aware sampling for the joint fine-tuning stage.

The largest dataset must not dominate. Each dataset gets (approximately) the
same expected number of draws per epoch regardless of its size; within a
dataset the samples are still drawn in proportion to the number of records.
Class imbalance is handled separately by the class-weighted cross-entropy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    from torch.utils.data import WeightedRandomSampler
except ImportError:  # pragma: no cover
    WeightedRandomSampler = None  # type: ignore[assignment,misc]


def build_dataset_balanced_sampler(
    dataset_sizes: Sequence[int],
    *,
    samples_per_epoch: int | None = None,
    replacement: bool = True,
    generator_seed: int = 42,
) -> WeightedRandomSampler:
    """Return a sampler with equal expected mass per dataset."""
    if WeightedRandomSampler is None:
        raise ImportError("PyTorch is required for sampling")
    sizes = [int(size) for size in dataset_sizes]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"dataset_sizes must be positive, got {sizes}")
    if samples_per_epoch is None:
        samples_per_epoch = int(round(sum(sizes) / len(sizes))) * len(sizes)
    if samples_per_epoch <= 0:
        raise ValueError("samples_per_epoch must be positive")

    weights: list[float] = []
    for size in sizes:
        per_sample = 1.0 / (len(sizes) * size)
        weights.extend([per_sample] * size)
    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array = weight_array / weight_array.sum()

    generator = None
    try:
        import torch

        generator = torch.Generator()
        generator.manual_seed(generator_seed)
    except ImportError:  # pragma: no cover
        pass

    return WeightedRandomSampler(
        weights=weight_array.tolist(),
        num_samples=int(samples_per_epoch),
        replacement=replacement,
        generator=generator,
    )


def dataset_share_report(
    dataset_sizes: Sequence[int], *, samples_per_epoch: int | None = None
) -> dict[str, object]:
    """Describe the expected sampling shares (useful in training logs)."""
    sizes = [int(size) for size in dataset_sizes]
    if samples_per_epoch is None:
        samples_per_epoch = int(round(sum(sizes) / len(sizes))) * len(sizes)
    per_dataset = samples_per_epoch / len(sizes)
    return {
        "sizes": sizes,
        "samples_per_epoch": samples_per_epoch,
        "expected_draws_per_dataset": per_dataset,
        "expected_share_per_dataset": 1.0 / len(sizes),
        "epochs_to_cover_largest_once": samples_per_epoch / max(sizes),
    }
