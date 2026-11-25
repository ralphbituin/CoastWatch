"""Dataset-balanced sampler behaviour (joint fine-tuning stage)."""

from __future__ import annotations

import numpy as np

from ml.src.data.sampler import build_dataset_balanced_sampler, dataset_share_report


def test_each_dataset_gets_equal_expected_mass() -> None:
    sizes = [645, 2342, 1203]
    sampler = build_dataset_balanced_sampler(sizes, samples_per_epoch=1000, generator_seed=42)
    drawn = np.array(list(sampler), dtype=np.int64)
    assert drawn.size == 1000

    boundaries = np.cumsum(sizes)
    shares = []
    for start, end in zip([0, *boundaries[:-1]], boundaries, strict=True):
        shares.append(float(((drawn >= start) & (drawn < end)).mean()))
    assert all(abs(share - 1 / 3) < 0.08 for share in shares), shares


def test_sampler_is_deterministic_for_a_seed() -> None:
    first = list(build_dataset_balanced_sampler([10, 90], samples_per_epoch=50, generator_seed=7))
    second = list(build_dataset_balanced_sampler([10, 90], samples_per_epoch=50, generator_seed=7))
    assert first == second


def test_share_report_is_honest_about_coverage() -> None:
    report = dataset_share_report([10, 1000], samples_per_epoch=200)
    assert report["expected_share_per_dataset"] == 0.5
    assert report["expected_draws_per_dataset"] == 100
    assert report["epochs_to_cover_largest_once"] == 0.2
