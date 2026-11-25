"""Manifest integrity: no split leakage, deterministic rebuilds, documented counts.

These tests need the generated manifests (``python -m ml.src.train manifests``).
They are skipped when the datasets are absent, e.g. on a clean CI checkout.
"""

from __future__ import annotations

import pytest

from ml.src.data.splits import load_manifest
from ml.src.utils import project_root, resolve_path

MANIFESTS = project_root() / "ml" / "manifests"
DATASETS = project_root() / "ml" / "datasets"

pytestmark = pytest.mark.skipif(
    not MANIFESTS.is_dir() or not DATASETS.is_dir(),
    reason="manifests/datasets not present in this checkout",
)


def _records(dataset: str, split: str):
    path = MANIFESTS / f"{dataset}_{split}.csv"
    if not path.is_file():
        pytest.skip(f"missing manifest {path.name}")
    return load_manifest(path)


def test_expected_manifest_counts() -> None:
    assert len(_records("india_sandy_beach", "train")) == 652
    assert len(_records("india_sandy_beach", "val")) == 217
    assert len(_records("india_sandy_beach", "test")) == 219
    total = sum(len(_records("sclabels", split)) for split in ("train", "val", "test"))
    assert total == 1717


def test_splits_do_not_overlap() -> None:
    for dataset in ("india_sandy_beach", "sclabels"):
        paths = {
            split: {record.image_path for record in _records(dataset, split)}
            for split in ("train", "val", "test")
        }
        assert not paths["train"] & paths["val"], dataset
        assert not paths["train"] & paths["test"], dataset
        assert not paths["val"] & paths["test"], dataset


def test_sclabels_split_is_chronological_per_site() -> None:
    train = _records("sclabels", "train")
    val = _records("sclabels", "val")
    for site in {record.group for record in train + val}:
        train_dates = sorted(r.timestamp for r in train if r.group == site)
        val_dates = sorted(r.timestamp for r in val if r.group == site)
        if train_dates and val_dates:
            assert max(train_dates) <= min(val_dates), site


def test_all_manifest_files_exist_on_disk() -> None:
    for dataset in ("india_sandy_beach", "sclabels"):
        for split in ("train", "val", "test"):
            for record in _records(dataset, split):
                assert resolve_path(record.image_path).is_file(), record.image_path
                assert resolve_path(record.mask_path).is_file(), record.mask_path
