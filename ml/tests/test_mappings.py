"""Label-mapping invariants.

The mapping is the most dangerous part of this pipeline: a wrong mapping
silently trains the model on fabricated labels. These tests pin the documented
decisions and prove unmapped values fail loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ml.src.data.mappings import (
    IGNORE_INDEX,
    build_mapping,
    load_class_config,
    mapping_table_summary,
)


def test_schema_is_four_classes_with_ignore_255() -> None:
    config = load_class_config()
    schema = config["schema"]
    assert schema["num_classes"] == 4
    assert schema["ignore_index"] == IGNORE_INDEX == 255
    assert [schema["classes"][i]["name"] for i in range(4)] == [
        "unknown",
        "land",
        "water",
        "foam",
    ]
    assert schema["water_classes"] == [2, 3]


def test_india_only_sand_is_land_everything_else_ignored() -> None:
    mapping = build_mapping("india_sandy_beach")
    raw = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    mapped = mapping.apply(raw)
    assert mapped[0, 0] == IGNORE_INDEX
    assert mapped[0, 1] == 1
    assert mapped[1, 0] == 1


def test_sclabels_masks() -> None:
    mapping = build_mapping("sclabels")
    raw = np.array([[0, 25, 75, 150, 255]], dtype=np.uint8)
    mapped = mapping.apply(raw)
    assert mapped.tolist() == [[IGNORE_INDEX, IGNORE_INDEX, 1, 2, IGNORE_INDEX]]


def test_unmapped_value_raises_instead_of_guessing() -> None:
    mapping = build_mapping("sclabels")
    with pytest.raises(ValueError, match="unmapped source mask values"):
        mapping.apply(np.array([[0, 42]], dtype=np.uint8))


def test_non_integer_mask_raises() -> None:
    mapping = build_mapping("sclabels")
    with pytest.raises(ValueError, match="non-integer"):
        mapping.apply(np.array([[0.5, 75.0]], dtype=np.float32))


def test_ignored_classes_never_become_class_zero() -> None:
    """0 means "unknown prediction"; ignored labels must use 255 instead."""
    for dataset in ("india_sandy_beach", "sclabels"):
        mapping = build_mapping(dataset)
        ignored = mapping.lookup[mapping.lookup == 0]
        assert ignored.size == 0, f"{dataset} maps some source value to class 0"


def test_mapping_table_is_renderable() -> None:
    for dataset in ("india_sandy_beach", "sclabels"):
        rows = mapping_table_summary(build_mapping(dataset))
        assert rows
        assert all("target" in row and "source_value" in row for row in rows)
