"""Create leakage-safe train/val/test splits for the unified dataset.

Grouping is performed by ``prepare_dataset.py`` (exact duplicates, perceptual
near-duplicates, timestamp sequence). This script only assigns whole groups to
splits and verifies that no group — and therefore no exact/near-duplicate —
crosses a split boundary.

Usage:
    python -m ml.scripts.split_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ml.src.dataset import load_config, resolve_path, unified_paths

SPLITS = ("train", "val", "test")
CLASS_KEYS = ("beach_pixels", "ocean_pixels", "sky_pixels")


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} is empty")
    for row in rows:
        row["width"] = int(row["width"])
        row["height"] = int(row["height"])
        row["group_size"] = int(row["group_size"])
        row["beach_pixels"] = int(row["beach_pixels"])
        row["ocean_pixels"] = int(row["ocean_pixels"])
        row["sky_pixels"] = int(row["sky_pixels"])
        row["ignore_pixels"] = int(row["ignore_pixels"])
        row["mean_luma"] = float(row["mean_luma"])
    return rows


def _write_manifest(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group_id = row.get("group_id") or f"singleton_{row['id']}"
        groups[group_id].append(row)
    return dict(groups)


def _global_stats(rows: list[dict[str, Any]]) -> tuple[np.ndarray, float, float]:
    class_counts = np.array(
        [sum(row[key] for row in rows) for key in CLASS_KEYS], dtype=np.float64
    )
    total = class_counts.sum()
    fractions = class_counts / total if total else np.zeros_like(class_counts)
    lumas = np.array([row["mean_luma"] for row in rows], dtype=np.float64)
    luma_mean = float(np.nanmean(lumas)) if lumas.size else 0.0
    luma_std = float(np.nanstd(lumas)) if lumas.size else 1.0
    if not np.isfinite(luma_std) or luma_std < 1e-6:
        luma_std = 1.0
    return fractions, luma_mean, luma_std


def _stats(
    assignment: dict[str, str],
    groups: dict[str, list[dict[str, Any]]],
    targets: dict[str, float],
    global_fractions: np.ndarray,
    global_luma: float,
    luma_std: float,
) -> tuple[float, dict[str, dict[str, Any]]]:
    details: dict[str, dict[str, Any]] = {}
    score = 0.0
    total_images = sum(len(groups[group_id]) for group_id in assignment)
    for split in SPLITS:
        members = [
            row
            for group_id, assigned in assignment.items()
            if assigned == split
            for row in groups[group_id]
        ]
        count = len(members)
        class_counts = np.array(
            [sum(row[key] for row in members) for key in CLASS_KEYS], dtype=np.float64
        )
        total = class_counts.sum()
        fractions = class_counts / total if total else np.zeros_like(class_counts)
        lumas = np.array([row["mean_luma"] for row in members], dtype=np.float64)
        mean_luma = float(np.nanmean(lumas)) if lumas.size else 0.0
        group_count = sum(1 for value in assignment.values() if value == split)
        score += 8.0 * ((count - targets[split]) / max(targets[split], 1.0)) ** 2
        score += 3.0 * float(np.sum((fractions - global_fractions) ** 2))
        score += 0.25 * ((mean_luma - global_luma) / luma_std) ** 2
        if total == 0:
            score += 5.0
        for index, key in enumerate(CLASS_KEYS):
            if class_counts[index] <= 0:
                score += 1.0
        if group_count < 2:
            score += 2.0
        details[split] = {
            "images": count,
            "groups": group_count,
            "class_pixels": {
                "Beach": int(class_counts[0]),
                "Ocean": int(class_counts[1]),
                "Sky": int(class_counts[2]),
            },
            "class_fraction": {
                "Beach": float(fractions[0]),
                "Ocean": float(fractions[1]),
                "Sky": float(fractions[2]),
            },
            "mean_luma": mean_luma,
            "strength": float(len(members) / max(total_images, 1)),
        }
    return score, details


def _greedy_start(
    group_ids: list[str],
    groups: dict[str, list[dict[str, Any]]],
    targets: dict[str, float],
    rng: random.Random,
) -> dict[str, str]:
    order = list(group_ids)
    rng.shuffle(order)
    counts = {split: 0 for split in SPLITS}
    assignment: dict[str, str] = {}
    for group_id in order:
        size = len(groups[group_id])
        # Prefer the split furthest below its target; a small overflow penalty
        # prevents a late huge group from destroying the ratio.
        def penalty(split: str) -> float:
            after = counts[split] + size
            overflow = max(0.0, after - targets[split] - 2.0)
            deficit = max(0.0, targets[split] - counts[split])
            return overflow * 3.0 - deficit / max(targets[split], 1.0)

        chosen = min(SPLITS, key=penalty)
        assignment[group_id] = chosen
        counts[chosen] += size
    return assignment


def _optimise(
    groups: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    *,
    seed: int,
    restarts: int = 24,
    swaps: int = 6000,
) -> tuple[dict[str, str], dict[str, dict[str, Any]], float]:
    targets = {
        "train": 0.70 * len(rows),
        "val": 0.15 * len(rows),
        "test": 0.15 * len(rows),
    }
    global_fractions, global_luma, luma_std = _global_stats(rows)
    group_ids = sorted(groups)
    rng = random.Random(seed)
    best_assignment: dict[str, str] | None = None
    best_details: dict[str, dict[str, Any]] | None = None
    best_score = math.inf
    for _ in range(restarts):
        assignment = _greedy_start(group_ids, groups, targets, rng)
        score, details = _stats(
            assignment, groups, targets, global_fractions, global_luma, luma_std
        )
        # Hill-climb by swapping two groups in different splits. This preserves
        # whole-group integrity and improves count/class/luma balance.
        for _ in range(swaps):
            left, right = rng.sample(group_ids, 2)
            if assignment[left] == assignment[right]:
                continue
            candidate = dict(assignment)
            candidate[left], candidate[right] = candidate[right], candidate[left]
            candidate_score, candidate_details = _stats(
                candidate, groups, targets, global_fractions, global_luma, luma_std
            )
            if candidate_score < score - 1e-12:
                assignment, score, details = candidate, candidate_score, candidate_details
        if score < best_score:
            best_assignment = assignment
            best_details = details
            best_score = score
    if best_assignment is None or best_details is None:
        raise RuntimeError("split optimisation failed")
    return best_assignment, best_details, best_score


def _verify_no_leakage(
    rows: list[dict[str, Any]], assignment: dict[str, str]
) -> list[str]:
    violations: list[str] = []
    # Group integrity.
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        group_id = row["group_id"]
        group_splits[group_id].add(assignment[group_id])
    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            violations.append(f"group {group_id} appears in multiple splits: {sorted(splits)}")
    # Exact-duplicate SHA-256 cross-split check. prepare_dataset.py already
    # removes extra copies, but this catches accidental future regressions.
    sha_splits: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        sha_splits[row["sha256"]].add(assignment[row["group_id"]])
    for sha, splits in sha_splits.items():
        if len(splits) > 1:
            violations.append(f"exact SHA-256 {sha[:12]} appears in multiple splits")
    return violations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ml/configs/segformer_b2.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    paths = unified_paths(config)
    seed = int(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))

    rows = _read_manifest(paths["manifest"])
    fieldnames = list(rows[0].keys())
    groups = _group_rows(rows)
    assignment, details, score = _optimise(groups, rows, seed=seed)
    violations = _verify_no_leakage(rows, assignment)

    for row in rows:
        row["split"] = assignment[row["group_id"]]
    _write_manifest(paths["manifest"], rows, fieldnames)

    by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in rows:
        by_split[row["split"]].append(row)
    for split, members in by_split.items():
        split_path = paths["splits"] / f"{split}.txt"
        names = sorted(row["image"] for row in members)
        with split_path.open("w", encoding="utf-8") as handle:
            handle.write(f"# {split}: {len(names)} images; source dataset(s): "
                         f"{', '.join(sorted({row['source_dataset'] for row in members}))}\n")
            for name in names:
                handle.write(name + "\n")

    group_summary = []
    for group_id in sorted(groups):
        members = groups[group_id]
        group_summary.append(
            {
                "group_id": group_id,
                "split": assignment[group_id],
                "size": len(members),
                "method": members[0].get("group_method", ""),
                "members": sorted(row["image"] for row in members),
            }
        )
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_images": len(rows),
        "total_groups": len(groups),
        "target_fraction": {"train": 0.70, "val": 0.15, "test": 0.15},
        "actual": details,
        "objective_score": score,
        "leakage_violations": violations,
        "group_policy": (
            "Whole groups assigned to one split. Groups come from "
            "prepare_dataset.py: exact duplicates excluded, perceptual "
            "near-duplicates clustered, timestamp frames kept together."
        ),
        "limitations": [
            "No camera/session metadata exists for UUID and videoframe images; "
            "camera-independent generalisation cannot be guaranteed.",
            "The test split is small, so metric confidence intervals are wide.",
        ],
        "groups": group_summary,
    }
    (paths["splits"] / "split_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (paths["splits"] / "README.md").write_text(
        "\n".join(
            [
                "# Split policy",
                "",
                f"- Images: {len(rows)}; groups: {len(groups)}",
                f"- Target: train 70% / val 15% / test 15%",
                f"- Actual: train {details['train']['images']} / "
                f"val {details['val']['images']} / test {details['test']['images']}",
                f"- Group integrity violations: {len(violations)}",
                "",
                "Whole groups are assigned to one split. Exact duplicates are",
                "excluded in `prepare_dataset.py`; perceptual near-duplicates are",
                "clustered; timestamp-named frames stay together. No UUID/videoframe",
                "camera or session metadata exists, so cross-camera generalisation",
                "cannot be proven from this split alone.",
                "",
                "The test split is never used for checkpoint selection.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("[split] actual counts:", {split: details[split]["images"] for split in SPLITS})
    print("[split] leakage violations:", violations or "none")
    print(f"[split] wrote {paths['splits']}")
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
