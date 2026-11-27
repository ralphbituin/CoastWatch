"""Plots for the Phase 5/6 tide-aware anomaly prototype.

Run with the ML virtualenv (matplotlib is installed there)::

    ml\\.venv\\Scripts\\python.exe ml\\scripts\\plot_phase5_results.py

Reads observations.csv / anomaly_results.csv / processing_report.json and
writes PNGs under <output-dir>/plots. Missing data is shown as a gap or an
explicit "unavailable" annotation; it is never interpolated into a fake line.
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "phase5_tide_anomaly"
TIMEZONE = ZoneInfo("Asia/Bangkok")
ANOMALY_COLORS = {
    "UNUSUAL_SHORELINE_POSITION": "#e67e22",
    "RAPID_SHORELINE_RETREAT": "#c0392b",
    "PERSISTENT_DRAWDOWN": "#8e44ad",
    "DATA_UNRELIABLE": "#7f8c8d",
    "PENDING_PERSISTENCE": "#f1c40f",
    "EXPLORATORY_MOVEMENT_ONLY": "#3498db",
    "NORMAL": "#2ecc71",
}


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _bool(value: str | None) -> bool:
    return (value or "").lower() in {"true", "1", "yes"}


def _time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=TIMEZONE)
    return moment.astimezone(TIMEZONE)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _anomaly_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output = Path(args.output_dir).resolve()
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    observations = _rows(output / "observations.csv")
    anomalies = _anomaly_rows(output / "anomaly_results.csv")
    report = json.loads((output / "processing_report.json").read_text(encoding="utf-8"))

    times = [_time(row.get("capture_timestamp")) for row in observations]
    signed = [
        _float(row.get("signed_position_px")) if _bool(row.get("shoreline_available")) else None
        for row in observations
    ]
    valid_times = [
        moment for moment, value in zip(times, signed, strict=False) if moment and value is not None
    ]
    valid_positions = [value for value in signed if value is not None]
    axis = next(
        (row.get("position_axis") for row in observations if row.get("position_axis")),
        "y",
    )

    # 1. Shoreline position over time + anomaly markers.
    figure, axis_obj = plt.subplots(figsize=(13, 5.4), dpi=150)
    if valid_times:
        axis_obj.plot(
            [
                mdates.date2num(moment.astimezone(TIMEZONE)) if moment else np.nan
                for moment in times
            ],
            [np.nan if value is None else value for value in signed],
            color="#2c3e50",
            linewidth=1.3,
            marker=".",
            markersize=3,
            label=f"observed shoreline (signed {axis}, + = seaward)",
        )
    for row in anomalies:
        classification = row.get("classification")
        if classification not in {
            "UNUSUAL_SHORELINE_POSITION",
            "RAPID_SHORELINE_RETREAT",
            "PERSISTENT_DRAWDOWN",
        }:
            continue
        moment = _time(row.get("capture_timestamp"))
        value = _float(row.get("observed_position_px"))
        if moment is None or value is None:
            continue
        axis_obj.scatter(
            mdates.date2num(moment),
            value,
            s=34,
            color=ANOMALY_COLORS.get(classification, "#c0392b"),
            edgecolor="black",
            linewidth=0.4,
            zorder=5,
            label=classification,
        )
    handles, labels = axis_obj.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    axis_obj.legend(unique.values(), unique.keys(), loc="best", fontsize=8)
    axis_obj.set_title(
        "Shoreline position over time (signed pixels; positive = seaward)\n"
        "Gaps are frames with no usable Ocean/Beach contact"
    )
    axis_obj.set_ylabel("signed shoreline position (px)")
    axis_obj.grid(alpha=0.25)
    axis_obj.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TIMEZONE))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(plots / "position_timeseries.png")
    plt.close(figure)

    # 2. Tide and residual (or explicit unavailable annotations).
    expected = [_float(row.get("expected_position_px")) for row in anomalies]
    residual = [_float(row.get("residual_px")) for row in anomalies]
    tide = [
        _float(row.get("tide_height_m")) if _bool(row.get("tide_data_available")) else None
        for row in observations
    ]
    anomaly_times = [_time(row.get("capture_timestamp")) for row in anomalies]
    anomaly_x = [
        mdates.date2num(moment.astimezone(TIMEZONE)) if moment else np.nan
        for moment in anomaly_times
    ]
    observation_x = [
        mdates.date2num(moment.astimezone(TIMEZONE)) if moment else np.nan for moment in times
    ]
    figure, (left, right) = plt.subplots(
        2, 1, figsize=(13, 7.2), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    if any(value is not None for value in tide):
        left.plot(observation_x, tide, color="#1f77b4", linewidth=1.4, label="predicted tide (m)")
        left.set_ylabel("tide height (m)")
        left.legend(loc="best", fontsize=8)
    else:
        left.text(
            0.5,
            0.5,
            "No verified, geographically appropriate Thai tide data\n"
            "(NAMRIA / PHIVOLCS stations are not valid for Karon Beach)",
            ha="center",
            va="center",
            fontsize=10,
            transform=left.transAxes,
            color="#7f8c8d",
        )
        left.set_yticks([])
    valid_x = [value for value in observation_x if np.isfinite(value)]
    if valid_x:
        left.set_xlim(min(valid_x), max(valid_x))
    left.set_title("Tide association (separate axis) and tide-adjusted residual")
    left.grid(alpha=0.2)

    if any(value is not None for value in expected):
        right.plot(
            anomaly_x,
            expected,
            color="#16a085",
            linewidth=1.1,
            label="expected (provisional baseline)",
        )
    if any(value is not None for value in residual):
        right.plot(
            anomaly_x,
            residual,
            color="#c0392b",
            linewidth=1.1,
            label="residual (observed - expected)",
        )
        right.axhline(0, color="#7f8c8d", linewidth=0.8)
        right.set_ylabel("residual (px; + = seaward)")
    else:
        right.text(
            0.5,
            0.5,
            "Tide-adjusted expected shoreline and residual: UNAVAILABLE\n"
            "Baseline status: "
            + str(report.get("baseline", {}).get("status"))
            + "\nReason: "
            + textwrap.fill(str(report.get("baseline", {}).get("reason")), 88),
            ha="center",
            va="center",
            fontsize=9,
            transform=right.transAxes,
            color="#7f8c8d",
        )
        right.set_yticks([])
    if valid_x:
        right.set_xlim(min(valid_x), max(valid_x))
    right.set_title("Expected shoreline / residual (pixels; no metric calibration)")
    right.grid(alpha=0.2)
    for panel in (left, right):
        panel.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TIMEZONE))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(plots / "tide_and_residual.png")
    plt.close(figure)

    # 3. Classification timeline and movement rate.
    movement = [_float(row.get("movement_rate_px_s")) for row in anomalies]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7.0), dpi=150, sharex=True)
    classifications = [row.get("classification") or "unknown" for row in anomalies]
    class_order = sorted(set(classifications))
    class_rank = {name: index for index, name in enumerate(class_order)}
    for classification in class_order:
        points = [
            (moment, class_rank[classification])
            for moment, value in zip(anomaly_times, classifications, strict=False)
            if value == classification and moment is not None
        ]
        if points:
            top.scatter(
                [mdates.date2num(point[0]) for point in points],
                [point[1] for point in points],
                s=24,
                color=ANOMALY_COLORS.get(classification, "#95a5a6"),
                label=classification,
            )
    top.set_yticks(list(class_rank.values()))
    top.set_yticklabels(class_order, fontsize=8)
    top.set_title("Classification timeline (one point per sampled observation)")
    top.legend(loc="upper left", fontsize=7, ncol=2)
    top.grid(alpha=0.2)
    if valid_x:
        top.set_xlim(min(valid_x), max(valid_x))

    bottom.plot(anomaly_x, movement, color="#34495e", linewidth=1.0, marker=".", markersize=2)
    bottom.axhline(0, color="#7f8c8d", linewidth=0.8)
    threshold = report.get("thresholds", {}).get("rapid_threshold_px_s")
    if threshold is not None:
        bottom.axhline(
            float(threshold),
            color="#c0392b",
            linestyle="--",
            linewidth=1.0,
            label=f"rapid threshold {threshold:.3f} px/s",
        )
        bottom.axhline(-float(threshold), color="#c0392b", linestyle="--", linewidth=0.7)
    bottom.set_ylabel("movement rate (px/s; + = seaward)")
    bottom.set_title("Frame-to-frame movement rate and data-driven threshold")
    bottom.legend(loc="best", fontsize=8)
    bottom.grid(alpha=0.2)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TIMEZONE))
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(plots / "anomaly_timeline.png")
    plt.close(figure)

    # 4. Availability reasons and class fractions.
    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=150)
    reasons = Counter(
        (row.get("reason_code") or "unknown")
        for row in observations
        if not _bool(row.get("shoreline_available"))
    )
    if reasons:
        labels, values = zip(*reasons.most_common(), strict=False)
        left.barh(range(len(labels)), values, color="#95a5a6")
        left.set_yticks(range(len(labels)))
        left.set_yticklabels([label[:38] for label in labels], fontsize=8)
        left.invert_yaxis()
    else:
        left.text(
            0.5,
            0.5,
            "No unavailable shorelines",
            ha="center",
            va="center",
            transform=left.transAxes,
        )
    left.set_title("Unavailable shoreline reason codes")
    left.set_xlabel("frames")

    class_series = {"Beach": [], "Ocean": [], "Sky": []}
    for row in observations:
        try:
            fractions = json.loads(row.get("class_fractions") or "{}")
        except json.JSONDecodeError:
            fractions = {}
        for name in class_series:
            class_series[name].append(float(fractions.get(name, np.nan)))
    if any(np.isfinite(values).any() for values in class_series.values()):
        for name, color in (("Beach", "#d6b589"), ("Ocean", "#2070be"), ("Sky", "#87ceeb")):
            right.plot(observation_x, class_series[name], color=color, linewidth=1.0, label=name)
        right.set_title("Predicted class fractions over time")
        right.set_ylabel("fraction of frame")
        right.legend(loc="best", fontsize=8)
        right.grid(alpha=0.2)
        right.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TIMEZONE))
    else:
        right.text(
            0.5,
            0.5,
            "No class fractions available",
            ha="center",
            va="center",
            transform=right.transAxes,
        )
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(plots / "availability_and_classes.png")
    plt.close(figure)

    # 5. Movement-rate distribution with threshold.
    rates = [value for value in movement if value is not None]
    figure, axis_obj = plt.subplots(figsize=(9, 4.8), dpi=150)
    if rates:
        axis_obj.hist(rates, bins=60, color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.3)
        axis_obj.axvline(0, color="#2c3e50", linewidth=0.9)
        if threshold is not None:
            axis_obj.axvline(
                float(threshold),
                color="#c0392b",
                linestyle="--",
                label=f"rapid threshold {threshold:.3f} px/s",
            )
            axis_obj.legend(loc="best", fontsize=8)
        axis_obj.set_title("Distribution of seaward-positive movement rates")
        axis_obj.set_xlabel("movement rate (px/s)")
        axis_obj.set_ylabel("observations")
        axis_obj.grid(alpha=0.2)
    else:
        axis_obj.text(
            0.5,
            0.5,
            "No valid movement pairs",
            ha="center",
            va="center",
            transform=axis_obj.transAxes,
        )
    figure.tight_layout()
    figure.savefig(plots / "movement_rate_distribution.png")
    plt.close(figure)

    summary = {
        "plots": [
            str(plots / "position_timeseries.png"),
            str(plots / "tide_and_residual.png"),
            str(plots / "anomaly_timeline.png"),
            str(plots / "availability_and_classes.png"),
            str(plots / "movement_rate_distribution.png"),
        ],
        "observations": len(observations),
        "anomaly_rows": len(anomalies),
        "valid_positions": len(valid_positions),
    }
    (plots / "plot_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
