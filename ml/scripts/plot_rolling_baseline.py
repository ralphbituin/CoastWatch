"""Plots for the rolling-baseline outputs (ML venv, matplotlib).

The main phase5/6 runner stores all numbers without requiring matplotlib. This
script renders the required figures from the saved CSV outputs using the ML
virtualenv:

    ml\\.venv\\Scripts\\python.exe ml\\scripts\\plot_rolling_baseline.py `
      --output-dir ml\\reports\\phase5_tide_anomaly

Synthetic injection output is explicitly labelled as synthetic. Nothing here
touches the database, the model or the main result CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "phase5_tide_anomaly"
TIMEZONE = ZoneInfo("Asia/Bangkok")
FLOOR_CLASSIFICATIONS = {
    "POSSIBLE_DRAWDOWN",
    "RAPID_SHORELINE_RETREAT",
    "PERSISTENT_DRAWDOWN",
    "DATA_UNRELIABLE",
}


def _float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


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


def plot_main(output_dir: Path) -> Path | None:
    anomaly_path = output_dir / "anomaly_results.csv"
    if not anomaly_path.is_file():
        return None
    rows = _rows(anomaly_path)
    if not rows or "rolling_baseline_px" not in rows[0]:
        return None
    times = [_time(row.get("capture_timestamp")) for row in rows]
    observed = [_float(row.get("observed_position_px")) for row in rows]
    baseline = [_float(row.get("rolling_baseline_px")) for row in rows]
    residual = [_float(row.get("residual_px")) for row in rows]
    z_values = [_float(row.get("robust_z")) for row in rows]
    cumulative = [_float(row.get("cumulative_drawdown_px")) for row in rows]
    threshold = 3.5
    report_path = output_dir / "processing_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            threshold = float(
                report.get("detector", {}).get("config", {}).get("z_threshold", 3.5)
            )
        except (OSError, ValueError, TypeError):
            pass

    path = output_dir / "plots" / "rolling_baseline_drawdown.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(13, 7),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]},
    )
    top.plot(
        times,
        observed,
        color="#2c3e50",
        linewidth=1.2,
        marker=".",
        markersize=3,
        label="observed (px, + = seaward)",
    )
    top.plot(
        times,
        baseline,
        color="#0f766e",
        linewidth=1.6,
        linestyle="--",
        label="Rolling baseline (unverified; no tide data)",
    )
    for row, moment in zip(rows, times, strict=False):
        if moment is None or row.get("classification") not in FLOOR_CLASSIFICATIONS:
            continue
        value = _float(row.get("observed_position_px")) or _float(
            row.get("rolling_baseline_px")
        )
        if value is None:
            continue
        color = "#c0392b" if row.get("classification") != "DATA_UNRELIABLE" else "#7f8c8d"
        top.scatter([moment], [value], color=color, edgecolor="black", s=34, zorder=5)
    top.set_ylabel("shoreline position (px)")
    top.set_title(
        "Tide-independent rolling-median drawdown detector\n"
        "Rolling baseline is self-referential and unverified; it is not a tide model"
    )
    handles, labels = top.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    top.legend(unique.values(), unique.keys(), fontsize=8, loc="best")
    top.grid(alpha=0.25)

    bottom.plot(times, residual, color="#c0392b", linewidth=1.2, label="residual (px)")
    bottom.plot(times, z_values, color="#2563eb", linewidth=1.0, label="robust z", alpha=0.8)
    bottom.plot(
        times,
        cumulative,
        color="#f97316",
        linewidth=1.1,
        label="cumulative drawdown (px)",
        alpha=0.85,
    )
    bottom.axhline(threshold, color="#2563eb", linestyle=":", linewidth=1)
    bottom.axhline(0.0, color="#94a3b8", linewidth=1)
    bottom.set_ylabel("residual / z / drawdown")
    bottom.grid(alpha=0.25)
    bottom.legend(fontsize=8, loc="best")
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TIMEZONE))
    figure.autofmt_xdate()
    figure.text(
        0.5,
        0.01,
        "Experimental rolling-baseline detector — not validated for tsunami detection.",
        ha="center",
        fontsize=8,
        color="#7f8c8d",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path)
    plt.close(figure)
    return path


def plot_injection(output_dir: Path) -> Path | None:
    path_in = output_dir / "injection" / "injection_results.csv"
    if not path_in.is_file():
        return None
    cells = _rows(path_in)
    if not cells:
        return None
    path = output_dir / "plots" / "injection_detection_rate.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.2), dpi=150)
    amplitudes = sorted({float(cell["amplitude_px"]) for cell in cells})
    for amplitude in amplitudes:
        subset = sorted(
            [cell for cell in cells if float(cell["amplitude_px"]) == amplitude],
            key=lambda cell: float(cell["ramp_seconds"]),
        )
        axis.plot(
            [float(cell["ramp_seconds"]) for cell in subset],
            [float(cell["detection_rate"]) for cell in subset],
            marker="o",
            label=f"A={amplitude:.0f} px",
        )
    axis.set_xlabel("ramp duration T (s)")
    axis.set_ylabel("detection rate")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    axis.set_title(
        "Synthetic injection — not real tsunami performance\n"
        "Seaward ramp-then-hold into the real shoreline series"
    )
    figure.text(
        0.5,
        0.01,
        "synthetic injection — not real tsunami performance",
        ha="center",
        fontsize=8,
        color="#7f8c8d",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    figure.savefig(path)
    plt.close(figure)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    main_plot = plot_main(output_dir)
    injection_plot = plot_injection(output_dir)
    print(f"[plot] rolling={main_plot}")
    print(f"[plot] injection={injection_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
