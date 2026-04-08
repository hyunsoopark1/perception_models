# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Plot developmental stage predictions over time from extract_description.py output.

Each clip is plotted as a point: x = recording date, y = developmental stage.
Multiple clips from the same date are jittered vertically to avoid overlap.

Usage:
    python plot_development.py descriptions.json
    python plot_development.py descriptions.json --output growth.png
    python plot_development.py descriptions.json --smooth       # add trend line
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Ordered developmental stages (youngest → oldest)
STAGE_ORDER = [
    "0-3 months",
    "4-6 months",
    "7-9 months",
    "10-12 months",
    "13-18 months",
    "19-24 months",
    "25-36 months",
]

# Approximate midpoint in months for each stage (used for trend line fitting)
STAGE_MONTHS = [1.5, 5, 8, 11, 15.5, 21.5, 30.5]


def _parse_stage(raw: str) -> int | None:
    """Return 0-based index into STAGE_ORDER, or None if unrecognised."""
    if not raw:
        return None
    raw = raw.strip().lower()
    for i, label in enumerate(STAGE_ORDER):
        if label.lower() in raw or raw in label.lower():
            return i
    # Try matching just the numeric part, e.g. "13-18" inside a longer string
    m = re.search(r"\d+[-–]\d+", raw)
    if m:
        token = m.group().replace("–", "-")
        for i, label in enumerate(STAGE_ORDER):
            if token in label:
                return i
    return None


def _extract_date(clip_path: str) -> datetime | None:
    """Extract recording date from clip filename like 2025-06-15_clip001.mp4."""
    name = Path(clip_path).stem
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", name)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    return None


def load_data(json_path: str):
    with open(json_path) as f:
        records = json.load(f)

    dates, stage_idxs, labels, paths = [], [], [], []
    skipped = 0
    for clip_path, entry in records.items():
        stage_raw = entry.get("stage", "")
        dt = _extract_date(clip_path)
        idx = _parse_stage(stage_raw)
        if dt is None or idx is None:
            skipped += 1
            continue
        dates.append(dt)
        stage_idxs.append(idx)
        labels.append(STAGE_ORDER[idx])
        paths.append(clip_path)

    if skipped:
        print(f"Skipped {skipped} clip(s) with missing date or unrecognised stage.")

    order = np.argsort(dates)
    dates      = [dates[i]       for i in order]
    stage_idxs = [stage_idxs[i] for i in order]
    labels     = [labels[i]      for i in order]
    paths      = [paths[i]       for i in order]
    return dates, stage_idxs, labels, paths


def add_jitter(stage_idxs, dates, jitter_y=0.18):
    """Vertically jitter points that share the same date+stage."""
    from collections import Counter
    y = np.array(stage_idxs, dtype=float)
    x_num = mdates.date2num(dates)
    counter: dict[tuple, int] = {}
    for i, (xi, yi) in enumerate(zip(x_num, stage_idxs)):
        key = (xi, yi)
        n = counter.get(key, 0)
        # Alternate above/below: 0, +1, -1, +2, -2, ...
        sign = 1 if n % 2 == 1 else -1
        offset = sign * ((n + 1) // 2) * jitter_y
        y[i] += offset
        counter[key] = n + 1
    return y


def plot(dates, stage_idxs, labels, output_path=None, smooth=False, title=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    y_jittered = add_jitter(stage_idxs, dates)

    # Color by stage index
    cmap = plt.get_cmap("plasma", len(STAGE_ORDER))
    colors = [cmap(idx / (len(STAGE_ORDER) - 1)) for idx in stage_idxs]

    ax.scatter(dates, y_jittered, c=colors, s=60, alpha=0.75, zorder=3,
               edgecolors="white", linewidths=0.4)

    # Optional LOWESS-style trend: fit a degree-1 poly over time
    if smooth and len(dates) >= 4:
        x_num = mdates.date2num(dates)
        coeffs = np.polyfit(x_num, stage_idxs, deg=1)
        x_fit = np.linspace(x_num[0], x_num[-1], 200)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(mdates.num2date(x_fit), y_fit, color="steelblue",
                lw=2, alpha=0.7, linestyle="--", label="trend (linear fit)")
        ax.legend(fontsize=9)

    # y axis: stage labels
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(STAGE_ORDER, fontsize=9)
    ax.set_ylim(-0.6, len(STAGE_ORDER) - 0.4)

    # x axis: dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=9)

    ax.set_xlabel("Recording date", fontsize=11)
    ax.set_ylabel("Predicted developmental stage", fontsize=11)
    ax.set_title(title or "Developmental stage over time", fontsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot developmental stage predictions over time.")
    parser.add_argument("descriptions_json",
                        help="JSON file produced by extract_description.py")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save plot to this file (PNG/PDF/SVG). Displays interactively if omitted.")
    parser.add_argument("--smooth", action="store_true",
                        help="Overlay a linear trend line.")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title.")
    args = parser.parse_args()

    dates, stage_idxs, labels, paths = load_data(args.descriptions_json)
    if not dates:
        print("No plottable data found — check that 'stage' fields are populated.")
        sys.exit(1)

    print(f"Plotting {len(dates)} clip(s) spanning "
          f"{dates[0].strftime('%Y-%m-%d')} → {dates[-1].strftime('%Y-%m-%d')}")

    plot(dates, stage_idxs, labels,
         output_path=args.output,
         smooth=args.smooth,
         title=args.title)


if __name__ == "__main__":
    main()
