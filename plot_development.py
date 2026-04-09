# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
Plot developmental stage predictions over time from extract_description.py output.

x axis: child's age in months (recording date − birthdate)
y axis: predicted developmental stage (discrete)

Usage:
    python plot_development.py descriptions.json
    python plot_development.py descriptions.json --output growth.png --smooth
    python plot_development.py descriptions.json --birthdate 2023-08-10
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

BIRTHDATE_DEFAULT = datetime(2023, 8, 10)

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


def _parse_stage(raw: str) -> int | None:
    """Return 0-based index into STAGE_ORDER, or None if unrecognised."""
    if not raw:
        return None
    raw = raw.strip().lower()
    for i, label in enumerate(STAGE_ORDER):
        if label.lower() in raw or raw in label.lower():
            return i
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


def _age_months(dt: datetime, birthdate: datetime) -> float:
    return (dt - birthdate).days / 30.4375


def load_data(json_path: str, birthdate: datetime):
    with open(json_path) as f:
        records = json.load(f)

    ages, stage_idxs, paths = [], [], []
    skipped = 0
    for clip_path, entry in records.items():
        stage_raw = entry.get("stage", "")
        dt = _extract_date(clip_path)
        idx = _parse_stage(stage_raw)
        if dt is None or idx is None:
            skipped += 1
            continue
        ages.append(_age_months(dt, birthdate))
        stage_idxs.append(idx)
        paths.append(clip_path)

    if skipped:
        print(f"Skipped {skipped} clip(s) with missing date or unrecognised stage.")

    order = np.argsort(ages)
    ages       = [ages[i]       for i in order]
    stage_idxs = [stage_idxs[i] for i in order]
    paths      = [paths[i]      for i in order]
    return ages, stage_idxs, paths


def plot(ages, stage_idxs, paths, birthdate, output_path=None, smooth=False, title=None):
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.array(ages, dtype=float)
    y = np.array(stage_idxs, dtype=int)

    cmap = plt.get_cmap("plasma", len(STAGE_ORDER))
    colors = [cmap(idx / (len(STAGE_ORDER) - 1)) for idx in stage_idxs]

    ax.scatter(x, y, c=colors, s=60, alpha=0.75, zorder=3,
               edgecolors="white", linewidths=0.4)

    # Optional linear trend line
    if smooth and len(ages) >= 4:
        coeffs = np.polyfit(ages, stage_idxs, deg=1)
        x_fit = np.linspace(min(ages), max(ages), 200)
        y_fit = np.polyval(coeffs, x_fit)
        ax.plot(x_fit, y_fit, color="steelblue", lw=2, alpha=0.7,
                linestyle="--", label="trend (linear fit)")
        ax.legend(fontsize=9)

    # y axis: discrete stage labels only
    ax.set_yticks(range(len(STAGE_ORDER)))
    ax.set_yticklabels(STAGE_ORDER, fontsize=9)
    ax.set_ylim(-0.6, len(STAGE_ORDER) - 0.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)

    # x axis: age in months
    ax.set_xlabel(f"Child's age (months)  [born {birthdate.strftime('%Y-%m-%d')}]", fontsize=11)
    ax.set_ylabel("Predicted developmental stage", fontsize=11)
    ax.set_title(title or "Developmental stage by age", fontsize=13)
    ax.xaxis.grid(True, linestyle=":", alpha=0.3)

    # Minor tick every month, major every 3
    max_age = max(ages) if ages else 36
    ax.set_xlim(max(0, min(ages) - 1), max_age + 1)
    ax.xaxis.set_major_locator(plt.MultipleLocator(3))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot developmental stage predictions by child age.")
    parser.add_argument("descriptions_json",
                        help="JSON file produced by extract_description.py")
    parser.add_argument("--birthdate", type=str, default="2023-08-10",
                        help="Child's birthdate as YYYY-MM-DD (default: 2023-08-10).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save plot to file (PNG/PDF/SVG). Shows interactively if omitted.")
    parser.add_argument("--smooth", action="store_true",
                        help="Overlay a linear trend line.")
    parser.add_argument("--title", type=str, default=None,
                        help="Custom plot title.")
    args = parser.parse_args()

    try:
        birthdate = datetime.strptime(args.birthdate, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid birthdate format: {args.birthdate!r}. Use YYYY-MM-DD.")
        sys.exit(1)

    ages, stage_idxs, paths = load_data(args.descriptions_json, birthdate)
    if not ages:
        print("No plottable data found — check that 'stage' fields are populated.")
        sys.exit(1)

    print(f"Plotting {len(ages)} clip(s), age range "
          f"{min(ages):.1f} → {max(ages):.1f} months")

    plot(ages, stage_idxs, paths, birthdate,
         output_path=args.output,
         smooth=args.smooth,
         title=args.title)


if __name__ == "__main__":
    main()
