#!/usr/bin/env python3
"""Render the fixed-length UniRL/VERL per-step timing comparison."""

import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "unirl-report-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "unirl-report-cache"))

BUNDLE = Path(__file__).resolve().parent
INPUT = BUNDLE / "curve.csv"
OUTPUT = BUNDLE / "curve.png"


def load_rows():
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 8
    assert {row["framework"] for row in rows} == {"UniRL", "VERL"}
    assert all(row["workload"] == "fixed_4096" for row in rows)
    return rows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows()
    colors = {"UniRL": "#256b9c", "VERL": "#b65e2b"}
    panels = [
        ("total_s", "End-to-end"),
        ("generation_s", "Generation"),
        ("train_s", "Training"),
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#626975",
            "axes.linewidth": 0.7,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.35))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.19, top=0.72, wspace=0.31)

    for ax, (field, title) in zip(axes, panels):
        for framework in ("UniRL", "VERL"):
            selected = sorted(
                (row for row in rows if row["framework"] == framework),
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [int(row["step"]) for row in selected],
                [float(row[field]) for row in selected],
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=colors[framework],
                label=framework,
            )
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel("Seconds")
        ax.set_xticks([2, 3, 4, 5])
        ax.grid(axis="y", color="#d9dee6", linewidth=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Qwen3-4B DRPO fixed 4096-token benchmark",
        x=0.065,
        y=0.96,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.065,
        0.84,
        "16× H20 · 64 prompts × 8 samples · lower is faster",
        fontsize=9.5,
        color="#525a65",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False)
    fig.savefig(
        OUTPUT,
        dpi=200,
        metadata={"Software": "UniRLReport plot_performance_curve.py"},
    )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
