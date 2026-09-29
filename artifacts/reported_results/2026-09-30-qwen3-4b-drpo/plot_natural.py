#!/usr/bin/env python3
"""Render the natural-length UniRL/VERL per-step comparison."""

import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "unirl-report-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "unirl-report-cache"))

BUNDLE = Path(__file__).resolve().parent
INPUT = BUNDLE / "natural_curve.csv"
OUTPUT = BUNDLE / "natural_curve.png"


def load_rows():
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 42
    assert {row["framework"] for row in rows} == {
        "UniRL TP1",
        "UniRL TP2 + FlashInfer",
        "VERL TP2",
    }
    assert all(row["workload"] == "natural_eos_8192_cap" for row in rows)
    return rows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load_rows()
    colors = {
        "UniRL TP1": "#7d8b99",
        "UniRL TP2 + FlashInfer": "#256b9c",
        "VERL TP2": "#b65e2b",
    }
    panels = [
        ("total_s", "End-to-end", "Seconds"),
        ("generation_s", "Generation", "Seconds"),
        ("train_s", "Training", "Seconds"),
        ("response_length_mean", "Mean response length", "Tokens"),
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
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.15))
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.1, top=0.78, hspace=0.42, wspace=0.25)

    for ax, (field, title, ylabel) in zip(axes.flat, panels):
        for framework in ("UniRL TP1", "UniRL TP2 + FlashInfer", "VERL TP2"):
            selected = sorted(
                (row for row in rows if row["framework"] == framework),
                key=lambda row: int(row["step"]),
            )
            ax.plot(
                [int(row["step"]) for row in selected],
                [float(row[field]) for row in selected],
                marker="o",
                markersize=4,
                linewidth=1.7,
                color=colors[framework],
                label=framework,
            )
        ax.set_title(title, loc="left", fontweight="semibold")
        ax.set_xlabel("Rollout step")
        ax.set_ylabel(ylabel)
        ax.set_xticks([2, 4, 6, 8, 10, 12, 15])
        ax.grid(axis="y", color="#d9dee6", linewidth=0.6)
        ax.set_axisbelow(True)

    fig.suptitle(
        "Qwen3-4B DRPO natural-length benchmark",
        x=0.08,
        y=0.965,
        ha="left",
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.08,
        0.9,
        "EOS enabled · 8,192-token cap · 16× H20 · 64 prompts × 8 samples",
        fontsize=9.5,
        color="#525a65",
    )
    fig.text(
        0.08,
        0.855,
        "Timing and sampled response length are shown together because natural workloads differ by step.",
        fontsize=9.1,
        color="#525a65",
    )
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.985, 0.985), frameon=False)
    fig.savefig(
        OUTPUT,
        dpi=200,
        metadata={"Software": "UniRLReport plot_natural.py"},
    )
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
