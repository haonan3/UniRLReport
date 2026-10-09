#!/usr/bin/env python3
"""Plot long-run natural-length timing, length, phases, and throughput."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def rolling(values: list[float], window: int) -> list[float]:
    output = []
    for index in range(len(values)):
        start = max(0, index + 1 - window)
        output.append(sum(values[start : index + 1]) / (index + 1 - start))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with args.csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[row["framework"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["step"]))
    min_step = min(int(row["step"]) for rows in grouped.values() for row in rows)
    max_step = max(int(row["step"]) for rows in grouped.values() for row in rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    colors = {"UniRL": "#d95f02", "VERL": "#1b9e77"}
    for framework, rows in grouped.items():
        steps = [int(row["step"]) for row in rows]
        color = colors.get(framework)
        e2e = [float(row["e2e_s"]) for row in rows]
        lengths = [float(row["response_length_mean"]) for row in rows]
        generation = [float(row["generation_s"]) for row in rows]
        training = [float(row["training_s"]) for row in rows]
        throughput = [float(row["e2e_output_tokens_per_s"]) for row in rows]
        axes[0, 0].plot(steps, rolling(e2e, args.window), label=framework, color=color)
        axes[0, 1].plot(steps, rolling(lengths, args.window), label=framework, color=color)
        axes[1, 0].plot(steps, rolling(generation, args.window), label=f"{framework} generation", color=color)
        axes[1, 0].plot(
            steps,
            rolling(training, args.window),
            label=f"{framework} training",
            color=color,
            linestyle="--",
        )
        axes[1, 1].plot(steps, rolling(throughput, args.window), label=framework, color=color)

    labels = (
        (axes[0, 0], "End-to-end time", "seconds / step"),
        (axes[0, 1], "Mean response length", "tokens / response"),
        (axes[1, 0], "Phase time", "seconds / step"),
        (axes[1, 1], "End-to-end output throughput", "output tokens / second"),
    )
    for axis, title, ylabel in labels:
        axis.set_title(f"{title} ({args.window}-step rolling mean)")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        for boundary in (100, 200, 300, 400):
            if min_step < boundary < max_step:
                axis.axvline(boundary, color="#777777", alpha=0.18, linewidth=0.8)
        axis.set_xlim(min_step, max_step)
    axes[1, 0].set_xlabel("training step")
    axes[1, 1].set_xlabel("training step")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
