#!/usr/bin/env python3
"""Export matched long-run natural-length timings by step and training band."""

from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
import re
import statistics

import wandb


ENTITY = "leviking98z-zhejiang-university"
PROJECT = "unirl-grpo"
RESPONSES_PER_STEP = 64 * 8
BANDS = (
    (1, 100, "0-100"),
    (101, 200, "100-200"),
    (201, 300, "200-300"),
    (301, 400, "300-400"),
    (401, 500, "400-500"),
    (501, 600, "500-600"),
    (601, 1000, "600-1000"),
)

SCHEMAS = {
    "UniRL": {
        "step": "rollout/step",
        "e2e_s": ("perf/step_time_s", "perf/rollout_time_s"),
        "generation_s": ("perf/generate_time_s",),
        "reward_s": ("perf/reward_time_s",),
        "advantage_s": (),
        "training_s": ("perf/train_time_s",),
        "weight_sync_s": ("perf/weight_sync_time_s",),
        "wake_up_s": ("perf/wake_up_time_s",),
        "sleep_s": ("perf/sleep_time_s",),
        "response_length_mean": ("rollout/response_len_mean",),
        "response_length_min": ("rollout/response_len_min",),
        "response_length_max": ("rollout/response_len_max",),
        "response_length_std": ("rollout/response_len_std",),
        "trunc_ratio": ("rollout/trunc_ratio",),
    },
    "VERL": {
        "step": "training/global_step",
        "e2e_s": ("timing_s/step", "perf/time_per_step"),
        "generation_s": ("timing_s/gen",),
        "reward_s": ("timing_s/reward",),
        "advantage_s": ("timing_s/adv",),
        "training_s": ("timing_s/update_actor",),
        "weight_sync_s": ("timing_s/update_weights",),
        "wake_up_s": (),
        "sleep_s": (),
        "response_length_mean": ("response_length/mean",),
        "response_length_min": ("response_length/min",),
        "response_length_max": ("response_length/max",),
        "response_length_std": (),
        "trunc_ratio": ("response_length/clip_ratio",),
    },
}

REQUIRED = (
    "e2e_s",
    "generation_s",
    "reward_s",
    "training_s",
    "weight_sync_s",
    "response_length_mean",
    "response_length_min",
    "response_length_max",
    "trunc_ratio",
)
PHASE_FIELDS = (
    "generation_s",
    "reward_s",
    "advantage_s",
    "training_s",
    "weight_sync_s",
    "wake_up_s",
    "sleep_s",
)

NUMBER_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def resolve_run(api: wandb.Api, reference: str):
    try:
        return api.run(f"{ENTITY}/{PROJECT}/{reference}")
    except Exception:
        matches = list(api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": reference}, per_page=100))
        if not matches:
            raise RuntimeError(f"W&B run not found: {reference}")
        return max(matches, key=lambda run: run.created_at)


def band_for_step(step: int) -> str:
    for start, end, label in BANDS:
        if start <= step <= end:
            return label
    return "outside"


def load_verl_log_history(path: Path, schema: dict[str, object]):
    """Yield VERL metrics captured in the console log but not flushed to W&B."""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    for line in text.splitlines():
        if "training/global_step:" not in line:
            continue
        source: dict[str, float] = {}
        keys = [schema["step"]]
        keys.extend(candidate for candidates in schema.values() if isinstance(candidates, tuple) for candidate in candidates)
        for key in keys:
            match = re.search(rf"{re.escape(key)}:(?:np\.[A-Za-z0-9_]+\()?({NUMBER_PATTERN})", line)
            if match:
                source[key] = float(match.group(1))
        if schema["step"] in source:
            yield source


def load_rows(
    api: wandb.Api,
    reference: str,
    framework: str,
    start_step: int,
    max_step: int,
    supplemental_log: Path | None = None,
):
    run = resolve_run(api, reference)
    schema = SCHEMAS[framework]
    by_step: dict[int, dict[str, float]] = {}
    history_sources = [run.scan_history(page_size=1000), run.history(samples=100_000, pandas=False)]
    if supplemental_log is not None:
        history_sources.append(load_verl_log_history(supplemental_log, schema))
    history = itertools.chain(*history_sources)
    for source in history:
        raw_step = source.get(schema["step"])
        if not finite(raw_step):
            continue
        step = int(raw_step)
        if not start_step <= step <= max_step:
            continue
        destination = by_step.setdefault(step, {})
        for output, candidates in schema.items():
            if output == "step":
                continue
            for candidate in candidates:
                value = source.get(candidate)
                if finite(value):
                    # Prefer W&B history when present. Supplemental console logs
                    # only fill metrics that were not flushed before shutdown.
                    destination.setdefault(output, float(value))
                    break

    rows = []
    for step, metrics in sorted(by_step.items()):
        missing = [field for field in REQUIRED if field not in metrics]
        if missing:
            continue
        output_tokens = metrics["response_length_mean"] * RESPONSES_PER_STEP
        phase_sum = sum(metrics.get(field, 0.0) for field in PHASE_FIELDS)
        rows.append(
            {
                "framework": framework,
                "run_id": run.id,
                "run_name": run.name,
                "step": step,
                "step_band": band_for_step(step),
                **{field: metrics.get(field, "") for field in SCHEMAS[framework] if field != "step"},
                "phase_sum_s": phase_sum,
                "unaccounted_s": metrics["e2e_s"] - phase_sum,
                "output_tokens": output_tokens,
                "e2e_output_tokens_per_s": output_tokens / metrics["e2e_s"],
                "generation_output_tokens_per_s": output_tokens / metrics["generation_s"],
            }
        )
    return run, rows


def numeric(rows: list[dict[str, object]], field: str) -> list[float]:
    return [float(row[field]) for row in rows if finite(row.get(field))]


def summarize_band(framework: str, run_id: str, label: str, rows: list[dict[str, object]]):
    output_tokens = sum(numeric(rows, "output_tokens"))
    e2e = numeric(rows, "e2e_s")
    generation = numeric(rows, "generation_s")
    result: dict[str, object] = {
        "framework": framework,
        "run_id": run_id,
        "step_band": label,
        "first_step": min(int(row["step"]) for row in rows),
        "last_step": max(int(row["step"]) for row in rows),
        "n_steps": len(rows),
    }
    for field in (*PHASE_FIELDS, "phase_sum_s", "unaccounted_s", "trunc_ratio"):
        values = numeric(rows, field)
        result[f"{field}_mean"] = statistics.fmean(values) if values else ""
        result[f"{field}_population_std"] = statistics.pstdev(values) if values else ""
    lengths = numeric(rows, "response_length_mean")
    result["response_length_mean"] = statistics.fmean(lengths)
    result["response_length_step_population_std"] = statistics.pstdev(lengths)
    result["e2e_s_mean"] = statistics.fmean(e2e)
    result["e2e_s_population_std"] = statistics.pstdev(e2e)
    result["e2e_output_tokens_per_s"] = output_tokens / sum(e2e)
    result["generation_output_tokens_per_s"] = output_tokens / sum(generation)
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unirl-run", required=True, help="W&B run ID or exact display name")
    parser.add_argument("--verl-run", required=True, help="W&B run ID or exact display name")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-step", type=int, default=2, help="step 1 is the startup warmup by default")
    parser.add_argument("--max-step", type=int, default=500)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--verl-log",
        type=Path,
        help="VERL console log used only to supplement history rows missing from W&B",
    )
    args = parser.parse_args()

    api = wandb.Api(timeout=120)
    all_rows: list[dict[str, object]] = []
    run_ids = {}
    for framework, reference in (("UniRL", args.unirl_run), ("VERL", args.verl_run)):
        supplemental_log = args.verl_log if framework == "VERL" else None
        run, rows = load_rows(
            api,
            reference,
            framework,
            args.start_step,
            args.max_step,
            supplemental_log=supplemental_log,
        )
        steps = {int(row["step"]) for row in rows}
        expected = set(range(args.start_step, args.max_step + 1))
        if not args.allow_partial and steps != expected:
            missing = sorted(expected - steps)
            raise RuntimeError(f"{framework}: missing {len(missing)} steps; first missing={missing[:10]}")
        run_ids[framework] = run.id
        all_rows.extend(rows)

    summaries = []
    for framework in ("UniRL", "VERL"):
        framework_rows = [row for row in all_rows if row["framework"] == framework]
        for _start, _end, label in BANDS:
            band_rows = [row for row in framework_rows if row["step_band"] == label]
            if band_rows:
                summaries.append(summarize_band(framework, run_ids[framework], label, band_rows))
        if framework_rows:
            summaries.append(summarize_band(framework, run_ids[framework], "overall", framework_rows))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "natural_long_per_step.csv", all_rows)
    write_csv(args.output_dir / "natural_long_by_phase.csv", summaries)
    comparisons = []
    by_framework_band = {(row["framework"], row["step_band"]): row for row in summaries}
    labels = [label for _start, _end, label in BANDS] + ["overall"]
    for label in labels:
        unirl = by_framework_band.get(("UniRL", label))
        verl = by_framework_band.get(("VERL", label))
        if unirl is None or verl is None:
            continue
        comparisons.append(
            {
                "step_band": label,
                "n_steps": min(int(unirl["n_steps"]), int(verl["n_steps"])),
                "verl_e2e_s_per_step": verl["e2e_s_mean"],
                "verl_response_length": verl["response_length_mean"],
                "unirl_e2e_s_per_step": unirl["e2e_s_mean"],
                "unirl_response_length": unirl["response_length_mean"],
                "unirl_vs_verl_e2e_percent": (
                    float(unirl["e2e_s_mean"]) / float(verl["e2e_s_mean"]) - 1.0
                )
                * 100.0,
                "unirl_vs_verl_e2e_throughput_percent": (
                    float(unirl["e2e_output_tokens_per_s"])
                    / float(verl["e2e_output_tokens_per_s"])
                    - 1.0
                )
                * 100.0,
                "unirl_vs_verl_generation_throughput_percent": (
                    float(unirl["generation_output_tokens_per_s"])
                    / float(verl["generation_output_tokens_per_s"])
                    - 1.0
                )
                * 100.0,
            }
        )
    if comparisons:
        write_csv(args.output_dir / "natural_long_comparison.csv", comparisons)
    print(
        f"wrote {len(all_rows)} per-step rows, {len(summaries)} phase rows, "
        f"and {len(comparisons)} comparison rows"
    )


if __name__ == "__main__":
    main()
