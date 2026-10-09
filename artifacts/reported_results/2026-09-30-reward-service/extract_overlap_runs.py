#!/usr/bin/env python3
"""Extract per-rollout phase timings from the overlap-gate status summaries into overlap_runs.csv (see README.md)."""

from __future__ import annotations

import csv
import glob
import os
import re
import sys

SESSIONS = {
    "20260922-163432": ("A", "bagel_it2i", "editscore_8b", 8, "0.27.0", "evaluation pass enabled; the gate's POST-count check did not account for it (rc=0)"),
    "20260922-170146": ("A", "bagel_it2i", "editscore_72b", 8, "0.27.0", ""),
    "20260922-175009": ("A", "bagel_it2i", "editscore_8b", 8, "0.27.0", ""),
    "20260922-181520": ("A", "bagel_it2i", "editscore_72b", 8, "0.27.0", ""),
    "20260922-190018": ("A", "bagel_it2i", "editscore_8b", 8, "0.27.0", ""),
    "20260923-225836": ("A", "bagel_it2i", "editscore_8b", 8, "0.27.0", "event-trace instrumentation enabled"),
    "20260923-230352": ("A", "bagel_it2i", "editscore_72b", 8, "0.27.0", "event-trace instrumentation enabled"),
    "20260928-093100": ("B", "bagel_it2i", "editscore_8b", 4, "0.29.0", ""),
    "20260928-093606": ("B", "bagel_t2i", "pickscore_local", 0, "", ""),
    "20260928-114630": ("B", "bagel_it2i", "editscore_72b", 8, "0.27.0", ""),
    "20260928-124606": ("B", "bagel_it2i", "editscore_72b", 8, "0.27.0", ""),
    "20260928-131525": ("B", "bagel_it2i", "editscore_8b", 4, "0.29.0", ""),
    "20260930-141053": ("B", "bagel_it2i", "editscore_8b", 4, "0.29.0", "final tree: a micro may go below a generation group"),
}
DEPTH1_COMMITS = {"1fd8f59", "97e83f6"}
ARMS = {
    "xbase": "baseline", "xdeep_base": "baseline", "d_base": "baseline", "l_base": "baseline",
    "xstack": "serial", "xdeep_stack": "serial", "d_serial": "serial", "q_serial": "serial", "l_serial": "serial",
    "d_serial_ec": "serial",
    "xstack_ov": "overlap", "xdeep_stack_ov": "overlap", "d_ov": "overlap", "q_ov": "overlap", "l_ov": "overlap",
    "d_ov_ec": "overlap", "l_ov_ec": "overlap",
    "xrbs": "request_batch_only",
}

HEAD = re.compile(r"node=\S+ code=\S+ @ (\w+)")
START = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d START (\S+)\s+.*overrides=(.*)$")
END = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d END\s+(\S+)\s+(PASS|FAIL\S*)\s+rc=(\S+)")
PERF = re.compile(r"perf phases: (.*)$")
TIMING = re.compile(r"reward stack timing: ranks=(\d+) rows=(\d+) micros=(\d+)")
ROLLOUT = re.compile(r"rollout (\d+)/(\d+)\s+reward=")
FIELDS = [
    "session", "cluster", "unirl_commit", "workload", "reward", "judge_tp", "judge_vllm", "arm", "overlap_impl",
    "flush_after_rollout", "prompts", "samples_per_prompt", "rows_per_rank", "micro_batch_size", "micros_per_rank",
    "rollout", "steady", "generate_s", "reward_s", "generate_reward_s", "stack_generate_s", "stack_score_s",
    "train_s", "weight_sync_s", "gate_result", "session_note",
]


def _f(value: str | None) -> str:
    return "" if value is None else f"{float(value):.2f}"


def extract(log_dir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(glob.glob(os.path.join(log_dir, "pr445-*.status"))):
        stamp = os.path.basename(path)[: -len(".status")].rsplit("-", 2)
        stamp = f"{stamp[-2]}-{stamp[-1]}"
        if stamp not in SESSIONS:
            continue
        cluster, workload, reward, judge_tp, judge_vllm, note = SESSIONS[stamp]
        commit, arm, overrides, perf, micros, results, first = "", "", "", {}, "", {}, len(rows)
        for line in open(path, errors="replace"):
            line = line.rstrip("\n")
            if not commit and (m := HEAD.search(line)):
                commit = m.group(1)
            elif m := START.match(line):
                arm, overrides, perf, micros = m.group(1), m.group(2), {}, ""
            elif m := END.match(line):
                results[m.group(1)] = m.group(2) if m.group(3) == "0" else f"{m.group(2)} rc={m.group(3)}"
            elif m := TIMING.search(line):
                micros = m.group(3)
            elif m := PERF.search(line):
                perf = dict(kv.split("=", 1) for kv in m.group(1).split())
            elif (m := ROLLOUT.search(line)) and arm in ARMS:
                kind = ARMS[arm]
                prompts = re.search(r"(?:^| )batch_size=(\d+)", overrides)
                spp = re.search(r"samples_per_prompt=(\d+)", overrides)
                micro = re.search(r"micro_batch_size=(\d+)", overrides)
                rows_per_rank = int(prompts.group(1)) * int(spp.group(1)) // 8
                wall = perf.get("rollout_score")
                if wall is None:
                    wall = float(perf["generate"]) + float(perf["reward"])
                rows.append({
                    "session": stamp, "cluster": cluster, "unirl_commit": commit, "workload": workload,
                    "reward": reward, "judge_tp": str(judge_tp or ""), "judge_vllm": judge_vllm, "arm": kind,
                    "overlap_impl": ("" if kind != "overlap" else "depth1" if commit in DEPTH1_COMMITS else "queued"),
                    "flush_after_rollout": "1" if arm.endswith("_ec") else "0",
                    "prompts": prompts.group(1), "samples_per_prompt": spp.group(1), "rows_per_rank": str(rows_per_rank),
                    "micro_batch_size": micro.group(1) if micro else "", "micros_per_rank": micros,
                    "rollout": m.group(1), "steady": "1" if int(m.group(1)) >= 2 else "0",
                    "generate_s": _f(perf.get("generate")), "reward_s": _f(perf.get("reward")),
                    "generate_reward_s": _f(str(wall)), "stack_generate_s": _f(perf.get("stack_generate")),
                    "stack_score_s": _f(perf.get("stack_score")), "train_s": _f(perf.get("train")),
                    "weight_sync_s": _f(perf.get("weight_sync")), "gate_result": arm, "session_note": note,
                })
                perf, micros = {}, ""
        for row in rows[first:]:
            row["gate_result"] = results.get(row["gate_result"], "")
    return sorted(rows, key=lambda row: row["session"])


def main() -> None:
    log_dir, out = sys.argv[1], sys.argv[2]
    rows = extract(log_dir)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
