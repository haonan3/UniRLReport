#!/usr/bin/env python3
"""Join driver-side rank events with judge-side request logs into overlap_trace.csv (see README.md)."""

from __future__ import annotations

import csv
import json
import os
import re
import sys

RUNS = [
    ("20260923-225836", "editscore_8b", "pr445t", "xdeep_base", "baseline", "", "xnode-editscore8b-t-server.log"),
    ("20260923-225836", "editscore_8b", "pr445t", "xdeep_stack", "serial", "", "xnode-editscore8b-t-server.log"),
    ("20260923-225836", "editscore_8b", "pr445t", "xdeep_stack_ov", "overlap", "depth1", "xnode-editscore8b-t-server.log"),
    ("20260923-230352", "editscore_72b", "pr445t", "xdeep_base", "baseline", "", "xnode-editscore72b-t-server.log"),
    ("20260923-230352", "editscore_72b", "pr445t", "xdeep_stack", "serial", "", "xnode-editscore72b-t-server.log"),
    ("20260923-230352", "editscore_72b", "pr445t", "xdeep_stack_ov", "overlap", "depth1", "xnode-editscore72b-t-server.log"),
    ("20260928-093100", "editscore_8b", "pr445d", "d_base", "baseline", "", "xnode-editscore8b-gen-server.log"),
    ("20260928-093100", "editscore_8b", "pr445d", "d_serial", "serial", "", "xnode-editscore8b-gen-server.log"),
    ("20260928-093100", "editscore_8b", "pr445d", "d_ov", "overlap", "depth1", "xnode-editscore8b-gen-server.log"),
    ("20260928-114630", "editscore_72b", "pr445q", "q_serial", "serial", "", "xnode-editscore72b-p2-server.log"),
    ("20260928-114630", "editscore_72b", "pr445q", "q_ov", "overlap", "queued", "xnode-editscore72b-p2-server.log"),
    ("20260928-124606", "editscore_72b", "pr445d", "d_ov", "overlap", "depth1", "xnode-editscore72b-p2-server.log"),
    ("20260928-131525", "editscore_8b", "pr445q", "q_ov", "overlap", "queued", "xnode-editscore8b-gen-server.log"),
]
SCORE = re.compile(r"trace score n=(\d+) recv=([\d.]+) done=([\d.]+)")
FIELDS = ["session", "reward", "arm", "overlap_impl", "rollout", "lane", "kind", "index", "start_s", "end_s", "recv_s", "rows"]


def judge_requests(path: str) -> list[tuple[float, float, int]]:
    events = []
    for line in open(path, errors="replace"):
        if m := SCORE.search(line):
            events.append((float(m.group(2)), float(m.group(3)), int(m.group(1))))
    return events


def extract(log_dir: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    judges: dict[str, list[tuple[float, float, int]]] = {}
    for session, reward, prefix, gate_arm, arm, impl, judge_log in RUNS:
        if judge_log not in judges:
            judges[judge_log] = judge_requests(os.path.join(log_dir, judge_log))
        for line in open(os.path.join(log_dir, f"{prefix}-{gate_arm}-{session}.log"), errors="replace"):
            at = line.find("trace json: ")
            if at < 0:
                continue
            trace = json.loads(line[at + len("trace json: "):])
            rollout = int(trace["rollout_id"]) + 1
            if rollout < 2:
                continue
            origin = min(e["t0"] for rank in trace["rollout"] for e in rank["events"] if e["kind"] == "gen")
            base = {"session": session, "reward": reward, "arm": arm, "overlap_impl": impl, "rollout": str(rollout)}

            def add(lane: str, kind: str, index: int, t0: float, t1: float, recv: float | None, n: int) -> None:
                rows.append({
                    **base, "lane": lane, "kind": kind, "index": str(index),
                    "start_s": f"{t0 - origin:.2f}", "end_s": f"{t1 - origin:.2f}",
                    "recv_s": "" if recv is None else f"{recv - origin:.2f}", "rows": str(n),
                })

            for key, kind, want in (("rollout", "generate", "gen"), ("reward", "reward_wait", "score")):
                for rank in trace.get(key) or []:
                    events = [e for e in rank["events"] if e["kind"] == want]
                    for index, event in enumerate(sorted(events, key=lambda e: e["t0"]), start=1):
                        add(f"rank{rank['rank']}", kind, index, event["t0"], event["t1"], None, int(event.get("rows") or 0))
            window = (trace["driver"]["gen_t0"] - 1.0, trace["driver"]["score_t1"] + 1.0)
            served = sorted((r for r in judges[judge_log] if r[0] >= window[0] and r[1] <= window[1]), key=lambda r: r[1])
            previous = None
            for index, (recv, done, n) in enumerate(served, start=1):
                start = recv if previous is None else max(recv, previous)
                add("judge", "judge_service", index, start, done, recv, n)
                previous = done
    return rows


def main() -> None:
    rows = extract(sys.argv[1])
    with open(sys.argv[2], "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows -> {sys.argv[2]}")


if __name__ == "__main__":
    main()
