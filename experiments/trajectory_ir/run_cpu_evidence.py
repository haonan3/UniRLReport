#!/usr/bin/env python3
"""Run executable correctness checks and CPU microbenchmarks for UniRL's trajectory IR.

This script imports the pinned UniRL checkout directly.  It does not emulate rollout,
training, transport, or GPU execution, and its timings are regression evidence rather
than end-to-end system-performance measurements.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import hashlib
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable


PINNED_COMMIT = "f5d710406b215bb7a0b387fdd37e4d4778b92338"
RUN_ID = "cpu_f5d7104_macos_arm64"
ROOT_COUNTS = (8, 32, 128)
AR_BRANCH = 4
DIFFUSION_BRANCH = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unirl-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", default=PINNED_COMMIT)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Permit a modified UniRL checkout (disabled for paper evidence).",
    )
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("--warmup must be nonnegative; --repeats and --threads must be positive")
    return args


def git(repo: Path, *args: str, allow_failure: bool = False) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode and not allow_failure:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def sysctl(name: str) -> str | None:
    proc = subprocess.run(["sysctl", "-n", name], check=False, capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical(value: Any) -> Any:
    """Build a stable, data-sensitive representation of a UniRL Batch tree."""
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "tensor": True,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "sha256": sha256_bytes(tensor.numpy().tobytes()),
        }
    if dataclasses.is_dataclass(value):
        result = {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: canonical(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }
        cu = getattr(value, "cu_seqlens", None)
        if cu is not None:
            result["cu_seqlens"] = canonical(cu)
        return result
    if isinstance(value, dict):
        return {str(key): canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def digest(value: Any) -> str:
    payload = json.dumps(canonical(value), sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def replace_part(sample: Any, part_index: int, **updates: Any) -> Any:
    parts = list(sample.parts)
    parts[part_index] = dataclasses.replace(parts[part_index], **updates)
    return type(sample)(parts=parts, reward_compute_s=sample.reward_compute_s)


def build_sample(n_roots: int, ar_branch: int = AR_BRANCH, diffusion_branch: int = DIFFUSION_BRANCH) -> Any:
    root_ids = [f"root-{i:04d}" for i in range(n_roots)]
    root = Part.input(
        root_ids,
        primitives={"text": Texts([f"prompt {i}" for i in range(n_roots)])},
        metadata=[{"prompt_index": i} for i in range(n_roots)],
        role="user",
    )
    sample = Sample.request(root)
    sample = sample.fork(ar_branch, sampling_params=ARSamplingParams(samples_per_prompt=ar_branch))
    n_ar = n_roots * ar_branch
    ar_part = sample.parts[-1].fill(
        primitives={"text": Texts([f"answer {i}" for i in range(n_ar)])},
        output_version=7,
    )
    sample = replace_part(sample, -1, **{field.name: getattr(ar_part, field.name) for field in dataclasses.fields(ar_part)})
    sample = sample.observe(Texts([f"observation {i}" for i in range(n_ar)]), role="tool")
    sample = sample.fork(
        diffusion_branch,
        sampling_params=DiffusionSamplingParams(
            samples_per_prompt=diffusion_branch,
            num_inference_steps=4,
            height=2,
            width=2,
        ),
    )
    n_leaf = n_ar * diffusion_branch
    pixels = torch.arange(n_leaf * 4, dtype=torch.float32).reshape(n_leaf, 1, 2, 2)
    leaf = sample.parts[-1].fill(primitives={"image": Images.from_dense(pixels)}, output_version=8)
    sample = replace_part(sample, -1, **{field.name: getattr(leaf, field.name) for field in dataclasses.fields(leaf)})
    rewards = torch.arange(n_leaf, dtype=torch.float32)
    return replace_part(sample, -1, rewards=rewards)


def expect_raises(name: str, exception: type[BaseException], fn: Callable[[], Any]) -> dict[str, Any]:
    try:
        fn()
    except exception as exc:
        return {"name": name, "passed": True, "observed": type(exc).__name__}
    except Exception as exc:  # pragma: no cover - recorded as a hard failure
        raise AssertionError(f"{name}: expected {exception.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"{name}: expected {exception.__name__}, but no exception was raised")


@dataclasses.dataclass
class LocalHandle:
    tensor: Any

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.tensor.shape)

    @property
    def dtype(self) -> Any:
        return self.tensor.dtype

    @property
    def device(self) -> Any:
        return self.tensor.device

    def local(self) -> Any:
        return self.tensor


def make_ref(n_rows: int, width: int = 4) -> tuple[Any, list[LocalHandle]]:
    split = max(1, n_rows // 3)
    tensors = [
        torch.arange(0, split * width, dtype=torch.float32).reshape(split, width),
        torch.arange(split * width, n_rows * width, dtype=torch.float32).reshape(n_rows - split, width),
    ]
    handles = [LocalHandle(tensor) for tensor in tensors if tensor.shape[0] > 0]
    return TensorRef.from_handles(handles), handles


def run_correctness_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    sample = build_sample(3, ar_branch=2, diffusion_branch=2)
    expected_sizes = [3, 6, 6, 12]
    assert [part.batch_size for part in sample.parts] == expected_sizes
    checks.append(
        {
            "name": "cross_stage_lineage_validation",
            "passed": True,
            "detail": {"part_batch_sizes": expected_sizes, "modalities": ["text", "text", "text", "image"]},
        }
    )

    split_samples = sample.split()
    assert len(split_samples) == 3
    assert all([part.batch_size for part in shard.parts] == [1, 2, 2, 4] for shard in split_samples)
    joined = Sample.concat(split_samples)
    assert digest(joined) == digest(sample)
    checks.append(
        {
            "name": "split_concat_exact_roundtrip",
            "passed": True,
            "detail": {"digest": digest(sample), "root_shards": len(split_samples)},
        }
    )

    selected = sample.select(torch.tensor([2, 0], dtype=torch.long))
    selected_expected = Sample.concat([split_samples[2], split_samples[0]])
    assert digest(selected) == digest(selected_expected)
    sliced = sample.slice(1, 3)
    sliced_expected = Sample.concat(split_samples[1:3])
    assert digest(sliced) == digest(sliced_expected)
    assert [part.batch_size for part in selected.parts] == [2, 4, 4, 8]
    checks.append(
        {
            "name": "whole_tree_select_and_slice",
            "passed": True,
            "detail": {"selected_roots": selected.parts[0].sample_ids, "part_batch_sizes": [2, 4, 4, 8]},
        }
    )

    propagated = sample.propagate_rewards("mean")
    expected_leaf = torch.arange(12, dtype=torch.float32)
    expected_ar = expected_leaf.reshape(6, 2).mean(dim=1)
    expected_root = expected_ar.reshape(3, 2).mean(dim=1)
    assert torch.equal(propagated.parts[-1].rewards, expected_leaf)
    assert torch.equal(propagated.parts[2].rewards, expected_ar)
    assert torch.equal(propagated.parts[1].rewards, expected_ar)
    assert torch.equal(propagated.parts[0].rewards, expected_root)
    with_advantages = propagated.parts[1].compute_advantages()
    grouped_advantages = with_advantages.advantages.reshape(3, 2)
    assert torch.allclose(grouped_advantages.mean(dim=1), torch.zeros(3), atol=1e-6)
    assert torch.allclose(
        torch.sqrt(torch.mean(grouped_advantages.square(), dim=1)), torch.ones(3), atol=1e-6
    )
    checks.append(
        {
            "name": "reward_propagation_and_group_advantages",
            "passed": True,
            "detail": {
                "root_rewards": expected_root.tolist(),
                "generation_rewards": expected_ar.tolist(),
                "advantages": with_advantages.advantages.tolist(),
            },
        }
    )

    bad_child = Part(sample_ids=["missing-parent/0"])
    checks.append(
        expect_raises("invalid_lineage_rejected", ValueError, lambda: Sample(parts=[sample.parts[0], bad_child]))
    )
    checks.append(
        expect_raises(
            "primitive_modality_mismatch_rejected",
            ValueError,
            lambda: Part.input(["one"], primitives={"image": Texts(["not an image"])}),
        )
    )
    checks.append(
        expect_raises(
            "primitive_batch_mismatch_rejected",
            ValueError,
            lambda: Part.input(["one", "two"], primitives={"text": Texts(["only one"])}),
        )
    )

    packed = TextSegment.pack(
        tokens=[torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4, 5, 6])],
        log_probs=[torch.tensor([0.1, 0.2]), torch.tensor([0.3]), torch.tensor([0.4, 0.5, 0.6])],
        loss_mask=[torch.tensor([1, 1]), torch.tensor([1]), torch.tensor([0, 1, 1])],
    )
    packed_selected = packed.select(torch.tensor([2, 0], dtype=torch.long))
    assert packed_selected.lengths.tolist() == [3, 2]
    assert packed_selected.tokens.tolist() == [4, 5, 6, 1, 2]
    packed_joined = TextSegment.concat([packed.slice(0, 1), packed.slice(1, 3)])
    assert digest(packed_joined) == digest(packed)
    checks.append(
        {
            "name": "packed_variable_length_roundtrip",
            "passed": True,
            "detail": {"lengths": packed.lengths.tolist(), "selected_lengths": [3, 2]},
        }
    )

    tensor_ref, handles = make_ref(7)
    indices = torch.tensor([1, 2, 3, 6, 0], dtype=torch.long)
    selected_ref = tensor_ref.select(indices)
    assert all(any(span.handle is handle for handle in handles) for span in selected_ref.spans)
    assert torch.equal(selected_ref.materialize(), tensor_ref.materialize().index_select(0, indices))
    checks.append(
        {
            "name": "tensorref_view_selection_and_materialization",
            "passed": True,
            "detail": {
                "rows": selected_ref.batch_size,
                "spans": len(selected_ref.spans),
                "source_handles": len(handles),
                "handle_identity_preserved": True,
            },
        }
    )
    checks.append(
        expect_raises(
            "tensorref_bounds_rejected", IndexError, lambda: tensor_ref.select(torch.tensor([7]))
        )
    )

    assert checks and all(check["passed"] for check in checks)
    return checks


def nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def consume(value: Any) -> int:
    if hasattr(value, "batch_size"):
        return int(value.batch_size)
    if isinstance(value, list):
        return len(value)
    return 0


def time_operation(fn: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    sink = 0
    for _ in range(warmup):
        sink ^= consume(fn())
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        trials_us: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            value = fn()
            elapsed = time.perf_counter_ns() - start
            sink ^= consume(value)
            trials_us.append(elapsed / 1_000.0)
    finally:
        if was_enabled:
            gc.enable()
    if sink == -1:  # keep the result consumption visibly live to static analyzers
        raise AssertionError("unreachable")
    return trials_us


def run_benchmarks(warmup: int, repeats: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for n_roots in ROOT_COUNTS:
        sample = build_sample(n_roots)
        splits = sample.split()
        half_indices = torch.arange(0, n_roots, 2, dtype=torch.long)
        ref, _ = make_ref(sample.parts[-1].batch_size)
        n_rows = ref.batch_size
        contiguous_start = n_rows // 4
        contiguous_stop = 3 * n_rows // 4
        strided_indices = torch.arange(0, n_rows, 2, dtype=torch.long)
        operations: dict[str, Callable[[], Any]] = {
            "sample.split": sample.split,
            "sample.split_concat": lambda splits=splits: Sample.concat(splits),
            "sample.select_half": lambda sample=sample, indices=half_indices: sample.select(indices),
            "sample.propagate_rewards": sample.propagate_rewards,
            "tensorref.select_contiguous": lambda ref=ref, start=contiguous_start, stop=contiguous_stop: ref.slice(
                start, stop
            ),
            "tensorref.select_strided": lambda ref=ref, indices=strided_indices: ref.select(indices),
        }
        for operation, fn in operations.items():
            values = time_operation(fn, warmup, repeats)
            raw.append(
                {
                    "root_groups": n_roots,
                    "leaf_rows": sample.parts[-1].batch_size,
                    "operation": operation,
                    "unit": "microseconds",
                    "trials": values,
                }
            )
            summary.append(
                {
                    "root_groups": n_roots,
                    "leaf_rows": sample.parts[-1].batch_size,
                    "operation": operation,
                    "median_us": statistics.median(values),
                    "p90_us": nearest_rank(values, 0.90),
                    "min_us": min(values),
                    "max_us": max(values),
                    "repeats": repeats,
                }
            )
    return raw, summary


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_summary_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    fields = ["root_groups", "leaf_rows", "operation", "median_us", "p90_us", "min_us", "max_us", "repeats"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def write_tex_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    selected_operations = {"sample.split_concat", "sample.select_half", "tensorref.select_strided"}
    selected = [row for row in rows if row["operation"] in selected_operations]
    lines = [
        "% Generated by experiments/trajectory_ir/run_cpu_evidence.py; do not edit by hand.",
        *[
            f"{row['root_groups']} & {row['leaf_rows']} & \\texttt{{{tex_escape(row['operation'])}}} & "
            f"{row['median_us']:.1f} & {row['p90_us']:.1f} \\\\"
            for row in selected
        ],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tex_table(path: Path, rows: list[dict[str, Any]]) -> None:
    selected_operations = {"sample.split_concat", "sample.select_half", "tensorref.select_strided"}
    selected = [row for row in rows if row["operation"] in selected_operations]
    lines = [
        "% Generated by experiments/trajectory_ir/run_cpu_evidence.py; do not edit by hand.",
        r"\begin{tabular}{@{}rrlrr@{}}",
        r"\toprule",
        r"Roots & Leaf rows & Operation & Median & p90 \\",
        r"\midrule",
        *[
            f"{row['root_groups']} & {row['leaf_rows']} & \\texttt{{{tex_escape(row['operation'])}}} & "
            f"{row['median_us']:.1f} & {row['p90_us']:.1f} \\\\"
            for row in selected
        ],
        r"\bottomrule",
        r"\end{tabular}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(
    path: Path,
    manifest: dict[str, Any],
    checks: list[dict[str, Any]],
    summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# UniRL trajectory-IR CPU evidence",
        "",
        f"- Run ID: `{manifest['run_id']}`",
        f"- UniRL commit: `{manifest['unirl']['commit']}`",
        f"- Checkout clean: `{manifest['unirl']['clean']}`",
        f"- Python: `{manifest['software']['python']}`",
        f"- PyTorch: `{manifest['software']['torch']}`",
        f"- Platform: `{manifest['hardware']['platform']}`",
        f"- Timed device: `CPU` ({manifest['benchmark']['threads']} thread)",
        f"- Protocol: {manifest['benchmark']['warmup']} warm-up + {manifest['benchmark']['repeats']} measured repetitions",
        "",
        "## Correctness",
        "",
        f"All {len(checks)} executable checks passed. The process aborts on the first failed assertion.",
        "",
        *[f"- PASS — `{check['name']}`" for check in checks],
        "",
        "## Timing summary",
        "",
        "| Root groups | Leaf rows | Operation | Median (us) | p90 (us) |",
        "|---:|---:|---|---:|---:|",
        *[
            f"| {row['root_groups']} | {row['leaf_rows']} | `{row['operation']}` | "
            f"{row['median_us']:.1f} | {row['p90_us']:.1f} |"
            for row in summary
        ],
        "",
        "## Scope",
        "",
        "These are synthetic, single-process CPU measurements of container and reference operations. "
        "They do not measure rollout throughput, learner throughput, network transport, GPU utilization, "
        "training quality, or synchronous/asynchronous convergence, and must not be extrapolated to those claims.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_checksums(output_dir: Path) -> None:
    targets = sorted(path for path in output_dir.iterdir() if path.is_file() and path.name != "checksums.sha256")
    lines = [f"{sha256_bytes(path.read_bytes())}  {path.name}" for path in targets]
    (output_dir / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    unirl_root = args.unirl_root.resolve()
    output_dir = args.output_dir.resolve()
    if not (unirl_root / "unirl").is_dir():
        raise RuntimeError(f"--unirl-root does not contain the unirl package: {unirl_root}")

    commit = git(unirl_root, "rev-parse", "HEAD")
    dirty_lines = git(unirl_root, "status", "--porcelain").splitlines()
    if commit != args.expected_commit:
        raise RuntimeError(f"UniRL commit {commit} does not match expected {args.expected_commit}")
    if dirty_lines and not args.allow_dirty:
        raise RuntimeError("UniRL checkout is dirty; commit or stash changes, or pass --allow-dirty")

    sys.path.insert(0, str(unirl_root))
    global PIL, np, ray, torch
    global ARSamplingParams, DiffusionSamplingParams, Images, Part, Sample, Texts, TextSegment, TensorRef
    import PIL
    import numpy as np
    import ray
    import torch
    from unirl.distributed.tensor.ref import TensorRef
    from unirl.types.primitives import Images, Texts
    from unirl.types.sample import Part, Sample
    from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
    from unirl.types.segments import TextSegment

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(args.threads)
    except RuntimeError:
        pass

    output_dir.mkdir(parents=True, exist_ok=True)
    checks = run_correctness_checks()
    raw, summary = run_benchmarks(args.warmup, args.repeats)

    manifest = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "unirl": {
            "root": str(unirl_root),
            "commit": commit,
            "expected_commit": args.expected_commit,
            "clean": not dirty_lines,
            "dirty_entries": dirty_lines,
            "remote_origin": git(unirl_root, "config", "--get", "remote.origin.url", allow_failure=True),
        },
        "software": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "torch": torch.__version__,
            "ray": ray.__version__,
            "numpy": np.__version__,
            "pillow": PIL.__version__,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_brand": sysctl("machdep.cpu.brand_string"),
            "logical_cpu_count": os.cpu_count(),
            "memory_bytes": int(sysctl("hw.memsize")) if sysctl("hw.memsize") else None,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        },
        "benchmark": {
            "device": "cpu",
            "threads": args.threads,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "warmup": args.warmup,
            "repeats": args.repeats,
            "timer": "time.perf_counter_ns",
            "percentile": "nearest-rank",
            "root_counts": list(ROOT_COUNTS),
            "ar_branch": AR_BRANCH,
            "observation_branch": 1,
            "diffusion_branch": DIFFUSION_BRANCH,
            "image_payload_shape": [1, 2, 2],
        },
        "scope": "synthetic single-process CPU trajectory-container regression evidence; not GPU or end-to-end performance",
    }

    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "correctness.json", {"all_passed": True, "checks": checks})
    write_json(output_dir / "raw_trials.json", raw)
    write_summary_csv(output_dir / "summary.csv", summary)
    write_tex_rows(output_dir / "summary_rows.tex", summary)
    write_tex_table(output_dir / "summary_table.tex", summary)
    write_report(output_dir / "report.md", manifest, checks, summary)
    status_output = f"PASS: {len(checks)} correctness checks\nWROTE: {output_dir}\n"
    (output_dir / "stdout.txt").write_text(status_output, encoding="utf-8")
    (output_dir / "stderr.txt").write_text("", encoding="utf-8")
    write_checksums(output_dir)
    sys.stdout.write(status_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
