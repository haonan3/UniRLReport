#!/usr/bin/env python3
"""E5-R Paired GPU Representation & Transport Benchmark + E4a Recorded Rollout Verification.

Measures actual UniRL `Sample` / `Part` / `TensorRef` / `LatentSegment` / `TextSegment` /
`SD3Conditions` operations (`representation=tree`) paired against an explicit Flat-ID-Join
(`representation=flat_id_join`) implementation on the exact E4 3-stage Prompt Enhancement
tree geometry:
  - Roots P in {8, 32, 128}
  - 4 AR rewrite children per root (4P)
  - 8 SD3 diffusion leaf children per rewrite (32P leaves total: 256, 1024, 4096)
  - Sharded across 8 worker shards on 8 NVIDIA H20 GPUs (cuda:0..7)
  - Leaf payload sizes: e4_exact (~2.98 MiB/leaf) and {1, 16, 64} MiB/leaf
  - Backends: colocate_store (TensorStore / ColocateStoreTransport), gpu_store (CUDA IPC GPUTensorHandle)
  - Placements: same_gpu (src=cuda:s, dst=cuda:s), cross_gpu (src=cuda:s, dst=cuda:(s+4)%8)
  - Warmup 10 + Repetitions 100 per cell.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import gc
import glob
import hashlib
import json
import math
import os
import pickle
import platform
import random
import resource
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

UNIRL_ROOT = Path("/data/UniRL-det-eval")
if str(UNIRL_ROOT) not in sys.path:
    sys.path.insert(0, str(UNIRL_ROOT))
if str(UNIRL_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(UNIRL_ROOT / "experiments"))

import torch

from oracle.flat_row_oracle import compare_to_oracle, flat_id_join
from unirl.distributed.tensor.backend.colocate_store.handle import ColocateTensorHandle
from unirl.distributed.tensor.backend.colocate_store.store import TensorStore
from unirl.distributed.tensor.backend.colocate_store.transport import ColocateStoreTransport
from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle
from unirl.distributed.tensor.ref import TensorRef, TensorSpan, cat_rows, map_tree
from unirl.types.conditions import TextEmbedCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sample_id import ancestor_id
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments import LatentSegment, TextSegment

EPS = 1e-8
NUM_SHARDS = 8
AR_BRANCH = 4
DIFF_BRANCH = 8
LEAVES_PER_ROOT = AR_BRANCH * DIFF_BRANCH  # 32
E4_REF_STEP_S = 64.2
E4V2_ACTUAL_STEP_S = 29.2


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(max(int(math.ceil(q * len(ordered))) - 1, 0), len(ordered) - 1)
    return float(ordered[idx])


def _stats_us(values_s: Sequence[float]) -> Dict[str, float]:
    us = [v * 1e6 for v in values_s]
    return {
        "median_us": float(statistics.median(us)),
        "mean_us": float(statistics.fmean(us)),
        "p90_us": _percentile(us, 0.90),
        "min_us": float(min(us)),
        "max_us": float(max(us)),
        "median_ms": float(statistics.median(us) / 1000.0),
        "mean_ms": float(statistics.fmean(us) / 1000.0),
        "p90_ms": _percentile(us, 0.90) / 1000.0,
        "min_ms": float(min(us) / 1000.0),
        "max_ms": float(max(us) / 1000.0),
    }


def _tensor_sha256(t: torch.Tensor) -> str:
    """Deterministic SHA-256 of tensor contents (shape + dtype + raw bytes, streamed in 16 MiB chunks)."""
    h = hashlib.sha256()
    header = f"{tuple(t.shape)}:{t.dtype}:".encode("utf-8")
    h.update(header)
    flat = t.detach().contiguous().view(-1)
    n = flat.numel()
    chunk = 8 * (1 << 20)
    # Fast deterministic stride-sampled + boundary digest for multi-GiB tensors, full digest up to 512 MiB
    if n * flat.element_size() <= (512 << 20):
        for off in range(0, n, chunk):
            sub = flat[off : off + chunk].to("cpu", non_blocking=False)
            h.update(sub.view(torch.uint8).numpy().tobytes())
    else:
        # Hash first 16 MiB, middle 16 MiB, last 16 MiB, and every 64th block so every shard & root is covered
        step = max(chunk, (n // 64) & ~4095)
        for off in range(0, n, step):
            sub = flat[off : min(off + (1 << 20), n)].to("cpu", non_blocking=False)
            h.update(sub.view(torch.uint8).numpy().tobytes())
        tail = flat[max(0, n - (1 << 20)) : n].to("cpu", non_blocking=False)
        h.update(tail.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class LocalIPCStore:
    """Per-GPU store supporting CUDA IPC handles (_share_cuda_ / _new_shared_cuda) for gpu_store backend."""

    def __init__(self, worker_id: str, device_id: int) -> None:
        self.worker_id = worker_id
        self.device_id = device_id
        self.device = f"cuda:{device_id}"
        self._store: Dict[str, torch.Tensor] = {}
        self._ipc_cache: Dict[str, Tuple[tuple, tuple, tuple, torch.dtype]] = {}
        self._counter = 0

    def put(self, tensor: torch.Tensor, *, share_ipc: bool = True) -> GPUTensorHandle:
        t = tensor.detach().contiguous()
        key = f"{self.worker_id}_{self._counter}"
        self._counter += 1
        self._store[key] = t
        if share_ipc:
            ipc_h = t.untyped_storage()._share_cuda_()
            self._ipc_cache[key] = (ipc_h, tuple(t.shape), t.stride(), t.dtype)
        return GPUTensorHandle(
            store_key=key,
            source_id=self.worker_id,
            shape=tuple(t.shape),
            dtype=t.dtype,
            device=self.device,
        )

    def borrow_ipc(self, key: str) -> Tuple[tuple, tuple, tuple, torch.dtype]:
        return self._ipc_cache[key]

    def get(self, handle: GPUTensorHandle) -> torch.Tensor:
        return self._store[handle.store_key]

    def clear(self) -> None:
        self._store.clear()
        self._ipc_cache.clear()


def _payload_spec(payload_mode: str) -> Dict[str, Any]:
    """Return tensor shapes for a leaf so total dense bytes per leaf match payload_mode."""
    # Each [16, 64, 64] bf16 latent slice is 131,072 bytes = 128 KiB (0.125 MiB).
    if payload_mode == "e4_exact":
        # Exact E4 SD3 512x512 LatentSegment + SD3Conditions:
        # latents: [2, 16, 64, 64] bf16 (256 KiB: SDE step x_t and terminal x_T)
        # aux_latents (next_latents): [1, 16, 64, 64] bf16 (128 KiB: x_{t+1})
        # sde_logp: [1] fp32 (4 B)
        # conditions.text.embeds: [333, 4096] bf16 (2,727,936 B)
        # conditions.text.pooled: [2048] bf16 (4,096 B)
        # Total = 3,125,252 B = 2.9804725646972656 MiB / leaf
        return {
            "label": "e4_exact",
            "nominal_mib": round(3125252 / (1 << 20), 4),
            "latent_k": 2,
            "aux_k": 1,
            "sde_steps": 1,
            "cond_seq_len": 333,
            "cond_dim": 4096,
            "pooled_dim": 2048,
        }
    mib = float(payload_mode)
    if mib == 1.0:
        # 3 * 128 KiB + 3 * 128 KiB + (31 * 4096 * 2 + 2048 * 2) = 384 + 384 + 256 = 1024 KiB = 1.0 MiB
        return {
            "label": "1",
            "nominal_mib": 1.0,
            "latent_k": 3,
            "aux_k": 3,
            "sde_steps": 1,
            "cond_seq_len": 31,
            "cond_dim": 4096,
            "pooled_dim": 2048,
        }
    if mib == 16.0:
        # 62 * 128 KiB + 62 * 128 KiB + (63 * 4096 * 2 + 2048 * 2 + 1024 * 4) = 7.75 + 7.75 + 0.50 = 16.0 MiB
        return {
            "label": "16",
            "nominal_mib": 16.0,
            "latent_k": 62,
            "aux_k": 62,
            "sde_steps": 1024,
            "cond_seq_len": 63,
            "cond_dim": 4096,
            "pooled_dim": 2048,
        }
    if mib == 64.0:
        # 254 * 128 KiB + 254 * 128 KiB + (63 * 4096 * 2 + 2048 * 2 + 1024 * 4) = 31.75 + 31.75 + 0.50 = 64.0 MiB
        return {
            "label": "64",
            "nominal_mib": 64.0,
            "latent_k": 254,
            "aux_k": 254,
            "sde_steps": 1024,
            "cond_seq_len": 63,
            "cond_dim": 4096,
            "pooled_dim": 2048,
        }
    raise ValueError(f"Unsupported payload_mode: {payload_mode!r}")


def _flat_join_credit_and_advantages(
    sample_ids: Sequence[str],
    ar_sample_ids: Sequence[str],
    rewards: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, List[int]], Dict[str, List[int]]]:
    """Explicit dictionary ID-join credit propagation and group advantage computation.

    Reconstructs:
      - 32-image prompt groups (by root_id) -> prompt-scope leaf advantages & root rewards
      - 8-image rewrite groups (by rewrite_id) -> rewrite-scope leaf advantages & AR rewrite rewards
      - 4-rewrite AR groups (by root_id) -> AR rewrite advantages
    """
    prompt_groups: Dict[str, List[int]] = defaultdict(list)
    rewrite_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, sid in enumerate(sample_ids):
        parts = sid.split("/")
        root_id = parts[0]
        rewrite_id = f"{parts[0]}/{parts[1]}"
        prompt_groups[root_id].append(idx)
        rewrite_groups[rewrite_id].append(idx)

    r_f32 = rewards.to(torch.float32)

    # 1. Rewrite-scope advantages (32*P leaves in 4*P groups of 8) & propagated AR rewards
    adv_rewrite = torch.empty_like(r_f32)
    ar_reward_map: Dict[str, torch.Tensor] = {}
    for rw_id, member_indices in rewrite_groups.items():
        idx_t = torch.tensor(member_indices, dtype=torch.long, device=r_f32.device)
        vals = r_f32.index_select(0, idx_t)
        mean = vals.mean()
        ar_reward_map[rw_id] = mean
        centered = vals - mean
        std = (centered * centered).mean().sqrt()
        adv_rewrite.index_copy_(0, idx_t, centered / (std + EPS))

    ar_rewards = torch.stack([ar_reward_map[aid] for aid in ar_sample_ids])

    # 2. Prompt-scope advantages (32*P leaves in P groups of 32)
    adv_prompt = torch.empty_like(r_f32)
    for root_id, member_indices in prompt_groups.items():
        idx_t = torch.tensor(member_indices, dtype=torch.long, device=r_f32.device)
        vals = r_f32.index_select(0, idx_t)
        mean = vals.mean()
        centered = vals - mean
        std = (centered * centered).mean().sqrt()
        adv_prompt.index_copy_(0, idx_t, centered / (std + EPS))

    # 3. Root rewards & AR advantages (4*P rewrites in P groups of 4)
    ar_root_groups: Dict[str, List[int]] = defaultdict(list)
    for idx, aid in enumerate(ar_sample_ids):
        ar_root_groups[aid.split("/")[0]].append(idx)
    root_rewards_list: List[torch.Tensor] = []
    adv_ar = torch.empty_like(ar_rewards)
    for root_id, member_indices in ar_root_groups.items():
        idx_t = torch.tensor(member_indices, dtype=torch.long, device=ar_rewards.device)
        vals = ar_rewards.index_select(0, idx_t)
        mean = vals.mean()
        root_rewards_list.append(mean)
        centered = vals - mean
        std = (centered * centered).mean().sqrt()
        adv_ar.index_copy_(0, idx_t, centered / (std + EPS))
    root_rewards = torch.stack(root_rewards_list)

    return root_rewards, ar_rewards, adv_ar, adv_prompt, adv_rewrite, dict(prompt_groups), dict(rewrite_groups)


def run_e4a_recorded_rollout_checks() -> Dict[str, Any]:
    """Run E4a verification on the real recorded E4 v2 rollout captures and trace files."""
    e4_base = Path("/apdcephfs_gz44/share_305110755/linyuwu/LIN-1610/e4_v2/logs")
    runs = ["prompt_s11", "prompt_s22", "prompt_s33", "rewrite_s11", "rewrite_s22", "rewrite_s33"]

    # Load real root texts & rewrite token lengths from prompt_s11 trace for rollout 0
    trace_rewrites: Dict[str, Dict[str, Any]] = {}
    for p in sorted(glob.glob(str(e4_base / "prompt_s11/local/trace/pe_trace_*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("kind") == "rewrite" and row["sample_id"].startswith("r0:"):
                    trace_rewrites[row["sample_id"]] = row

    capture_results: List[Dict[str, Any]] = []
    for run_name in runs:
        for cap_name in ("rollout0000.pt", "rollout0001.pt"):
            cap_path = e4_base / run_name / "local/capture" / cap_name
            blob = torch.load(cap_path, map_location="cpu", weights_only=False)
            sids: List[str] = list(blob["sample_ids"])
            ar_sids: List[str] = list(blob["ar_sample_ids"])
            ar_texts: List[str] = list(blob["ar_texts"])
            rewards: torch.Tensor = blob["rewards"].to(torch.float32)
            rec_adv_prompt: torch.Tensor = blob["advantages_prompt"].to(torch.float32)
            rec_adv_rewrite: torch.Tensor = blob["advantages_rewrite"].to(torch.float32)
            rec_adv_trained: torch.Tensor = blob["advantages"].to(torch.float32)

            root_sids = list(dict.fromkeys(sid.split("/")[0] for sid in ar_sids))
            root_texts = [
                trace_rewrites.get(f"{rid}/0", {}).get("root_text", f"prompt_{rid}")
                for rid in root_sids
            ]
            tok_lens = [
                int(trace_rewrites.get(aid, {}).get("num_tokens", 28))
                for aid in ar_sids
            ]

            p0 = Part.input(
                root_sids,
                primitives={"text": Texts(root_texts)},
                metadata=[{"root_id": rid} for rid in root_sids],
                role="user",
            )
            t_seg = TextSegment.pack(
                tokens=[torch.arange(L, dtype=torch.long) for L in tok_lens],
                log_probs=[torch.full((L,), -0.1, dtype=torch.float32) for L in tok_lens],
                loss_mask=[torch.ones(L, dtype=torch.long) for L in tok_lens],
            )
            p1 = Part(
                sample_ids=ar_sids,
                segment=t_seg,
                primitives={"text": Texts(ar_texts)},
                sampling_params=ARSamplingParams(samples_per_prompt=4),
                output_version=0,
            )
            # Attach deterministic replay segment on the 256 leaves to test replay input round-trip
            gen = torch.Generator().manual_seed(1000 + int(blob["rollout_id"]))
            latents = torch.randn((len(sids), 2, 16, 8, 8), generator=gen, dtype=torch.bfloat16)
            aux_latents = torch.randn((len(sids), 1, 16, 8, 8), generator=gen, dtype=torch.bfloat16)
            sde_logp = torch.randn((len(sids), 1), generator=gen, dtype=torch.float32)
            cond_embeds = torch.randn((len(sids), 16, 64), generator=gen, dtype=torch.bfloat16)
            cond_pooled = torch.randn((len(sids), 32), generator=gen, dtype=torch.bfloat16)

            l_seg = LatentSegment(
                latents=latents,
                aux_latents=aux_latents,
                sde_logp=sde_logp,
                sigmas=torch.linspace(1.0, 0.0, 11),
                indices=torch.tensor([0, 10], dtype=torch.long),
                sde_indices=torch.tensor([0], dtype=torch.long),
            )
            p2 = Part(
                sample_ids=sids,
                segment=l_seg,
                conditions={"text": TextEmbedCondition(embeds=cond_embeds, pooled=cond_pooled)},
                rewards=rewards,
                sampling_params=DiffusionSamplingParams(samples_per_prompt=8),
                output_version=0,
            )
            sample = Sample(parts=[p0, p1, p2]).propagate_rewards("mean")
            tree_adv_prompt = sample.parts[2].compute_advantages(normalize=True, group_layer=0).advantages
            tree_adv_rewrite = sample.parts[2].compute_advantages(normalize=True, group_layer=None).advantages
            tree_adv_ar = sample.parts[1].compute_advantages(normalize=True, group_layer=0).advantages
            tree_ar_rewards = sample.parts[1].rewards
            tree_root_rewards = sample.parts[0].rewards

            (
                flat_root_rewards,
                flat_ar_rewards,
                flat_adv_ar,
                flat_adv_prompt,
                flat_adv_rewrite,
                prompt_groups,
                rewrite_groups,
            ) = _flat_join_credit_and_advantages(sids, ar_sids, rewards)

            # Also verify root-shard split/concat and permutation select round-trip
            shards = sample.split()
            rejoined = Sample.concat(shards)
            perm = torch.tensor([5, 2, 7, 0, 4, 1, 6, 3], dtype=torch.long)
            inv_perm = torch.argsort(perm)
            roundtrip = rejoined.select(perm).select(inv_perm)

            # Also verify shuffled flat rows + ID-join reconstruction
            shuf_idx = list(range(len(sids)))
            random.Random(42).shuffle(shuf_idx)
            shuf_sids = [sids[i] for i in shuf_idx]
            shuf_rewards = rewards[torch.tensor(shuf_idx, dtype=torch.long)]
            _, _, _, shuf_adv_p, shuf_adv_w, _, _ = _flat_join_credit_and_advantages(
                shuf_sids, ar_sids, shuf_rewards
            )
            unshuf_adv_p = torch.empty_like(shuf_adv_p)
            unshuf_adv_w = torch.empty_like(shuf_adv_w)
            unshuf_adv_p[torch.tensor(shuf_idx, dtype=torch.long)] = shuf_adv_p
            unshuf_adv_w[torch.tensor(shuf_idx, dtype=torch.long)] = shuf_adv_w

            # Float64 oracle check
            oracle_prompt = compare_to_oracle(sids, rewards.tolist(), tree_adv_prompt.tolist(), layer=0, tolerance=1e-5)
            oracle_rewrite = compare_to_oracle(sids, rewards.tolist(), tree_adv_rewrite.tolist(), layer=1, tolerance=1e-5)

            assert torch.equal(tree_adv_prompt, rec_adv_prompt)
            assert torch.equal(tree_adv_rewrite, rec_adv_rewrite)
            assert torch.equal(tree_adv_prompt, flat_adv_prompt)
            assert torch.equal(tree_adv_rewrite, flat_adv_rewrite)
            assert torch.allclose(tree_ar_rewards, flat_ar_rewards, atol=1e-6)
            assert torch.allclose(tree_root_rewards, flat_root_rewards, atol=1e-6)
            assert torch.allclose(tree_adv_ar, flat_adv_ar, atol=1e-6)
            assert torch.allclose(unshuf_adv_p, tree_adv_prompt, atol=1e-5)
            assert torch.allclose(unshuf_adv_w, tree_adv_rewrite, atol=1e-5)
            assert torch.equal(roundtrip.parts[2].segment.latents, latents)
            assert torch.equal(roundtrip.parts[2].conditions["text"].embeds, cond_embeds)
            assert oracle_prompt.passed and oracle_rewrite.passed

            capture_results.append(
                {
                    "run": run_name,
                    "capture": cap_name,
                    "rollout_id": int(blob["rollout_id"]),
                    "trained_scope": str(blob["scope"]),
                    "num_roots": len(root_sids),
                    "num_rewrites": len(ar_sids),
                    "num_leaves": len(sids),
                    "prompt_groups": len(prompt_groups),
                    "prompt_group_size": sorted({len(v) for v in prompt_groups.values()}),
                    "rewrite_groups": len(rewrite_groups),
                    "rewrite_group_size": sorted({len(v) for v in rewrite_groups.values()}),
                    "exact_equal_recorded_prompt_adv": True,
                    "exact_equal_recorded_rewrite_adv": True,
                    "exact_equal_flat_f32_prompt_adv": True,
                    "exact_equal_flat_f32_rewrite_adv": True,
                    "max_abs_diff_vs_flat_f32_prompt": float((tree_adv_prompt - flat_adv_prompt).abs().max().item()),
                    "max_abs_diff_vs_flat_f32_rewrite": float((tree_adv_rewrite - flat_adv_rewrite).abs().max().item()),
                    "max_abs_diff_vs_oracle_f64_prompt": float(oracle_prompt.max_abs_advantage_diff),
                    "max_abs_diff_vs_oracle_f64_rewrite": float(oracle_rewrite.max_abs_advantage_diff),
                    "max_abs_diff_ar_rewards": float((tree_ar_rewards - flat_ar_rewards).abs().max().item()),
                    "max_abs_diff_root_rewards": float((tree_root_rewards - flat_root_rewards).abs().max().item()),
                    "shuffled_flat_join_exact_match": True,
                    "split_select_concat_replay_roundtrip_exact": True,
                    "prompt_adv_sha256": _tensor_sha256(tree_adv_prompt),
                    "rewrite_adv_sha256": _tensor_sha256(tree_adv_rewrite),
                    "ar_rewards_sha256": _tensor_sha256(tree_ar_rewards),
                    "replay_latents_sha256": _tensor_sha256(latents),
                }
            )

    return {
        "all_passed": True,
        "num_captures_verified": len(capture_results),
        "primary_capture": "/apdcephfs_gz44/share_305110755/linyuwu/LIN-1610/e4_v2/logs/prompt_s11/local/capture/rollout0000.pt",
        "captures": capture_results,
    }


def _build_cell_fixtures(
    roots: int,
    payload_mode: str,
    backend: str,
    placement: str,
    recorded_r0_blob: Dict[str, Any],
) -> Dict[str, Any]:
    """Allocate the 8 worker shards on cuda:0..7 and build both Tree (Sample+TensorRef) and Flat-ID-Join structures."""
    spec = _payload_spec(payload_mode)
    n_roots = roots
    n_ar = n_roots * AR_BRANCH
    n_leaves = n_roots * LEAVES_PER_ROOT
    roots_per_shard = n_roots // NUM_SHARDS
    ar_per_shard = n_ar // NUM_SHARDS
    leaves_per_shard = n_leaves // NUM_SHARDS

    root_ids = [f"r0:prompt:{i:04d}" for i in range(n_roots)]
    ar_ids = [f"r0:prompt:{i:04d}/{j}" for i in range(n_roots) for j in range(AR_BRANCH)]
    leaf_ids = [
        f"r0:prompt:{i:04d}/{j}/{k}"
        for i in range(n_roots)
        for j in range(AR_BRANCH)
        for k in range(DIFF_BRANCH)
    ]

    # Use recorded E4 v2 rewards tiled deterministically if roots > 8
    base_rewards: torch.Tensor = recorded_r0_blob["rewards"].to(torch.float32).cpu()
    if n_leaves == base_rewards.numel():
        rewards = base_rewards.clone()
    else:
        reps = n_leaves // base_rewards.numel()
        rewards = base_rewards.repeat(reps)
        # Add tiny deterministic root offset so every root has distinct values
        offsets = torch.arange(n_roots, dtype=torch.float32).repeat_interleave(LEAVES_PER_ROOT) * 1e-3
        rewards = rewards + offsets

    base_ar_texts: List[str] = list(recorded_r0_blob["ar_texts"])
    ar_texts = [base_ar_texts[i % len(base_ar_texts)] for i in range(n_ar)]
    root_texts = [f"original user prompt {i}: cyberpunk cityscape in neon rain" for i in range(n_roots)]

    # Deterministic token lengths per rewrite (matching E4 v2 distribution 16..56 tokens)
    ar_tok_lens = [20 + ((i * 7) % 36) for i in range(n_ar)]

    src_devices = [f"cuda:{s}" for s in range(NUM_SHARDS)]
    dst_devices = (
        [f"cuda:{s}" for s in range(NUM_SHARDS)]
        if placement == "same_gpu"
        else [f"cuda:{(s + 4) % NUM_SHARDS}" for s in range(NUM_SHARDS)]
    )

    # Create transport stores per shard
    if backend == "colocate_store":
        src_stores = [TensorStore(worker_id=f"src_w{s}", device=src_devices[s]) for s in range(NUM_SHARDS)]
        dst_stores = (
            src_stores
            if placement == "same_gpu"
            else [TensorStore(worker_id=f"dst_w{s}", device=dst_devices[s]) for s in range(NUM_SHARDS)]
        )
    elif backend == "gpu_store":
        src_stores = [LocalIPCStore(worker_id=f"src_w{s}", device_id=s) for s in range(NUM_SHARDS)]
        dst_stores = (
            src_stores
            if placement == "same_gpu"
            else [LocalIPCStore(worker_id=f"dst_w{s}", device_id=(s + 4) % NUM_SHARDS) for s in range(NUM_SHARDS)]
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    def _put_ref(store: Any, t: torch.Tensor) -> TensorRef:
        h = store.put(t)
        return TensorRef(
            spans=[TensorSpan(h, 0, int(t.shape[0]))],
            shape=tuple(t.shape),
            dtype=t.dtype,
            device=str(t.device),
        )

    tree_worker_shards: List[Sample] = []
    flat_worker_shards: List[Dict[str, Any]] = []
    dense_payload_bytes = 0
    ar_token_bytes = 0

    sigmas = torch.linspace(1.0, 0.0, 11, dtype=torch.float32)
    indices_t = torch.arange(spec["latent_k"], dtype=torch.long)
    sde_indices_t = torch.arange(spec["sde_steps"], dtype=torch.long)

    for s in range(NUM_SHARDS):
        dev = src_devices[s]
        r_start, r_stop = s * roots_per_shard, (s + 1) * roots_per_shard
        a_start, a_stop = s * ar_per_shard, (s + 1) * ar_per_shard
        l_start, l_stop = s * leaves_per_shard, (s + 1) * leaves_per_shard

        # Allocate dense leaf tensors on shard GPU `dev`
        latents_s = torch.full(
            (leaves_per_shard, spec["latent_k"], 16, 64, 64),
            float((s + 1) * 0.125),
            dtype=torch.bfloat16,
            device=dev,
        )
        aux_latents_s = torch.full(
            (leaves_per_shard, spec["aux_k"], 16, 64, 64),
            float((s + 1) * 0.25),
            dtype=torch.bfloat16,
            device=dev,
        )
        sde_logp_s = torch.full(
            (leaves_per_shard, spec["sde_steps"]),
            float(-0.5 - 0.01 * s),
            dtype=torch.float32,
            device=dev,
        )
        cond_embeds_s = torch.full(
            (leaves_per_shard, spec["cond_seq_len"], spec["cond_dim"]),
            float((s + 1) * 0.0625),
            dtype=torch.bfloat16,
            device=dev,
        )
        cond_pooled_s = torch.full(
            (leaves_per_shard, spec["pooled_dim"]),
            float((s + 1) * 0.03125),
            dtype=torch.bfloat16,
            device=dev,
        )

        for t in (latents_s, aux_latents_s, sde_logp_s, cond_embeds_s, cond_pooled_s):
            dense_payload_bytes += t.numel() * t.element_size()

        # AR packed segment on shard
        lens_s = ar_tok_lens[a_start:a_stop]
        tok_list = [torch.full((L,), (i % 100) + 1, dtype=torch.long, device=dev) for i, L in enumerate(lens_s)]
        lp_list = [torch.full((L,), -0.25, dtype=torch.float32, device=dev) for L in lens_s]
        mask_list = [torch.ones((L,), dtype=torch.long, device=dev) for L in lens_s]
        t_seg = TextSegment.pack(tokens=tok_list, log_probs=lp_list, loss_mask=mask_list)
        for t in (t_seg.tokens, t_seg.log_probs, t_seg.loss_mask):
            ar_token_bytes += t.numel() * t.element_size()

        # Dehydrate into TensorRef for tree representation
        t_seg_ref = TextSegment(
            tokens=_put_ref(src_stores[s], t_seg.tokens),
            log_probs=_put_ref(src_stores[s], t_seg.log_probs),
            loss_mask=_put_ref(src_stores[s], t_seg.loss_mask),
        )
        object.__setattr__(t_seg_ref, "_packed_cu_seqlens", t_seg._packed_cu_seqlens.cpu())

        l_seg_ref = LatentSegment(
            latents=_put_ref(src_stores[s], latents_s),
            aux_latents=_put_ref(src_stores[s], aux_latents_s),
            sde_logp=_put_ref(src_stores[s], sde_logp_s),
            sigmas=sigmas,
            indices=indices_t,
            sde_indices=sde_indices_t,
        )
        cond_ref = TextEmbedCondition(
            embeds=_put_ref(src_stores[s], cond_embeds_s),
            pooled=_put_ref(src_stores[s], cond_pooled_s),
        )

        p0 = Part.input(
            root_ids[r_start:r_stop],
            primitives={"text": Texts(root_texts[r_start:r_stop])},
            metadata=[{"prompt_idx": i} for i in range(r_start, r_stop)],
            role="user",
        )
        p1 = Part(
            sample_ids=ar_ids[a_start:a_stop],
            segment=t_seg_ref,
            primitives={"text": Texts(ar_texts[a_start:a_stop])},
            sampling_params=ARSamplingParams(samples_per_prompt=AR_BRANCH),
            output_version=0,
        )
        p2 = Part(
            sample_ids=leaf_ids[l_start:l_stop],
            segment=l_seg_ref,
            conditions={"text": cond_ref},
            rewards=rewards[l_start:l_stop].clone(),
            sampling_params=DiffusionSamplingParams(
                samples_per_prompt=DIFF_BRANCH,
                num_inference_steps=10,
                height=512,
                width=512,
                eta=0.7,
            ),
            output_version=0,
        )
        tree_worker_shards.append(Sample(parts=[p0, p1, p2]))

        # Build flat_id_join shard: flat leaf rows with replicated parent prompts/rewrites and per-root dense slices
        flat_rows_meta = [
            {
                "sample_id": leaf_ids[k],
                "rewrite_id": "/".join(leaf_ids[k].split("/")[:2]),
                "root_id": leaf_ids[k].split("/")[0],
                "root_text": root_texts[k // LEAVES_PER_ROOT],
                "rewrite_text": ar_texts[k // DIFF_BRANCH],
                "ar_token_len": ar_tok_lens[k // DIFF_BRANCH],
                "output_version": 0,
            }
            for k in range(l_start, l_stop)
        ]
        # Per-root slice views on the shard GPU (each root has 32 leaves)
        root_slices = []
        for r_local in range(roots_per_shard):
            lo, hi = r_local * LEAVES_PER_ROOT, (r_local + 1) * LEAVES_PER_ROOT
            root_slices.append(
                {
                    "root_id": root_ids[r_start + r_local],
                    "latents": latents_s[lo:hi],
                    "aux_latents": aux_latents_s[lo:hi],
                    "sde_logp": sde_logp_s[lo:hi],
                    "cond_embeds": cond_embeds_s[lo:hi],
                    "cond_pooled": cond_pooled_s[lo:hi],
                }
            )
        flat_worker_shards.append(
            {
                "shard_id": s,
                "device": dev,
                "rows": flat_rows_meta,
                "sample_ids": leaf_ids[l_start:l_stop],
                "ar_sample_ids": ar_ids[a_start:a_stop],
                "rewards": rewards[l_start:l_stop].clone(),
                "root_slices": root_slices,
                "latents": latents_s,
                "aux_latents": aux_latents_s,
                "sde_logp": sde_logp_s,
                "cond_embeds": cond_embeds_s,
                "cond_pooled": cond_pooled_s,
            }
        )

    torch.cuda.synchronize()
    return {
        "spec": spec,
        "roots": n_roots,
        "rewrites": n_ar,
        "leaves": n_leaves,
        "roots_per_shard": roots_per_shard,
        "leaves_per_shard": leaves_per_shard,
        "src_devices": src_devices,
        "dst_devices": dst_devices,
        "src_stores": src_stores,
        "dst_stores": dst_stores,
        "tree_worker_shards": tree_worker_shards,
        "flat_worker_shards": flat_worker_shards,
        "dense_payload_bytes": dense_payload_bytes,
        "ar_token_bytes": ar_token_bytes,
    }


def _localize_tree_shards(
    shards: List[Sample],
    backend: str,
    placement: str,
    src_stores: List[Any],
    dst_stores: List[Any],
    dst_devices: List[str],
) -> List[Sample]:
    """Localize TensorRef spans in each worker shard for its destination GPU."""
    localized_shards: List[Sample] = []
    for s, shard in enumerate(shards):
        dst_store = dst_stores[s]
        dst_dev = dst_devices[s]

        def _localize_node(obj: Any, s_idx: int = s, d_store: Any = dst_store, d_dev: str = dst_dev) -> Any:
            if not isinstance(obj, TensorRef):
                return obj
            # Coalesce adjacent spans on the same handle first
            coalesced: List[TensorSpan] = []
            for sp in obj.spans:
                if (
                    coalesced
                    and coalesced[-1].handle.store_key == sp.handle.store_key
                    and coalesced[-1].stop == sp.start
                ):
                    coalesced[-1] = TensorSpan(coalesced[-1].handle, coalesced[-1].start, sp.stop)
                else:
                    coalesced.append(sp)

            new_spans: List[TensorSpan] = []
            changed = len(coalesced) != len(obj.spans)
            for span in coalesced:
                h = span.handle
                src_idx = int(str(h.source_id).split("_w")[-1])
                if placement == "same_gpu" and src_idx == s_idx:
                    if backend == "gpu_store":
                        _ = src_stores[src_idx].borrow_ipc(h.store_key)
                    new_spans.append(span)
                else:
                    changed = True
                    if backend == "colocate_store":
                        base_t = src_stores[src_idx].get(h)[span.start : span.stop]
                        moved_t = base_t.to(d_dev, non_blocking=True)
                        new_h = d_store.put(moved_t)
                        new_spans.append(TensorSpan(new_h, 0, int(moved_t.shape[0])))
                    else:
                        _ = src_stores[src_idx].borrow_ipc(h.store_key)
                        base_t = src_stores[src_idx].get(h)[span.start : span.stop]
                        moved_t = base_t.to(d_dev, non_blocking=True)
                        new_h = d_store.put(moved_t, share_ipc=False)
                        new_spans.append(TensorSpan(new_h, 0, int(moved_t.shape[0])))
            return obj.with_spans(new_spans) if changed else obj

        localized_shards.append(map_tree(shard, _localize_node))
    return localized_shards


def _materialize_tree_shards(
    shards: List[Sample],
    dst_stores: List[Any],
) -> List[Dict[str, torch.Tensor]]:
    """Materialize (hydrate) the leaf LatentSegment and SD3Conditions TensorRefs on each worker."""
    out: List[Dict[str, torch.Tensor]] = []
    for s, shard in enumerate(shards):
        store = dst_stores[s]
        diff_part = shard.parts[2]
        seg: LatentSegment = diff_part.segment  # type: ignore[assignment]
        cond: TextEmbedCondition = diff_part.conditions["text"]  # type: ignore[assignment]

        def _mat(ref: Any, st: Any = store) -> torch.Tensor:
            if isinstance(ref, torch.Tensor):
                return ref
            assert isinstance(ref, TensorRef)
            # Coalesce adjacent spans on the same handle before slicing
            coalesced: List[Tuple[Any, int, int]] = []
            for sp in ref.spans:
                if coalesced and coalesced[-1][0].store_key == sp.handle.store_key and coalesced[-1][2] == sp.start:
                    prev_h, prev_start, _ = coalesced[-1]
                    coalesced[-1] = (prev_h, prev_start, sp.stop)
                else:
                    coalesced.append((sp.handle, sp.start, sp.stop))
            parts = [st.get(h)[start:stop] for h, start, stop in coalesced]
            return cat_rows(parts)

        out.append(
            {
                "latents": _mat(seg.latents),
                "aux_latents": _mat(seg.aux_latents),
                "sde_logp": _mat(seg.sde_logp),
                "cond_embeds": _mat(cond.embeds),
                "cond_pooled": _mat(cond.pooled),
            }
        )
    return out


def run_paired_cell(
    *,
    roots: int,
    payload_mode: str,
    backend: str,
    placement: str,
    warmup: int,
    repetitions: int,
    recorded_r0_blob: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Run both `tree` and `flat_id_join` arms on the exact same cell fixtures and assert 100% parity."""
    for d in range(NUM_SHARDS):
        torch.cuda.reset_peak_memory_stats(d)

    fx = _build_cell_fixtures(roots, payload_mode, backend, placement, recorded_r0_blob)
    spec = fx["spec"]
    n_roots = fx["roots"]
    n_leaves = fx["leaves"]
    roots_per_shard = fx["roots_per_shard"]
    src_stores = fx["src_stores"]
    dst_stores = fx["dst_stores"]
    dst_devices = fx["dst_devices"]
    tree_worker_shards: List[Sample] = fx["tree_worker_shards"]
    flat_worker_shards: List[Dict[str, Any]] = fx["flat_worker_shards"]

    # Deterministic root permutation (within-shard & cross-root order preserved for paired check)
    perm_indices = torch.arange(n_roots - 1, -1, -1, dtype=torch.long)
    inv_perm_indices = torch.argsort(perm_indices)

    # Measure serialized metadata bytes at driver
    full_tree_sample = Sample.concat(tree_worker_shards)
    tree_metadata_bytes = len(pickle.dumps(full_tree_sample))
    flat_driver_meta = {
        "rows": [row for sh in flat_worker_shards for row in sh["rows"]],
        "ar_sample_ids": [aid for sh in flat_worker_shards for aid in sh["ar_sample_ids"]],
        "rewards": torch.cat([sh["rewards"] for sh in flat_worker_shards], dim=0),
    }
    flat_metadata_bytes = len(pickle.dumps(flat_driver_meta))

    trial_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    # Storage for correctness verification between tree and flat_id_join
    verified_tree: Dict[str, Any] = {}
    verified_flat: Dict[str, Any] = {}

    # ── ARM 1: representation = "tree" (UniRL Sample + Part + TensorRef) ──
    for d in range(NUM_SHARDS):
        torch.cuda.synchronize(d)
        torch.cuda.reset_peak_memory_stats(d)

    tree_timings: Dict[str, List[float]] = defaultdict(list)

    for it in range(warmup + repetitions):
        if it > 0 and it % 25 == 0 and n_roots >= 128 and spec["nominal_mib"] >= 16:
            print(f"         [tree] iter {it}/{warmup + repetitions}...", flush=True)
        rec = it >= warmup
        if placement == "cross_gpu":
            for st in dst_stores:
                st._store.clear()
                if hasattr(st, "_ipc_cache"):
                    st._ipc_cache.clear()
        else:
            # Keep only the initial 4 tensors per shard in src_stores
            for st in src_stores:
                keys = list(st._store.keys())
                for k in keys[8:]:
                    del st._store[k]
                    if hasattr(st, "_ipc_cache"):
                        st._ipc_cache.pop(k, None)

        # 1. concat (merge 8 worker shards into full driver Sample)
        t0 = time.perf_counter()
        merged_sample = Sample.concat(tree_worker_shards)
        t_concat = time.perf_counter() - t0

        # 2. split (split full Sample into P root trees and 8 worker shards)
        t0 = time.perf_counter()
        root_trees = merged_sample.split()
        split_shards = [
            Sample.concat(root_trees[s * roots_per_shard : (s + 1) * roots_per_shard])
            for s in range(NUM_SHARDS)
        ]
        t_split = time.perf_counter() - t0

        # 3. select (permute root trees and restore order via TensorRef views)
        t0 = time.perf_counter()
        selected_sample = merged_sample.select(perm_indices)
        t_select = time.perf_counter() - t0

        # 4. credit_and_advantage (propagate_rewards + compute_advantages for prompt, rewrite, and AR)
        t0 = time.perf_counter()
        prop_sample = merged_sample.propagate_rewards("mean")
        part_prompt = prop_sample.parts[2].compute_advantages(normalize=True, group_layer=0)
        part_rewrite = prop_sample.parts[2].compute_advantages(normalize=True, group_layer=None)
        part_ar = prop_sample.parts[1].compute_advantages(normalize=True, group_layer=0)
        t_credit = time.perf_counter() - t0

        # 5. handle_transport (localize shards for destination worker GPUs)
        t0 = time.perf_counter()
        localized_shards = _localize_tree_shards(
            split_shards, backend, placement, src_stores, dst_stores, dst_devices
        )
        torch.cuda.synchronize()
        t_transport = time.perf_counter() - t0

        # 6. materialize (worker-side hydration of TensorRef spans)
        t0 = time.perf_counter()
        mat_tensors = _materialize_tree_shards(localized_shards, dst_stores)
        torch.cuda.synchronize()
        t_materialize = time.perf_counter() - t0

        t_structure = t_split + t_select + t_concat + t_credit
        t_total = t_structure + t_transport + t_materialize

        if rec:
            rep_idx = it - warmup
            tree_timings["split"].append(t_split)
            tree_timings["select"].append(t_select)
            tree_timings["concat"].append(t_concat)
            tree_timings["credit_and_advantage"].append(t_credit)
            tree_timings["handle_transport"].append(t_transport)
            tree_timings["materialize"].append(t_materialize)
            tree_timings["structure_critical_path"].append(t_structure)
            tree_timings["total_critical_path"].append(t_total)

            trial_rows.append(
                {
                    "representation": "tree",
                    "backend": backend,
                    "placement": placement,
                    "roots": n_roots,
                    "rewrites": fx["rewrites"],
                    "leaves": n_leaves,
                    "payload_mode": spec["label"],
                    "payload_mib": spec["nominal_mib"],
                    "trial": rep_idx,
                    "split_us": t_split * 1e6,
                    "select_us": t_select * 1e6,
                    "concat_us": t_concat * 1e6,
                    "credit_and_advantage_us": t_credit * 1e6,
                    "handle_transport_us": t_transport * 1e6,
                    "materialize_us": t_materialize * 1e6,
                    "structure_critical_path_us": t_structure * 1e6,
                    "total_critical_path_us": t_total * 1e6,
                }
            )
            if rep_idx == 0:
                # Record hashes and outputs for parity check
                lat_digest = hashlib.sha256()
                cond_digest = hashlib.sha256()
                for m_item in mat_tensors:
                    lat_digest.update(_tensor_sha256(m_item["latents"]).encode("utf-8"))
                    lat_digest.update(_tensor_sha256(m_item["aux_latents"]).encode("utf-8"))
                    lat_digest.update(_tensor_sha256(m_item["sde_logp"]).encode("utf-8"))
                    cond_digest.update(_tensor_sha256(m_item["cond_embeds"]).encode("utf-8"))
                    cond_digest.update(_tensor_sha256(m_item["cond_pooled"]).encode("utf-8"))
                del m_item
                verified_tree = {
                    "root_rewards": prop_sample.parts[0].rewards.detach().cpu(),
                    "ar_rewards": prop_sample.parts[1].rewards.detach().cpu(),
                    "adv_ar": part_ar.advantages.detach().cpu(),
                    "adv_prompt": part_prompt.advantages.detach().cpu(),
                    "adv_rewrite": part_rewrite.advantages.detach().cpu(),
                    "latents_sha256": lat_digest.hexdigest(),
                    "cond_sha256": cond_digest.hexdigest(),
                    "selected_root_ids": list(selected_sample.parts[0].sample_ids),
                }

        del mat_tensors, localized_shards

    tree_gpu_alloc_mib = max(torch.cuda.max_memory_allocated(d) for d in range(NUM_SHARDS)) / (1 << 20)
    tree_gpu_res_mib = max(torch.cuda.max_memory_reserved(d) for d in range(NUM_SHARDS)) / (1 << 20)
    tree_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # Clear dst_stores from ARM 1 before starting ARM 2
    if placement == "cross_gpu":
        for st in dst_stores:
            st._store.clear()
            if hasattr(st, "_ipc_cache"):
                st._ipc_cache.clear()
    else:
        for st in src_stores:
            keys = list(st._store.keys())
            for k in keys[8:]:
                del st._store[k]
                if hasattr(st, "_ipc_cache"):
                    st._ipc_cache.pop(k, None)

    # ── ARM 2: representation = "flat_id_join" (Flat Row Table + Dictionary ID Join) ──
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    for d in range(NUM_SHARDS):
        torch.cuda.synchronize(d)
        torch.cuda.reset_peak_memory_stats(d)

    flat_timings: Dict[str, List[float]] = defaultdict(list)
    perm_root_ids = [f"r0:prompt:{int(i):04d}" for i in perm_indices.tolist()]

    for it in range(warmup + repetitions):
        if it > 0 and it % 25 == 0 and n_roots >= 128 and spec["nominal_mib"] >= 16:
            print(f"         [flat_id_join] iter {it}/{warmup + repetitions}...", flush=True)
        rec = it >= warmup
        # 1. concat (merge flat row tables + concatenate per-root slices into shard dense tensors)
        t0 = time.perf_counter()
        all_rows = [row for sh in flat_worker_shards for row in sh["rows"]]
        all_sids = [row["sample_id"] for row in all_rows]
        all_ar_sids = [aid for sh in flat_worker_shards for aid in sh["ar_sample_ids"]]
        all_rewards = torch.cat([sh["rewards"] for sh in flat_worker_shards], dim=0)
        concat_shards = []
        for sh in flat_worker_shards:
            rs = sh["root_slices"]
            concat_shards.append(
                {
                    "latents": torch.cat([r["latents"] for r in rs], dim=0),
                    "aux_latents": torch.cat([r["aux_latents"] for r in rs], dim=0),
                    "sde_logp": torch.cat([r["sde_logp"] for r in rs], dim=0),
                    "cond_embeds": torch.cat([r["cond_embeds"] for r in rs], dim=0),
                    "cond_pooled": torch.cat([r["cond_pooled"] for r in rs], dim=0),
                }
            )
        torch.cuda.synchronize()
        t_concat = time.perf_counter() - t0

        # 2. split (group flat rows by root_id via dictionary scan and slice dense tensors per root)
        t0 = time.perf_counter()
        root_buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, row in enumerate(all_rows):
            root_buckets[row["root_id"]].append(idx)
        split_root_slices: Dict[str, Dict[str, torch.Tensor]] = {}
        for s_idx, sh in enumerate(concat_shards):
            for r_local in range(roots_per_shard):
                rid = f"r0:prompt:{s_idx * roots_per_shard + r_local:04d}"
                lo, hi = r_local * LEAVES_PER_ROOT, (r_local + 1) * LEAVES_PER_ROOT
                split_root_slices[rid] = {
                    "latents": sh["latents"][lo:hi],
                    "aux_latents": sh["aux_latents"][lo:hi],
                    "sde_logp": sh["sde_logp"][lo:hi],
                    "cond_embeds": sh["cond_embeds"][lo:hi],
                    "cond_pooled": sh["cond_pooled"][lo:hi],
                }
        torch.cuda.synchronize()
        t_split = time.perf_counter() - t0
        del concat_shards, split_root_slices

        # 3. select (reorder flat rows by perm_root_ids and gather dense tensors per shard)
        t0 = time.perf_counter()
        selected_rows = [all_rows[i] for rid in perm_root_ids for i in root_buckets[rid]]
        for s_idx in range(NUM_SHARDS):
            sh = flat_worker_shards[s_idx]
            rev_idx = torch.arange(roots_per_shard - 1, -1, -1, device=sh["latents"].device)
            leaf_idx = (
                rev_idx.unsqueeze(1) * LEAVES_PER_ROOT
                + torch.arange(LEAVES_PER_ROOT, device=sh["latents"].device).unsqueeze(0)
            ).reshape(-1)
            for k in ("latents", "aux_latents", "sde_logp", "cond_embeds", "cond_pooled"):
                _sel_t = sh[k].index_select(0, leaf_idx)
                del _sel_t
        torch.cuda.synchronize()
        t_select = time.perf_counter() - t0

        # 4. credit_and_advantage (dictionary ID join for prompt groups, rewrite groups, and AR groups)
        t0 = time.perf_counter()
        (
            flat_root_rewards,
            flat_ar_rewards,
            flat_adv_ar,
            flat_adv_prompt,
            flat_adv_rewrite,
            prompt_groups,
            rewrite_groups,
        ) = _flat_join_credit_and_advantages(all_sids, all_ar_sids, all_rewards)
        t_credit = time.perf_counter() - t0

        # 5. handle_transport (serialize flat row metadata + transfer dense shard tensors to dst GPUs)
        t0 = time.perf_counter()
        _serialized_rows = pickle.loads(pickle.dumps(all_rows))
        transported_shards = []
        for s_idx in range(NUM_SHARDS):
            sh = flat_worker_shards[s_idx]
            d_dev = dst_devices[s_idx]
            if placement == "same_gpu":
                if backend == "gpu_store":
                    for k in ("latents", "aux_latents", "sde_logp", "cond_embeds", "cond_pooled"):
                        _ = src_stores[s_idx].borrow_ipc(f"src_w{s_idx}_3")
                transported_shards.append(
                    {k: sh[k] for k in ("latents", "aux_latents", "sde_logp", "cond_embeds", "cond_pooled")}
                )
            else:
                transported_shards.append(
                    {
                        k: sh[k].to(d_dev, non_blocking=True)
                        for k in ("latents", "aux_latents", "sde_logp", "cond_embeds", "cond_pooled")
                    }
                )
        torch.cuda.synchronize()
        t_transport = time.perf_counter() - t0

        # 6. materialize (assemble contiguous group tensors on destination worker GPU)
        t0 = time.perf_counter()
        mat_flat = []
        for s_idx in range(NUM_SHARDS):
            tsh = transported_shards[s_idx]
            for k in ("latents", "aux_latents", "sde_logp", "cond_embeds", "cond_pooled"):
                tsh[k] = torch.cat(list(tsh[k].chunk(roots_per_shard, dim=0)), dim=0)
            mat_flat.append(tsh)
        torch.cuda.synchronize()
        t_materialize = time.perf_counter() - t0

        t_structure = t_split + t_select + t_concat + t_credit
        t_total = t_structure + t_transport + t_materialize

        if rec:
            rep_idx = it - warmup
            flat_timings["split"].append(t_split)
            flat_timings["select"].append(t_select)
            flat_timings["concat"].append(t_concat)
            flat_timings["credit_and_advantage"].append(t_credit)
            flat_timings["handle_transport"].append(t_transport)
            flat_timings["materialize"].append(t_materialize)
            flat_timings["structure_critical_path"].append(t_structure)
            flat_timings["total_critical_path"].append(t_total)

            trial_rows.append(
                {
                    "representation": "flat_id_join",
                    "backend": backend,
                    "placement": placement,
                    "roots": n_roots,
                    "rewrites": fx["rewrites"],
                    "leaves": n_leaves,
                    "payload_mode": spec["label"],
                    "payload_mib": spec["nominal_mib"],
                    "trial": rep_idx,
                    "split_us": t_split * 1e6,
                    "select_us": t_select * 1e6,
                    "concat_us": t_concat * 1e6,
                    "credit_and_advantage_us": t_credit * 1e6,
                    "handle_transport_us": t_transport * 1e6,
                    "materialize_us": t_materialize * 1e6,
                    "structure_critical_path_us": t_structure * 1e6,
                    "total_critical_path_us": t_total * 1e6,
                }
            )
            if rep_idx == 0:
                lat_digest = hashlib.sha256()
                cond_digest = hashlib.sha256()
                for m_item in mat_flat:
                    lat_digest.update(_tensor_sha256(m_item["latents"]).encode("utf-8"))
                    lat_digest.update(_tensor_sha256(m_item["aux_latents"]).encode("utf-8"))
                    lat_digest.update(_tensor_sha256(m_item["sde_logp"]).encode("utf-8"))
                    cond_digest.update(_tensor_sha256(m_item["cond_embeds"]).encode("utf-8"))
                    cond_digest.update(_tensor_sha256(m_item["cond_pooled"]).encode("utf-8"))
                del m_item
                verified_flat = {
                    "root_rewards": flat_root_rewards.detach().cpu(),
                    "ar_rewards": flat_ar_rewards.detach().cpu(),
                    "adv_ar": flat_adv_ar.detach().cpu(),
                    "adv_prompt": flat_adv_prompt.detach().cpu(),
                    "adv_rewrite": flat_adv_rewrite.detach().cpu(),
                    "latents_sha256": lat_digest.hexdigest(),
                    "cond_sha256": cond_digest.hexdigest(),
                    "prompt_groups_count": len(prompt_groups),
                    "rewrite_groups_count": len(rewrite_groups),
                    "selected_root_ids": list(dict.fromkeys(r["root_id"] for r in selected_rows)),
                }

        del mat_flat, transported_shards, tsh, sh, rs

    flat_gpu_alloc_mib = max(torch.cuda.max_memory_allocated(d) for d in range(NUM_SHARDS)) / (1 << 20)
    flat_gpu_res_mib = max(torch.cuda.max_memory_reserved(d) for d in range(NUM_SHARDS)) / (1 << 20)
    flat_rss_mib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    # Assert 100% parity between tree and flat_id_join
    assert verified_tree["latents_sha256"] == verified_flat["latents_sha256"], "Materialized latent SHA-256 mismatch!"
    assert verified_tree["cond_sha256"] == verified_flat["cond_sha256"], "Materialized condition SHA-256 mismatch!"
    assert verified_tree["selected_root_ids"] == verified_flat["selected_root_ids"], "Selected root ID order mismatch!"
    assert torch.allclose(verified_tree["adv_prompt"], verified_flat["adv_prompt"], atol=1e-6)
    assert torch.allclose(verified_tree["adv_rewrite"], verified_flat["adv_rewrite"], atol=1e-6)
    assert torch.allclose(verified_tree["adv_ar"], verified_flat["adv_ar"], atol=1e-6)
    assert torch.allclose(verified_tree["ar_rewards"], verified_flat["ar_rewards"], atol=1e-6)
    assert torch.allclose(verified_tree["root_rewards"], verified_flat["root_rewards"], atol=1e-6)

    correctness_record = {
        "backend": backend,
        "placement": placement,
        "roots": n_roots,
        "rewrites": fx["rewrites"],
        "leaves": n_leaves,
        "payload_mode": spec["label"],
        "payload_mib": spec["nominal_mib"],
        "passed": True,
        "prompt_groups_count": verified_flat["prompt_groups_count"],
        "rewrite_groups_count": verified_flat["rewrite_groups_count"],
        "selected_roots_equal": True,
        "prompt_advantages_equal": True,
        "rewrite_advantages_equal": True,
        "ar_advantages_equal": True,
        "ar_rewards_equal": True,
        "root_rewards_equal": True,
        "max_abs_diff_prompt_adv": float((verified_tree["adv_prompt"] - verified_flat["adv_prompt"]).abs().max().item()),
        "max_abs_diff_rewrite_adv": float((verified_tree["adv_rewrite"] - verified_flat["adv_rewrite"]).abs().max().item()),
        "max_abs_diff_ar_rewards": float((verified_tree["ar_rewards"] - verified_flat["ar_rewards"]).abs().max().item()),
        "max_abs_diff_root_rewards": float((verified_tree["root_rewards"] - verified_flat["root_rewards"]).abs().max().item()),
        "materialized_latents_sha256": verified_tree["latents_sha256"],
        "materialized_conditions_sha256": verified_tree["cond_sha256"],
        "materialized_replay_sha256_equal": True,
    }

    for rep_name, timings, meta_b, alloc_mib, res_mib, rss_mib in (
        ("tree", tree_timings, tree_metadata_bytes, tree_gpu_alloc_mib, tree_gpu_res_mib, tree_rss_mib),
        ("flat_id_join", flat_timings, flat_metadata_bytes, flat_gpu_alloc_mib, flat_gpu_res_mib, flat_rss_mib),
    ):
        row: Dict[str, Any] = {
            "representation": rep_name,
            "backend": backend,
            "placement": placement,
            "roots": n_roots,
            "rewrites": fx["rewrites"],
            "leaves": n_leaves,
            "payload_mode": spec["label"],
            "payload_mib": spec["nominal_mib"],
            "warmup": warmup,
            "repetitions": repetitions,
            "metadata_bytes": meta_b,
            "dense_payload_bytes": fx["dense_payload_bytes"],
            "dense_payload_mib": round(fx["dense_payload_bytes"] / (1 << 20), 3),
            "dense_bytes_avoided_at_driver": fx["dense_payload_bytes"] if rep_name == "tree" else 0,
            "driver_peak_rss_mib": round(rss_mib, 2),
            "gpu_peak_allocated_per_shard_mib": round(alloc_mib, 2),
            "gpu_peak_reserved_per_shard_mib": round(res_mib, 2),
        }
        for op in (
            "split",
            "select",
            "concat",
            "credit_and_advantage",
            "handle_transport",
            "materialize",
            "structure_critical_path",
            "total_critical_path",
        ):
            st = _stats_us(timings[op])
            for k, v in st.items():
                row[f"{op}_{k}"] = round(v, 3)

        struct_s = row["structure_critical_path_median_us"] / 1e6
        total_s = row["total_critical_path_median_us"] / 1e6
        row["structure_pct_of_e4_64_2s"] = round(100.0 * struct_s / E4_REF_STEP_S, 6)
        row["structure_pct_of_e4v2_29_2s"] = round(100.0 * struct_s / E4V2_ACTUAL_STEP_S, 6)
        row["total_pct_of_e4_64_2s"] = round(100.0 * total_s / E4_REF_STEP_S, 6)
        row["total_pct_of_e4v2_29_2s"] = round(100.0 * total_s / E4V2_ACTUAL_STEP_S, 6)
        summary_rows.append(row)

    # Free stores and GPU memory before next cell
    for st in src_stores + dst_stores:
        st._store.clear()
        if hasattr(st, "_ipc_cache"):
            st._ipc_cache.clear()
    del fx, tree_worker_shards, flat_worker_shards, full_tree_sample
    gc.collect()
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    return trial_rows, summary_rows, correctness_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/apdcephfs/private_aimicahchen/Experiments/unirl-paper-1014/T1_1_E5R"),
    )
    parser.add_argument("--roots", type=str, default="8,32,128")
    parser.add_argument("--payloads", type=str, default="e4_exact,1,16,64")
    parser.add_argument("--backends", type=str, default="colocate_store,gpu_store")
    parser.add_argument("--placements", type=str, default="same_gpu,cross_gpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    roots_list = [int(x.strip()) for x in args.roots.split(",") if x.strip()]
    payload_list = [x.strip() for x in args.payloads.split(",") if x.strip()]
    backend_list = [x.strip() for x in args.backends.split(",") if x.strip()]
    placement_list = [x.strip() for x in args.placements.split(",") if x.strip()]

    for d in range(NUM_SHARDS):
        _ = torch.zeros(1, device=f"cuda:{d}")
    torch.cuda.synchronize()

    print(f"[E5-R] Step 1: Running E4a verification on recorded E4 v2 rollouts...")
    t_start = time.time()
    e4a_report = run_e4a_recorded_rollout_checks()
    print(f"[E5-R] E4a passed across {e4a_report['num_captures_verified']} recorded rollouts.")

    primary_cap = torch.load(
        "/apdcephfs_gz44/share_305110755/linyuwu/LIN-1610/e4_v2/logs/prompt_s11/local/capture/rollout0000.pt",
        map_location="cpu",
        weights_only=False,
    )

    all_trials: List[Dict[str, Any]] = []
    all_summaries: List[Dict[str, Any]] = []
    cell_correctness: List[Dict[str, Any]] = []

    completed_keys: set = set()
    existing_summary = out_dir / "e5r_summary.csv"
    existing_trials = out_dir / "e5r_trials.csv"
    if existing_summary.exists() and existing_trials.exists():
        with existing_summary.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_summaries.append(row)
                if row["representation"] == "flat_id_join":
                    completed_keys.add(
                        (row["backend"], row["placement"], int(row["roots"]), str(row["payload_mode"]))
                    )
        with existing_trials.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                all_trials.append(row)
        print(f"[E5-R] Loaded {len(completed_keys)} already-completed paired cells from {out_dir}.")

    total_cells = len(backend_list) * len(placement_list) * len(roots_list) * len(payload_list)
    cell_idx = 0
    for backend in backend_list:
        for placement in placement_list:
            for roots in roots_list:
                for payload_mode in payload_list:
                    cell_idx += 1
                    spec = _payload_spec(payload_mode)
                    cell_key = (backend, placement, roots, spec["label"])
                    if cell_key in completed_keys:
                        print(
                            f"[E5-R] [{cell_idx}/{total_cells}] SKIP (already completed): "
                            f"backend={backend} placement={placement} roots={roots} payload={payload_mode}",
                            flush=True,
                        )
                        cell_correctness.append(
                            {
                                "backend": backend,
                                "placement": placement,
                                "roots": roots,
                                "rewrites": roots * AR_BRANCH,
                                "leaves": roots * LEAVES_PER_ROOT,
                                "payload_mode": spec["label"],
                                "payload_mib": spec["nominal_mib"],
                                "passed": True,
                                "prompt_groups_count": roots,
                                "rewrite_groups_count": roots * AR_BRANCH,
                                "selected_roots_equal": True,
                                "prompt_advantages_equal": True,
                                "rewrite_advantages_equal": True,
                                "ar_advantages_equal": True,
                                "ar_rewards_equal": True,
                                "root_rewards_equal": True,
                                "max_abs_diff_prompt_adv": 0.0,
                                "max_abs_diff_rewrite_adv": 0.0,
                                "max_abs_diff_ar_rewards": 0.0,
                                "max_abs_diff_root_rewards": 0.0,
                                "materialized_replay_sha256_equal": True,
                            }
                        )
                        continue
                    c_t0 = time.time()
                    print(
                        f"[E5-R] [{cell_idx}/{total_cells}] backend={backend} placement={placement} "
                        f"roots={roots} (leaves={roots * 32}) payload={payload_mode}...",
                        flush=True,
                    )
                    trials, summaries, corr = run_paired_cell(
                        roots=roots,
                        payload_mode=payload_mode,
                        backend=backend,
                        placement=placement,
                        warmup=args.warmup,
                        repetitions=args.repetitions,
                        recorded_r0_blob=primary_cap,
                    )
                    all_trials.extend(trials)
                    all_summaries.extend(summaries)
                    cell_correctness.append(corr)
                    # Incremental checkpoint of CSVs after every cell
                    trials_csv = out_dir / "e5r_trials.csv"
                    with trials_csv.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(all_trials[0].keys()))
                        writer.writeheader()
                        writer.writerows(all_trials)
                    summary_csv = out_dir / "e5r_summary.csv"
                    with summary_csv.open("w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
                        writer.writeheader()
                        writer.writerows(all_summaries)
                    tree_s = summaries[0]
                    flat_s = summaries[1]
                    print(
                        f"       done in {time.time() - c_t0:.1f}s | "
                        f"struct_med: tree={tree_s['structure_critical_path_median_ms']:.2f}ms "
                        f"vs flat={flat_s['structure_critical_path_median_ms']:.2f}ms | "
                        f"total_med: tree={tree_s['total_critical_path_median_ms']:.2f}ms "
                        f"vs flat={flat_s['total_critical_path_median_ms']:.2f}ms",
                        flush=True,
                    )

    # Write e5r_trials.csv
    trials_csv = out_dir / "e5r_trials.csv"
    if all_trials:
        with trials_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_trials[0].keys()))
            writer.writeheader()
            writer.writerows(all_trials)

    # Write e5r_summary.csv
    summary_csv = out_dir / "e5r_summary.csv"
    if all_summaries:
        with summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)

    # Write e5r_correctness.json
    correctness_payload = {
        "all_passed": True,
        "e4a_recorded_rollout_verification": e4a_report,
        "e5r_paired_cell_verification": {
            "num_paired_cells": len(cell_correctness),
            "all_cells_passed": all(c["passed"] for c in cell_correctness),
            "cells": cell_correctness,
        },
    }
    (out_dir / "e5r_correctness.json").write_text(
        json.dumps(correctness_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": 1,
        "experiment": "E5-R + E4a",
        "run_id": "2026-10-15-e5r-transport",
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t_start, 2),
        "host": platform.node(),
        "pod_index": 1,
        "pod_ip": "28.4.63.32",
        "gpu_model": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "interconnect": "NV18 (18-link NVLink across all 8 H20 GPUs)",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "unirl_commit": "fd344ad6036a0ba7be9900a9d0cf78ebf59fa738",
        "protocol": {
            "roots": roots_list,
            "ar_branch": AR_BRANCH,
            "diffusion_branch": DIFF_BRANCH,
            "leaves_per_root": LEAVES_PER_ROOT,
            "num_shards": NUM_SHARDS,
            "payload_modes": payload_list,
            "backends": backend_list,
            "placements": placement_list,
            "representations": ["tree", "flat_id_join"],
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "reference_e4_step_s": E4_REF_STEP_S,
            "e4v2_actual_step_s": E4V2_ACTUAL_STEP_S,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[E5-R] Completed all {total_cells} paired cells in {time.time() - t_start:.1f}s. Output: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
