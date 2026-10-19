# E5-R Paired GPU Trajectory-IR Representation & Transport Evidence + E4a Verification (`2026-10-15-e5r-transport`)

This bundle contains the raw trial records, summary statistics, correctness hashes, E4a recorded-rollout verification results, and deterministic table/macro/figure generator (`generate.py`) for **RQ3 — E5-R (Paired Representation and Transport Cost)** and **E4a (Cross-Stage Grouping & Replay Parity)**.

## Hardware & Software Provenance

- **Cluster Allocation**: `occupy-128h20-gz-1012-p1` (`8b1d8998a1250fa701a13063c9df0f0f`), Pod 1 (`worker-0`, IP `28.4.63.32`, host `TENCENT64.site`)
- **GPUs**: $8 \times \text{NVIDIA H20}$ (`97,871 MiB` / `95.6 GiB` HBM3 per GPU), full `NV18` (18-link NVLink) mesh across all 8 GPUs
- **Software**: Python `3.12.11`, PyTorch `2.13.0+cu130`, CUDA `13.0`
- **Audited UniRL Commit**: `fd344ad6036a0ba7be9900a9d0cf78ebf59fa738`

## Contents

1. **Task 1 — Single-GPU / Two-GPU Transport Harness (`run_gpu_transport_evidence.py`)**:
   - `transport_{tree,flat_id_join}_{colocate_store,gpu_store}_{same_gpu,cross_gpu}.json`: 8 JSON files (48 cells each: `roots in {8,32,128}`, `branch_factor in {1,4,8,32}`, `payload_mib in {1,16,64,256}`, 10 warmup + 100 measured repetitions).
   - Demonstrates that at 64 GiB total leaf payload (`leaves * payload_mib = 64 GiB`), `tree` succeeds on a single 96 GiB H20 GPU because leaf references avoid dense driver concatenation, whereas `flat_id_join` triggers `torch.cuda.OutOfMemoryError` when `torch.cat` duplicates the 64 GiB buffer (`128 GiB > 95 GiB`).

2. **Task 2 — Full 8-Shard Paired E5-R + E4a Benchmark (`run_e5r_paired_full.py`)**:
   - `e5r_trials.csv`: All `9,600` individual measured trial records (`48` paired cells $\times 2$ arms $\times 100$ repetitions after 10 warmup iterations).
   - `e5r_summary.csv`: Per-configuration summary statistics (`median`, `mean`, `p90`, `min`, `max` in both $\mu\text{s}$ and $\text{ms}$) for `split`, `select`, `concat`, `credit_and_advantage`, `handle_transport`, `materialize`, `structure_critical_path`, and `total_critical_path`, plus memory/byte metrics and fraction of E4 step time (`64.2s` reference and `29.2s` E4 v2 actual step time).
   - `e5r_correctness.json`:
     - **E4a Recorded Rollout Verification**: Verified across all 12 recorded E4 v2 rollout captures (`rollout0000.pt` and `rollout0001.pt` across `prompt_s11`, `prompt_s22`, `prompt_s33`, `rewrite_s11`, `rewrite_s22`, `rewrite_s33`, joined with real rewrite traces from `pe_trace_*.jsonl`). Achieves `0.0` max absolute difference (`torch.equal == True`, bit-identical SHA-256) against recorded `advantages_prompt` (8 groups of 32) and `advantages_rewrite` (32 groups of 8) and against `flat_id_join` (`f32`), plus exact replay tensor round-trip after `Sample.split -> Sample.select -> Sample.concat`.
     - **E5-R Paired Cell Verification**: 100% agreement across all 48 paired cells between `tree` (`Sample` + `Part` + `TensorRef` + `LatentSegment` + `TextSegment` + `SD3Conditions`) and `flat_id_join` on group memberships, prompt-scope advantages, rewrite-scope advantages, propagated AR rewards, root rewards, and materialized replay tensor SHA-256 digests.

3. **Task 3 — Deterministic Regeneration (`generate.py`)**:
   - `e5r_table.tex`: Booktabs LaTeX table of paired E5-R operation timings and E4 step-time fractions.
   - `macros.tex`: LaTeX macros (`\EFiveR...`) for manuscript inclusion.
   - `figures/e5r_overhead.pdf` and `figures/e5r_overhead.png`: Three-panel figure showing (a) payload-size invariance of `tree` (`TensorRef`) on same-GPU operations, (b) phase breakdown across backends and placements at 64 GiB batch scale, and (c) total overhead as a percentage of E4 step time.

## Regeneration

```bash
python3 generate.py
```
