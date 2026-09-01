# UniRL Paper Experiment Runbook

Last updated: 2026-08-31

This document turns the paper's evaluation plan into a GPU-cluster execution and
evidence protocol. It contains **no experimental results**. Values below are either
prespecified study choices or fields read from executable source/configuration at
UniRL commit `f5d710406b215bb7a0b387fdd37e4d4778b92338`. Before any run, re-audit
the checked-out commit and regenerate the resolved configuration.

## 1. Evidence-admission gates

A result may enter the paper only when all applicable gates pass:

1. **Source identity:** archive UniRL and baseline commits, submodule commits, and
   dirty patches. A named commit plus an unrecorded dirty tree is insufficient.
2. **Resolved configuration:** archive Hydra's fully resolved config after all CLI
   and environment overrides. Source YAML alone is insufficient.
3. **Input identity:** record model/tokenizer/reward revisions, dataset content
   hashes, prompt preprocessing, chat template, and evaluator revision.
4. **Semantic parity:** compare effective work and algorithm semantics field by
   field. A topology experiment may vary topology, transport, residency, and
   concurrency—but not the estimator, prompt set, group size, generation budget,
   optimizer work, or evaluator.
5. **Correctness before speed:** rollout–replay parity, post-publication parity,
   complete behavior-version provenance, and hard-boundary checks must pass before
   throughput is interpreted.
6. **Raw evidence:** archive unaggregated per-step metrics, stdout/stderr, failures,
   retries, excluded iterations, checkpoints, and evaluation outputs.
7. **Regeneration:** every paper number and plot must be reproducible from the
   archived bundle by a committed parser. Hand-copied dashboard values are not
   admissible.

## 2. Source-derived measurement readiness

The current code already emits some—but not all—metrics required by the paper:

| Evidence | Current executable source | Readiness / action |
|---|---|---|
| End-to-end step time | `UniRLWandBLogger.log_rollout_step` → `perf/step_time_s` | Ready when structured logging is enabled. |
| Coarse phase time | `install_phase_timing` wraps wake, generate, sleep, weight sync, reward, and `train_track` | Partial. Split credit, replay/anchor, backward, optimizer, publication barrier, and checkpoint time before RQ2. |
| Reward/length/group diagnostics | `compute_rollout_sample_metrics` | Ready; archive all `rollout/*` series. |
| GPU peak allocated/reserved memory | `MemoryMonitor` → `perf/*` memory keys | Ready; verify enabled for every role and archive role/rank maxima. |
| Optimizer and publication versions | `training_version_metrics` | Ready: `async/train_version`, `published_version`, `publish_lag`, and `batches_since_sync`. |
| Realized rollout lag | `rollout_version_metrics` | Ready: `async/output_version`, `staleness_updates`, and `staleness_batches`. |
| Buffer occupancy, rejected/discarded work, barrier duration | Manager logs exist, but no complete structured per-step series is emitted | **Instrumentation TODO before RQ4.** Add root/prompt counts and time for admission, ready, carry, filter rejection, suspension, finish, quiesce, and publication barrier. |
| GPU utilization/power | No authoritative internal time series | Collect externally (DCGM or equivalent), synchronized to run timestamps. |
| External text/image quality | `benchmarks.run` and `BenchmarkSpec` registry | Ready after dataset/reward availability is verified; archive completions/images, scores, and `summary.json`. |

For phase and async metrics, set `logging.report_to_wandb=true`. On clusters where
network logging is undesirable, use W&B offline mode and archive the entire offline
run directory. Console timestamps alone are acceptable only as a cross-check, not as
the primary phase-breakdown source.

## 3. Run-directory contract

Use one immutable directory per process group:

```text
artifact/<study>/<system>/<run_id>/
  manifest.json
  source/
    unirl_commit.txt
    baseline_commit.txt
    dirty.patch
    submodules.txt
  config/
    command.txt
    environment.txt
    resolved.yaml
    parity.tsv
  inputs/
    models.json
    datasets.json
    evaluators.json
  logs/
    stdout.log
    stderr.log
    wandb_offline/          # or lossless export with history
    gpu_telemetry.csv
  metrics/
    steps.jsonl
    phases.jsonl
    memory.jsonl
    async.jsonl
    failures.jsonl
  checkpoints/
    index.json
  evaluation/
    <checkpoint>/<benchmark>/...
  checksums.sha256
```

`manifest.json` must state the research question, workload ID, seed, start/end time,
host allocation, GPU model/count, interconnect, driver/CUDA/PyTorch/engine versions,
warm-up rule, expected and completed optimizer work, failure/retry policy, and every
excluded sample or step.

## 4. RQ1 — Training reproducibility and final quality

### 4.1 AR reasoning

**Pinned starting recipe:** `examples/ar/qwen3_grpo_4b_base_dapo_sglang.yaml`.
The executable fields specify Qwen3-4B-Base, DAPO-Math JSONL input, group size 8,
maximum 8192 new tokens, textbook GRPO normalization, four optimizer updates per
rollout, and periodic AIME evaluation. Do not infer data availability or successful
training from the recipe.

Required run set:

- Base checkpoint evaluation before training.
- At least three independent UniRL seeds and the same seeds in the selected reference
  implementation where seed control is comparable.
- A reference run with aligned data order, prompt/chat template, verifier, group size,
  sampling, token budget, effective batch, GRPO normalization, clipping, optimizer,
  update count, and evaluation decoding.
- Frozen checkpoint evaluations at a prespecified cadence plus final/best checkpoints.

External evaluation is driven by executable `benchmarks/core/registry.py`:

- `text/math500`: 500 problems, avg@4, temperature 0.6, top-p 0.95, 16384-token cap,
  `math_verify` grader.
- `text/aime24` and `text/aime25`: 30 problems each, avg@16, temperature 0.6,
  top-p 0.95, 32768-token cap, `math_verify` grader.

Example evaluation command after serving a frozen checkpoint:

```bash
python -m benchmarks.run \
  -b text/math500,text/aime24,text/aime25 \
  --endpoint http://127.0.0.1:30000 \
  --tag <run_id>-u<optimizer_update> \
  --out <artifact_dir>/evaluation
```

Primary paper outputs:

- reward, external accuracy, response length, truncation ratio, zero-std group ratio,
  KL/ratio/clip diagnostics, and loss versus optimizer update and wall time;
- base, final, and best-checkpoint external metrics with seed-level points;
- mean and 95% confidence interval over independent seeds;
- failure/retry and checkpoint-selection rules fixed before examining the final curves.

### 4.2 Diffusion image generation

Use one FlowGRPO workload that can be aligned to a current public reference. The
source-aligned SD3.5 starting point is
`examples/diffusion/sd3/sd3_vllmomni.yaml`; its executable configuration uses
SD3.5-Medium, LoRA rank 32/alpha 64 on eight attention projections, PickScore reward,
ten denoising steps, three scheduled SDE steps, and two optimizer updates per rollout.
Freeze all overridden values in the resolved manifest rather than citing these
defaults indirectly.

Required run set:

- base checkpoint;
- at least three UniRL seeds;
- aligned reference seeds or a clearly separated published-reference comparison;
- the same prompts, resolution, denoising/sampling schedule, initial-noise policy,
  SDE indices, reward revision, LoRA targets, optimizer work, and evaluator.

External evaluation should include:

- `image/geneval` or `image/geneval2` for compositional behavior;
- `image/preference` for independent preference/reward views;
- a diversity/quality guard metric fixed before training.

Archive every generated image, prompt/sample index, seed, checkpoint, reward output,
and evaluator error. Reward improvement alone is not sufficient evidence.

### 4.3 Cross-stage representation test

Exercise one prompt-enhancement or tool-interaction path with complete serialized
lineage. Archive root/part IDs, stage/model identity, decoded-value hashes, output
versions, reward components, propagated credit, and replay-segment metadata. Compare
against a controlled flat-row or broken-lineage handoff. This tests C1 representation
and integration; it is not a state-of-the-art quality claim.

## 5. RQ2 — End-to-end systems performance

### 5.1 SD3.5 + FlowGRPO aligned pair

Pinned executable launchers:

- UniRL: `benchmarks/speed_benchmarks/verl_omni/run_unirl_sd35_aligned.sh`.
- VeRL-Omni: `benchmarks/speed_benchmarks/verl_omni/run_verlomni_sd35_aligned.sh`.
- Pinned baseline submodule: `01c87ee595874c313f9f296525fb5b4389678451`
  (currently recorded but not initialized in the audited checkout).

The launchers prescribe 48 prompts × 16 samples, 384² output, ten denoising steps,
three early-window SDE steps, two optimizer updates, microbatch 8, LoRA r32/alpha64
on the same eight projections, learning rate/weight decay 1e-4, clip 1e-5, PickScore,
and one 8-GPU node. Confirm these facts again from the resolved configuration of both
runs; launcher prose is not evidence.

Two comparison rows are required:

1. **backend-aligned:** SDPA-class attention on both sides;
2. **best valid:** each system's best disclosed supported backend.

Known inherent differences must be measured and disclosed rather than normalized
away: the engine/version stack, engine-emitted versus recomputed old log-probability,
and the exact SDE kernel implementation. Do not subtract a phase from only one
system's end-to-end result.

### 5.2 AR systems pair

Use the RQ1 Qwen3 GRPO workload and select one maintained reference only after an
exact semantic-parity audit. Match tokens generated and replayed, group/effective
batch, optimizer updates, reward placement, model precision, attention kernel class,
weight-update semantics, and checkpoint/evaluation cadence.

### 5.3 Required measurements

- At least 20 steady-state iterations after a prespecified warm-up; report all
  measured iterations, median, mean, p90, and failure-inclusive sensitivity.
- End-to-end seconds/iteration, samples/s, generated tokens or pixels/s,
  samples/GPU-hour, and GPU-hours to a fixed external-quality threshold.
- Coarse phases already available plus the finer instrumentation TODOs in Section 2.
- Per-role/rank GPU memory, CPU RSS, GPU utilization and power, idle fraction, and
  wake/sleep/publication/barrier cost.
- Strong scaling at fixed global work and weak scaling at fixed per-GPU work. State
  how global optimizer semantics change, if at all.

Run systems one at a time on the same reserved hosts. Record clocks/power settings,
other GPU processes, filesystem/cache state, and whether model/data caches were warm.

## 6. RQ3 — Abstraction and transport

The checked-in CPU artifact is a regression test only. The GPU study must separately
measure:

- tree split/concat/select and flat-row handoff at matched payloads;
- reference metadata bytes, driver RSS, dense bytes avoided, and materialization
  hashes;
- same-GPU, cross-GPU, and cross-node transfer latency/bandwidth for
  `colocate_store`, `gpu_store`, and `transfer_queue` where supported;
- end-to-end topology/transport ablations with identical model work;
- held-out model/workflow integration effort and reused versus changed modules.

Mooncake/TransferQueue results require explicit protocol, NIC/RDMA topology, queue
configuration, and failure handling. Omit the row if the hardware cannot support a
valid run.

## 7. RQ4 — Bounded staleness

### 7.1 Mandatory semantic-parity correction

The stock recipes below are **not** currently a controlled sync/async pair:

- sync: `examples/ar/qwen3_grpo_4b_base_dapo_sglang.yaml`;
- async: `examples/ar/qwen3_grpo_4b_base_dapo_sglang_async.yaml`.

Source-value mismatches that must be resolved before measurement:

| Field | Sync | Async stock | Required experiment action |
|---|---|---|---|
| `normalize_adv_by_std` | `true` | `false` | Choose one estimator and use it on both sides. |
| `algorithm.clip_range_high` | `null` | `0.28` | Match symmetric/asymmetric clipping. |
| `algorithm.loss_agg_mode` | `seq-mean-token-mean` | `seq-mean-token-sum-norm` | Match sequence-length weighting. |
| `bundle.config.attn_implementation` | `flex_attention` | absent/default | Match for the backend-aligned systems row. |
| `stack.micro_planner` | token-budget planner, 10240 tokens | absent/default count planner | Match packing for the controlled comparison. |

These differences change the estimator or learner execution and therefore confound
both convergence and throughput. Apply explicit async overrides, archive the resolved
configs, and attach a field-by-field parity table. The intended independent variables
are topology, concurrency, publication interval, and allowed lag.

### 7.2 Sweep and outputs

Use the same total GPU allocation and identical effective optimizer work. Prespecify
a feasible grid over in-flight capacity, publication interval, and lag budget after
one capacity pilot; retain the synchronous point as zero-overlap/zero-lag reference.
Do not tune the grid separately for each reported metric.

For every step archive:

- generation latency distribution and engine-slot occupancy;
- end-to-end and phase time, learner/engine idle fraction, publication and barrier
  time;
- output/train/published versions and realized lag;
- admitted, ready, carried, suspended, completed, rejected, discarded, and retried
  roots/prompts;
- reward/external evaluation versus optimizer update and wall time.

The headline plot is a time-to-quality versus realized-lag Pareto frontier. A
throughput gain without matched quality/time-to-quality is not sufficient.

## 8. Paper fill-in checklist

Before removing any `TODO[data]`:

- link the exact run IDs and artifact paths;
- regenerate the table/figure from raw data;
- state sample/seed counts and uncertainty;
- include base and reference results;
- report failed/retried/excluded work;
- update limitations with observed rather than anticipated failure modes;
- rerun LaTeX compilation and page-by-page PDF inspection.
