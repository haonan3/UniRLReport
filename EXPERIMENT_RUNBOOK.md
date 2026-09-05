# UniRL Paper Experiment Runbook

Last updated: 2026-09-05

This document turns the paper's evaluation plan into a GPU-cluster execution and
evidence protocol. It contains **no experimental results**. Values below are either
prespecified study choices or fields read from executable source/configuration at
UniRL commit `f5d710406b215bb7a0b387fdd37e4d4778b92338`. Before any run, re-audit
the checked-out commit and regenerate the resolved configuration.

`GPU_HANDOFF_README.md` is the command-oriented companion to this document. This
runbook defines what evidence is scientifically admissible; the handoff defines
the intended execution order and concrete launch/evaluation commands.

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

## 2. Experiment-to-claim matrix

| ID | Priority | Controlled study | Claim tested | Manuscript output |
|---|---:|---|---|---|
| E0 | P0 gate | Environment, semantic parity, rollout/replay, publication, checkpoint/resume, and instrumentation | Later numbers are interpretable and regenerable | Evaluation preflight and correctness appendix |
| E1 | P0 | Qwen3-4B-Base + DAPO-Math textbook GRPO, seeds 11/22/33 | Credible AR learning through the common trajectory path | RQ1 AR row and learning curves |
| E2 | P0 | SD3.5-Medium + aligned FlowGRPO, seeds 11/22/33 | Credible diffusion learning without independent-quality/diversity collapse | RQ1 diffusion row and learning curves |
| E3 | P0 | UniRL versus pinned VeRL-Omni on one 8-GPU SD3.5 node | End-to-end systems efficiency under matched work | RQ2 table and phase breakdown |
| E4 | P0 | Frozen-Qwen rewrite -> SD3, original-prompt versus rewrite-local grouping | Lineage selects an executable cross-stage objective | RQ3 lineage/enablement result |
| E5 | P0 minimum / P2 extensions | IR/transport costs plus a matched UniRL topology pair | Representation and execution cost | RQ3 table/scaling plot |
| E6 | P1 | Matched sync/async Qwen3 sweep | Bounded staleness has a measurable useful region | RQ4 Pareto plot and table |

E1/E2 establish training validity, E3 establishes comparative full-stack efficiency,
and E4 directly tests the paper's lineage thesis. The minimum E5 topology/cost slice
bounds the price of the design and is also required for paper completion. Complete
primary E0-E5 before adding model families, objectives, E6, or broad scaling points;
broader E5 backends/scaling are extensions.

## 3. Source-derived measurement readiness

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

## 4. Run-directory contract

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

## 5. RQ1 — Training reproducibility and final quality

### 5.1 AR reasoning (E1)

**Pinned starting recipe:** `examples/ar/qwen3_grpo_4b_base_dapo_sglang.yaml`.
The executable fields specify Qwen3-4B-Base, DAPO-Math JSONL input, 64 prompts x
8 samples = 512 trajectories per rollout, maximum 8192 new tokens, textbook GRPO
normalization, symmetric clip 0.2, `seq-mean-token-mean`, four optimizer updates
per rollout, a 10,240-token microplanner, AdamW lr 1e-6/weight decay 0.01, full
weight publication every rollout, and periodic AIME evaluation. The prespecified
formal budget is 800 rollouts on 32 total GPUs, with seeds 11/22/33 and full
checkpoints every 200 rollouts. GPU type, node layout, and interconnect remain blank
until allocation. Do not infer data availability or successful training from the
recipe.

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

Full-training checkpoints must first be exported with
`python -m unirl.tools.export_full --library transformers`; then serve the frozen
HF folder through an OpenAI-compatible SGLang endpoint. Example evaluation command:

```bash
python -m benchmarks.run \
  -b text/math500 -b text/aime24 -b text/aime25 \
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

The expected signal is improvement over the base on independent MATH-500/AIME
evaluation across most seeds without collapse in response length, truncation,
within-group variance, ratio, or clip diagnostics. This is a hypothesis, not a
filled result. Reward improvement without external-evaluation improvement is
treated as reward overfitting. Stable non-learning after all correctness gates pass
requires a human research decision; it is not repaired by silent tuning.

### 5.2 Diffusion image generation (E2)

Use one FlowGRPO workload that can be aligned to a current public reference. The
source-aligned SD3.5 starting point is
`examples/diffusion/sd3/sd3_vllmomni.yaml`, with the aligned launcher overrides in
`benchmarks/speed_benchmarks/verl_omni/run_unirl_sd35_aligned.sh`. The prespecified
formal setting uses SD3.5-Medium, 48 prompts x 16 images = 768 images/rollout,
384x384 training resolution, ten denoising steps, three SDE steps in the first
half, eta 0.8, guidance 1, distinct initial noise, PickScore, LoRA rank 32/alpha 64
on eight attention projections, two optimizer updates, microbatch 8, lr and weight
decay 1e-4, clip 1e-5, publication every rollout, 300 rollouts, one 8-GPU node,
and seeds 11/22/33. Freeze all values in the resolved manifest rather than citing
defaults indirectly.

Required run set:

- base checkpoint;
- at least three UniRL seeds;
- aligned reference seeds or a clearly separated published-reference comparison;
- the same prompts, resolution, denoising/sampling schedule, initial-noise policy,
  SDE indices, reward revision, LoRA targets, optimizer work, and evaluator.

External evaluation should include:

- `image/geneval` or `image/geneval2` for compositional behavior;
- `image/preference`, reporting HPSv3 and ImageReward separately from the PickScore
  training reward (the registry's PickScore view is not independent evidence);
- within-prompt mean pairwise LPIPS with AlexNet features over a fixed 16-image
  seed set per prompt as the diversity guard.

Use identical evaluation prompts and image seeds for the base and every checkpoint.
Report prompt-level paired LPIPS changes and a 95% bootstrap interval. A greater than
10% relative decrease from the base is preregistered as material diversity loss;
LPIPS is a collapse guard, not a perceptual-quality score. Implement and smoke-test
the metric before formal training. Unknown final scores, model snapshot hashes, GPU
identity, and evaluator hashes remain blank.

Archive every generated image, prompt/sample index, seed, checkpoint, reward output,
and evaluator error. Reward improvement alone is not sufficient evidence.

The expected signal is higher PickScore without material collapse in GenEval2,
independent PartiPrompts preference views, or the preregistered diversity guard.
If reward rises while the guards fall, report reward overfitting. If rollout/replay
or LoRA publication checks fail, fix correctness and create a new run ID. If the
implementation is correct but learning is absent, escalate the recipe/claim decision
to a researcher.

## 6. RQ2 — End-to-end systems performance

### 6.1 SD3.5 + FlowGRPO aligned pair (E3)

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

Use three process-level repetitions per configuration. Each repetition runs 30
steps; exactly the first five timing observations are warm-up, leaving at least 20
steady-state observations. Run one system at a time on the same reserved 8-GPU
node. The exact GPU, CPU, interconnect, driver, and cache state remain blank until
the allocation manifest is captured.

The expected result is competitive or better UniRL throughput with a phase breakdown
that attributes the outcome. A genuine slowdown after parity, cache, utilization,
and failure-accounting checks is retained and reported; no phase may be subtracted
from only one system.

### 6.2 AR systems pair

Use the RQ1 Qwen3 GRPO workload and select one maintained reference only after an
exact semantic-parity audit. Match tokens generated and replayed, group/effective
batch, optimizer updates, reward placement, model precision, attention kernel class,
weight-update semantics, and checkpoint/evaluation cadence.

The official veRL Qwen3-4B FSDP example is only a scaffold: as currently published,
it differs in model variant, data, group size, response budget, and KL/objective
settings. The baseline row remains blocked until a pinned `<VERL_COMMIT>` is adapted
and its resolved parity report passes. If parity is impossible, omit this head-to-head
row rather than substituting a nearby workload.

### 6.3 Required measurements

- At least 20 steady-state iterations after a prespecified warm-up; report all
  measured iterations, median, mean, p90, and failure-inclusive sensitivity.
- End-to-end seconds/iteration, samples/s, generated tokens or pixels/s,
  and samples/GPU-hour. The 30-step E3 protocol does not produce time-to-quality;
  that metric belongs to E6 unless a separate full-budget baseline study is registered.
- Coarse phases already available plus the finer instrumentation TODOs in Section 3.
- Per-role/rank GPU memory, CPU RSS, GPU utilization and power, idle fraction, and
  wake/sleep/publication/barrier cost.
- Strong scaling at fixed global work and weak scaling at fixed per-GPU work. State
  how global optimizer semantics change, if at all.

Run systems one at a time on the same reserved hosts. Record clocks/power settings,
other GPU processes, filesystem/cache state, and whether model/data caches were warm.

## 7. RQ3 — Lineage, abstraction, and transport (E4/E5)

### 7.1 Cross-stage representation test (E4)

Use `examples/pe/pe_trainside_pickscore_frozenllm_promptgroup.yaml`: eight original
prompts x four frozen-Qwen3 rewrites x eight SD3 images = 256 image descendants per
rollout, with only the diffusion side trained. Compare
`diffusion_group_scope=prompt` against the same configuration with
`diffusion_group_scope=rewrite`. The former normalizes all 32 image descendants of
an original prompt; the latter normalizes eight images within each rewrite. Invalid
or deliberately corrupted lineage belongs in E0 tests, not the quality baseline.

Archive root/part/parent IDs, stage/model identity, decoded-value hashes, output
versions, reward components, group membership, propagated credit, and replay-segment
metadata. Real trajectory dumping and deterministic trainside-AR seed plumbing are
instrumentation prerequisites. The current PickScore request is conditioned on each
rewrite, so an original-intent claim requires a separate evaluation join that scores
each generated image against its root prompt. Report rewrite-conditioned training
reward alongside root-prompt GenEval2 where applicable and HPSv3/ImageReward views
that are distinct from PickScore. If both groupings are correct but root-prompt
quality is equal or worse, narrow the claim to semantic expressiveness and integration
rather than discarding the result.

### 7.2 Representation, transport, and topology cost (E5)

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

At the audited commit there is no complete GPU transport evidence harness. Its
implementation and review are therefore an explicit prerequisite, not an implied
runnable result. The expected signal is modest structural metadata overhead and
avoidance of dense driver materialization. Span explosion or unfavorable transfer
regimes must be retained and reported.

The minimum end-to-end topology pair reuses E3's colocated UniRL SD3.5 row and adds
`examples/diffusion/sd3/sd3_vllmomni_lora_separate.yaml` with the same aligned
48 x 16, 384-square, ten-step, three-SDE-step, two-update work unit. Both consume
eight GPUs total; the separate row assigns four to training and four to rollout.
This is an allocation-level comparison, not a pure transport ablation: train DP,
residency, wake/sleep, and local-versus-remote LoRA publication change together.
Report those phase differences explicitly. Same-/cross-GPU backend and cross-node
transport studies are extensions after this minimum pair.

## 8. RQ4 — Bounded staleness (E6)

### 8.1 Mandatory semantic-parity correction

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
configs, and attach a field-by-field parity table. Topology is not isolated by the
stock sync/async pair: sync uses all 32 GPUs in a colocated time-shared path, whereas
async uses disjoint 16-GPU training and rollout slabs. D0 below is therefore required
to separate the cost of disaggregation from the incremental effect of overlap.

### 8.2 Prespecified primary sweep and outputs

Use 32 total GPUs and identical effective optimizer work. Disaggregated points fix
`train_fraction=0.5` and `per_worker_inflight=14`. `buffer_max_staleness` is measured
in consumed rollout batches, not optimizer-update versions; because each batch makes
four updates, a lag budget of one batch permits four optimizer-version steps. After
one capacity pilot, retain the following primary points; do not tune the grid
separately for each reported metric:

| Point | Trainer/topology | `max_inflight` | publication interval (batches) | maximum lag (batches) | Isolated purpose |
|---|---|---:|---:|---:|---|
| S0 | sync/colocated | n/a | 1 | 0 | total-resource reference |
| D0 | async/disaggregated | 1 | 1 | 0 | same-topology no-overlap control |
| A1 | async/disaggregated | 2 | 1 | 1 | onset of overlap relative to D0 |
| A2 | async | 2 | 1 | 2 | lag allowance |
| A3 | async | 2 | 2 | 2 | publication cadence |
| A4 | async | 2 | 2 | 4 | aggressive bounded point |

The formal quality/time sweep uses 800 rollouts and seeds 11/22/33. Only after the
primary grid is complete may `max_inflight={1,4}` be added at one fixed cadence/lag
point. Natural generation variance is stratified by prompt/response length; any
synthetic delay injector is a separately labeled stress test.

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

Interpret S0 versus D0 as an allocation/residency comparison and D0 versus A1 as
the onset-of-overlap comparison; do not attribute S0--A1 differences solely to
asynchrony. The expected result is that moderate overlap lowers idle time and improves
time-to-quality, while aggressive lag raises rejection/discard or hurts quality.
If version/buffer invariants fail, fix the scheduler or instrumentation. If they
pass and no asynchronous point Pareto-improves on its relevant control, preserve the
negative result and conclude that this workload/allocation has no demonstrated useful
async region.

## 9. Failure ownership and paper fill-in checklist

Codex/engineering work may correct paths, dependencies, Hydra mistakes, parity
reports, logging/parsers, deterministic divisibility, smoke-only OOM geometry,
version accounting, rollout/replay, publication, checkpoint/resume, and lineage
bugs. A human research decision is required before changing the formal optimizer,
reward, estimator, model/task, training budget, evaluation metric, baseline
definition, or headline claim.

Before removing any `TODO[data]`:

- link the exact run IDs and artifact paths;
- regenerate the table/figure from raw data;
- state sample/seed counts and uncertainty;
- include base and reference results;
- report failed/retried/excluded work;
- update limitations with observed rather than anticipated failure modes;
- rerun LaTeX compilation and page-by-page PDF inspection.
