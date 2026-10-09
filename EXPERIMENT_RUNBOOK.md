# UniRL Paper Experiment Runbook

Last updated: 2026-10-09

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

E0 applies at the actual caller boundaries. At the pinned commit, the `Sample`
constructor checks parent adjacency but not equal concat depth, uniform contiguous
reward blocks, or general shared-field equality. Generic concat retains the first
shared value; the async assembler separately checks behavior versions. Validate
those preconditions and compare with an ID-joined reference before real runs.
Data/sampling seed fields do not alone control worker model/LoRA initialization:
audit and seed every stochastic component before construction, preserve identical
trainable initialization across replicas, and document RNG restoration on resume.
Record any residual scheduling nondeterminism without claiming bitwise reproduction.

## 2. Experiment-to-claim matrix

| ID | Priority | Controlled study | Claim tested | Manuscript output |
|---|---:|---|---|---|
| E0 | P0 gate | Environment, semantic parity, rollout/replay, publication, checkpoint/resume, and instrumentation | Later numbers are interpretable and regenerable | Evaluation preflight and correctness appendix |
| E1 | P0 | Qwen3-4B-Base + DAPO-Math textbook GRPO, seeds 11/22/33 | Credible AR learning through the common trajectory path | RQ1 AR row and learning curves |
| E2 | P0 | SD3.5-Medium + full-budget UniRL/VeRL-Omni reference, seeds 11/22/33 | Learning validity and related-recipe context; reproduction requires estimator parity | RQ1 diffusion row and learning curves |
| E3 | P0 | UniRL versus pinned VeRL-Omni on one 8-GPU SD3.5 node | Shape-matched native-recipe system cost; same-algorithm parity pending | RQ2 table and phase breakdown |
| E4 | P0 | Frozen-Qwen rewrite -> SD3, original-prompt versus rewrite-local grouping | Lineage selects an executable cross-stage objective | RQ3 lineage/enablement result |
| E5 | P0 minimum / P2 extensions | IR/transport costs plus a matched UniRL topology pair | Representation and execution cost | RQ3 table/scaling plot |
| E6 | P1 | Matched sync/async Qwen3 sweep | Bounded staleness has a measurable useful region | RQ4 Pareto plot and table |
| E7 | P1 | HunyuanImage3 shared-backbone AR recaption -> image diffusion training | One trajectory can carry two replay types and credit through a jointly updated AR/diffusion backbone | RQ1/RQ3 generalization row and appendix trace |
| E8 | P1 | MiniMax H3 text-to-video-with-audio (T2AV) post-training on the pinned UniRL MiniMax H3 path | The trajectory contract extends from image diffusion to joint temporal video and audio values and replay state | RQ1 video/audio extension and modality-cost appendix |

E1/E2 establish training validity, E3 establishes comparative full-stack efficiency,
and E4 directly tests the paper's lineage thesis. The minimum E5 topology/cost slice
bounds the price of the design and is also required for paper completion. Complete
primary E0-E5 before adding model families, objectives, E6, or broad scaling points;
broader E5 backends/scaling and E7/E8 are extensions. E7 and E8 answer different
questions and neither substitutes for the other: E7 tests a shared AR/diffusion
backbone, while E8 tests a joint video-audio diffusion model with temporal outputs
and rewards.
At least one full-budget E1/E2 reference must pass mathematical and effective-work
parity before the paper claims reference-implementation reproduction. Omitting
both references leaves that author-requested evidence obligation open.

## 3. Source-derived measurement readiness

The current code already emits some—but not all—metrics required by the paper:

| Evidence | Current executable source | Readiness / action |
|---|---|---|
| Complete driver elapsed time | `perf/step_time_s` is logged inside the train batch, before outer async publication/eval/save | **Partial, not end-to-end.** Add monotonic driver events covering initialization, whole iterations, publication, evaluation, and checkpoints; never sum overlapping phase times. |
| Coarse phase time | `install_phase_timing` wraps wake, generate, sleep, weight sync, reward, and `train_track` | Partial. Split credit, replay/anchor, backward, optimizer, publication barrier, and checkpoint time before RQ2. |
| Reward/length/group diagnostics | `compute_rollout_sample_metrics` | Ready; archive all `rollout/*` series. |
| GPU peak allocated/reserved memory | `MemoryMonitor` → `perf/*` memory keys | Ready; verify enabled for every role and archive role/rank maxima. |
| Optimizer and publication versions | `training_version_metrics` | Ready: `async/train_version`, `published_version`, `publish_lag`, and `batches_since_sync`. |
| Realized rollout lag | `rollout_version_metrics` | Ready: `async/output_version`, `staleness_updates`, and `staleness_batches`. |
| Buffer occupancy, rejected/discarded work, barrier duration | Manager logs exist, but no complete structured per-step series is emitted | **Instrumentation TODO before RQ4.** Add root/prompt counts and time for admission, ready, carry, filter rejection, suspension, finish, quiesce, and publication barrier. |
| GPU utilization/power | No authoritative internal time series | Collect externally (DCGM or equivalent), synchronized to run timestamps. |
| External text/image quality | `benchmarks.run` and `BenchmarkSpec` registry | Ready after dataset/reward availability is verified; archive completions/images, scores, and `summary.json`. |
| External video/audio quality | Video primitives and reward-service video routing exist; VideoAlign/VideoReward has a vendored inference path; the MiniMax H3 path trains on `T2AVCompositeScorer` (ImageBind audio-video on a middle frame plus CLAP text-audio) | **Partial.** The T2AV training reward does not score prompt-video alignment. GPU-smoke the exact scorers and video/audio serialization path, freeze frame sampling/FPS and audio sampling, and add held-out prompt-video and audio evaluators distinct from the optimized training reward before E8. |

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
8 samples = 512 trajectories per rollout, maximum 8192 new tokens, GRPO population-standard-deviation
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
- Frozen checkpoint evaluations at the prespecified cadence; final checkpoint is primary.
- Internal AIME is validation: an AIME-selected best checkpoint is not independent AIME test evidence. Use final MATH-500 avg@4 as primary, fixed-budget AIME curves as secondary, and audit normalized prompt/problem overlap before interpretation. `avg@k` means mean accuracy, not pass@k.
- Separate training and evaluation RNG: fix evaluation seed 42, archive actual server chat template/token IDs and reasoning/content handling, and complete per-request RNG/resume auditing in E0.

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

The expected signal is improvement over the base on the primary external MATH-500
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
and seeds 11/22/33. Center advantages by prompt group and divide by batch-wide
reward std. Freeze all values in the resolved manifest rather than citing
defaults indirectly.

Required run set:

- base checkpoint;
- at least three UniRL seeds;
- 300-step VeRL-Omni reference runs with seeds 11/22/33, saves every 50 steps, and the same frozen evaluator; verify native-checkpoint export first (handoff Section 8.3);
- stock flow/CPS, sparse-index/window, normalization and anchor differences mean this is currently a related-recipe reference, not a same-estimator reproduction;
- the same prompts, resolution, denoising/sampling schedule, initial-noise policy,
  SDE indices, reward revision, LoRA targets, optimizer work, and evaluator.

External evaluation should include:

- `image/geneval2` for the explicitly named **UniRL synthetic compositional set / Qwen3-VL Soft-TIFA-style score**. This registry key is not evidence of official GenEval2 data/scorer parity;
- `image/preference`, using final PartiPrompts HPSv3 as the primary endpoint, with ImageReward as a complementary frozen judge. These automatic judges are distinct from optimized PickScore, not independent human evaluations;
- within-prompt mean pairwise LPIPS with AlexNet features over a fixed 16-image
  seed set per prompt as the diversity guard.

Use evaluation seed 42 across all training seeds/checkpoints, explicit unique output
tags, 512x512, 40 steps, guidance 1. The synthetic endpoint additionally uses its
registered linear sigma grid, prompt-hashed seeds and encoder length 256; preference
uses the pinned pipeline scheduler and index-based seeds. Record full resolved
settings, data/scorer revisions and prompt overlap; `summary.json` alone omits
settings and may average only successful scores. Require complete scoring or
explicit missingness analysis, and never reuse tags after changing inputs.

For LPIPS, generate 16 images per PartiPrompts prompt separately from the one-image
preference endpoint; use RGB [-1,1], frozen AlexNet LPIPS, all 120 pairs per prompt,
and then equal prompt means. Report relative change of mean diversity, with a
paired prompt bootstrap (10,000 resamples, RNG 20260905). A >10% drop is a
study-specific investigation trigger, not a literature-derived quality threshold.
If the base is near zero, report absolute change and flag the relative criterion
undefined. Implement and smoke-test the generator/metric before formal E2.
Unknown scores, allocation facts, and model/data/evaluator hashes remain blank.

Archive every generated image, prompt/sample index, seed, checkpoint, reward output,
and evaluator error. Reward improvement alone is not sufficient evidence.

The expected signal is higher PickScore without material collapse in synthetic composition,
independent PartiPrompts preference views, or the preregistered diversity guard.
If reward rises while the guards fall, report reward overfitting. If rollout/replay
or LoRA publication checks fail, fix correctness and create a new run ID. If the
implementation is correct but learning is absent, escalate the recipe/claim decision
to a researcher.

## 6. RQ2 — End-to-end systems performance

### 6.1 SD3.5 shape-matched native-recipe pair (E3)

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

The stock pair is not algorithm-equivalent: UniRL uses `FlowSDEStrategy` with
three sampled indices among early steps; VeRL-Omni requests CPS and a three-step
window. Transition means/variances and likelihoods can differ, not just kernels.
Audit exact SDE indices, anchor, advantage normalization, effective replay work,
and reward-device placement. Count all reward GPUs. Until a new mathematical-parity
pair passes, label these shape-matched native-recipe rows; SDPA alignment alone
cannot justify framework-overhead or same-algorithm superiority claims.
Do not subtract a phase from only one system's end-to-end result.

Use three process-level repetitions per configuration. Each repetition runs 30
steps; exactly the first five timing observations are warm-up, leaving at least 20
steady-state observations. Run one system at a time on the same reserved 8-GPU
node, randomizing framework order within each repetition block. The exact GPU,
CPU, interconnect, driver, and cache state remain blank until
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
  E2-R supplies full-budget quality context and E6 supplies the prespecified threshold study.
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

Fix E4 independently of E2: 512x512, ten steps, one SDE index among 0--8,
eta 0.7, LoRA 16/32, lr 3e-4, weight decay 0, clip 1e-4, replay anchor, two
updates, microbatch one, 300 rollouts, saves every 50, seeds 11/22/33 per scope.
E4a first requires equality of groups, advantages, and replay inputs on identical
recorded payloads against an explicit flat-row ID join. E4b then studies learning
under 32-image versus 8-image normalization groups; any quality difference is an
objective effect, not a causal measure of IR superiority.

Verify no AR optimizer and unchanged AR parameter hashes. Evaluate base and all
50-rollout diffusion checkpoints on held-out PickScore test roots after overlap
audit, using a cached common set of four rewrites per root and eight image seeds
per rewrite. Fix evaluation at 512x512, 40 steps, guidance 1, seed 42. PE adapters
live at `checkpoint-<rollout>/diffusion`. The stock default `eval_interval=0` does
not create these images: the required generation/scoring harness is specified in
the handoff and is not yet implemented. Score against both root and rewrite;
aggregate images, rewrites, then roots, and bootstrap roots separately from
training-seed uncertainty. Training trajectory dumps are not held-out evaluation.

Archive root/part/parent IDs, stage/model identity, decoded-value hashes, output
versions, reward components, group membership, propagated credit, and replay-segment
metadata. Worker/LoRA initialization, deterministic trainside-AR sampling,
real trace dumping and held-out checkpoint generation are E0 prerequisites. The current PickScore request is conditioned on each
rewrite, so an original-intent claim requires a separate evaluation join that scores
each generated image against its root prompt. Report rewrite-conditioned training
reward alongside held-out root-prompt HPSv3/ImageReward views
that are distinct from PickScore. If both groupings are correct but root-prompt
quality is equal or worse, narrow the claim to semantic expressiveness and integration
rather than discarding the result.

### 7.2 Representation, transport, and topology cost (E5)

The checked-in CPU artifact is a regression test only.

The required E5-R paired slice uses the same recorded E4 tensors, device,
transport and operations for tree versus `flat_id_join`, with exact E4a semantics.
Record metadata/driver memory/transfer/critical-path costs and their fraction of
the same model-work interval. E5-T below measures deployment instead. Broad
transport backends and cross-node scaling are extensions after these two slices.

The wider measurement inventory is:

- tree split/concat/select and flat-row handoff at matched payloads;
- reference metadata bytes, driver RSS, dense bytes avoided, and materialization
  hashes;
- same-GPU, cross-GPU, and cross-node transfer latency/bandwidth for
  `colocate_store`, `gpu_store`, and `transfer_queue` where supported;
- end-to-end topology/transport ablations with identical model work;
- an inventory of modules reused by E4; a held-out integration-effort study is an optional extension, not established by the existing workflow.

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

The plot is a time-to-quality versus realized-lag Pareto frontier. Before the formal
grid, fix MATH-500 avg@4 target `<Q_TARGET>` using only a separate pilot and archive
the cadence. Report the first observed crossing and bracket by adjacent checkpoint
evaluation times; non-crossing runs are right-censored. The full driver clock must
include initialization, publication, barriers, evaluation and checkpoint costs.
The old `perf/step_time_s` cannot supply that clock. D0/A1 changes permitted lag
and concurrency together: it tests allowed overlap, not either knob in isolation.

Interpret S0 versus D0 as an allocation/residency comparison and D0 versus A1 as
the onset-of-overlap comparison; do not attribute S0--A1 differences solely to
asynchrony. The expected result is that moderate overlap lowers idle time and improves
time-to-quality, while aggressive lag raises rejection/discard or hurts quality.
If version/buffer invariants fail, fix the scheduler or instrumentation. If they
pass and no asynchronous point Pareto-improves on its relevant control, preserve the
negative result and conclude that this workload/allocation has no demonstrated useful
async region.

## 9. Model- and modality-generalization extensions (E7/E8)

E7 and E8 follow the primary E0--E5 evidence. They strengthen different parts of
the heterogeneous-model claim and must remain separate experiments. E7 asks whether
one shared backbone can consume AR and diffusion replay in one lineage-aware update.
E8 asks whether the diffusion trajectory contract survives the substantially larger
temporal state, serialization, reward, and evaluation surface of joint video and
audio generation.
Neither extension is required to validate the primary SD3.5 result, and neither may
be represented by a configuration file or inference-only demo without the gates below.

### 9.1 HunyuanImage3 shared-backbone AR and diffusion training (E7)

**Starting path:** `examples/unified_model/hi3/hi3_vllmomni.yaml` with
`python -m unirl.train_unified_model`. This implementation was observed in a source
checkout newer than the manuscript's audited commit. Before E7, pin the exact source
commit, archive its diff from the audited paper commit, and re-run the applicable E0
gates. Do not describe E7 as functionality of commit `f5d7104` unless that exact tree
contains and passes the exercised path.

E7 uses one HunyuanImage3 backbone in AR text-generation and DiT image-generation
modes. The intended trajectory is
`original prompt -> N reasoning/recaption children -> M image children per recaption`.
ARGRPO replays each `TextSegment`; DiffusionGRPO replays each `LatentSegment`; both
backward passes accumulate into one shared LoRA adapter before one optimizer step.

#### E7a: semantic and lifecycle gate

Run a dump-first smoke before any learning claim and require:

- a real `P x N x M` trace with `N >= 2` and `M >= 2`, unique root/part IDs, exact
  parent IDs, distinct recaptions, and distinct per-image initial-noise identities;
- equality of root/recaption/image groups, propagated rewards, per-track advantages,
  and replay inputs against an explicit flat ID-join oracle on the same payload;
- AR rollout/replay token-log-prob consistency and diffusion rollout/replay
  transition-log-prob consistency at initialization;
- proof that the same adapter update is published to both AR and DiT rollout engines,
  followed by post-publication parity probes for both modes;
- checkpoint/resume preservation of the shared optimizer, adapter, RNG state, both
  replay paths, and behavior-policy versions;
- a trace showing that image reward reaches only the correct recaption ancestors and
  that the evaluator joins every image to the intended original prompt.

If the engine produces only one image for each recaption, or reconstructs the
`N x M` structure from positional assumptions rather than archived IDs, E7a fails.
Fix and pin the fan-out/response-lineage implementation before E7b. Configuration
comments are not evidence of the runtime trajectory shape.

#### E7b: training-validity comparison

Use the same prompt manifest, generation geometry, initial adapter, noise manifest,
reward revision, evaluator, optimizer work, and checkpoint cadence for:

1. frozen base inference;
2. image-only updating, with the AR loss disabled by an explicit, logged stage-loss
   control while retaining the same AR rollout/recaption path;
3. joint ARGRPO + DiffusionGRPO updating through the shared adapter.

The current shared stack has no paper-audited stage-loss weighting control. Implement
and test an explicit switch before the image-only row; zeroing advantages or silently
skipping a backward pass is not an admissible substitute. After E7a and one
capacity-only pilot, register prompts/rollout, `N`, `M`, image size, denoising/SDE
schedule, LoRA targets/rank, loss weights, optimizer budget, checkpoint cadence, GPU
allocation, and the exact HunyuanImage3 checkpoint. Formal learning summaries use
seeds 11/22/33 and may not change those fields per condition.

The primary endpoint is held-out image quality scored against the original root
prompt by a frozen evaluator distinct from the optimized reward. Also report
rewrite-conditioned quality, within-recaption image diversity, recaption validity,
length and duplication, AR/diffusion ratio and clipping diagnostics, per-stage
gradient norms, and complete system time/memory. Gradient cosine similarity between
the two stage losses is a useful diagnostic if it can be collected without changing
the update. A valid but non-improving joint row supports execution generality, not a
claim that joint optimization improves image quality.

A HunyuanImage3 text-to-image diffusion run without the AR rollout path (for example,
a trainside FlowGRPO recipe with `sys_type: en_vanilla`) is neither the image-only
E7b row nor E7 evidence. Report it, if at all, as single-stage diffusion
generalization under its own name.

### 9.2 MiniMax H3 text-to-video-with-audio post-training (E8)

MiniMax H3 generates video with stereo audio from text (T2AV). The UniRL path is
not in the manuscript's audited commit `f5d7104`; it was introduced in UniRL PR403
(commit `d8e4aefa`), which adds `MiniMaxH3Bundle`, `MiniMaxH3Conditions`,
`MiniMaxH3Pipeline`, trainside rollout, and the recipe
`diffusion/minimax_h3/minimax_h3_t2va_trainside`. Before E8, pin the exact source
commit (PR403 or its merged successor), archive its diff from the audited paper
commit, and re-run the applicable E0 gates. Also pin the MiniMax H3 checkpoint
revision, license, text encoder and hidden layer, scheduler and video/audio shifts,
video and audio VAEs, frame-count constraint, frame rate, audio sample rate and
channels, and resolution. Do not substitute HunyuanImage3, HunyuanVideo, or WAN
results for MiniMax H3.

#### E8-access: determine the admissible claim

The PR403 path loads local weights and trains a LoRA adapter, so the trainable case
below applies to it. Record the weight revision and license with every run.

- **Trainable weights or an official gradient/update interface:** E8 may proceed to
  rollout/replay integration and multi-seed RL training.
- **Inference API with controllable seeds but no parameter updates:** only an
  inference/trajectory interoperability row is admissible. It cannot establish
  post-training, replay correctness, weight publication, or learning validity.
- **API without reproducible version, seed, raw video, or complete error accounting:**
  omit the row from quantitative evidence and retain only a stated limitation.

A local imitation using another video model must carry that model's actual name.
The training reward's license is part of the access contract: ImageBind is
CC-BY-NC-SA, so the current T2AV recipe is non-commercial.

#### E8a: video/audio trajectory and replay gate

Run the gate on the pinned UniRL MiniMax H3 path. The first admitted smoke must use
more than one frame and more than one sample per prompt. Require:

- `Videos` decoded values with exact sample/root IDs plus archived frame count, FPS,
  duration, resolution, codec/container, and preprocessing metadata, and the paired
  audio with sample rate, channel count, and duration;
- temporal video and audio latent/replay tensors with recorded shapes, dtypes,
  selected SDE indices, sigma schedules for both modalities, initial-noise identity,
  and byte counts;
- exact split/concat/select and serialization round trips without dropping temporal
  order, desynchronizing audio from its video, or associating a reward with the wrong
  sample;
- rollout/replay transition-log-prob agreement over the joint video-audio SDE, finite
  gradients, optimizer movement, and post-publication parity for every supported
  weight-sync path;
- a reward smoke that demonstrably consumes multiple frames and the audio track.
  First-frame or middle-frame image scores may be retained as diagnostics but are not
  sufficient evidence for temporal video quality;
- a decoded-output sanity check on every reward smoke (for example, blank-frame or
  luma-collapse detection and silent-audio detection). The current composite reward
  can rise while decoded frames collapse toward black, so the reward alone cannot
  admit a run;
- GPU validation of the selected training reward and any held-out evaluator,
  including frame-sampling/FPS and audio-sampling policy, missing/error handling,
  and deterministic scoring tolerance on repeated inputs.

Exploratory single-seed runs on this path, including matched UniRL/veRL-Omni
comparisons, are pipeline and systems evidence only. They do not satisfy E8a or
E8b and must not be reported as E8.

Use an existing stable UniRL video recipe such as WAN 2.2 only as a pipeline control
for `Videos`, reward routing, artifact serialization, and metric generation. It is
not a MiniMax H3 quality or efficiency baseline unless effective work, model access,
generation geometry, reward, and hardware are explicitly aligned.

#### E8b: training-validity and modality-cost study

After E8a and one capacity-only pilot, freeze in the registration manifest:

- prompt dataset and held-out split, MiniMax H3 revision, prompt preprocessing, and
  overlap audit;
- frames, FPS, duration, resolution, audio sample rate and channels, denoising
  steps, SDE step selection, video/audio shifts, guidance, samples per prompt, and
  initial-noise policy;
- reward models/revisions, component weights, frame and audio samplers,
  group-normalization population, replay
  anchor, optimizer/update count, LoRA or full-weight target, precision, checkpoint
  cadence, total rollout budget, and allocation;
- seeds 11/22/33 for the formal row, with identical base and checkpoint evaluation
  manifests.

Required rows are the frozen base and at least three trained seeds. A framework or
model baseline is optional and must not delay the within-model learning-validity
result. The primary external endpoint must be a frozen prompt-video alignment or
video-quality evaluator distinct from the optimized reward; because the current
training reward does not relate the prompt to the video, an improvement in it
cannot stand in for this endpoint. Report temporal consistency/motion, visual
quality, prompt alignment, audio quality, text-audio alignment, audio-video
synchronization, safety/error rate, and within-prompt diversity as separate views;
do not collapse them into an unversioned single score. Archive every generated
video with its audio and its prompt/sample/seed/checkpoint identity, evaluator
output, and failure.

Because video changes the payload scale, also report latent and decoded bytes per
trajectory for each modality, video and audio VAE encode/decode time, rollout time,
reward-service time, transfer and materialization time, peak memory by role,
samples/GPU-hour, frames/GPU-hour, and failed/retried generations. Normalize any
image-versus-video cost discussion by explicit pixels, frames, audio samples,
denoising work, and allocated resources; raw iteration-time
ratios across SD3.5 and MiniMax H3 do not isolate modality overhead.

Reward improvement without held-out improvement, or improvement accompanied by
temporal collapse, decoded-frame collapse, audio degradation, or diversity loss, is
reported as reward overfitting. If E8a passes but formal training does not improve,
retain the negative result and limit the claim to video/audio-path execution. If trainable MiniMax H3 access is unavailable, report the
inference-only result separately and do not count E8 as diffusion training validity.

## 10. Failure ownership and paper fill-in checklist

Statistics use training seeds and process repetitions as independent units. Report
all seeds/repetitions, means and 95% t intervals; pooled timing steps show within-run
variation and are not independent replicate counts for a speed-ratio CI. Primary
endpoints are final MATH-500 for E1 and final PartiPrompts HPSv3 for E2. Secondary
metrics remain visible even when they disagree; no post hoc endpoint switching.
E7/E8 primary endpoints and budgets must be registered after their correctness and
capacity gates but before formal multi-seed runs or inspection of treatment results.

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
