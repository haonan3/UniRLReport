# UniRL Paper Plan

Last updated: 2026-09-01

## 0. Evidence boundary

- **Code source of truth:** a clean sibling checkout at `../UniRL`
- **Audited commit:** `f5d710406b215bb7a0b387fdd37e4d4778b92338`
- **Commit date / subject:** 2026-08-20, `feat(async): dynamically dispatch prompt rollouts for AR and DiT (#289)`
- The code worktree was clean and matched `origin/main` when this plan was created.
- Repository README files, documentation prose, and existing architecture images were **not** used as design evidence. They may be used only to locate code. Claims below cite executable source, configs, or required experiment artifacts.
- No checked-in raw training logs currently substantiate the performance numbers written in `benchmarks/speed_benchmarks/verl_omni/README.md`. Those numbers therefore do **not** enter the paper until logs, configs, environment manifests, and parser outputs are archived and independently checked.
- The GPU-cluster protocol is now specified in `EXPERIMENT_RUNBOOK.md`. It includes evidence-admission gates, resolved-config parity, run-directory contents, metric requirements, and paper fill-in rules. The author has explicitly deferred GPU execution to suitable hardware; this draft does not treat the absence of a local GPU as evidence.
- The stock Qwen3 GRPO sync and async recipes are not a controlled comparison: they differ in advantage normalization, high clipping bound, loss aggregation, attention implementation, and microplanner. RQ4 results are inadmissible until explicit overrides make all non-topology semantics equal and a resolved-config parity report is archived.
- Current structured logging is sufficient for end-to-end step time, six coarse phases, rollout diagnostics, memory, and policy-version/lag fields. Fine-grained credit/replay/backward/optimizer/publication timing and complete buffer/drop/barrier accounting remain instrumentation prerequisites for RQ2/RQ4; GPU utilization/power requires synchronized external telemetry.

## 1. Central thesis

**Working title:** *UniRL: Trajectory-Centric Post-Training across Heterogeneous Generative Models*

**Central thesis:** A post-training system can make model family, rollout engine, and resource topology largely orthogonal when the system boundary is a lineage-preserving, modality-aware trajectory IR rather than an LLM-token batch. In UniRL, the same `Sample -> Sample` contract carries branching, rewards, policy versions, sparse AR or diffusion replay state, and multimodal values through synchronous, disaggregated, and bounded-staleness execution.

This is a systems thesis, not a claim that every algorithm is mathematically unified. Model-specific stages and algorithm-specific replay remain explicit.

### What the paper must not claim

- “Multimodal support,” “diffusion RL,” “agentic RL,” “Ray,” “asynchronous RL,” or a long algorithm list as standalone novelty.
- Bitwise reproducibility, convergence equivalence, performance superiority, or scale beyond the experiments actually run.
- A capability based only on a config filename or import path. A supported path must reach an entry point and complete an exercised run.
- Numbers copied from README prose without primary log artifacts.
- That UniRL subsumes veRL/HybridFlow, VeRL-Omni, Relax, AReaL, ROLL, NeMo-RL, or slime. These systems establish strong overlapping baselines.

## 2. Main claims and evidence contracts

| ID | Claim to test | Current code evidence at audited commit | Experimental evidence required before final paper |
|---|---|---|---|
| C1 | **One trajectory IR represents heterogeneous post-training workflows without flattening away lineage or modality.** | `unirl/types/sample.py`: `Part` stores lineage IDs, typed primitives, model-facing conditions/segments, rewards/advantages, and `output_version`; `Sample` checks parent adjacency and implements tree-complete `split`, `fork`, `observe`, and reward propagation. `unirl/types/primitives.py` defines packed text/image/video/audio values. `unirl/types/segments/{text,latent}.py` encode packed tokens and sparse diffusion trajectories. | At least three exercised traces dumped in a versioned artifact: AR group rollout, diffusion SDE rollout, and composed or multi-turn rollout. Add invariant tests demonstrating tree-preserving split/concat and version/group preservation. Report integration effort for a held-out model/workflow. |
| C2 | **Semantic components and physical execution are orthogonal enough that the same trainer-level loop can run under multiple valid topologies.** | `unirl/distributed/group/{device_pool,placement,handle,dispatch}.py` separate logical roles, placement slabs/slots, and dispatch. `unirl/trainer/{ar,diffusion}.py` build the same bundle/pipeline/backend/algorithm/stack contracts for colocated or separate rollout. Hydra `_target_` recipes instantiate roles. | Run the same model/algorithm/workload under trainside-colocated, external-colocated, and separate rollout where supported. Verify matched reward/loss behavior and output schema; report throughput, memory, and topology-specific setup. Do not compare modes with different effective work. |
| C3 | **Trajectory movement is representation-preserving while tensor movement is backend-selectable.** | `unirl/distributed/tensor/{batch,ref,transport,worker_local}.py`: dataclass field algebra for concat/packed/shared/reductions; `TensorRef` row-window views; dehydrate/hydrate; transport selection. Backends exist under `unirl/distributed/tensor/backend/{colocate_store,gpu_store,transfer_queue}`. | Payload-size and transfer-time microbenchmarks on same/cross GPU and cross node; correctness hashes after split/concat/localize; peak driver/worker memory; end-to-end transport ablation. Exercise Mooncake only on suitable hardware and report protocol/RDMA details. |
| C4 | **Policy publication and lifecycle transitions make rollout/training interaction explicit and measurable.** | `unirl/rollout/engine/base.py` stamps generated parts with policy version and defines sleep/wake/sync hooks. `unirl/distributed/weight_sync/{lora,full}` implements LoRA and full-weight paths including tensor, IPC, NCCL, and checkpoint variants. Trainers order wake/sync/offload/generate/sleep/onload and fail closed on invalid combinations. | Measure weight-publication time/bytes, wake/sleep cost, peak memory, and iteration impact for LoRA vs full weights and colocate vs separate. Add rollout-vs-replay log-prob consistency checks after each supported sync path. Inject lifecycle failures to confirm the runtime does not silently train on unstamped or partially updated outputs. |
| C5 | **A version-aware scheduler can overlap eligible work while enforcing bounded policy lag and hard boundaries.** | `unirl/trainer/async_rollout.py` is shared by async AR and diffusion; `unirl/rollout/manager/{dispatch,rollout,buffers,filters}.py` dynamically fills engine slots, forms complete prompt groups, filters by version lag, quiesces before publication, and requires empty eval/checkpoint boundaries. | Synchronous vs async runs with identical optimizer work; throughput/utilization vs lag; lag histograms; tail-latency sensitivity; convergence/time-to-quality at lag budgets; boundary and resume tests. Compare against AReaL/Relax or another current async baseline only with aligned semantics. |
| C6 | **The system reproduces credible training behavior for representative AR and diffusion workloads.** | Algorithms and entry paths exist, but code presence is not empirical evidence. Candidate paths: `unirl/train_ar.py`, `unirl/train_diffusion.py`; GRPO/PPO/GSPO and FlowGRPO/FlowDPPO/DiffusionNFT implementations; benchmark loaders under `benchmarks/`. | Multi-seed learning curves, final benchmark metrics, base-model and reference-implementation comparisons, seed CIs, exact checkpoints/data/rewards, and released raw logs. Minimum matrix in §6.1. |
| C7 | **UniRL has competitive end-to-end efficiency under fair, disclosed comparisons.** | Timing and memory instrumentation exist in `unirl/utils/{profiling,memory_monitor,wandb_logger}.py`; aligned launcher scripts exist under `benchmarks/speed_benchmarks/verl_omni/`, but there are no checked-in raw logs. | Re-run and archive all primary artifacts. Compare with closest current baselines on identical hardware and work. Report median, p90, samples/s, samples/GPU-hour, phase breakdown, memory, scaling efficiency, and time-to-quality—not an isolated best-case number. |

### Evidence acquired on 2026-08-30

- Run `cpu_f5d7104_macos_arm64` directly imported a clean UniRL checkout at the audited commit under Python 3.12.14, PyTorch 2.11.0, Ray 2.55.1, NumPy 2.5.2, and Pillow 12.3.0.
- Ten fail-fast checks passed: a mixed AR–observation–diffusion lineage; exact tree split/concat; whole-tree selection/slicing; mean reward propagation and group advantages; malformed-lineage, wrong-modality, and wrong-batch rejection; ragged packed-text round trips; and lazy `TensorRef` selection/materialization/bounds.
- A one-thread CPU microbenchmark recorded all 100 raw trials after ten warm-ups for 8/32/128 roots and six IR operations. This is preliminary RQ3 regression evidence only—not rollout, learner, transport, GPU, training-quality, or end-to-end evidence.
- Reproduction code: `experiments/trajectory_ir/run_cpu_evidence.py`. Evidence bundle: `artifacts/trajectory_ir/cpu_f5d7104_macos_arm64/`, including manifest, structured correctness results, raw trials, captured stdout/stderr, CSV/TeX summaries, report, and SHA-256 checksums.
- C1 is partially exercised with a synthetic mixed-modality lineage, but still requires real AR, diffusion-SDE, and cross-stage rollout traces plus a held-out integration. C3 now has local selection/materialization and tree-reassembly evidence, but still requires byte/memory measurements and same-GPU, cross-GPU, and cross-node transport experiments.

## 3. Section and subsection questions

### Abstract

Question: What fragmentation problem does UniRL address, what is the trajectory-centric mechanism, and what do the experiments establish quantitatively?

Status: design summary and narrowly scoped preliminary RQ3 statement written; headline quantitative sentences remain TODO until C6/C7 evidence exists.

### 1. Introduction

- Why do post-training systems become model-family-specific even when the outer loop is rollout–score–update?
- Why are “more algorithms/models” and “async” insufficient paper stories?
- What is the testable trajectory-centric thesis?
- What are the contributions in terms of abstractions, execution, and evidence?

### 2. Problem and design requirements

- Which information must survive a heterogeneous rollout? Branch lineage, grouping, modality, sparse replay state, reward/credit, and policy version.
- Which axes should be orthogonal? Semantics, rollout engine, training backend, placement, transfer, and publication cadence.
- Which correctness invariants are endangered by batching and asynchrony?
- What is explicitly outside scope (universal algorithm API, automatic optimal placement, fault-tolerant global services)?

### 3. Trajectory-centric programming model

#### 3.1 `Sample` and `Part`: a lineage-preserving trajectory IR

Question: How does a sequence of typed parts encode branching trajectories and preserve whole prompt trees during sharding?

#### 3.2 Modality values, model conditions, and replay segments

Question: Why distinguish user-facing primitives, model-facing conditions, and optimization-facing segments? How do packed text tokens and sparse latent trajectories coexist under one batch algebra?

#### 3.3 Rollout as `Sample -> Sample`

Question: How can trainside, SGLang, SGLang diffusion, vLLM-Omni, composed, and agentic engines share one boundary while retaining engine-specific internals?

#### 3.4 Algorithm and train-stack boundary

Question: Which responsibilities are generic (microplanning, loss normalization, optimizer step, anchor freezing) and which remain algorithm/stage specific (AR replay, diffusion replay, reference terms)?

### 4. From a program to a distributed execution

#### 4.1 Roles, placement slabs, and dispatch

Question: How do logical roles map to device slabs and worker slots, and how are broadcast/DP-scatter/merge semantics expressed?

#### 4.2 Tensor references and transport

Question: How does UniRL avoid sending dense tensors through the driver while preserving structural operations on the trajectory?

#### 4.3 Colocated and disaggregated phase lifecycles

Question: How are wake/sleep, train offload/onload, reward placement, and weight publication ordered? Which configurations are rejected?

#### 4.4 Synchronous and bounded-staleness scheduling

Question: How does the async manager fill lanes, form complete groups, stamp/filter versions, and establish eval/checkpoint barriers?

### 5. Instantiations, not a feature catalogue

- **AR reasoning:** typed text segment, group fan-out, scalar or GAE credit, replay loss.
- **Diffusion generation:** latent segment, selected SDE steps, rollout or replay anchor, black-box reward.
- **Cross-stage trajectories:** composed AR→diffusion and tool observations show why lineage—not a flat tensor dictionary—is the central unit.
- For each, explain the minimal specialized code left after reusing the common contracts. Do not claim quality from implementation existence.

### 6. Evaluation

#### 6.1 RQ1 — Training behavior and final quality

Question: Can UniRL reproduce expected optimization behavior and credible final performance across at least one AR and one diffusion workload?

#### 6.2 RQ2 — End-to-end system efficiency

Question: Where does wall-clock time go, how much does each topology cost, and how does UniRL compare with an aligned reference?

#### 6.3 RQ3 — What does the trajectory abstraction cost and enable?

Question: What are IR/transport overheads, and does the same workload remain semantically stable across execution choices?

#### 6.4 RQ4 — When is asynchrony useful?

Question: How do throughput and convergence change with generation variance and policy lag?

### 7. Related work

- LLM RL systems: HybridFlow/veRL, OpenRLHF, ROLL, NeMo-RL.
- Asynchronous systems: AReaL, Relax, slime. These make “async” non-novel by itself.
- Multimodal/diffusion systems: VeRL-Omni and Relax. These make “omni-modal support” non-novel by itself.
- Diffusion RL algorithms: DDPO, Flow-GRPO, DanceGRPO, DiffusionNFT. UniRL implements/adapts algorithms; it does not claim their mathematical contributions.
- Positioning target: one lineage-aware trajectory program crossing generative families and execution modes, evaluated as a system property.

### 8. Limitations and conclusion

Question: Which model integrations remain architecture-specific, which topology choices are manual, and which claims await broader scale/failure evidence?

## 4. Figures plan

| Fig. | Purpose / claim | Construction | Required data | Status |
|---|---|---|---|---|
| 1 | Explain `Sample` lineage for AR, diffusion, and composed/tool trajectories, including replay state and policy version. | New TikZ schematic with three trajectory shapes, reconstructed from audited types and execution paths. | Held-out trace dumps remain part of the empirical validation. | Drafted and visually verified. |
| 2 | Show the central separation: one trajectory IR through semantic roles, mapped onto execution topologies. | New TikZ diagram generated from audited code structure; no old repository image. | None for the mechanism diagram. | Drafted and visually verified. |
| 3 | Make the evaluation contract explicit: four RQs feed one archived evidence bundle before claims are admitted. | New TikZ claim-to-evidence diagram. | None for the protocol diagram. | Drafted and visually verified. |
| 4 | RQ1 learning curves: reward and external eval vs optimizer updates/wall time. | Schema and evidence rules specified in `EXPERIMENT_RUNBOOK.md`; generator is a GPU-stage TODO. | ≥3 seeds per primary workload, reference curves. | Protocol complete; data/generator pending. |
| 5 | RQ2 phase breakdown and throughput comparison. | Stacked bars + table; parser/generator follows the runbook's structured metric contract. | Raw per-step phase logs for all systems. | Protocol complete; instrumentation/data/generator pending. |
| 6 | Scaling/utilization and policy-lag tradeoff. | Scaling curves and lag histogram from resolved-parity runs only. | 1/2/4/8 GPUs or 1/2/4 nodes; async sweeps. | Protocol complete; instrumentation/data/generator pending. |

## 5. Tables plan

| Table | Question answered | Rule |
|---|---|---|
| 1. Requirements and mechanisms | What information/decision does each mechanism preserve? | Design table, no performance claims. |
| 2. Compared system scope | Which adjacent systems target LLM, diffusion/omni, async, or lineage-aware cross-stage workflows? | Use papers/current official sources; avoid checkmarks not verified at a pinned version. Prefer prose categories and access dates. |
| 3. RQ1 reproducibility | Does each workload improve over base and match a credible reference? | Mean ± CI over seeds; exact benchmark version and decoding/eval config. |
| 4. RQ2 performance | How fast and memory-efficient is each aligned setup? | Same effective work/hardware; include phase times, peak memory, GPU-hours, and environment pins. |
| 5. Ablations | Which design choices matter? | Topology, engine, transport, sync mode, weight path, staleness, microplanner. One claim per row group. |
| 6. Integration effort | Does the IR reduce model/workflow-specific changes? | Held-out integration; changed LOC/modules are supporting evidence, not a quality metric alone. |
| 7. Preliminary RQ3 CPU evidence | Do audited IR invariants execute, and how do selected container operations scale on one CPU process? | Generated only from archived raw trials; must remain explicitly separated from GPU, distributed, and end-to-end performance. |

## 6. Missing experiment program

### 6.1 Minimum RQ1 matrix

1. **AR reasoning / GRPO**
   - Candidate: Qwen3-4B on a documented math dataset using an exercised `examples/ar/qwen3_grpo_4b_*.yaml` path.
   - External eval: MATH-500 and AIME (fixed benchmark snapshots, pass@1/avg@k as appropriate).
   - Reference: current veRL or the algorithm/model recipe from a public implementation, aligned for data, prompts, batch, group size, reward, maximum tokens, and optimizer work.
   - Evidence: base score; 3+ UniRL seeds; reward/loss/KL/clip diagnostics; final and best-checkpoint scores; exact model revision.

2. **Diffusion image / FlowGRPO**
   - Candidate A: SD3.5-Medium + PickScore for closest aligned systems comparison.
   - Candidate B: SD3.5-Medium or Qwen-Image + GenEval for externally interpretable compositional quality.
   - Reference: Flow-GRPO code or VeRL-Omni at pinned commits, with identical reward, prompts, resolution, inference steps, SDE window, LoRA targets, and update count.
   - Evidence: base and trained reward; GenEval/independent quality metrics; diversity/quality guard metrics; 3+ seeds or justified lower count with bootstrap intervals if prohibitively expensive.

3. **One cross-stage case**
   - Candidate: prompt enhancement (AR rewrite → diffusion output → image reward propagated to rewrite) or agentic tool use.
   - Purpose: validate C1 beyond parallel single-stage modalities; this is an abstraction test, not necessarily the headline quality result.
   - Evidence: complete serialized lineage traces, per-stage reward/credit, and an ablation that breaks lineage or trains only one stage.

### 6.2 Minimum RQ2 matrix

1. **Aligned end-to-end baseline:** UniRL vs VeRL-Omni for SD3.5 FlowGRPO on the same 8-GPU node.
2. **AR baseline:** UniRL vs current veRL/NeMo-RL/another close framework on one Qwen3 GRPO workload.
3. **Execution modes:** trainside-colocated, external-colocated, separate synchronous, and separate async where each is supported.
4. **Scale:** strong scaling at fixed global work and weak scaling at fixed per-GPU work. Report communication topology.
5. **Breakdown:** data, wake/onload, weight sync, rollout, reward, advantage, replay/anchor, backward/optimizer, checkpoint, idle/barrier.
6. **Memory/resources:** per-role peak allocated/reserved GPU memory, CPU/RAM, GPU utilization, and GPU-hours to a fixed eval threshold.
7. **Transfer/publication:** payload bytes and time for trajectory tensors and for LoRA/full-weight publication.

### 6.3 Required ablations

- Typed trajectory IR overhead vs an equivalent dense/local handoff for one controlled step.
- Whole-tree split/concat vs naive row sharding on a multi-stage grouped trajectory (correctness + overhead).
- Trainside vs external rollout with matched kernels where possible.
- Colocate vs separate; FSDP offload on/off; reward colocate/separate.
- `colocate_store` vs `gpu_store`; `transfer_queue`/Mooncake only when hardware supports a fair run.
- LoRA tensor/IPC/NCCL path vs full-weight publication when both are valid.
- Sync vs async, varying inflight and `weight_sync_interval`; report actual lag distribution and rejection/drop rates.
- AR token balancing and diffusion replay anchor source, if these appear in headline workloads.

### 6.4 Artifact checklist for every reported run

- UniRL commit and dirty diff; baseline commit/submodule commit.
- Complete resolved Hydra config and launch command.
- Python/CUDA/driver/PyTorch/engine versions; GPU model/count, interconnect, node count.
- Dataset snapshot/hash and preprocessing command.
- Model/reward checkpoint revisions and access date.
- Raw console logs, structured per-step metrics, W&B export or equivalent, and parser version.
- Checkpoint/eval outputs and generated-sample manifest.
- Warmup exclusion rule, failure/retry accounting, number of runs/seeds, and statistical aggregation.

## 7. Immediate author-only inputs eventually required

These do not block drafting but block a publishable arXiv release:

- Author names, affiliations, ordering, and corresponding author.
- Which private clusters/checkpoints/reward endpoints may be named and released.
- Compute budget for the minimum RQ1/RQ2 matrix.
- Whether the measured benchmark artifacts referenced in repository prose can be recovered; if not, all numbers must be rerun.
- Preferred licensing/release plan for configs, logs, and processed benchmark data.

## 8. Current manuscript status

- TMLR style is retained in `preprint` mode so the PDF is de-anonymized from the venue while remaining easy to convert for a later TMLR submission.
- Three conceptual figures are newly reconstructed in TikZ from code and the claim-to-evidence plan; none reuse legacy architecture art.
- Introduction, requirements, design, execution, instantiations, evaluation protocol, related work, limitations, references, and appendices have been rewritten from scratch.
- A reproducible source-importing CPU evidence harness and checksummed raw artifact bundle now support a limited subset of RQ3; the paper includes its generated table rows with an explicit non-GPU scope statement.
- `EXPERIMENT_RUNBOOK.md` now translates RQ1--RQ4 into a GPU-cluster protocol and records the stock sync/async semantic mismatches plus measurement-instrumentation gaps before any future run is admitted.
- The revised 15-page draft compiles with Tectonic and has been rendered and inspected page by page after integrating the experiment preflight material; no unresolved references, clipping, overlap, or table overflow was observed.
- All missing empirical values use explicit `TODO[data]` markers; no placeholder number should resemble a result.
