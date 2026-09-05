# GPU Experiment Handoff for the UniRL Paper

Last updated: 2026-09-05

This is the operational handoff for the next Codex or researcher working on a
GPU machine. It is intentionally written before the GPU runs. It contains
prespecified settings, commands, expected signals, admission gates, and paper
destinations, but no invented measurements.

The paper's job is not to show that UniRL has many recipes. It must establish two
primary claims and two explanatory claims:

1. **Training validity (RQ1):** the same trajectory-centric system produces
   credible learning for one AR and one diffusion workload.
2. **Systems efficiency (RQ2):** under matched work and hardware, UniRL has
   competitive end-to-end cost, with transparent phase and resource accounting.
3. **Abstraction value (RQ3):** lineage-aware trajectory handling selects a real
   cross-stage objective; a matched UniRL topology pair and transport measurements
   bound the cost of preserving and moving that structure.
4. **Bounded staleness (RQ4):** overlap is useful only in the region where it
   improves time-to-quality without unacceptable policy lag or discarded work.

Do not silently tune a failed experiment until it looks good. Preserve the failed
run, diagnose it, and register any changed configuration as a new experiment.

## 1. Fixed source state and repository roles

- UniRL source audited by the manuscript:
  `f5d710406b215bb7a0b387fdd37e4d4778b92338`.
- Manuscript repository: this `UniRLReport` checkout. Record its commit at run
  time; do not assume the commit in this handoff is still the paper head.
- VeRL-Omni submodule pin for the SD3.5 systems pair:
  `01c87ee595874c313f9f296525fb5b4389678451`.
- The AR veRL baseline commit is deliberately blank until cloned and parity
  audited: `<VERL_COMMIT>`.

The two repositories serve different purposes. Run training and benchmark code
from the UniRL source checkout. Put immutable raw run artifacts outside both Git
worktrees, then copy only compact generated tables/plots or public artifacts into
the paper repository.

At the start of every allocation:

```bash
export UNIRL_ROOT=/path/to/UniRL
export PAPER_ROOT=/path/to/UniRLReport
export ARTIFACT_ROOT=/shared/path/unirl-paper-artifacts
mkdir -p "$ARTIFACT_ROOT"
cd "$UNIRL_ROOT"

git rev-parse HEAD
git status --short
git diff --binary > "$ARTIFACT_ROOT/preflight-unirl-dirty.patch"
git submodule status --recursive
git -C "$PAPER_ROOT" rev-parse HEAD
git -C "$PAPER_ROOT" status --short
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version,pstate,power.limit --format=csv
```

If the UniRL tree is not exactly the audited commit, archive the commit and diff
and re-run the source/config audit. A dirty tree is allowed only when its patch is
part of the run artifact.

## 2. Experiment map and execution order

| ID | Priority | Experiment | Why it is needed | Paper destination |
|---|---:|---|---|---|
| E0 | P0 gate | Environment, config parity, replay/publication correctness, instrumentation | Prevents invalid quality or speed claims | Evaluation preflight; Appendix `Correctness status` |
| E1 | P0 | Qwen3-4B-Base GRPO on DAPO-Math, >=3 seeds | Establishes AR training validity | RQ1 AR; RQ1 table; learning-curve figure |
| E2 | P0 | SD3.5-Medium FlowGRPO + independent image eval, >=3 seeds | Establishes diffusion training validity and collapse guards | RQ1 diffusion; RQ1 table; learning-curve figure |
| E3 | P0 | UniRL vs VeRL-Omni aligned SD3.5 performance pair | Supplies the cleanest end-to-end systems comparison | RQ2 table; phase-breakdown figure |
| E4 | P0 | Prompt enhancement, `prompt` vs `rewrite` grouping | Directly tests whether lineage selects a meaningful cross-stage objective | RQ3 lineage/enablement result |
| E5 | P0 minimum / P2 extensions | IR/transport costs plus matched colocated/separate UniRL rows | Quantifies abstraction and execution costs | RQ3 transport/topology table and scaling plot |
| E6 | P1 | Matched sync/async AR sweep | Tests throughput-lag-quality trade-off | RQ4 Pareto plot and async table |

Order is mandatory: E0 -> E1/E2 -> E3/E4 -> minimum E5 -> E6/E5 extensions. E3 may run as
soon as its E0 systems gates pass; it does not have to wait for all E1/E2 seeds.
If compute is constrained, complete primary E0-E5 before broadening models or algorithms:
E4 directly tests lineage, and the minimum E5 slice bounds the abstraction's cost.

## 3. Evidence directory contract

Every process group gets one immutable directory:

```text
<ARTIFACT_ROOT>/<experiment>/<system>/<run_id>/
  manifest.json
  source/{commit.txt,dirty.patch,submodules.txt}
  config/{command.txt,resolved.yaml,parity.tsv,environment.txt}
  inputs/{models.json,datasets.json,evaluators.json}
  logs/{stdout.log,stderr.log,wandb/,gpu_telemetry.csv}
  metrics/{steps.jsonl,phases.jsonl,memory.jsonl,async.jsonl,failures.jsonl}
  checkpoints/{index.json,checkpoint-*}
  evaluation/<checkpoint>/<benchmark>/...
  checksums.sha256
```

`manifest.json` must include the experiment ID, system, run ID, seed, start/end
time, allocation, GPU model/count, nodes/interconnect, clocks/power settings,
driver/CUDA/PyTorch/engine versions, exact input hashes, expected/completed work,
warm-up exclusion, failures/retries, and parser commit.

The following paper-setting fields are unknown until the GPU allocation and must
remain blank rather than guessed:

- GPU model: `<GPU_MODEL>`
- node count and GPUs/node: `<NODES>` / `<GPUS_PER_NODE>`
- NVLink/PCIe/InfiniBand topology: `<INTERCONNECT>`
- driver and host CUDA compatibility stack: `<DRIVER>` / `<CUDA_RUNTIME>`
- exact local model snapshot revisions and hashes: `<MODEL_REVISIONS>`
- final dataset/evaluator content hashes: `<INPUT_HASHES>`
- AR veRL baseline commit: `<VERL_COMMIT>`
- W&B entity or offline artifact location: `<LOGGER_DESTINATION>`

## 4. Environments: keep SGLang and vLLM-Omni separate

UniRL's extras intentionally conflict: SGLang uses torch 2.11 + CUDA 13.0,
whereas vLLM/vLLM-Omni uses torch 2.11 + CUDA 12.9. Never install both engines
in the same venv.

SGLang environment for E1 and E6:

```bash
cd "$UNIRL_ROOT"
uv venv --python 3.12 --seed .venv-sglang
source .venv-sglang/bin/activate
uv pip install -e ".[sglang,train,infer]" --prerelease=allow
python -c "import torch,sglang; print(torch.__version__, sglang.__version__)"
```

vLLM-Omni environment for E2 and E3:

```bash
cd "$UNIRL_ROOT"
uv venv --python 3.12 --seed .venv
source .venv/bin/activate
export VLLM_USE_PRECOMPILED=1
uv pip install -e ".[vllm,train,infer]"
python -c "import torch,vllm; import importlib.metadata as m; print(torch.__version__, vllm.__version__, m.version('vllm-omni'))"
```

Use the vLLM environment for the trainside prompt-enhancement E4 unless a smaller
purpose-built environment has been validated. Follow `INSTALL.md` if the cluster
needs CUDA-13 forward-compat libraries for SGLang.

Archive `python -m pip freeze`, `uname -a`, `/etc/os-release`, `ldd --version`,
`nvidia-smi`, and the output of the import/version probes in `environment.txt`.

## 5. Inputs and one-time preparation

Prepare DAPO-Math and AIME JSONL:

```bash
cd "$UNIRL_ROOT"
python datasets/dapo_math/prepare_dapo_math.py --out-dir data/dapo_math
sha256sum data/dapo_math/train.jsonl data/dapo_math/aime_eval.jsonl
```

Required model paths:

```bash
export QWEN3_PATH=/local/path/Qwen3-4B-Base
export SD35=/local/path/stable-diffusion-3.5-medium
export LLM_MODEL=/local/path/Qwen3-0.6B
export DATA_PATH="$UNIRL_ROOT/data/dapo_math/train.jsonl"
export EVAL_DATA_PATH="$UNIRL_ROOT/data/dapo_math/aime_eval.jsonl"
```

Pin snapshot revisions, not only model names. Confirm `config.json` or
`model_index.json`, tokenizer files, PickScore model/processor revisions, and
prompt-file hashes. Do not train from a network-mounted model if one system uses
a warm node-local copy and the other does not.

## 6. E0: gates before any admissible measurement

### 6.1 Resolve configurations without training

Hydra can print the effective job config. Run the exact formal overrides and save
the result before the training command. Example for E1 seed 11:

```bash
source "$UNIRL_ROOT/.venv-sglang/bin/activate"
cd "$UNIRL_ROOT"
python -m unirl.train_ar \
  --config-name=ar/qwen3_grpo_4b_base_dapo_sglang \
  --cfg job --resolve \
  num_devices=32 \
  data_source.args.run.seed=11 \
  ++rollout.config.engine_kwargs.random_seed=11 \
  logging.project_name=unirl-paper logging.run_name=E1-unirl-qwen3-grpo-s11 \
  +save_interval=200 +save_dir=/abs/path/checkpoints +save_mode=full \
  > /abs/path/resolved.yaml
```

Run `DRY_RUN=1` through the selected launcher as a second command-construction
check. Resolved sync/async and UniRL/baseline configs must also be converted into
`parity.tsv`; source YAML comments are not parity evidence.

### 6.2 Required correctness tests

Before E1-E6 results are interpreted, verify and archive:

1. rollout-engine versus train-side replay log probabilities/transition
   statistics before the first update, with a stated tolerance;
2. the same parity check after every weight-publication path used;
3. no optimizer batch mixes behavior versions when the estimator assumes one;
4. checkpoint/eval starts only at an empty async boundary, and resume restores
   optimizer version before new rollout;
5. generated/replayed tokens or pixels, scored samples, SDE steps, and committed
   optimizer updates agree across compared systems;
6. fixed seeds actually change across seed IDs and reproduce when the same seed
   is rerun.

SGLang 0.5.12.post1 accepts `random_seed` as a server argument; UniRL forwards
unknown engine settings through `rollout.config.engine_kwargs`, hence the explicit
override above. The trainside AR sampler used by E4 currently has no equivalent
top-level seed field. Add and verify deterministic seed plumbing before calling E4
a multi-seed result; otherwise label it as repeated runs with uncontrolled sampler
state.

### 6.3 Instrumentation blockers

Do not start formal E3 or E6 until structured output includes:

- credit, replay/anchor, backward, optimizer, checkpoint, quiesce, and publication
  barrier durations;
- per-step buffer roots/prompts in admission, ready, carry, suspended, completed,
  rejected, discarded, and retry states;
- sampled, redacted real trajectory records containing root/part/parent IDs,
  modality, shapes/dtypes, output versions, reward components, advantages, and
  replay index metadata, without private decoded payloads;
- an E2 within-prompt LPIPS diversity evaluator and an E4 lineage join that can
  score each image against its root prompt as well as its immediate rewrite;
- external GPU utilization/power telemetry synchronized to the run clock.

The current logger already has end-to-end step time, six coarse phases, rollout
diagnostics, GPU peak memory, and policy versions. Missing fields must not be
reconstructed later from screenshots or isolated console lines.

### 6.4 Smoke runs

Smoke runs prove execution only and never enter result tables. Adjust batch sizes
to satisfy the allocation's divisibility rules.

AR smoke on 8 GPUs (32 samples / rollout, four updates):

```bash
source "$UNIRL_ROOT/.venv-sglang/bin/activate"
cd "$UNIRL_ROOT"
ENTRY=train_ar GPUS_PER_NODE=8 REPORT_TO_WANDB=false \
QWEN3_PATH="$QWEN3_PATH" DATA_PATH="$DATA_PATH" EVAL_DATA_PATH="$EVAL_DATA_PATH" \
bash examples/run_experiment_single_node.sh ar/qwen3_grpo_4b_base_dapo_sglang \
  num_rollouts=2 batch_size=8 sampling.samples_per_prompt=4 eval_interval=0 \
  sampling.max_new_tokens=1024 rollout.config.max_new_tokens=1024 \
  stack.micro_planner.token_budget=2048 +save_interval=2 \
  +save_dir="$ARTIFACT_ROOT/E0/ar-smoke/checkpoints" +save_mode=full
```

Diffusion smoke on 8 GPUs (16 images / rollout, two updates):

```bash
source "$UNIRL_ROOT/.venv/bin/activate"
cd "$UNIRL_ROOT"
ENTRY=train_diffusion GPUS_PER_NODE=8 REPORT_TO_WANDB=false PRETRAINED_MODEL="$SD35" \
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_vllmomni \
  +num_rollouts=2 batch_size=4 sampling.samples_per_prompt=4 \
  sampling.height=256 sampling.width=256 sampling.num_inference_steps=4 \
  sampling.scheduler.num_sde_steps=1 stack.micro_batch_size=1 \
  +save_interval=2 +save_dir="$ARTIFACT_ROOT/E0/sd35-smoke/checkpoints" +save_mode=adapter
```

If these smoke geometries do not fit the actual GPU type, record the failure and
change only the smoke configuration. Do not silently transfer smoke-only memory
changes into a formal row.

## 7. E1: AR training validity

### 7.1 Prespecified setting

- model: Qwen3-4B-Base, full fine-tuning, fp32 parameter storage/bf16 compute;
- data: DAPO-Math-17k JSONL; 64 prompts x 8 samples = 512 trajectories/rollout;
- generation: SGLang TP=1, thinking enabled, temperature/top-p 1.0, top-k off,
  maximum 8192 new tokens;
- objective: textbook GRPO, group-std normalization, symmetric clip 0.2,
  `seq-mean-token-mean`, no stated KL term;
- learner: four disjoint optimizer updates/rollout, 10,240-token planner,
  AdamW lr 1e-6, weight decay 0.01;
- schedule: 800 rollouts, full-weight publication every rollout, internal AIME
  evaluation every 10 rollouts, full checkpoints every 200 rollouts;
- formal allocation: 32 GPUs total; GPU model/nodes/interconnect remain `<blank>`;
- independent seed IDs: 11, 22, 33, applied to data order and SGLang server RNG.

### 7.2 Formal command

Start or attach to the cluster's Ray head, then run once for each seed. On a
single 32-GPU node, the single-node launcher is sufficient; on multiple nodes use
the site's supported multi-node launcher with the same Hydra overrides.

```bash
export SEED=11
export RUN_ID=E1-unirl-qwen3-grpo-s${SEED}
export RUN_DIR="$ARTIFACT_ROOT/E1/unirl/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

source "$UNIRL_ROOT/.venv-sglang/bin/activate"
cd "$UNIRL_ROOT"
REPORT_TO_WANDB=true WANDB_PROJECT=unirl-paper WANDB_RUN_NAME="$RUN_ID" \
QWEN3_PATH="$QWEN3_PATH" DATA_PATH="$DATA_PATH" EVAL_DATA_PATH="$EVAL_DATA_PATH" \
python -m unirl.train_ar \
  --config-name=ar/qwen3_grpo_4b_base_dapo_sglang \
  num_devices=32 data_source.args.run.seed="$SEED" \
  ++rollout.config.engine_kwargs.random_seed="$SEED" \
  logging.project_name=unirl-paper logging.run_name="$RUN_ID" \
  +save_interval=200 +save_dir="$RUN_DIR/checkpoints" +save_mode=full \
  >"$RUN_DIR/logs/stdout.log" 2>"$RUN_DIR/logs/stderr.log"
```

Repeat with seeds 22 and 33. A retry keeps the same seed but gets a new run ID and
links to the failed run. The formal budget is fixed after the first seed starts.

### 7.3 Frozen-checkpoint evaluation

Evaluate the base plus saved checkpoints 200/400/600/800. UniRL full-training
checkpoints are resume artifacts, so export them to a standard Hugging Face folder:

```bash
python -m unirl.tools.export_full \
  --checkpoint "$RUN_DIR/checkpoints/checkpoint-800" \
  --library transformers --base "$QWEN3_PATH" \
  --output "$RUN_DIR/checkpoints/hf-800"
```

Serve one frozen checkpoint in the SGLang environment:

```bash
python -m sglang.launch_server \
  --model-path "$RUN_DIR/checkpoints/hf-800" \
  --host 127.0.0.1 --port 30000 --tp-size 1 --random-seed "$SEED"
```

Then run the registered fixed protocols (repeat `-b`; a comma-separated name is
not valid):

```bash
python -m benchmarks.run \
  -b text/math500 -b text/aime24 -b text/aime25 \
  --endpoint http://127.0.0.1:30000 \
  --tag "$RUN_ID-u3200" --out "$RUN_DIR/evaluation"
```

At 800 rollouts and four updates/rollout, the final optimizer update is 3200.
Checkpoint selection is best among the prespecified saved checkpoints according to
the internal evaluation; report both best and final, never best alone.

### 7.4 AR reference baseline gate

The current official veRL Qwen3-4B FSDP example is a scaffold, not an admissible
baseline as written: it uses a different model variant, GSM8K, group size, token
budget, and KL/objective settings. Before any head-to-head AR row:

```bash
git clone https://github.com/verl-project/verl.git third_party/verl
git -C third_party/verl checkout <VERL_COMMIT>
cp third_party/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
  experiments/baselines/run_verl_qwen3_4b_dapo_aligned.sh
```

The copied launcher must align Qwen3-4B-Base, DAPO-Math-17k, 64 x 8 = 512
samples, 8192 response tokens, temperature/top-p 1, top-k off, math verifier,
no KL term, symmetric clip 0.2, `seq-mean-token-mean`, lr 1e-6, weight decay
0.01, four optimizer updates, token packing, checkpoint cadence, and external
evaluation. If any estimator or effective-work field cannot be matched, mark the
AR baseline **blocked** and omit the head-to-head row; do not substitute a nearby
recipe and call it aligned.

### 7.5 Expected signal and failure meaning

The hypothesis is that external MATH-500/AIME performance improves from the base
across most seeds while reward, length, truncation, group variance, ratio, and clip
diagnostics remain stable. Comparable reference behavior would establish that the
generic trajectory path does not prevent credible AR training.

- Divergence, NaNs, replay mismatch, ratio drift before learning, zero-advantage
  collapse, or publication mismatch: Codex/engineer diagnoses implementation,
  configuration, or numerical correctness first.
- Stable execution but no external improvement after all correctness gates pass:
  preserve the result and escalate to a human research decision about the recipe,
  reward, budget, or claim. Do not auto-tune the headline run.
- Reward improves but external eval does not: treat as reward overfitting; the RQ1
  claim is not established.

## 8. E2: diffusion training validity

### 8.1 Prespecified setting

Use the same aligned SD3.5 setting later used in E3: SD3.5-Medium, 48 prompts x
16 images = 768 samples/rollout, 384 x 384 training resolution, ten denoising
steps, three SDE steps in the first half, eta 0.8, guidance 1, distinct initial
noise, PickScore, LoRA r32/alpha64 on the same eight attention projections, two
optimizer updates, microbatch 8, lr/weight decay 1e-4, clip 1e-5, and publication
every rollout. Prespecify 300 rollouts and seeds 11/22/33. The allocation is one
8-GPU node; exact hardware remains `<blank>`.

### 8.2 Formal command

```bash
export SEED=11
export RUN_ID=E2-unirl-sd35-flowgrpo-s${SEED}
export RUN_DIR="$ARTIFACT_ROOT/E2/unirl/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

source "$UNIRL_ROOT/.venv/bin/activate"
cd "$UNIRL_ROOT"
SD35="$SD35" STEPS=300 REPORT_TO_WANDB=true \
WANDB_PROJECT=unirl-paper WANDB_RUN_NAME="$RUN_ID" \
bash benchmarks/speed_benchmarks/verl_omni/run_unirl_sd35_aligned.sh \
  data_source.args.run.seed="$SEED" sampling.seed="$SEED" \
  logging.report_to_wandb=true logging.project_name=unirl-paper logging.run_name="$RUN_ID" \
  +save_interval=50 +save_dir="$RUN_DIR/checkpoints" +save_mode=adapter \
  >"$RUN_DIR/logs/stdout.log" 2>"$RUN_DIR/logs/stderr.log"
```

Repeat with seeds 22 and 33. If a 300-rollout budget is changed before the first
formal run for a documented compute reason, update the paper and this file first.
Never extend only a seed that looks promising.

### 8.3 External evaluation and guard metrics

Evaluate the base and checkpoints 50/100/150/200/250/300. `benchmarks.run`
automatically exports a UniRL LoRA checkpoint passed via `--lora`.

GenEval2 compositional evaluation with the registered local Qwen3-VL scorer:

```bash
python -m benchmarks.run -b image/geneval2 \
  --ckpt "$SD35" --lora "$RUN_DIR/checkpoints/checkpoint-300" \
  --out "$RUN_DIR/evaluation" --local-geneval2 --seed "$SEED"
```

Preference views over PartiPrompts (requires a pinned reward service):

```bash
export REWARD_SERVICE_URL=http://<reward-host>:8080
python -m benchmarks.run -b image/preference \
  --ckpt "$SD35" --lora "$RUN_DIR/checkpoints/checkpoint-300" \
  --out "$RUN_DIR/evaluation" --reward-url "$REWARD_SERVICE_URL" --seed "$SEED"
```

Run the same commands without `--lora` for the base. Archive images, prompt/sample
indices, seeds, scores, evaluator errors, and summaries. Report HPSv3 and ImageReward
as views independent of the PickScore training reward; the registry's PickScore
output is a useful consistency check but is not independent evidence.

The preregistered diversity guard is within-prompt mean pairwise LPIPS with AlexNet
features over 16 fixed image seeds per evaluation prompt. Use identical prompts and
image seeds for base and checkpoints; report the prompt-level paired relative change
and 95% bootstrap interval. A greater than 10% relative decline from base is material
diversity loss. Before formal E2, implement and smoke-test a stable CLI such as:

```bash
python experiments/evaluation/compute_lpips_diversity.py \
  --base-images "$RUN_DIR/evaluation/base/diversity/images" \
  --candidate-images "$RUN_DIR/evaluation/checkpoint-300/diversity/images" \
  --group-manifest "$RUN_DIR/evaluation/diversity_groups.jsonl" \
  --network alex --bootstrap 10000 --seed 20260905 \
  --output "$RUN_DIR/evaluation/checkpoint-300/lpips_diversity.json"
```

This is an interface requirement, not a checked-in executable at the audited commit.
LPIPS guards against within-prompt collapse; do not present it as image quality.

### 8.4 Expected signal and failure meaning

The hypothesis is that PickScore improves while GenEval2, independent preference
scores, and the prespecified diversity guard do not collapse. This is a stronger
claim than reward improvement alone.

- NaNs, invalid SDE indices, rollout/replay transition mismatch, LoRA checksum
  mismatch, or missing images: Codex/engineer fixes correctness and reruns under a
  new run ID.
- PickScore rises while independent metrics/diversity fall materially: preserve
  the result as reward overfitting; a human decides whether to change the reward
  or narrow the claim.
- No learning after parity passes: human research decision. Do not silently switch
  model, reward, eta, or SDE window.

## 9. E3: aligned end-to-end SD3.5 systems comparison

Initialize the pinned VeRL-Omni submodule and build its environment according to
its pinned installation guide:

```bash
cd "$UNIRL_ROOT"
git submodule update --init benchmarks/speed_benchmarks/verl_omni/upstream
git -C benchmarks/speed_benchmarks/verl_omni/upstream rev-parse HEAD
python benchmarks/speed_benchmarks/verl_omni/make_pickscore_parquet.py
```

Run each system alone on the same reserved 8-GPU node. Use three process-level
replicates per configuration, 30 steps each, discard exactly the first five timing
observations, and retain at least 20 observations. Keep caches/clocks consistent
and record any failed step rather than restarting it out of the distribution.

UniRL aligned row:

```bash
source "$UNIRL_ROOT/.venv/bin/activate"
cd "$UNIRL_ROOT"
SD35="$SD35" STEPS=30 REPORT_TO_WANDB=true \
bash benchmarks/speed_benchmarks/verl_omni/run_unirl_sd35_aligned.sh \
  logging.report_to_wandb=true logging.run_name=E3-unirl-aligned-r1 \
  >"$ARTIFACT_ROOT/E3/unirl-aligned-r1.log" 2>&1

python benchmarks/speed_benchmarks/parse_perf.py \
  "$ARTIFACT_ROOT/E3/unirl-aligned-r1.log" \
  --skip 5 --samples-per-step 768 --gpus 8
```

VeRL-Omni backend-aligned row:

```bash
cd "$UNIRL_ROOT"
SD35="$SD35" STEPS=30 ATTN=sdpa \
bash benchmarks/speed_benchmarks/verl_omni/run_verlomni_sd35_aligned.sh \
  >"$ARTIFACT_ROOT/E3/verlomni-sdpa-r1.log" 2>&1

python benchmarks/speed_benchmarks/verl_omni/parse_verl_timing.py \
  "$ARTIFACT_ROOT/E3/verlomni-sdpa-r1.log" \
  --skip 5 --samples-per-step 768 --gpus 8
```

VeRL-Omni best-valid attention row:

```bash
SD35="$SD35" STEPS=30 ATTN=fa3 \
bash benchmarks/speed_benchmarks/verl_omni/run_verlomni_sd35_aligned.sh \
  >"$ARTIFACT_ROOT/E3/verlomni-fa3-r1.log" 2>&1
```

Only call a UniRL configuration “best valid” if it was registered before looking
at the final comparison. The launchers align effective work, but inherent engine,
log-probability, and SDE-kernel differences remain and must be disclosed. The
current console parsers are cross-checks; paper phase plots must come from the
structured instrumentation required by E0.

Expected result: UniRL is competitive or faster and its phase breakdown explains
why. If UniRL is slower, first audit parity, cache state, failures, utilization,
and phase accounting. If the gap is genuine, report it and use the breakdown to
bound the contribution; do not remove an expensive phase from only one system.

## 10. E4: lineage-aware cross-stage prompt enhancement

Use `examples/pe/pe_trainside_pickscore_frozenllm_promptgroup.yaml`: eight roots x
four frozen-Qwen3 rewrites x eight SD3 images = 256 images/rollout. Only the SD3
side trains. The primary condition groups all 32 descendant images by original
prompt; the executable control changes only `diffusion_group_scope=rewrite`, which
groups eight images under each rewritten prompt.

Do not implement a deliberately corrupted lineage as the quality baseline. Invalid
lineage belongs in E0 fail-fast tests; prompt-scope versus rewrite-scope is the
scientifically interpretable ablation.

After adding redacted trajectory dumping and deterministic trainside-AR seed
plumbing, run seeds 11/22/33 for both scopes:

```bash
export SCOPE=prompt
export SEED=11
export RUN_ID=E4-pe-${SCOPE}-s${SEED}
export RUN_DIR="$ARTIFACT_ROOT/E4/$SCOPE/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

source "$UNIRL_ROOT/.venv/bin/activate"
cd "$UNIRL_ROOT"
ENTRY=train_pe GPUS_PER_NODE=8 REPORT_TO_WANDB=true \
WANDB_PROJECT=unirl-paper WANDB_RUN_NAME="$RUN_ID" \
PRETRAINED_MODEL="$SD35" LLM_MODEL="$LLM_MODEL" \
bash examples/run_experiment_single_node.sh \
  pe/pe_trainside_pickscore_frozenllm_promptgroup \
  diffusion_group_scope="$SCOPE" data_source.args.run.seed="$SEED" \
  sampling.diffusion.seed="$SEED" num_rollouts=300 \
  logging.report_to_wandb=true logging.project_name=unirl-paper logging.run_name="$RUN_ID" \
  +save_interval=50 +save_dir="$RUN_DIR/checkpoints" +save_mode=adapter \
  >"$RUN_DIR/logs/stdout.log" 2>"$RUN_DIR/logs/stderr.log"
```

Expected result: both conditions preserve complete parent/child traces and produce
the intended advantage groups. The current PickScore request uses the immediate
rewrite as text conditioning, so it cannot by itself establish preservation of the
root user's intent. Build a lineage-keyed evaluation manifest with one row per image,
including `root_prompt`, `rewrite`, image path/hash, root/part IDs, scope, seed, and
checkpoint. Score the same image once against its root prompt and once against its
rewrite, using frozen evaluator revisions. A stable interface should be:

```bash
python experiments/evaluation/evaluate_pe_lineage.py \
  --trace "$RUN_DIR/metrics/trajectory_samples.jsonl" \
  --images "$RUN_DIR/evaluation/images" \
  --evaluators hpsv3,imagereward \
  --output "$RUN_DIR/evaluation/lineage_alignment.jsonl"
```

This evaluator is an interface requirement, not an existing executable. Prompt-scope
grouping may improve root-prompt alignment; that direction is a hypothesis, not an
invariant. Rewrite-conditioned reward and root-conditioned evaluation must be shown
separately.
If group membership or reward propagation is wrong, Codex/engineer fixes it. If
both are correct but root-conditioned quality is equal or worse, retain the result
and narrow the enablement claim to semantic expressiveness/integration rather than
quality.

## 11. E5: abstraction, transport, publication, and topology

The minimum topology evidence is executable without the new transport harness.
Reuse E3's colocated UniRL row as T0, then run T1 with the same total eight GPUs
and effective work but separate four-GPU training and rollout slabs:

```bash
export RUN_ID=E5-sd35-separate-r1
export RUN_DIR="$ARTIFACT_ROOT/E5/topology/$RUN_ID"
mkdir -p "$RUN_DIR/logs"

source "$UNIRL_ROOT/.venv/bin/activate"
cd "$UNIRL_ROOT"
PRETRAINED_MODEL="$SD35" REPORT_TO_WANDB=true \
python -m unirl.train_diffusion \
  --config-name=diffusion/sd3/sd3_vllmomni_lora_separate \
  num_devices=8 train_fraction=0.5 batch_size=48 \
  sampling.samples_per_prompt=16 sampling.height=384 sampling.width=384 \
  sampling.eta=0.8 backend.optimizer_cfg.learning_rate=1e-4 \
  backend.optimizer_cfg.weight_decay=1e-4 algorithm.clip_range=1e-5 \
  stack.micro_batch_size=8 +num_rollouts=30 \
  +logging.report_to_wandb=true +logging.project_name=unirl-paper \
  +logging.run_name="$RUN_ID" \
  >"$RUN_DIR/logs/stdout.log" 2>"$RUN_DIR/logs/stderr.log"
```

Run three process repetitions, remove the same first five observations, and reuse
all E3 instrumentation/parity rules. T0 uses eight time-shared train/rollout GPUs
and local LoRA publication; T1 uses four training plus four resident rollout GPUs
and remote LoRA publication. This is an allocation-level topology comparison, not
a pure communication ablation: training DP, residency, and publication path change
together, so the paper must show the phase breakdown and may not attribute the
entire difference to `layout` alone.

The existing `experiments/trajectory_ir/run_cpu_evidence.py` is only a local
regression. There is no checked-in end-to-end GPU transport harness at the audited
commit. Before E5, implement and review a harness with a stable CLI such as:

```bash
python experiments/trajectory_ir/run_gpu_transport_evidence.py \
  --backend <colocate_store|gpu_store|transfer_queue> \
  --placement <same_gpu|cross_gpu|cross_node> \
  --roots 8,32,128 --branch-factor 1,4,8 \
  --payload-mib 1,16,256 --warmup 10 --repetitions 100 \
  --output "$ARTIFACT_ROOT/E5/<run_id>"
```

This command is a required interface specification, not an existing executable.
The harness must report structure-operation time separately from materialization,
reference metadata bytes, dense bytes avoided at the driver, driver/worker peak
RSS and GPU memory, transfer latency/bandwidth, and end-to-end correctness hashes.
TransferQueue/Mooncake is omitted if the cluster lacks a valid RDMA configuration.

Expected result: structural metadata overhead is small relative to model work and
`TensorRef` avoids dense driver materialization. If metadata/span explosion occurs
for strided selection, report the regime and optimize only in a separately named
follow-up. Genuine overhead belongs in RQ3, even if unfavorable.

## 12. E6: matched bounded-staleness sweep

The stock sync and async AR recipes are not comparable. The async recipe must be
overridden to match the sync estimator and learner:

```text
normalize_adv_by_std=true
algorithm.clip_range_high=null
algorithm.loss_agg_mode=seq-mean-token-mean
bundle.config.attn_implementation=flex_attention
stack.micro_planner._target_=unirl.train.stack.TokenBudgetPlanner
stack.micro_planner.token_budget=10240
```

Use 32 total GPUs for both sync and async; disaggregated points fix
`train_fraction=0.5` and `per_worker_inflight=14`. Keep 64 x 8 samples, four
updates, data/sampling, reward, and eval cadence identical. The configured
`buffer_max_staleness` unit is a consumed rollout batch, not an optimizer version;
one batch here performs four optimizer updates.
After a short capacity pilot, run the following registered primary points for the
full 800-rollout budget and seeds 11/22/33:

| Point | Trainer/topology | max inflight | publish interval (batches) | max lag (batches) | Purpose |
|---|---|---:|---:|---:|---|
| S0 | sync/colocated | n/a | 1 | 0 | total-resource reference |
| D0 | async/disaggregated | 1 | 1 | 0 | same-topology no-overlap control |
| A1 | async/disaggregated | 2 | 1 | 1 | isolate onset of overlap |
| A2 | async/disaggregated | 2 | 1 | 2 | isolate lag allowance |
| A3 | async/disaggregated | 2 | 2 | 2 | isolate publication cadence |
| A4 | async/disaggregated | 2 | 2 | 4 | aggressive but bounded point |

Only after the primary sweep may `max_inflight={1,4}` be added at one fixed
publication/lag setting. Characterize natural generation latency by prompt and
response-length strata. A synthetic delay injector is a secondary stress test and
must not replace primary end-to-end measurements.

Async command template:

```bash
export POINT=A2
export SEED=11
export MAX_INFLIGHT=2
export SYNC_INTERVAL=1
export MAX_STALENESS=2
export RUN_ID=E6-${POINT}-s${SEED}
export RUN_DIR="$ARTIFACT_ROOT/E6/$POINT/$RUN_ID"
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/checkpoints"

source "$UNIRL_ROOT/.venv-sglang/bin/activate"
cd "$UNIRL_ROOT"
QWEN3_PATH="$QWEN3_PATH" DATA_PATH="$DATA_PATH" EVAL_DATA_PATH="$EVAL_DATA_PATH" \
REPORT_TO_WANDB=true WANDB_PROJECT=unirl-paper WANDB_RUN_NAME="$RUN_ID" \
python -m unirl.train_async_ar \
  --config-name=ar/qwen3_grpo_4b_base_dapo_sglang_async \
  num_devices=32 train_fraction=0.5 \
  normalize_adv_by_std=true algorithm.clip_range_high=null \
  algorithm.loss_agg_mode=seq-mean-token-mean \
  ++bundle.config.attn_implementation=flex_attention \
  ++stack.micro_planner._target_=unirl.train.stack.TokenBudgetPlanner \
  ++stack.micro_planner.token_budget=10240 \
  max_inflight="$MAX_INFLIGHT" per_worker_inflight=14 \
  weight_sync_interval="$SYNC_INTERVAL" \
  buffer_max_staleness="$MAX_STALENESS" \
  data_source.args.run.seed="$SEED" \
  ++rollout.config.engine_kwargs.random_seed="$SEED" \
  logging.project_name=unirl-paper logging.run_name="$RUN_ID" \
  +save_interval=200 +save_dir="$RUN_DIR/checkpoints" +save_mode=full \
  >"$RUN_DIR/logs/stdout.log" 2>"$RUN_DIR/logs/stderr.log"
```

Run S0 with the E1 sync command and the same seed/input/eval rules. Run D0 with the
async template using `MAX_INFLIGHT=1`, `SYNC_INTERVAL=1`, and `MAX_STALENESS=0`;
the manager's freshness gate prevents refilling across that boundary, so D0 is the
non-overlapped disaggregated control. Compare S0 versus D0 as allocation/residency,
and D0 versus A1 as the onset of overlap. Compare reward and external evaluation
both versus optimizer update and versus wall time. Report realized lag in both batch
and optimizer-update units, not configured maxima alone.

Expected result: moderate lag/overlap reduces idle time and improves
time-to-quality when generation has variable latency; aggressive lag may increase
rejection/discard or harm quality. If async is faster per step but worse in
time-to-quality, it does not support the headline claim. If buffer/version
invariants fail, Codex/engineer fixes the scheduler or instrumentation. If
invariants pass and no asynchronous point Pareto-improves on its relevant control,
keep the negative result and state the useful operating region is absent for this
workload/allocation.

## 13. Failure ownership and stopping rules

Codex/engineer may autonomously diagnose and fix:

- bad paths, missing dependencies, Hydra override mistakes, config-parity reports;
- deterministic shape/divisibility failures and obvious smoke-only OOM geometry;
- logging/parser bugs, missing structured fields, incorrect version accounting;
- rollout/replay, checkpoint/resume, weight-sync, trace-lineage, or checksum bugs.

Escalate to a human researcher before changing:

- optimizer, learning rate, reward, clipping, estimator, SDE schedule, model/task,
  formal training budget, or evaluation metric;
- an aligned-baseline definition that cannot be matched;
- a stable but unfavorable quality/system result after correctness passes;
- the paper's central claim or which result is designated primary.

Three identical infrastructure failures may justify pausing that point, but the
manifest must retain all attempts. An OOM or crash is not an excluded timing
sample; report failure-inclusive sensitivity.

## 14. Returning evidence to the paper repository

For each completed experiment, give the paper-side agent:

1. immutable artifact URI/path and checksum;
2. run IDs and seed IDs;
3. resolved configurations and parity table;
4. raw metric/evaluation locations;
5. parser/generator commit and exact regeneration command;
6. a one-paragraph result statement separating observation from interpretation;
7. anomalies, retries, exclusions, and whether every admission gate passed.

Paper mapping:

- E0 -> Appendix `Correctness status and remaining checks`, reproducibility
  statement, and any evaluation limitation;
- E1/E2 -> RQ1 table, learning curves, abstract headline only after both pass;
- E3 -> RQ2 systems table and phase breakdown; abstract throughput claim only
  after aligned artifacts pass;
- E4 -> RQ3 lineage/enablement result, root-versus-rewrite evaluation, and real trace example;
- E5 -> RQ3 minimum colocated/separate pair, transport results, and design-cost discussion;
- E6 -> RQ4 lag/time-to-quality plot, async limitations, and conclusion.

Never type a result manually into `main.tex`. Generate a TeX fragment or figure
from the archived bundle, review the diff, compile the paper, and inspect the PDF
page by page.
