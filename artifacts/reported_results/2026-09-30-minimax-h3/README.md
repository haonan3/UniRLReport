# MiniMax-H3 UniRL reproduction and veRL-Omni comparison

This directory records the MiniMax-H3 text-to-video-with-audio reproduction in
UniRL and a configuration-matched veRL-Omni control.

## Recipes

- UniRL source: [PR403 commit `d8e4aefa`](https://github.com/Tencent-Hunyuan/UniRL/commit/d8e4aefa7a266b229e05a82f7ac17bbac537e639)
- UniRL run: [`x8b0x8al`](https://wandb.ai/leviking98z-zhejiang-university/unirl-minimax-h3-t2va/runs/x8b0x8al)
- veRL-Omni run: [`646mvxxs`](https://wandb.ai/leviking98z-zhejiang-university/unirl-minimax-h3-t2va/runs/646mvxxs)
- Hardware: one node with 8 NVIDIA H20 GPUs per run
- Work unit: 8 prompts × 16 samples = 128 trajectories per step
- Sampling: 256 × 384, 124 frames, 10 denoising/SDE transitions
- Objective: FlowGRPO, one optimizer update, learning rate 1e-4
- Reward: ImageBind audio-video + CLAP text-audio
- Seed: 42

The actual configurations are preserved as [`unirl_recipe.yaml`](unirl_recipe.yaml)
and [`verl_omni_recipe.sh`](verl_omni_recipe.sh). The veRL-Omni recipe explicitly
matches the prompt batch, group size, canvas, frame count, SDE transitions,
optimizer, LoRA targets, reward functions, and seed.

## Training curve

![MiniMax-H3 training reward](curve.png)

Both runs learn under the matched recipe. UniRL's mean reward averaged over the
first/last 10 recorded steps was **0.194 / 0.403**; veRL-Omni's was
**0.212 / 0.405**.

[`curve.csv`](curve.csv) uses each framework's logical step: UniRL
`rollout/step` and veRL-Omni `training/global_step`. W&B's default `_step` is
not comparable because one UniRL logical step produces multiple W&B records.

## Performance

![MiniMax-H3 end-to-end performance](performance.png)

[`performance.csv`](performance.csv) reports medians and interquartile ranges.
The long-run comparison uses the same logical steps 2–98, excluding step 1 as
warm-up. PR403 UniRL measured **1882 s/step**, versus **1567 s/step** for
veRL-Omni.

The PR403 gap is concentrated in rollout generation. Its frozen 32B Qwen3-VL
conditioner is recomputed for every sibling sample when
`forward_batch_size=1`. [PR410](https://github.com/Tencent-Hunyuan/UniRL/pull/410)
introduced a CPU prompt-embedding cache; the
exact [`conditioning_cache.patch`](conditioning_cache.patch) used here is a
cache-only backport of that fix. It is reported as a four-step regression check
(step 1 warm-up, steps 2–4 measured) once that run completes.

End-to-end step time is the primary comparison. Framework-native phase timers
have different boundaries, so they are retained in the CSV but are not stacked
as if they were identical.
