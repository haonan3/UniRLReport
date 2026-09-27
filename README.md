# UniRL Paper

Manuscript: **UniRL: Trajectory-Centric Post-Training Across Heterogeneous
Generative Models**.





This is the paper/Overleaf repository, not the training-code repository.
Overleaf's root document is `main.tex`; keep the TMLR style, bibliography and
relative figure/table inputs together. Existing `main.pdf` files predate the current
source edits; compile `main.tex` for the updated manuscript.

## Experiment results

[EXPERIMENT_RESULTS.md](EXPERIMENT_RESULTS.md) audits what the available results
support, what remains open, and which experiments to prioritize. Reviewed on
2026-09-27; the live Notion report still states **as of 2026-09-22**.

The manuscript now includes report-derived E2 quality/diversity and E3/E5-T
systems tables and figures. The compact
[evidence bundle](artifacts/reported_results/2026-09-22/README.md) preserves source
snapshots, numerical inputs, provenance limitations, and regeneration instructions.
The original GPU logs were not accessible during this integration; these figures
reproduce the reported summaries, not an independent reanalysis of those logs.

![E2 quality and diversity](artifacts/reported_results/2026-09-22/figures/e2_quality_diversity.png)

![E3 and E5-T iteration times](artifacts/reported_results/2026-09-22/figures/e3_e5_systems.png)

The PDF figures and generated TeX fragments are checked in. Overleaf only needs
the repository files: no Python execution, shell escape, or external downloads
are required during compilation. The generator is for updating assets locally.

## GPU handover

Start with [GPU_HANDOFF_README.md](GPU_HANDOFF_README.md). Its readiness table
distinguishes existing launchers from prerequisites that the GPU-side agent must
implement and validate before formal runs. Every experiment has settings,
commands, expected signals, failure ownership and a paper destination.
This is the historical prespecified protocol; use the current result audit above
for completed-run status.

- [EXPERIMENT_RUNBOOK.md](EXPERIMENT_RUNBOOK.md): evidence-admission and statistical rules.
- [PAPER_REVIEW.md](PAPER_REVIEW.md): substantive review, paragraph roles and remaining scientific obligations.
- [PAPER_PLAN.md](PAPER_PLAN.md): thesis, source audit, claim-to-evidence map and priorities.
- [experiments/trajectory_ir/README.md](experiments/trajectory_ir/README.md): historical CPU-only IR evidence and reproduction instructions.

Train and implement missing GPU harnesses in the separately pinned UniRL code
checkout. Store checkpoints, datasets, environments and raw GPU logs outside this
repository. Return compact, artifact-generated tables/figures and their manifests
here. Do not upload model weights, credentials, private prompts or large logs to
GitHub/Overleaf.

## Build

Compile `main.tex` in Overleaf, or with an installed LaTeX toolchain:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Keep unknown hardware, dataset statistics and final scores blank until supported
by admitted artifacts. Compile and inspect the PDF after changing the manuscript.
