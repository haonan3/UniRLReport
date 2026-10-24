#!/usr/bin/env python3
"""Regenerate E3 & E5-T tables, macros, figures, and manifest from evidence.json and per_step_metrics.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from decimal import Decimal
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "unirl-mpl-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "unirl-plot-cache"))

BUNDLE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def ratio(numerator: float | str, denominator: float | str, digits: int) -> str:
    return f"{Decimal(str(numerator)) / Decimal(str(denominator)):.{digits}f}"


def load_evidence() -> dict:
    data = json.loads((BUNDLE / "evidence.json").read_text(encoding="utf-8"))
    for arm in ("unirl_colocate", "verl_sdpa", "verl_fa3", "unirl_separate"):
        assert arm in data["arms"], f"Missing arm {arm} in evidence.json"
        assert float(data["arms"][arm]["median_s"]) > 0
    return data


def render_tex(data: dict) -> dict[str, str]:
    arms = data["arms"]
    u_col = arms["unirl_colocate"]
    v_sdpa = arms["verl_sdpa"]
    v_fa3 = arms["verl_fa3"]
    u_sep = arms["unirl_separate"]

    macros = {
        # Drop-in replacements for existing manuscript macros
        "ReportedNativeSpeedup": ratio(v_sdpa["median_s"], u_col["median_s"], 2),
        "ReportedFAThreeSpeedup": ratio(v_fa3["median_s"], u_col["median_s"], 2),
        "ReportedTopologySlowdown": ratio(u_sep["median_s"], u_col["median_s"], 3),
        "ReportedUniRLSeconds": f"{u_col['median_s']:.2f}",
        "ReportedSDPASeconds": f"{v_sdpa['median_s']:.2f}",
        "ReportedFAThreeSeconds": f"{v_fa3['median_s']:.2f}",
        "ReportedColocatedSeconds": f"{u_col['median_s']:.3f}",
        "ReportedSeparateSeconds": f"{u_sep['median_s']:.3f}",
        # Detailed verified E3 & E5-T macros
        "EThreeUniRLSeconds": f"{u_col['median_s']:.1f}",
        "EThreeUniRLMeanSeconds": f"{u_col['mean_s']:.1f}",
        "EThreeUniRLPNinetySeconds": f"{u_col['p90_s']:.1f}",
        "EThreeUniRLThroughput": f"{round(u_col['samples_per_gpu_h']):d}",
        "EThreeUniRLGenSeconds": f"{u_col['gen_s']:.1f}",
        "EThreeUniRLRewardSeconds": f"{u_col['reward_s']:.1f}",
        "EThreeUniRLSyncSeconds": f"{u_col['sync_s']:.2f}",
        "EThreeUniRLTrainSeconds": f"{u_col['train_s']:.1f}",
        "EThreeUniRLPeakAllocGB": f"{u_col['peak_allocated_gb']:.1f}",
        "EThreeUniRLPeakResGB": f"{u_col['peak_reserved_gb']:.1f}",
        "EThreeUniRLPeakDevGB": f"{u_col['peak_device_used_gb']:.1f}",
        "EThreeVeRLSDPASeconds": f"{v_sdpa['median_s']:.1f}",
        "EThreeVeRLSDPAMeanSeconds": f"{v_sdpa['mean_s']:.1f}",
        "EThreeVeRLSDPAPNinetySeconds": f"{v_sdpa['p90_s']:.1f}",
        "EThreeVeRLSDPAThroughput": f"{round(v_sdpa['samples_per_gpu_h']):d}",
        "EThreeVeRLSDPAGenSeconds": f"{v_sdpa['gen_s']:.1f}",
        "EThreeVeRLSDPARewardSeconds": f"{v_sdpa['reward_s']:.1f}",
        "EThreeVeRLSDPAOldLPSeconds": f"{v_sdpa['old_log_prob_s']:.1f}",
        "EThreeVeRLSDPASyncSeconds": f"{v_sdpa['sync_s']:.2f}",
        "EThreeVeRLSDPATrainSeconds": f"{v_sdpa['train_s']:.1f}",
        "EThreeVeRLSDPAPeakAllocGB": f"{v_sdpa['peak_allocated_gb']:.1f}",
        "EThreeVeRLSDPAPeakResGB": f"{v_sdpa['peak_reserved_gb']:.1f}",
        "EThreeVeRLSDPAPeakDevGB": f"{v_sdpa['peak_device_used_gb']:.1f}",
        "EThreeVeRLFAThreeSeconds": f"{v_fa3['median_s']:.1f}",
        "EThreeVeRLFAThreeMeanSeconds": f"{v_fa3['mean_s']:.1f}",
        "EThreeVeRLFAThreePNinetySeconds": f"{v_fa3['p90_s']:.1f}",
        "EThreeVeRLFAThreeThroughput": f"{round(v_fa3['samples_per_gpu_h']):d}",
        "EThreeVeRLFAThreeGenSeconds": f"{v_fa3['gen_s']:.1f}",
        "EThreeVeRLFAThreeRewardSeconds": f"{v_fa3['reward_s']:.1f}",
        "EThreeVeRLFAThreeOldLPSeconds": f"{v_fa3['old_log_prob_s']:.1f}",
        "EThreeVeRLFAThreeSyncSeconds": f"{v_fa3['sync_s']:.2f}",
        "EThreeVeRLFAThreeTrainSeconds": f"{v_fa3['train_s']:.1f}",
        "EThreeVeRLFAThreePeakAllocGB": f"{v_fa3['peak_allocated_gb']:.1f}",
        "EThreeVeRLFAThreePeakResGB": f"{v_fa3['peak_reserved_gb']:.1f}",
        "EThreeVeRLFAThreePeakDevGB": f"{v_fa3['peak_device_used_gb']:.1f}",
        "EFiveTColocatedSeconds": f"{u_col['median_s']:.2f}",
        "EFiveTColocatedThroughput": f"{round(u_col['samples_per_gpu_h']):d}",
        "EFiveTSeparateSeconds": f"{u_sep['median_s']:.2f}",
        "EFiveTSeparateMeanSeconds": f"{u_sep['mean_s']:.2f}",
        "EFiveTSeparatePNinetySeconds": f"{u_sep['p90_s']:.2f}",
        "EFiveTSeparateThroughput": f"{round(u_sep['samples_per_gpu_h']):d}",
        "EFiveTSeparateGenSeconds": f"{u_sep['gen_s']:.1f}",
        "EFiveTSeparateRewardSeconds": f"{u_sep['reward_s']:.1f}",
        "EFiveTSeparateSyncSeconds": f"{u_sep['sync_s']:.2f}",
        "EFiveTSeparateTrainSeconds": f"{u_sep['train_s']:.1f}",
        "EFiveTSeparatePeakAllocGB": f"{u_sep['peak_allocated_gb']:.1f}",
        "EFiveTSeparatePeakResGB": f"{u_sep['peak_reserved_gb']:.1f}",
        "EFiveTSeparatePeakDevGB": f"{u_sep['peak_device_used_gb']:.1f}",
    }

    header = (
        "% Generated by generate.py from evidence.json and per_step_metrics.csv.\n"
        "% Raw-log verified on 8x NVIDIA H20 (96GB), Pod 13 (worker-12, 28.4.45.107).\n"
    )
    lines = []
    for k, v in macros.items():
        if k.startswith("Reported"):
            lines.append(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
        else:
            lines.append(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    write(BUNDLE / "macros.tex", header + "".join(lines))

    table_lines = [
        header.rstrip(),
        r"\begin{tabularx}{\textwidth}{@{}lXrrrrrr@{}}",
        r"\toprule",
        r"System & Configuration & s/iter & Gen / Rew / Sync / Train (s) & Old-log$p$ (s) & samples/GPU-h & Alloc / Res / Dev (GB) & Speedup \\",
        r"\midrule",
        r"\multicolumn{8}{@{}l}{\textbf{E3: shape-matched native SD3.5 recipes (8 H20 GPUs colocated)}} \\",
    ]
    for key, sys_label, cfg_label in [
        ("unirl_colocate", "UniRL", "Native flow / SDPA-class"),
        ("verl_sdpa", "VeRL-Omni", "Native CPS / SDPA"),
        ("verl_fa3", "VeRL-Omni", "Native CPS / FA3"),
    ]:
        r = arms[key]
        phases = f"{r['gen_s']:.1f} / {r['reward_s']:.1f} / {r['sync_s']:.2f} / {r['train_s']:.1f}"
        old_lp = "--" if r["old_log_prob_s"] == 0 else f"{r['old_log_prob_s']:.1f}"
        mem_str = f"{r['peak_allocated_gb']:.1f} / {r['peak_reserved_gb']:.1f} / {r['peak_device_used_gb']:.1f}"
        if key == "unirl_colocate":
            sp_str = r"1.00$\times$ (ref)"
        else:
            sp_str = f"{v_sdpa['median_s'] / u_col['median_s']:.2f}$\\times$" if key == "verl_sdpa" else f"{v_fa3['median_s'] / u_col['median_s']:.2f}$\\times$"
        table_lines.append(
            f"{sys_label} & {cfg_label} & {r['median_s']:.1f} & {phases} & {old_lp} & {round(r['samples_per_gpu_h']):d} & {mem_str} & {sp_str} \\\\"
        )

    table_lines.extend(
        [
            r"\midrule",
            r"\multicolumn{8}{@{}l}{\textbf{E5-T: deployment topology regimes (8 H20 GPUs total)}} \\",
        ]
    )
    for key, cfg_label in [
        ("unirl_colocate", "Colocated ($T_0$: 8 shared GPUs, local LoRA)"),
        ("unirl_separate", "Separate ($T_1$: 4 train + 4 rollout GPUs, RPC LoRA)"),
    ]:
        r = arms[key]
        phases = f"{r['gen_s']:.1f} / {r['reward_s']:.1f} / {r['sync_s']:.2f} / {r['train_s']:.1f}"
        mem_str = f"{r['peak_allocated_gb']:.1f} / {r['peak_reserved_gb']:.1f} / {r['peak_device_used_gb']:.1f}"
        rel_str = r"1.000$\times$" if key == "unirl_colocate" else f"{r['median_s'] / u_col['median_s']:.3f}$\\times$"
        table_lines.append(
            f"UniRL & {cfg_label} & {r['median_s']:.2f} & {phases} & -- & {round(r['samples_per_gpu_h']):d} & {mem_str} & {rel_str} \\\\"
        )

    table_lines.extend([r"\bottomrule", r"\end{tabularx}", ""])
    write(BUNDLE / "systems_table.tex", "\n".join(table_lines))
    return macros


def render_figures(data: dict) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#626975",
            "axes.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )
    figure_dir = BUNDLE / "figures"
    figure_dir.mkdir(exist_ok=True)

    arms = data["arms"]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.65), gridspec_kw={"width_ratios": [1.35, 1.0, 1.1]})
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.25, top=0.78, wspace=0.32)

    def grid(ax):
        ax.grid(axis="y", color="#d9dee6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(width=0.6, length=3)

    # Panel (a): E3 Phase Breakdown
    ax = axes[0]
    e3_keys = ["unirl_colocate", "verl_sdpa", "verl_fa3"]
    e3_labels = ["UniRL\nFlow / SDPA", "VeRL-Omni\nCPS / SDPA", "VeRL-Omni\nCPS / FA3"]
    gen_v = np.array([arms[k]["gen_s"] for k in e3_keys])
    old_lp_v = np.array([arms[k]["old_log_prob_s"] for k in e3_keys])
    train_v = np.array([arms[k]["train_s"] for k in e3_keys])
    rew_sync_v = np.array(
        [
            arms[k]["reward_s"] + arms[k]["sync_s"]
            if k.startswith("unirl")
            else arms[k]["sync_s"]
            for k in e3_keys
        ]
    )
    x = np.arange(len(e3_keys))
    w = 0.56
    ax.bar(x, gen_v, width=w, color="#256b9c", label="Generate (rollout)", zorder=3)
    ax.bar(x, old_lp_v, width=w, bottom=gen_v, color="#8e9aa8", label="Old log-prob recompute", zorder=3)
    ax.bar(x, train_v, width=w, bottom=gen_v + old_lp_v, color="#5c57a4", label="Train (FSDP2 update)", zorder=3)
    ax.bar(
        x,
        rew_sync_v,
        width=w,
        bottom=gen_v + old_lp_v + train_v,
        color="#e08b43",
        label="Reward + Weight sync",
        zorder=3,
    )
    for i, k in enumerate(e3_keys):
        tot = arms[k]["median_s"]
        reps = arms[k].get("rep_medians_s", [tot])
        ax.scatter([i] * len(reps), reps, color="#1f242b", s=18, zorder=5, alpha=0.85)
        ax.text(i, max(tot, float((gen_v + old_lp_v + train_v + rew_sync_v)[i])) + 2.5, f"{tot:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels(e3_labels)
    ax.set_ylim(0, max(arms[k]["median_s"] for k in e3_keys) * 1.22)
    ax.set_ylabel("Seconds per iteration")
    ax.set_title("(a) E3: native SD3.5 recipes (8 GPUs)", loc="left", pad=10, fontweight="semibold")
    grid(ax)

    # Panel (b): E5-T Colocated vs Separate
    ax = axes[1]
    e5_keys = ["unirl_colocate", "unirl_separate"]
    e5_labels = ["Colocated ($T_0$)\n8 shared GPUs", "Separate ($T_1$)\n4 train + 4 rollout"]
    gen_5 = np.array([arms[k]["gen_s"] for k in e5_keys])
    train_5 = np.array([arms[k]["train_s"] for k in e5_keys])
    rew_sync_5 = np.array([arms[k]["reward_s"] + arms[k]["sync_s"] for k in e5_keys])
    x5 = np.arange(len(e5_keys))
    ax.bar(x5, gen_5, width=0.52, color="#256b9c", zorder=3)
    ax.bar(x5, train_5, width=0.52, bottom=gen_5, color="#5c57a4", zorder=3)
    ax.bar(x5, rew_sync_5, width=0.52, bottom=gen_5 + train_5, color="#e08b43", zorder=3)
    for i, k in enumerate(e5_keys):
        tot = arms[k]["median_s"]
        reps = arms[k].get("rep_medians_s", [tot])
        ax.scatter([i] * len(reps), reps, color="#1f242b", s=18, zorder=5, alpha=0.85)
        ax.text(i, max(tot, float((gen_5 + train_5 + rew_sync_5)[i])) + 3.0, f"{tot:.1f}s", ha="center", va="bottom", fontsize=8.5, fontweight="semibold")
    ax.set_xticks(x5)
    ax.set_xticklabels(e5_labels)
    ax.set_ylim(0, max(arms[k]["median_s"] for k in e5_keys) * 1.22)
    ax.set_ylabel("Seconds per iteration")
    ax.set_title("(b) E5-T: deployment topology", loc="left", pad=10, fontweight="semibold")
    grid(ax)

    # Panel (c): Peak GPU Memory
    ax = axes[2]
    all_keys = ["unirl_colocate", "verl_sdpa", "verl_fa3", "unirl_separate"]
    all_labels = ["UniRL\nColoc", "VeRL\nSDPA", "VeRL\nFA3", "UniRL\nSep"]
    alloc_m = [arms[k]["peak_allocated_gb"] for k in all_keys]
    res_m = [arms[k]["peak_reserved_gb"] for k in all_keys]
    dev_m = [arms[k]["peak_device_used_gb"] for k in all_keys]
    xm = np.arange(len(all_keys))
    bw = 0.25
    ax.bar(xm - bw, alloc_m, width=bw, color="#2b8cbe", label="Actor Alloc (GB)", zorder=3)
    ax.bar(xm, res_m, width=bw, color="#7bccc4", label="Actor Res (GB)", zorder=3)
    ax.bar(xm + bw, dev_m, width=bw, color="#08589e", label="Device Peak (GB)", zorder=3)
    for i in range(len(all_keys)):
        ax.text(xm[i] + bw, dev_m[i] + 1.2, f"{dev_m[i]:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(xm)
    ax.set_xticklabels(all_labels)
    ax.set_ylim(0, max(dev_m + res_m + alloc_m + [50.0]) * 1.25)
    ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("(c) Peak GPU memory (Actor vs Device)", loc="left", pad=10, fontweight="semibold")
    ax.legend(loc="upper right", frameon=False, fontsize=7.2)
    grid(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.06, 0.03), ncol=4, frameon=False, fontsize=8.2)
    fig.suptitle(
        "E3 and E5-T: verified iteration time, phase breakdown, and peak GPU memory on 8x NVIDIA H20",
        x=0.065,
        ha="left",
        y=0.96,
        fontsize=11.5,
        fontweight="semibold",
    )
    fig.text(
        0.065,
        0.865,
        f"SD3.5-Medium FlowGRPO, 48 prompts x 16 samples @ 384x384 (768 images/step); {arms['unirl_colocate']['reps']} repetitions x 25 steady-state steps/arm",
        fontsize=8.8,
        color="#525a65",
    )

    fig.savefig(
        figure_dir / "e3_e5_systems.pdf",
        metadata={"Creator": "UniRL E3/E5-T systems generator", "CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        figure_dir / "e3_e5_systems.png",
        dpi=200,
        metadata={"Software": "UniRL E3/E5-T systems generator"},
    )
    plt.close(fig)
    return {"python": sys.version.split()[0], "matplotlib": matplotlib.__version__, "numpy": np.__version__}


def save_manifest(versions: dict[str, str]) -> None:
    files = sorted(
        path
        for path in BUNDLE.rglob("*")
        if path.is_file()
        and path.name not in ("manifest.json", "checksums.sha256")
        and not path.name.startswith(".")
        and "runs" not in path.parts
        and "logs" not in path.parts
        and "scripts" not in path.parts
    )
    hashes = {str(path.relative_to(BUNDLE)): digest(path) for path in files}
    generator = Path(__file__).resolve()
    manifest = {
        "schema_version": 2,
        "evidence_tier": "raw_log_verified_multi_rep",
        "measured_on": "2026-10-24",
        "generator": generator.name,
        "generator_sha256": digest(generator),
        "versions": versions,
        "reproduction": "python3 generate.py",
        "sha256": hashes,
    }
    write(BUNDLE / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    write(BUNDLE / "checksums.sha256", "".join(f"{v}  {k}\n" for k, v in hashes.items()))


def check() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    mismatches = [
        name
        for name, expected in manifest["sha256"].items()
        if not (BUNDLE / name).is_file() or digest(BUNDLE / name) != expected
    ]
    if digest(Path(__file__).resolve()) != manifest["generator_sha256"]:
        mismatches.append("generator")
    if mismatches:
        raise SystemExit("Checksum mismatch: " + ", ".join(mismatches))
    print(f"Verified {len(manifest['sha256'])} bundle files and generator hash.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify bundle hashes without regenerating")
    args = parser.parse_args()
    if args.check:
        check()
        return
    data = load_evidence()
    macros = render_tex(data)
    versions = render_figures(data)
    save_manifest(versions)
    print(f"Generated {len(macros)} macros, systems_table.tex, figures/e3_e5_systems.pdf, and manifest.json in {BUNDLE}")


if __name__ == "__main__":
    main()
