#!/usr/bin/env python3
"""
Fig 2d: 瞬时功耗 Power (W) — BS=4 only, 1x2 layout (Qwen3-235B | Llama4-Maverick)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "plots"

STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new"]
LABELS = {
    "hmp_reo_new": "Ours", "rubin": "Rubin", "rubin_tp2": "Rubin TP2",
    "h100": "H100", "h100_tp2": "H100 TP2", "neupims": "NeuPIMs",
}
COLORS = {
    "h100": "#D4D4D4", "h100_tp2": "#DBDDEF",
    "rubin": "#92B1D9", "rubin_tp2": "#C1D8E9",
    "neupims": "#F4A582", "hmp_reo_new": "#E07850",
}

BS = 4

MODELS = [
    ("Qwen3-235B", str(BASE / f"reports/qwen3-235B/power/gqa_hybrid_merged_96T_bw1500_util96_bs{BS}_power.json")),
    ("Llama4-Maverick", str(BASE / f"reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_bs{BS}_power.json")),
]

#SEQS = [4096, 65536, 262144, 1048576]
SEQS = [4096, 262144, 1048576]

def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def load(path):
    with open(path) as f:
        return json.load(f)


def get_power_map(data, strat):
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["total_power_w"] for e in entries}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)

    for col, (model_label, path) in enumerate(MODELS):
        data = load(path)
        ax = axes[col]
        x = np.arange(n_seqs)

        total_width = 0.96
        bar_w = total_width / n_strats
        offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

        ours_power_map = get_power_map(data, "hmp_reo_new")
        neupims_power_map = {s: v + 760 for s, v in ours_power_map.items()}

        max_val = 0
        for si, sk in enumerate(STRATEGIES):
            if sk == "neupims":
                power_map = neupims_power_map
            else:
                power_map = get_power_map(data, sk)
            vals = [power_map.get(s, 0) for s in SEQS]

            ax.bar(x + offsets[si], vals, bar_w,
                   color=COLORS[sk], edgecolor="white", linewidth=0.3)

            for v in vals:
                if v > max_val:
                    max_val = v

        ax.set_ylim(top=max_val * 1.15)

        for si, sk in enumerate(STRATEGIES):
            if sk == "neupims":
                power_map = neupims_power_map
            else:
                power_map = get_power_map(data, sk)
            vals = [power_map.get(s, 0) for s in SEQS]
           #for i, v in enumerate(vals):
           #    if v > 0:
           #        ax.text(x[i] + offsets[si], v - max_val * 0.005,
           #                f"{v:.0f}", ha="center", va="top",
           #                fontsize=12, fontweight="normal", color="black", rotation=90)
            for i, v in enumerate(vals):
                if v > 0:
                    if sk in ["h100", "h100_tp2"]:
                        ax.text(x[i] + offsets[si], v + max_val * 0.01,
                                f"{v:.0f}", ha="center", va="bottom",
                                fontsize=12, fontweight="normal",
                                color="black", rotation=90)
                    else:
                        ax.text(x[i] + offsets[si], v - max_val * 0.03,
                                f"{v:.0f}", ha="center", va="top",
                                fontsize=12, fontweight="normal",
                                color="black", rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([seq_label(s) for s in SEQS], fontsize=15, fontweight="normal")
        ax.grid(axis="y", ls="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=15)
        ax.set_title(f"{model_label}, BS={BS}", fontsize=15, fontweight="normal")

    fig.text(0.5, -0.02, "Sequence Length", va="top", ha="center", fontsize=20, fontweight="normal")
    fig.text(-0.01, 0.5, "Power (W)", va="center", ha="center",
             rotation="vertical", fontsize=20, fontweight="normal")

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=15, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.05))

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fname = OUT / "fig2d_power_W_bs4_1x2.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
