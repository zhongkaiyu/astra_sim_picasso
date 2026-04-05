#!/usr/bin/env python3
"""
Plot GQA seq-scaling comparison from collected JSON data.

Produces:
  1. Wall Time comparison (line plot with speedup annotations)
  2. Module breakdown (stacked bar per strategy)
  3. Compute vs Comm ratio (grouped bar)

Usage:
  python3 plot_gqa_comparison.py                                    # defaults
  python3 plot_gqa_comparison.py -i reports/gqa_seq_scaling_data.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = SCRIPT_DIR / "reports" / "gqa_seq_scaling_data.json"

STRATEGY_ORDER = ["baseline", "hmp", "tp16", "rubin"]
STRATEGY_COLORS = {"baseline": "#4C72B0", "hmp": "#DD8452", "tp16": "#55A868", "rubin": "#C44E52"}
STRATEGY_MARKERS = {"baseline": "o", "hmp": "s", "tp16": "^", "rubin": "D"}

MODULE_ORDER = [
    "qkv_comp", "attn_comp", "output_comp", "norm_res", "other_comp",
    "weight_ag", "attn_comm", "output_comm", "other_comm",
]
MODULE_LABELS = {
    "qkv_comp": "QKV Projection",
    "attn_comp": "Attn Kernel",
    "output_comp": "Output Projection",
    "norm_res": "Norm + Residual",
    "other_comp": "Other Compute",
    "weight_ag": "Weight AllGather",
    "attn_comm": "Attn Comm",
    "output_comm": "Output Comm",
    "ffn_comp": "FFN Compute",
    "ffn_comm": "FFN Comm",
    "other_comm": "Other Comm",
}
MODULE_COLORS = {
    "qkv_comp": "#4C72B0",
    "attn_comp": "#DD8452",
    "output_comp": "#55A868",
    "norm_res": "#C44E52",
    "other_comp": "#8172B3",
    "weight_ag": "#CCB974",
    "attn_comm": "#E07B7B",
    "output_comm": "#B07AA1",
    "ffn_comp": "#76B7B2",
    "ffn_comm": "#FF9DA7",
    "other_comm": "#BAB0AC",
}


def load_data(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _seq_label(seq: int) -> str:
    if seq >= 1024:
        return f"{seq // 1024}K"
    return str(seq)


def plot_wall_time(data: dict, out_dir: Path):
    """Line plot: Wall Time vs Seq for three strategies with speedup annotations."""
    fig, ax = plt.subplots(figsize=(10, 6))

    all_walls = {}
    for key in STRATEGY_ORDER:
        if key not in data:
            continue
        info = data[key]
        seqs = [d["seq"] for d in info["data"]]
        walls = [d["wall_ns"] / 1000 for d in info["data"]]
        all_walls[key] = dict(zip(seqs, walls))
        ax.plot(seqs, walls, marker=STRATEGY_MARKERS[key], color=STRATEGY_COLORS[key],
                label=info["label"], linewidth=2, markersize=7)

    common_seqs = sorted(set.intersection(*(set(v.keys()) for v in all_walls.values())))
    for seq in common_seqs:
        vals = {k: all_walls[k][seq] for k in all_walls}
        best_key = min(vals, key=vals.get)
        worst_key = max(vals, key=vals.get)
        speedup = vals[worst_key] / vals[best_key]
        best_label = data[best_key]["label"].split("(")[0].strip()
        ax.annotate(
            f"{best_label}\n{speedup:.2f}x",
            xy=(seq, vals[best_key]),
            xytext=(0, -22), textcoords="offset points",
            fontsize=7, ha="center", color=STRATEGY_COLORS[best_key],
            fontweight="bold",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Wall Time (us)", fontsize=12)
    ax.set_title("Single-Layer GQA: Wall Time vs Sequence Length", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: _seq_label(int(x))))

    fig.tight_layout()
    out = out_dir / "gqa_wall_time_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def plot_module_breakdown(data: dict, out_dir: Path):
    """Stacked bar: per-module latency for each strategy in subplots."""
    strategies = [k for k in STRATEGY_ORDER if k in data]
    n = len(strategies)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    active_modules = set()
    for key in strategies:
        for d in data[key]["data"]:
            for m, v in d["modules"].items():
                if v > 0:
                    active_modules.add(m)
    ordered = [m for m in MODULE_ORDER if m in active_modules]

    for idx, key in enumerate(strategies):
        ax = axes[idx]
        info = data[key]
        seqs = [d["seq"] for d in info["data"]]
        x = np.arange(len(seqs))
        bottom = np.zeros(len(seqs))

        for mod in ordered:
            vals = np.array([d["modules"].get(mod, 0) / 1000 for d in info["data"]])
            ax.bar(x, vals, bottom=bottom, width=0.7,
                   label=MODULE_LABELS.get(mod, mod), color=MODULE_COLORS.get(mod, "#999"))
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([_seq_label(s) for s in seqs], rotation=45, fontsize=9)
        ax.set_xlabel("Sequence Length", fontsize=11)
        ax.set_title(info["label"], fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Latency (us)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(ordered), 5),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Single-Layer GQA: Module Breakdown vs Sequence Length",
                 fontsize=14, y=1.06)
    fig.tight_layout()
    out = out_dir / "gqa_module_breakdown.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_comp_comm_ratio(data: dict, out_dir: Path):
    """Grouped bar: compute vs comm for each strategy at each seq."""
    strategies = [k for k in STRATEGY_ORDER if k in data]
    all_seqs = sorted(set(d["seq"] for k in strategies for d in data[k]["data"]))
    common_seqs = []
    for s in all_seqs:
        if all(any(d["seq"] == s for d in data[k]["data"]) for k in strategies):
            common_seqs.append(s)

    n_strat = len(strategies)
    n_seq = len(common_seqs)
    bar_width = 0.8 / n_strat
    fig, ax = plt.subplots(figsize=(max(10, n_seq * 1.5), 6))

    for si, key in enumerate(strategies):
        info = data[key]
        seq_map = {d["seq"]: d for d in info["data"]}
        comps, comms = [], []
        for seq in common_seqs:
            d = seq_map[seq]
            comps.append(d["gpu_ns"] / 1000)
            comms.append(d["comm_ns"] / 1000)

        x = np.arange(n_seq) + si * bar_width
        ax.bar(x, comps, width=bar_width, color=STRATEGY_COLORS[key], alpha=0.85,
               label=f"{info['label']} Compute")
        ax.bar(x, comms, width=bar_width, bottom=comps, color=STRATEGY_COLORS[key],
               alpha=0.4, hatch="//", label=f"{info['label']} Comm")

    ax.set_xticks(np.arange(n_seq) + bar_width * (n_strat - 1) / 2)
    ax.set_xticklabels([_seq_label(s) for s in common_seqs], fontsize=10)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Latency (us)", fontsize=12)
    ax.set_title("Single-Layer GQA: Compute vs Communication", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / "gqa_comp_comm_ratio.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot GQA seq-scaling comparison")
    parser.add_argument("-i", "--input", default=str(DEFAULT_JSON))
    parser.add_argument("-d", "--out-dir", default=str(SCRIPT_DIR / "reports"))
    args = parser.parse_args()

    data = load_data(Path(args.input))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_wall_time(data, out_dir)
    plot_module_breakdown(data, out_dir)
    plot_comp_comm_ratio(data, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
