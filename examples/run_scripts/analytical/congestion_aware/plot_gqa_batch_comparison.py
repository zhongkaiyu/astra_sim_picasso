#!/usr/bin/env python3
"""
Plot GQA batch-scaling comparison from collected JSON data.

Produces:
  1. Wall Time comparison (line plot with speedup annotations)
  2. Module breakdown (stacked bar per strategy)
  3. Compute vs Comm ratio (grouped bar)

Usage:
  python3 plot_gqa_batch_comparison.py
  python3 plot_gqa_batch_comparison.py -i reports/gqa_batch_scaling_sl65536_data.json
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
    "other_comm": "#BAB0AC",
}


def load_data(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_wall_time(data: dict, seq: int, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    all_walls = {}
    for key in STRATEGY_ORDER:
        if key not in data:
            continue
        info = data[key]
        batches = [d["batch"] for d in info["data"]]
        walls = [d["wall_ns"] / 1000 for d in info["data"]]
        all_walls[key] = dict(zip(batches, walls))
        ax.plot(batches, walls, marker=STRATEGY_MARKERS[key], color=STRATEGY_COLORS[key],
                label=info["label"], linewidth=2, markersize=7)

    common_batches = sorted(set.intersection(*(set(v.keys()) for v in all_walls.values())))
    for bs in common_batches:
        vals = {k: all_walls[k][bs] for k in all_walls}
        best_key = min(vals, key=vals.get)
        worst_key = max(vals, key=vals.get)
        speedup = vals[worst_key] / vals[best_key]
        best_label = data[best_key]["label"].split("(")[0].strip()
        ax.annotate(
            f"{best_label}\n{speedup:.2f}x",
            xy=(bs, vals[best_key]),
            xytext=(0, -22), textcoords="offset points",
            fontsize=7, ha="center", color=STRATEGY_COLORS[best_key],
            fontweight="bold",
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Wall Time (us)", fontsize=12)
    ax.set_title(f"Single-Layer GQA (Seq={seq}): Wall Time vs Batch Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))

    fig.tight_layout()
    out = out_dir / f"gqa_batch_wall_time_sl{seq}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def plot_module_breakdown(data: dict, seq: int, out_dir: Path):
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
        batches = [d["batch"] for d in info["data"]]
        x = np.arange(len(batches))
        bottom = np.zeros(len(batches))

        for mod in ordered:
            vals = np.array([d["modules"].get(mod, 0) / 1000 for d in info["data"]])
            ax.bar(x, vals, bottom=bottom, width=0.7,
                   label=MODULE_LABELS.get(mod, mod), color=MODULE_COLORS.get(mod, "#999"))
            bottom += vals

        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in batches], rotation=45, fontsize=9)
        ax.set_xlabel("Batch Size", fontsize=11)
        ax.set_title(info["label"], fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Latency (us)", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(ordered), 5),
               fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Single-Layer GQA (Seq={seq}): Module Breakdown vs Batch Size",
                 fontsize=14, y=1.06)
    fig.tight_layout()
    out = out_dir / f"gqa_batch_module_breakdown_sl{seq}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out}")


def plot_comp_comm_ratio(data: dict, seq: int, out_dir: Path):
    strategies = [k for k in STRATEGY_ORDER if k in data]
    all_batches = sorted(set(d["batch"] for k in strategies for d in data[k]["data"]))
    common_batches = [b for b in all_batches
                      if all(any(d["batch"] == b for d in data[k]["data"]) for k in strategies)]

    n_strat = len(strategies)
    n_bs = len(common_batches)
    bar_width = 0.8 / n_strat
    fig, ax = plt.subplots(figsize=(max(10, n_bs * 1.5), 6))

    for si, key in enumerate(strategies):
        info = data[key]
        bs_map = {d["batch"]: d for d in info["data"]}
        comps, comms = [], []
        for bs in common_batches:
            d = bs_map[bs]
            comps.append(d["gpu_ns"] / 1000)
            comms.append(d["comm_ns"] / 1000)

        x = np.arange(n_bs) + si * bar_width
        ax.bar(x, comps, width=bar_width, color=STRATEGY_COLORS[key], alpha=0.85,
               label=f"{info['label']} Compute")
        ax.bar(x, comms, width=bar_width, bottom=comps, color=STRATEGY_COLORS[key],
               alpha=0.4, hatch="//", label=f"{info['label']} Comm")

    ax.set_xticks(np.arange(n_bs) + bar_width * (n_strat - 1) / 2)
    ax.set_xticklabels([str(b) for b in common_batches], fontsize=10)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Latency (us)", fontsize=12)
    ax.set_title(f"Single-Layer GQA (Seq={seq}): Compute vs Communication", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = out_dir / f"gqa_batch_comp_comm_sl{seq}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def plot_speedup(data: dict, seq: int, out_dir: Path):
    """Speedup curves: each strategy pair as a function of batch size."""
    strategies = [k for k in STRATEGY_ORDER if k in data]
    all_batches = sorted(set(d["batch"] for k in strategies for d in data[k]["data"]))
    common_batches = [b for b in all_batches
                      if all(any(d["batch"] == b for d in data[k]["data"]) for k in strategies)]

    wall_map = {}
    for key in strategies:
        wall_map[key] = {d["batch"]: d["wall_ns"] for d in data[key]["data"]}

    pairs = [
        ("baseline", "hmp", "Baseline / HMP", "#4C72B0"),
        ("tp16", "hmp", "TP16 / HMP", "#55A868"),
        ("tp16", "baseline", "TP16 / Baseline", "#C44E52"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for slow, fast, label, color in pairs:
        if slow not in wall_map or fast not in wall_map:
            continue
        speedups = [wall_map[slow][b] / wall_map[fast][b] for b in common_batches]
        ax.plot(common_batches, speedups, marker="o", color=color,
                label=label, linewidth=2, markersize=7)
        for b, sp in zip(common_batches, speedups):
            ax.annotate(f"{sp:.1f}x", xy=(b, sp),
                        xytext=(0, 8), textcoords="offset points",
                        fontsize=7, ha="center", color=color, fontweight="bold")

    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Speedup (x)", fontsize=12)
    ax.set_title(f"Single-Layer GQA (Seq={seq}): Speedup vs Batch Size", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))

    fig.tight_layout()
    out = out_dir / f"gqa_batch_speedup_sl{seq}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[OK] {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot GQA batch-scaling comparison")
    parser.add_argument("-i", "--input", default="")
    parser.add_argument("-d", "--out-dir", default=str(SCRIPT_DIR / "reports"))
    parser.add_argument("--seq", type=int, default=65536)
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else (
        SCRIPT_DIR / "reports" / f"gqa_batch_scaling_sl{args.seq}_data.json"
    )
    data = load_data(input_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq = args.seq
    plot_wall_time(data, seq, out_dir)
    plot_speedup(data, seq, out_dir)
    plot_module_breakdown(data, seq, out_dir)
    plot_comp_comm_ratio(data, seq, out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
