#!/usr/bin/env python3
"""Plot batch-size sweep results from multiple hybrid JSONs."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

STRAT_KEYS = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin", "rubin_tp2"]
STRAT_LABELS = {
    "HMP_reo": "HMP_RO", "hmp_reo_new": "RO_new",
    "hmp": "HMP", "tp16": "TP16", "rubin": "Rubin", "rubin_tp2": "Rubin_TP2",
}
STRAT_COLORS = {
    "HMP_reo": "#E67E22", "hmp_reo_new": "#8E44AD",
    "hmp": "#2E86C1", "tp16": "#27AE60",
    "rubin": "#C0392B", "rubin_tp2": "#922B21",
}
STRAT_MARKERS = {
    "HMP_reo": "s", "hmp_reo_new": "P", "hmp": "D",
    "tp16": "^", "rubin": "o", "rubin_tp2": "v",
}

MOD_COLORS = {
    "Proj_QKV": "#1B4F72",
    "Attention": "#3498DB",
    "Proj_O": "#A9CCE3",
    "Comm": "#E74C3C",
}


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024: return f"{s // 1024}K"
    return str(s)


def load_all(pattern, bs_list):
    """Load hybrid JSONs for each batch size."""
    data = {}
    for bs in bs_list:
        path = Path(pattern.replace("{BS}", str(bs)))
        if path.exists():
            with open(path) as f:
                data[bs] = json.load(f)
    return data


def get_entry(d, strat, seq):
    for e in d.get("strategies", {}).get(strat, {}).get("data", []):
        if e["seq"] == seq:
            return e
    return None


def plot_wall_vs_bs(data, bs_list, seq, out_dir):
    """Fig 1: Wall time vs batch size, per strategy."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for sk in STRAT_KEYS:
        xs, ys = [], []
        for bs in bs_list:
            if bs not in data: continue
            e = get_entry(data[bs], sk, seq)
            if e:
                xs.append(bs)
                ys.append(e["hybrid_wall_ns"] / 1e3)
        if xs:
            ax.plot(xs, ys, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                    color=STRAT_COLORS[sk], linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Wall Time (us)", fontsize=13)
    ax.set_title(f"Wall Time vs Batch Size (seq={seq_label(seq)})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = f"batch_sweep_wall_seq{seq_label(seq)}.png"
    fig.savefig(out_dir / fname, dpi=180)
    print(f"Saved: {out_dir / fname}")
    plt.close(fig)


def plot_per_token_vs_bs(data, bs_list, seq, out_dir):
    """Fig 2: Per-token latency vs batch size."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for sk in STRAT_KEYS:
        xs, ys = [], []
        for bs in bs_list:
            if bs not in data: continue
            e = get_entry(data[bs], sk, seq)
            if e:
                xs.append(bs)
                ys.append(e["hybrid_wall_ns"] / bs)
        if xs:
            ax.plot(xs, ys, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                    color=STRAT_COLORS[sk], linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Per-Token Latency (ns)", fontsize=13)
    ax.set_title(f"Per-Token Latency vs Batch Size (seq={seq_label(seq)})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = f"batch_sweep_per_token_seq{seq_label(seq)}.png"
    fig.savefig(out_dir / fname, dpi=180)
    print(f"Saved: {out_dir / fname}")
    plt.close(fig)


def plot_speedup_vs_bs(data, bs_list, seq, out_dir):
    """Fig 3: Speedup vs Rubin at different batch sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    compare = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin_tp2"]
    ax.axhline(1.0, color="red", ls="--", lw=1.2, label="Rubin = 1.0x")
    ax.axhspan(0, 1.0, color="#FFCCCC", alpha=0.2)
    ax.axhspan(1.0, 10, color="#CCFFCC", alpha=0.2)

    for sk in compare:
        xs, ys = [], []
        for bs in bs_list:
            if bs not in data: continue
            e = get_entry(data[bs], sk, seq)
            rb = get_entry(data[bs], "rubin", seq)
            if e and rb and e["hybrid_wall_ns"] > 0:
                xs.append(bs)
                ys.append(rb["hybrid_wall_ns"] / e["hybrid_wall_ns"])
        if xs:
            ax.plot(xs, ys, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                    color=STRAT_COLORS[sk], linewidth=2, markersize=7)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}x", (x, y), textcoords="offset points",
                            xytext=(0, 10), fontsize=7.5, ha="center",
                            color=STRAT_COLORS[sk], fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=13)
    ax.set_ylabel("Speedup vs Rubin", fontsize=13)
    ax.set_title(f"Speedup vs Rubin at Different Batch Sizes (seq={seq_label(seq)})", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fname = f"batch_sweep_speedup_seq{seq_label(seq)}.png"
    fig.savefig(out_dir / fname, dpi=180)
    print(f"Saved: {out_dir / fname}")
    plt.close(fig)


def plot_module_breakdown_vs_bs(data, bs_list, seq, out_dir):
    """Fig 4: Stacked bar of QKV/Attn/ProjO/Comm per batch, all strategies side by side."""
    plot_strats = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin", "rubin_tp2"]
    n_strats = len(plot_strats)
    n_bs = len(bs_list)

    fig, axes = plt.subplots(1, n_strats, figsize=(5 * n_strats, 7), sharey=True)
    if n_strats == 1:
        axes = [axes]

    global_ymax = 0
    for sk in plot_strats:
        for bs in bs_list:
            e = get_entry(data.get(bs, {}), sk, seq)
            if e:
                total = (e["hybrid_Proj_QKV_ns"] + e["hybrid_attn_ns"]
                         + e["hybrid_Proj_O_ns"] + e["comm_total_ns"]) / 1e3
                if total > global_ymax:
                    global_ymax = total
    y_limit = global_ymax * 1.12

    for idx, sk in enumerate(plot_strats):
        ax = axes[idx]
        x = np.arange(n_bs)
        w = 0.65

        qkv_vals, attn_vals, o_vals, comm_vals = [], [], [], []
        for bs in bs_list:
            e = get_entry(data.get(bs, {}), sk, seq)
            if e:
                qkv_vals.append(e["hybrid_Proj_QKV_ns"] / 1e3)
                attn_vals.append(e["hybrid_attn_ns"] / 1e3)
                o_vals.append(e["hybrid_Proj_O_ns"] / 1e3)
                comm_vals.append(e["comm_total_ns"] / 1e3)
            else:
                qkv_vals.append(0); attn_vals.append(0)
                o_vals.append(0); comm_vals.append(0)

        qkv_a = np.array(qkv_vals)
        attn_a = np.array(attn_vals)
        o_a = np.array(o_vals)
        comm_a = np.array(comm_vals)

        ax.bar(x, qkv_a, w, color=MOD_COLORS["Proj_QKV"])
        ax.bar(x, attn_a, w, bottom=qkv_a, color=MOD_COLORS["Attention"])
        ax.bar(x, o_a, w, bottom=qkv_a + attn_a, color=MOD_COLORS["Proj_O"])
        ax.bar(x, comm_a, w, bottom=qkv_a + attn_a + o_a, color=MOD_COLORS["Comm"])

        for i, bs in enumerate(bs_list):
            total = qkv_a[i] + attn_a[i] + o_a[i] + comm_a[i]
            if total > 0:
                comm_frac = comm_a[i] / total * 100
                ax.text(i, total + y_limit * 0.01, f"C{comm_frac:.0f}%",
                        ha="center", fontsize=7, color="#C0392B")

        ax.set_ylim(0, y_limit)
        ax.set_xticks(x)
        ax.set_xticklabels([str(bs) for bs in bs_list], fontsize=10)
        ax.set_xlabel("Batch Size", fontsize=12)
        ax.set_title(STRAT_LABELS[sk], fontsize=14, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.3)

    axes[0].set_ylabel("Latency (us)", fontsize=13)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=MOD_COLORS["Proj_QKV"]),
        plt.Rectangle((0, 0), 1, 1, fc=MOD_COLORS["Attention"]),
        plt.Rectangle((0, 0), 1, 1, fc=MOD_COLORS["Proj_O"]),
        plt.Rectangle((0, 0), 1, 1, fc=MOD_COLORS["Comm"]),
    ]
    fig.legend(handles, ["Proj_QKV", "Attention", "Proj_O", "Comm"],
               loc="upper center", ncol=4, fontsize=12, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Module Breakdown vs Batch Size (seq={seq_label(seq)})",
                 fontsize=16, fontweight="bold", y=1.06)
    fig.tight_layout()
    fname = f"batch_sweep_breakdown_seq{seq_label(seq)}.png"
    fig.savefig(out_dir / fname, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_dir / fname}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot batch-size sweep")
    parser.add_argument("--pattern", required=True,
                        help="Hybrid JSON pattern with {BS} placeholder, "
                             "e.g. reports/.../gqa_hybrid_merged_80T_bw1500_util80_bs{BS}.json")
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--seq", type=int, nargs="+", default=[65536])
    parser.add_argument("-o", "--out-dir", default="reports/qwen3-235B/hybrid/plots_batch_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_all(args.pattern, args.bs)
    print(f"Loaded {len(data)} batch sizes: {sorted(data.keys())}")

    for seq in args.seq:
        print(f"\n--- seq={seq_label(seq)} ---")
        plot_wall_vs_bs(data, args.bs, seq, out_dir)
        plot_per_token_vs_bs(data, args.bs, seq, out_dir)
        plot_speedup_vs_bs(data, args.bs, seq, out_dir)
        plot_module_breakdown_vs_bs(data, args.bs, seq, out_dir)


if __name__ == "__main__":
    main()
