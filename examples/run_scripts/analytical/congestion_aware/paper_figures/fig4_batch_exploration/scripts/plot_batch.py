#!/usr/bin/env python3
"""Fig 4: Batch Exploration — Throughput, e2e Latency, and Pareto curves."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy.interpolate import PchipInterpolator

BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "plots"

STRATEGIES = ["hmp_reo_new", "rubin", "rubin_tp2", "h100", "h100_tp2"]
LABELS = {"hmp_reo_new": "Ours (RO_new)", "rubin": "Rubin", "rubin_tp2": "Rubin TP2",
          "h100": "H100", "h100_tp2": "H100 TP2"}
COLORS = {"hmp_reo_new": "#4a7dba", "rubin": "#8fc7de", "rubin_tp2": "#ffd680",
          "h100": "#fa874f", "h100_tp2": "#d93026"}
MARKERS = {"hmp_reo_new": "P", "rubin": "o", "rubin_tp2": "v", "h100": "X", "h100_tp2": "p"}

BS_LIST = [1, 2, 4, 8, 16, 32, 64]
SEQS = [1024, 16384, 65536, 262144]


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024: return f"{s // 1024}K"
    return str(s)


def load_all_bs():
    data = {}
    for bs in BS_LIST:
        path = BASE / f"reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{bs}.json"
        if path.exists():
            with open(path) as f:
                data[bs] = json.load(f)
    return data


def get_wall(d, strat, seq):
    for e in d.get("strategies", {}).get(strat, {}).get("data", []):
        if e["seq"] == seq:
            return e["hybrid_wall_ns"]
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_all_bs()
    seq = 65536

    # ── Combined 1×3 figure: Throughput | e2e Latency | Pareto ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    colors = PARETO_COLORS  # unified color scheme for all panels
    AXIS_LABEL = 16
    TICK_SIZE = 13

    # (a) Throughput
    ax = axes[0]
    for sk in STRATEGIES:
        xs, ys = [], []
        for bs in BS_LIST:
            if bs not in data: continue
            w = get_wall(data[bs], sk, seq)
            if w and w > 0:
                xs.append(bs)
                ys.append(bs / (w / 1e3))  # tokens per us
        if xs:
            ax.plot(xs, ys, marker=MARKERS[sk], label=LABELS[sk],
                    color=colors[sk], linewidth=2, markersize=6,
                    linestyle="--" if "tp2" in sk else "-")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=AXIS_LABEL)
    ax.set_ylabel("Throughput (tokens/μs)", fontsize=AXIS_LABEL)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("(a)", fontsize=AXIS_LABEL, fontweight="bold")

    # (b) e2e Latency
    ax = axes[1]
    for sk in STRATEGIES:
        xs, ys = [], []
        for bs in BS_LIST:
            if bs not in data: continue
            w = get_wall(data[bs], sk, seq)
            if w and w > 0:
                xs.append(bs)
                ys.append(w)  # e2e latency in ns
        if xs:
            ax.plot(xs, ys, marker=MARKERS[sk], label=LABELS[sk],
                    color=colors[sk], linewidth=2, markersize=6,
                    linestyle="--" if "tp2" in sk else "-")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=AXIS_LABEL)
    ax.set_ylabel("e2e Latency (ns)", fontsize=AXIS_LABEL)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("(b)", fontsize=AXIS_LABEL, fontweight="bold")

    # (c) Pareto curve
    ax = axes[2]
    for sk in STRATEGIES:
        tps_user, tputs, bs_labels = [], [], []
        for bs in BS_LIST:
            if bs not in data:
                continue
            w = get_wall(data[bs], sk, seq)
            if w and w > 0:
                tputs.append(bs * 1e9 / w)   # tokens/s
                tps_user.append(1e9 / w)      # 1/s
                bs_labels.append(bs)

        if not tps_user:
            continue

        tps_user = np.array(tps_user)
        tputs = np.array(tputs)

        order = np.argsort(tps_user)
        tps_user = tps_user[order]
        tputs = tputs[order]
        bs_labels = [bs_labels[i] for i in order]

        ax.scatter(tps_user, tputs, marker=MARKERS[sk], color=colors[sk],
                   s=50, zorder=5, edgecolors="white", linewidths=0.5)

        if len(tps_user) >= 4:
            spl = PchipInterpolator(tps_user, tputs)
            x_smooth = np.linspace(tps_user.min(), tps_user.max(), 200)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, label=LABELS[sk],
                    color=colors[sk], linewidth=2.5,
                    linestyle="--" if "tp2" in sk else "-")
        elif len(tps_user) >= 2:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=2.5,
                    linestyle="--" if "tp2" in sk else "-")
        else:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=2.5, marker=MARKERS[sk])

        for i, bs in enumerate(bs_labels):
            if bs == bs_labels[0] or bs == bs_labels[-1]:
                ax.annotate(f"bs={bs}", (tps_user[i], tputs[i]),
                            textcoords="offset points", xytext=(6, -4),
                            fontsize=7, color=colors[sk])

    ax.set_xlabel("TPS/User (1/s)", fontsize=AXIS_LABEL)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=AXIS_LABEL)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("(c)", fontsize=AXIS_LABEL, fontweight="bold")

    fig.tight_layout()
    fname = OUT / "fig4_combined.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)

    # ── Also save individual plots (original behavior) ──
    plot_pareto(data, BS_LIST, seq, OUT)


def _fmt_k(x, _):
    """Format axis values with k/M suffix."""
    if x >= 1e6:
        return f"{x / 1e6:.0f}M" if x % 1e6 == 0 else f"{x / 1e6:.1f}M"
    if x >= 1e3:
        return f"{x / 1e3:.0f}k" if x % 1e3 == 0 else f"{x / 1e3:.1f}k"
    return f"{x:.0f}"


# Pareto-specific colors: green palette + bright red for Ours
PARETO_COLORS = {
    "hmp_reo_new": "#e8192c",   # bright red
    "rubin":       "#1b7a3d",   # dark green
    "rubin_tp2":   "#3aaa5c",   # medium green
    "h100":        "#6dca82",   # light green
    "h100_tp2":    "#a8e0b4",   # pale green
}


def plot_pareto(data, bs_list, seq, out_dir):
    """
    Pareto 曲线: x = TPS/User (1/s),  y = Throughput (tokens/s)
    log-log 坐标 + log 空间样条拟合，展示清晰趋势
    """
    fig, ax = plt.subplots(figsize=(10, 3.8))
    colors = PARETO_COLORS

    for sk in STRATEGIES:
        tps_user, tputs, bs_labels = [], [], []
        for bs in bs_list:
            if bs not in data:
                continue
            w = get_wall(data[bs], sk, seq)
            if w and w > 0:
                tputs.append(bs * 1e9 / w)   # tokens/s
                tps_user.append(1e9 / w)      # 1/s  (= 1 / latency_s)
                bs_labels.append(bs)

        if not tps_user:
            continue

        tps_user = np.array(tps_user)
        tputs = np.array(tputs)

        # 按 x 排序
        order = np.argsort(tps_user)
        tps_user = tps_user[order]
        tputs = tputs[order]
        bs_labels = [bs_labels[i] for i in order]

        # 散点
        ax.scatter(tps_user, tputs, marker=MARKERS[sk], color=colors[sk],
                   s=60, zorder=5, edgecolors="white", linewidths=0.5)

        # 样条拟合
        if len(tps_user) >= 4:
            spl = PchipInterpolator(tps_user, tputs)
            x_smooth = np.linspace(tps_user.min(), tps_user.max(), 200)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, label=LABELS[sk],
                    color=colors[sk], linewidth=2.5,
                    linestyle="--" if "tp2" in sk else "-")
        elif len(tps_user) >= 2:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=2.5,
                    linestyle="--" if "tp2" in sk else "-")
        else:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=2.5, marker=MARKERS[sk])

        # 标注 bs 值 (首尾)
        for i, bs in enumerate(bs_labels):
            if bs == bs_labels[0] or bs == bs_labels[-1]:
                ax.annotate(f"bs={bs}", (tps_user[i], tputs[i]),
                            textcoords="offset points", xytext=(8, -5),
                            fontsize=8, color=colors[sk])

    ax.set_xlabel("TPS/User (1/s)", fontsize=16)
    ax.set_ylabel("Throughput (1/s)", fontsize=16)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, ls="--", alpha=0.3)
    ax.tick_params(labelsize=15)

    fig.tight_layout()
    fname = out_dir / f"fig4_pareto_seq{seq_label(seq)}.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
