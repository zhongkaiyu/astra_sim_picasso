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

BASE = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "plots"

STRATEGIES = ["rubin", "rubin_tp2", "h100", "h100_tp2", "hmp_reo_new"]
LABELS = {"rubin": "Rubin", "rubin_tp2": "Rubin TP2",
          "h100": "H100", "h100_tp2": "H100 TP2", "hmp_reo_new": "Ours"}
COLORS = {"rubin": "#8fc7de", "rubin_tp2": "#ffd680",
          "h100": "#fa874f", "h100_tp2": "#d93026", "hmp_reo_new": "#4a7dba"}
MARKERS = {"rubin": "o", "rubin_tp2": "v", "h100": "X", "h100_tp2": "p", "hmp_reo_new": "P"}

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
    # 字号配置 (增大并加粗)
    AXIS_LABEL = 28
    TICK_SIZE = 26
    PANEL_LABEL = 28   # (a) (b) (c) 子图标签字号
    LEGEND_SIZE = 28

    fig, axes = plt.subplots(1, 3, figsize=(15.54, 6.5),
                             gridspec_kw={'width_ratios': [1.2, 1.2, 1.5]})
    colors = PARETO_COLORS  # unified color scheme for all panels

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
                    color=colors[sk], linewidth=3.5, markersize=8,
                    linestyle="--" if "tp2" in sk else "-")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=AXIS_LABEL, fontweight="normal")
    ax.set_ylabel("Tput(tok/μs)", fontsize=AXIS_LABEL, fontweight="normal")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE)
    # 子图标签放在下方 (使用 text + 相对坐标)
    ax.text(0.5, -0.32, "(a)", transform=ax.transAxes,
            fontsize=PANEL_LABEL, fontweight="normal", ha="center", va="top")

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
                    color=colors[sk], linewidth=3.5, markersize=8,
                    linestyle="--" if "tp2" in sk else "-")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Batch Size", fontsize=AXIS_LABEL, fontweight="normal")
    ax.set_ylabel("Latency (ns)", fontsize=AXIS_LABEL, fontweight="normal")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.text(0.5, -0.32, "(b)", transform=ax.transAxes,
            fontsize=PANEL_LABEL, fontweight="normal", ha="center", va="top")

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
                   s=70, zorder=5, edgecolors="white", linewidths=0.5)

        if len(tps_user) >= 4:
            spl = PchipInterpolator(tps_user, tputs)
            x_smooth = np.linspace(tps_user.min(), tps_user.max(), 200)
            y_smooth = spl(x_smooth)
            ax.plot(x_smooth, y_smooth, label=LABELS[sk],
                    color=colors[sk], linewidth=3.5,
                    linestyle="--" if "tp2" in sk else "-")
        elif len(tps_user) >= 2:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=3.5,
                    linestyle="--" if "tp2" in sk else "-")
        else:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=3.5, marker=MARKERS[sk])
        if sk in ("hmp_reo_new", "rubin_tp2"):
            for i, bs in enumerate(bs_labels):
                if bs == bs_labels[0] or bs == bs_labels[-1]:
                    is_last = (bs == bs_labels[-1])  # rightmost point
                    if sk == "rubin_tp2":
                        ofs = (-60, -18) if is_last else (6, -18)
                    else:
                        ofs = (-60, -4) if is_last else (6, -4)
                    ax.annotate(
                        f"BS={bs}",
                        (tps_user[i], tputs[i]),
                        textcoords="offset points",
                        xytext=ofs,
                        fontsize=20,
                        color=colors[sk]
                    )
    ax.set_xlabel("TPS/User (1/s)", fontsize=AXIS_LABEL, fontweight="normal")
    ax.set_ylabel("Tput(tok/s)", fontsize=AXIS_LABEL, fontweight="normal")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(_fmt_k))
    ax.grid(True, ls="--", alpha=0.3)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.yaxis.set_major_locator(ticker.FixedLocator([0, 200_000, 400_000]))
    plt.setp(ax.get_yticklabels(), rotation=90, va='center')
    ax.text(0.5, -0.32, "(c)", transform=ax.transAxes,
            fontsize=PANEL_LABEL, fontweight="normal", ha="center", va="top")

    # ── 共享图例 (横铺在顶部) ──
    handles = []
    for sk in STRATEGIES:
        handles.append(plt.Line2D([0], [0], marker=MARKERS[sk], color=colors[sk],
                                   linewidth=3.5, markersize=8,
                                   linestyle="--" if "tp2" in sk else "-",
                                   label=LABELS[sk]))
    fig.legend(handles=handles, loc="upper center", ncol=len(STRATEGIES),
               fontsize=LEGEND_SIZE, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.08))

    fig.tight_layout(rect=[0, 0.05, 1, 0.93], w_pad=0.5)
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
                    color=colors[sk], linewidth=3.5,
                    linestyle="--" if "tp2" in sk else "-")
        elif len(tps_user) >= 2:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=3.5,
                    linestyle="--" if "tp2" in sk else "-")
        else:
            ax.plot(tps_user, tputs, label=LABELS[sk],
                    color=colors[sk], linewidth=3.5, marker=MARKERS[sk])

        # 标注 bs 值 (首尾)
        for i, bs in enumerate(bs_labels):
            if bs == bs_labels[0] or bs == bs_labels[-1]:
                ax.annotate(f"BS={bs}", (tps_user[i], tputs[i]),
                            textcoords="offset points", xytext=(8, -5),
                            fontsize=20, color=colors[sk])

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
