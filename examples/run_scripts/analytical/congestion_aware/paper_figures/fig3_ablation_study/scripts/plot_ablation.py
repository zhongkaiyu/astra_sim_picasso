#!/usr/bin/env python3
"""
Fig 3: 策略消融实验 (Ablation Study)
- Fig 3a: E2E Latency Speedup vs TP16 (分组柱状图, 3 策略 × 3 seq)
- Fig 3b: Communication Speedup vs TP16 (分组柱状图, 3 策略 × 3 seq)
- 数据来源: 96T utilization 配置
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parents[3]  # congestion_aware/
OUT = Path(__file__).resolve().parents[1] / "plots"

# ── 数据源 ──
DATA_PATH = BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_split4_bw1500_util96.json"

# ── 策略定义 (顺序: TP16, HMP, RO_new) ──
STRATEGIES = ["tp16", "hmp", "hmp_reo_new"]
LABELS = {"hmp_reo_new": "RO_new", "hmp": "HMP", "tp16": "TP16"}
COLORS = {
    "tp16":        "#CDE2E8",
    "hmp":         "#C8D4E9",
    "hmp_reo_new": "#F59790",
}
BASELINE = "tp16"

# ── 序列长度 ──
SEQS = [8192, 262144, 1048576]


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def load(path):
    with open(path) as f:
        return json.load(f)


def get_wall_map(data, strat):
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["hybrid_wall_ns"] for e in entries}


def get_comm_map(data, strat):
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["comm_total_ns"] for e in entries}


def plot_speedup_chart(fig, ax, speedup_data, max_val, title):
    """绘制分组柱状图 (fig1 风格), speedup_data = {strat: [speedup_per_seq]}"""
    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)
    x = np.arange(n_seqs)

    total_width = 0.78
    bar_w = total_width / n_strats
    offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

    labels_by_group = {i: [] for i in range(n_seqs)}

    for si, sk in enumerate(STRATEGIES):
        ratios = speedup_data[sk]
        ax.bar(x + offsets[si], ratios, bar_w,
               color=COLORS[sk], edgecolor="white", linewidth=0.3)
        for i, r in enumerate(ratios):
            if r > 0:
                labels_by_group[i].append((x[i] + offsets[si], r, f"{r:.1f}"))  # 只显示数字

    ax.axhline(1.0, color="#7F8C8D", ls="--", lw=1, alpha=0.6)
    ax.set_ylim(top=max_val * 1.25)

    # 智能标注: 防重叠
    min_gap = max_val * 0.06
    for grp in labels_by_group.values():
        grp.sort(key=lambda t: t[1])
        y_positions = [item[1] for item in grp]
        for j in range(1, len(y_positions)):
            if y_positions[j] - y_positions[j - 1] < min_gap:
                y_positions[j] = y_positions[j - 1] + min_gap
        for j, (xp, _, txt) in enumerate(grp):
            ax.text(xp, y_positions[j] + 0.05,
                    txt, ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color="black",
                    rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([seq_label(s) for s in SEQS], fontsize=16)  # 横轴刻度加大
    ax.set_xlabel("Sequence Length", fontsize=18)                   # 横轴标题加大
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=15)                          # 纵轴刻度加大


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load(DATA_PATH)

    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)
    x = np.arange(n_seqs)
    total_width = 0.78
    bar_w = total_width / n_strats
    offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

    # ── 准备数据 ──
    baseline_wall = get_wall_map(data, BASELINE)
    wall_speedups = {}
    max_wall = 0
    for sk in STRATEGIES:
        wm = get_wall_map(data, sk)
        ratios = []
        for s in SEQS:
            bv = baseline_wall.get(s, 0)
            sv = wm.get(s, 0)
            r = bv / sv if sv > 0 and bv > 0 else 0
            ratios.append(r)
            if r > max_wall:
                max_wall = r
        wall_speedups[sk] = ratios

    baseline_comm = get_comm_map(data, BASELINE)
    comm_speedups = {}
    max_comm = 0
    for sk in STRATEGIES:
        cm = get_comm_map(data, sk)
        ratios = []
        for s in SEQS:
            bv = baseline_comm.get(s, 0)
            sv = cm.get(s, 0)
            r = bv / sv if sv > 0 and bv > 0 else 0
            ratios.append(r)
            if r > max_comm:
                max_comm = r
        comm_speedups[sk] = ratios

    # ── 组合图: 左 fig3a, 右 fig3b (断轴) ──
    BREAK_LO = 20
    BREAK_HI_PAD = 1.20
    upper_max = max_comm * BREAK_HI_PAD

    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 2.8))
    # 左列: 1 个 axes 跨 4 行;  右列: 上段 1 行 + 下段 3 行
    gs = GridSpec(4, 2, figure=fig, hspace=0.08, wspace=0.38,
                  height_ratios=[1, 1, 1, 1])
    ax_a     = fig.add_subplot(gs[:, 0])        # 左: 整列
    ax_r_top = fig.add_subplot(gs[0, 1])        # 右上
    ax_r_bot = fig.add_subplot(gs[1:, 1], sharex=ax_r_top)  # 右下

    # ==================================================================
    #  左: E2E Latency Speedup (fig3a)
    # ==================================================================
    plot_speedup_chart(fig, ax_a, wall_speedups, max_wall, "")
    ax_a.set_ylabel("Latency Speedup", fontsize=18)  # 纵轴标题加大

    # ==================================================================
    #  右: Comm Speedup (fig3b, 断轴)
    # ==================================================================
    for si, sk in enumerate(STRATEGIES):
        ratios = comm_speedups[sk]
        for a in (ax_r_top, ax_r_bot):
            a.bar(x + offsets[si], ratios, bar_w,
                  color=COLORS[sk], edgecolor="white", linewidth=0.3)

    ax_r_bot.set_ylim(0, BREAK_LO)
    ax_r_top.set_ylim(upper_max - (upper_max - BREAK_LO) * 1.1, upper_max)

    ax_r_top.spines["bottom"].set_visible(False)
    ax_r_bot.spines["top"].set_visible(False)
    ax_r_top.tick_params(bottom=False)

    # 断轴斜线
    d = 0.012
    kwargs = dict(transform=ax_r_top.transAxes, color="black", clip_on=False, lw=1)
    ax_r_top.plot((-d, +d), (0 - d, 0 + d), **kwargs)
    ax_r_top.plot((1 - d, 1 + d), (0 - d, 0 + d), **kwargs)
    kwargs["transform"] = ax_r_bot.transAxes
    ax_r_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_r_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    ax_r_bot.axhline(1.0, color="#7F8C8D", ls="--", lw=1, alpha=0.6)

    # 标注
    labels_by_group = {i: [] for i in range(n_seqs)}
    for si, sk in enumerate(STRATEGIES):
        for i, r in enumerate(comm_speedups[sk]):
            if r > 0:
                labels_by_group[i].append((x[i] + offsets[si], r, f"{r:.1f}"))  # 只显示数字

    min_gap_bot = BREAK_LO * 0.08
    min_gap_top = (upper_max - BREAK_LO) * 0.12
    for grp in labels_by_group.values():
        grp.sort(key=lambda t: t[1])
        y_positions = [item[1] for item in grp]
        for j in range(1, len(y_positions)):
            gap = min_gap_top if y_positions[j] > BREAK_LO else min_gap_bot
            if y_positions[j] - y_positions[j - 1] < gap:
                y_positions[j] = y_positions[j - 1] + gap
        for j, (xp, raw_y, txt) in enumerate(grp):
            target_ax = ax_r_top if raw_y > BREAK_LO else ax_r_bot
            target_ax.text(xp, y_positions[j] + 0.3,
                           txt, ha="center", va="bottom",
                           fontsize=8, fontweight="bold", color="black",
                           rotation=0)

    ax_r_bot.set_xticks(x)
    ax_r_bot.set_xticklabels([seq_label(s) for s in SEQS], fontsize=16)  # 横轴刻度加大
    ax_r_bot.set_xlabel("Sequence Length", fontsize=18)                   # 横轴标题加大
    ax_r_bot.grid(axis="y", ls="--", alpha=0.25)
    ax_r_top.grid(axis="y", ls="--", alpha=0.25)
    ax_r_bot.tick_params(axis="y", labelsize=15)                          # 纵轴刻度加大
    ax_r_top.tick_params(axis="y", labelsize=15)

    # 右侧 y 轴标签 — 与左侧对齐, 缩写 Communication → Comm
    ax_a.yaxis.set_label_coords(-0.12, 0.5)
    ax_r_bot.set_ylabel("Comm Speedup", fontsize=18)                      # 纵轴标题加大
    ax_r_bot.yaxis.set_label_coords(-0.12, 0.85)

    # ── 图例放在左侧图上方 ──
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    ax_a.legend(handles, legend_labels,
                loc="upper center", ncol=len(STRATEGIES),
                fontsize=12, frameon=True, fancybox=True,
                bbox_to_anchor=(0.5, 1.15))

    fname = OUT / "fig3_ablation.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
