#!/usr/bin/env python3
"""
Fig 5: 时间分解 (Time Breakdown)
- Fig 5a: 计算+通信分解 (Compute + Comm)
    - bs=8, 分组柱状图 (按 seq 分组, 策略并排)
    - Compute 部分用策略配色 (仿 fig2), Comm 用亮红
- Fig 5b: 通信细粒度分解
    - 只保留 Ours / H100 TP2 / Rubin TP2
- 数据来源: 96T utilization, AstraSim + Roofline 混合估算
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

# ── 策略与配色 (仿 fig2 策略色) ──
STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "hmp_reo_new"]
LABELS = {
    "hmp_reo_new": "Ours",
    "h100":        "H100",
    "h100_tp2":    "H100 TP2",
    "rubin":       "Rubin",
    "rubin_tp2":   "Rubin TP2",
}
# Compute 部分配色 (仿 fig3 色系: Ours 暖色, 其余冷色)
COMPUTE_COLORS = {
    "h100":        "#CDE2E8",   # 浅青
    "h100_tp2":    "#A8C8D8",   # 中青
    "rubin":       "#C8D4E9",   # 浅蓝紫
    "rubin_tp2":   "#9BB5D6",   # 中蓝
    "hmp_reo_new": "#F59790",   # 暖珊瑚 (与 fig3 RO_new 一致)
}
# Comm 统一用亮红
COMM_COLOR = "#FF4136"

# ── 通信子操作配色 (QKV AG / Attn RS / Final Red 用蓝色系) ──
COMM_COLORS = {
    "qkv_allgather":  "#A8C8D8",   # 蓝色系 - 浅青蓝
    "attn_rs":        "#5B9BD5",   # 蓝色系 - 中蓝
    "attn_comm":      "#E74C3C",
    "output_ar_comm": "#7B241C",
    "output_ag_comm": "#E67E22",
    "output_comm":    "#D35400",
    "final_reduce":   "#2E75B6",   # 蓝色系 - 深蓝
    "allreduce_attn": "#E74C3C",
}
COMM_LABELS = {
    "qkv_allgather":  "QKV AG",
    "attn_rs":        "Attn RS",
    "attn_comm":      "Attn AR",
    "output_ar_comm": "Out AR",
    "output_ag_comm": "Out AG",
    "output_comm":    "Out 2\u00d7AR",
    "final_reduce":   "Final Red",
    "allreduce_attn": "NVLink AR",
}

# ── Fig 5b 只保留这三个策略 ──
COMM_STRATEGIES = ["hmp_reo_new", "h100_tp2", "rubin_tp2"]
COMM_STRAT_LABELS = {
    "hmp_reo_new": "Ours",
    "h100_tp2":    "H100 TP2",
    "rubin_tp2":   "Rubin TP2",
}

# ── Compute 子操作配色 ──
COMP_PART_COLORS = {
    "Proj_QKV": "#5B9BD5",   # 蓝
    "Attn":     "#70AD47",   # 绿
    "Proj_O":   "#FFC000",   # 金
}
COMP_PART_LABELS = {
    "Proj_QKV": "Proj QKV",
    "Attn":     "Attention",
    "Proj_O":   "Proj O",
}

# ── 参数 ──
BS_LIST = [1, 8]
SEQS = [8192, 131072]  # 8K, 128K
BS = 8  # kept for fig5b


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def load_bs(bs):
    path = BASE / f"reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{bs}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_entry(data, strat, seq):
    for e in data.get("strategies", {}).get(strat, {}).get("data", []):
        if e["seq"] == seq:
            return e
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    #  Fig 5a: 2×2 Compute/Comm Breakdown
    #  Rows = seq (8K, 128K), Cols = batch (1, 8)
    #  3 panels share ylim=100; panel [1,1] (bs=8 seq=128K) uses broken axis
    # ==================================================================
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    bs_data = {}
    for bs in BS_LIST:
        d = load_bs(bs)
        if d:
            bs_data[bs] = d
        else:
            print(f"WARNING: bs={bs} data not found, skipping")

    n_strats = len(STRATEGIES)
    bar_w = 0.55
    YLIM = 100          # y-axis cap for 3 regular panels & broken-axis bottom
    BREAK_TOP = (260, 490)   # upper portion of broken axis

    # ── helper: draw stacked bars, return [(si, total, comm), ...] ──
    def _draw_bars(ax, bs, seq):
        data = bs_data.get(bs)
        if not data:
            return []
        xpos = np.arange(n_strats)
        results = []
        for si, sk in enumerate(STRATEGIES):
            e = get_entry(data, sk, seq)
            if not e:
                results.append((si, 0, 0))
                continue
            pq = e["hybrid_Proj_QKV_ns"] / 1e3
            at = e["hybrid_attn_ns"] / 1e3
            po = e["hybrid_Proj_O_ns"] / 1e3
            cm = e["comm_total_ns"] / 1e3
            bot = 0
            for val, key in [(pq, "Proj_QKV"), (at, "Attn"), (po, "Proj_O")]:
                ax.bar(xpos[si], val, bar_w, bottom=bot,
                       color=COMP_PART_COLORS[key], edgecolor="white", linewidth=0.3)
                bot += val
            ax.bar(xpos[si], cm, bar_w, bottom=bot,
                   color=COMM_COLOR, edgecolor="white", linewidth=0.3)
            results.append((si, pq + at + po + cm, cm))
        return results

    # ── helper: annotate Comm% ──
    def _annotate(ax, results, headroom):
        xpos = np.arange(n_strats)
        for si, total, cm in results:
            if total > 0 and cm / total > 0.01:
                ax.text(xpos[si], total + headroom,
                        f"C{cm / total * 100:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color="#C0392B")

    # ── helper: common axis styling ──
    def _style(ax, title, ylabel=False, xticks=True):
        xpos = np.arange(n_strats)
        if xticks:
            ax.set_xticks(xpos)
            ax.set_xticklabels([LABELS[sk] for sk in STRATEGIES],
                               fontsize=12, rotation=25, ha="right")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=11)
        if ylabel:
            ax.set_ylabel("Latency (\u03bcs)", fontsize=14)

    # ── layout ──
    fig_a = plt.figure(figsize=(11, 5.5))
    outer = GridSpec(2, 2, figure=fig_a, hspace=0.55, wspace=0.25)

    # 3 regular panels --------------------------------------------------
    regular = [
        (0, 0, SEQS[0], BS_LIST[0]),   # seq=8K,   bs=1
        (0, 1, SEQS[0], BS_LIST[1]),   # seq=8K,   bs=8
        (1, 0, SEQS[1], BS_LIST[0]),   # seq=128K, bs=1
    ]
    for ri, ci, seq, bs in regular:
        ax = fig_a.add_subplot(outer[ri, ci])
        res = _draw_bars(ax, bs, seq)
        ax.set_ylim(0, YLIM)
        _annotate(ax, res, YLIM * 0.01)
        _style(ax, f"Seq={seq_label(seq)}, BS={bs}", ylabel=(ci == 0))

    # Broken-axis panel [1,1]: seq=128K, bs=8 ---------------------------
    inner = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1, 1],
        height_ratios=[1, 2], hspace=0.10)
    ax_top = fig_a.add_subplot(inner[0])
    ax_bot = fig_a.add_subplot(inner[1])

    # draw bars on both halves (matplotlib clips to ylim)
    _draw_bars(ax_top, BS_LIST[1], SEQS[1])
    res_brk = _draw_bars(ax_bot, BS_LIST[1], SEQS[1])

    ax_top.set_ylim(*BREAK_TOP)
    ax_bot.set_ylim(0, YLIM)

    # annotations: tall bars on top axis, short bars on bottom
    xpos = np.arange(n_strats)
    for si, total, cm in res_brk:
        if total <= 0 or cm / total <= 0.01:
            continue
        pct_txt = f"C{cm / total * 100:.0f}%"
        if total > BREAK_TOP[0]:
            ax_top.text(xpos[si], total + (BREAK_TOP[1] - BREAK_TOP[0]) * 0.02,
                        pct_txt, ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color="#C0392B")
        else:
            ax_bot.text(xpos[si], total + YLIM * 0.01,
                        pct_txt, ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color="#C0392B")

    # cosmetics: hide spines between halves
    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_top.set_xticks([])
    ax_bot.set_xticks(xpos)
    ax_bot.set_xticklabels([LABELS[sk] for sk in STRATEGIES],
                           fontsize=12, rotation=25, ha="right")
    ax_top.set_title(f"Seq={seq_label(SEQS[1])}, BS={BS_LIST[1]}",
                     fontsize=14, fontweight="bold")
    for a in (ax_top, ax_bot):
        a.grid(axis="y", ls="--", alpha=0.25)
        a.tick_params(axis="y", labelsize=11)

    # diagonal break marks
    d = 0.015
    kw = dict(color="k", clip_on=False, lw=1)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)
    ax_bot.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bot.transAxes, **kw)

    # ── Legend ──
    handles_a = [plt.Rectangle((0, 0), 1, 1, fc=COMP_PART_COLORS[k])
                 for k in ["Proj_QKV", "Attn", "Proj_O"]]
    handles_a.append(plt.Rectangle((0, 0), 1, 1, fc=COMM_COLOR))
    legend_labels = [COMP_PART_LABELS[k]
                     for k in ["Proj_QKV", "Attn", "Proj_O"]] + ["Comm"]
    fig_a.legend(handles_a, legend_labels,
                 loc="upper center", ncol=4,
                 fontsize=12, frameon=True, fancybox=True,
                 bbox_to_anchor=(0.5, 1.02))

    fname_a = OUT / "fig5a_compute_breakdown.pdf"
    fig_a.savefig(fname_a, dpi=300, bbox_inches="tight")
    fig_a.savefig(fname_a.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname_a}")
    plt.close(fig_a)

    # ==================================================================
    #  Fig 5b: 通信细粒度分解
    #  只保留 Ours, H100 TP2, Rubin TP2; 只用 seq=8K
    # ==================================================================
    data = load_bs(BS)
    if not data:
        print(f"ERROR: bs={BS} data not found for fig5b")
        return
    SEQ_B = 8192
    n_comm_strats = len(COMM_STRATEGIES)

    # 收集所有 comm ops
    all_ops = set()
    for sk in COMM_STRATEGIES:
        e = get_entry(data, sk, SEQ_B)
        if e:
            for op in e.get("comm_breakdown", {}).get("ops", []):
                all_ops.add(op["name"])

    OP_ORDER = ["qkv_allgather", "attn_rs", "attn_comm",
                "output_ar_comm", "output_ag_comm", "output_comm",
                "final_reduce", "allreduce_attn"]
    op_order = [o for o in OP_ORDER if o in all_ops]

    # Y 轴上限
    comm_ymax = 0
    for sk in COMM_STRATEGIES:
        e = get_entry(data, sk, SEQ_B)
        if e:
            comm_ymax = max(comm_ymax, e["comm_total_ns"])
    comm_ylim = comm_ymax * 1.25

    fig_b, ax_b = plt.subplots(1, 1, figsize=(7, 3.5))
    x_b = np.arange(n_comm_strats)
    w = 0.55

    for sk_idx, sk in enumerate(COMM_STRATEGIES):
        e = get_entry(data, sk, SEQ_B)
        if not e:
            continue
        ops_map = {op["name"]: op["time_ns"]
                   for op in e.get("comm_breakdown", {}).get("ops", [])}
        total_comm = e["comm_total_ns"]

        bottom = 0
        for op_name in op_order:
            val = ops_map.get(op_name, 0)
            if val > 0:
                ax_b.bar(x_b[sk_idx], val, w, bottom=bottom,
                         color=COMM_COLORS.get(op_name, "#999"),
                         label=COMM_LABELS.get(op_name, op_name)
                               if sk_idx == 0 else "")
                bottom += val

        # 标注总通信
        if total_comm > 0:
            ax_b.text(x_b[sk_idx], total_comm + comm_ylim * 0.02,
                      f"{total_comm:.0f}", ha="center", va="bottom",
                      fontsize=9, fontweight="bold")
        else:
            ax_b.text(x_b[sk_idx], comm_ylim * 0.02, "0",
                      ha="center", va="bottom", fontsize=9, color="#999")

    # 箭头标注: 从 H100 TP2 柱顶指向 Ours 柱顶, 文字在线中间偏上
    ours_e = get_entry(data, "hmp_reo_new", SEQ_B)
    h100tp2_e = get_entry(data, "h100_tp2", SEQ_B)
    if ours_e and h100tp2_e:
        ours_comm = ours_e["comm_total_ns"]
        h100tp2_comm = h100tp2_e["comm_total_ns"]
        reduction = (1.0 - ours_comm / h100tp2_comm) * 100
        ours_idx = COMM_STRATEGIES.index("hmp_reo_new")
        h100tp2_idx = COMM_STRATEGIES.index("h100_tp2")
        x0, y0 = x_b[h100tp2_idx], h100tp2_comm * 0.85
        x1, y1 = x_b[ours_idx], ours_comm
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        # 上半段: 起点 → 中点 (无箭头)
        ax_b.annotate("", xy=(mid_x, mid_y), xytext=(x0, y0),
                       arrowprops=dict(arrowstyle="-", color="#C0392B", lw=2))
        # 下半段: 中点 → 终点 (带箭头)
        ax_b.annotate("", xy=(x1, y1), xytext=(mid_x, mid_y),
                       arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.15",
                                       color="#C0392B", lw=2))
        # 文字放在中点上方, 避免与线重合
        ax_b.text(mid_x + 0.08, mid_y + comm_ylim * 0.06,
                  f"\u2212{reduction:.0f}%",
                  fontsize=12, fontweight="bold", color="#C0392B",
                  ha="center", va="bottom")

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([COMM_STRAT_LABELS[sk] for sk in COMM_STRATEGIES],
                         fontsize=10)
    ax_b.set_ylabel("Communication (ns)", fontsize=12)
    ax_b.grid(axis="y", ls="--", alpha=0.25)
    ax_b.tick_params(axis="y", labelsize=10)
    ax_b.set_ylim(0, comm_ylim)

    # 图例
    seen = set()
    handles_b, labels_b = [], []
    for op_name in op_order:
        if op_name not in seen:
            handles_b.append(plt.Rectangle((0, 0), 1, 1,
                             fc=COMM_COLORS.get(op_name, "#999")))
            labels_b.append(COMM_LABELS.get(op_name, op_name))
            seen.add(op_name)
    ax_b.legend(handles_b, labels_b,
                loc="upper left", fontsize=9, frameon=True)
    fig_b.tight_layout()

    fname_b = OUT / "fig5b_comm_breakdown.pdf"
    fig_b.savefig(fname_b, dpi=300, bbox_inches="tight")
    fig_b.savefig(fname_b.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname_b}")
    plt.close(fig_b)


if __name__ == "__main__":
    main()
