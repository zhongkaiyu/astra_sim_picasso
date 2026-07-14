#!/usr/bin/env python3
"""
Fig 6: 设计空间探索 — Compute Power × D2D BW × Batch Size (含 utilization)

数据来源:
  sweep_with_util.json (由 sweep_with_util.py 生成)
  计算流程: util.py → roofline_gqa_calc.py → merge_gqa_results.py
  模型: Qwen3-235B, seq=64K, 策略: RO_new (hmp_reo_new)

图表:
  - Fig 6a: 热力图 (Compute × D2D BW), 4 个 batch
  - Fig 6b: 算力缩放 + GPU/Comm 分解
  - Fig 6c: D2D BW 缩放
  - Fig 6d: Raw Roofline vs Util-Adjusted 对比
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parents[4]  # congestion_aware/
DATA_UTIL = Path(__file__).resolve().parents[1] / "data" / "sweep_with_util.json"
DATA_RAW  = Path(__file__).resolve().parents[1] / "data" / "sweep_compute_bw_batch.json"
OUT = Path(__file__).resolve().parents[1] / "plots"

# ── fig6d: D2D 固定延时扫描所需的 roofline 后端 ──
sys.path.insert(0, str(BASE / "backend" / "roofline"))
from roofline_gqa_calc import calc_strategy
from attention import MODEL_CONFIGS


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    with open(DATA_UTIL) as f:
        sweep = json.load(f)
    with open(DATA_RAW) as f:
        raw_sweep = json.load(f)

    entries = sweep["data"]
    sa_vals = sweep["_parameters"]["num_sa_tflops"]
    lbw_vals = sweep["_parameters"]["link_bw_tbs"]
    bs_vals = sweep["_parameters"]["batch_sizes"]

    lookup = {(e["peak_perf"], e["link_bw"], e["batch"]): e for e in entries}
    raw_lookup = {(e["peak_perf"], e["link_bw"], e["batch"]): e for e in raw_sweep["data"]}

    colors_bs = {1: "#1B4F72", 4: "#2E86C1", 16: "#85C1E9", 32: "#D4E6F1"}

    # ==================================================================
    #  Fig 6a: 热力图 (只保留 BS=1, BS=16)
    #
    #  布局: 1×2 横向子图, 每个子图对应一个 batch size。
    #  - 横轴: D2D Link BW (TB/s)      — lbw_vals
    #  - 纵轴: Compute Power (TFLOPS)   — sa_vals
    #  - 色彩: 绿(低延迟/快) → 黄(中) → 红(高延迟/慢), 对数归一化
    #  - 格内数字: 延迟值 (μs), ≥10 取整, <10 保留一位小数
    #  - colorbar: 2 个子图共用同一个, 放在图的最右侧
    # ==================================================================
    bs_show = [bs for bs in (1, 16) if bs in bs_vals]
    n_bs = len(bs_show)

    # ── 第一遍: 遍历待显示 batch, 构建网格并收集全局 min/max ──
    #    用于建立统一的 LogNorm, 使 2 个子图共享同一色彩范围
    grids = {}
    global_min, global_max = np.inf, -np.inf
    for bs in bs_show:
        grid = np.zeros((len(sa_vals), len(lbw_vals)))
        for ci, pp in enumerate(sa_vals):
            for li, lbw in enumerate(lbw_vals):
                e = lookup.get((pp, lbw, bs))
                grid[ci, li] = e["wall_ns"] / 1e3 if e else 0
        grids[bs] = grid
        # 只统计 >0 的值来确定色彩范围
        valid = grid[grid > 0]
        if valid.size > 0:
            global_min = min(global_min, valid.min())
            global_max = max(global_max, valid.max())

    # ── 自定义色彩映射: 绿 → 黄 → 红 (10 个锚点, 256 级插值) ──
    custom_colors = [
       "#8FB4BE", "#AFC9CF", "#D5E1E3", "#EBBFC2", "#E28187", "#D93F49"
    ]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", custom_colors, N=256)

    # ── 全局统一的对数归一化 ──
    shared_norm = mcolors.LogNorm(vmin=max(global_min, 1), vmax=global_max)

    # ── 创建 figure, 预留右侧空间给共享 colorbar ──
    fig, axes = plt.subplots(1, n_bs, figsize=(5.5 * n_bs, 5.6))
    axes = np.atleast_1d(axes)

    for bi, bs in enumerate(bs_show):
        ax = axes[bi]
        grid = grids[bs]

        # ── 绘制热力图 ──
        #    aspect="auto": 格子自动拉伸填满子图区域
        #    origin="lower": y 轴从下到上递增 (低算力在底部)
        #    extent: 像素中心对齐到整数坐标 (-0.5 偏移)
        #    set_box_aspect(1): 强制子图本身为正方形
        ax.set_box_aspect(1)
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       cmap=custom_cmap, norm=shared_norm,
                       extent=[-0.5, len(lbw_vals)-0.5, -0.5, len(sa_vals)-0.5])

        # ── 在每个格子中标注延迟数值 ──
        for ci in range(len(sa_vals)):
            for li in range(len(lbw_vals)):
                val = grid[ci, li]
                if val > 0:
                    # 格式: ≥10 显示整数, <10 保留一位小数
                    txt = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
                    # 根据数值在 log 色彩范围中的位置选择字体颜色:
                    #   绿色区域 (ratio<0.25) 和红色区域 (ratio>0.65) 用白字
                    #   中间黄色区域用黑字, 保证可读性
                    ratio = (np.log(val) - np.log(global_min)) / (np.log(global_max) - np.log(global_min)) if global_max > global_min else 0.5
                    ax.text(li, ci, txt, ha="center", va="center",
                            fontsize=24, fontweight="normal",
                            color="white" if ratio > 0.65 or ratio < 0.25 else "black")

        # ── 坐标轴刻度与标签 ──
        #    1×2 单行布局: 两个子图都显示 x 刻度; 仅首列显示 y 刻度
        col = bi

        ax.set_xticks(range(len(lbw_vals)))
        ax.set_yticks(range(len(sa_vals)))

        ax.set_xticklabels([f"{v}" for v in lbw_vals], fontsize=24, fontweight="normal")
        ax.set_xlabel("")

        if col == 0:
            ax.set_yticklabels([f"{v}T" for v in sa_vals], fontsize=24, fontweight="normal")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.set_title(f"BS={bs}", fontsize=26, fontweight="normal")

    # ── 共享轴标题 ──
    fig.tight_layout(rect=[0.06, 0.08, 0.90, 1])
    fig.text(0.045, 0.54, "Compute Power (TFLOPS/NPU)", fontsize=24,
             ha="center", va="center", rotation=90)
    fig.text(0.46, 0.045, "D2D Link BW (TB/s)", fontsize=24,
             ha="center", va="center")
    # cbar_ax 位置: 与子图上下边界对齐
    pos = axes[-1].get_position()
    cbar_ax = fig.add_axes([0.915, pos.y0, 0.022, pos.y1 - pos.y0])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Latency (μs)", fontsize=26, fontweight="normal")
    cbar.ax.tick_params(labelsize=24)

    fname = OUT / "fig6a_heatmap_wall.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)

    # ==================================================================
    #  Fig 6d: 延时 vs D2D Link 固定延时 (hop latency), D2D 带宽固定
    #
    #  参数与 fig1_speedup_4panel.pdf 一致:
    #    - 模型: Qwen3-235B, 策略 Ours (hmp_reo_new)
    #    - 算力 96 TFLOPS/NPU, HBM 2.5TB/s, util96, D2D BW = 1.5 TB/s (固定)
    #    - util 调整后的 compute (hybrid_gpu_ns) 取自 fig1 的 hybrid report
    #  扫描: D2D Link 固定延时 (per-hop latency) 15→1000 ns
    #    wall(hop) = hybrid_gpu_ns(seq) + comm_total_ns(hop)
    #    comm 由 roofline_gqa_calc.calc_strategy 解析计算 (D2D BW=1.5 固定)
    #  布局: 1×2, 对应 BS=1 / BS=16, 每条折线为一个 sequence length
    # ==================================================================
    LINK_BW_FIXED = 1.5          # D2D 带宽固定 (TB/s), 与 fig1 一致
    HOP_VALS = [15, 100, 200, 400, 600, 800, 1000]   # D2D Link 固定延时 (ns)
    SEQS_D2D = [4096, 65536, 262144, 1048576]
    SEQ_LABELS = {4096: "4K", 65536: "64K", 262144: "256K", 1048576: "1M"}
    colors_seq = {4096: "#1B4F72", 65536: "#2E86C1",
                  262144: "#E07850", 1048576: "#C04020"}
    mc = MODEL_CONFIGS["qwen3"]
    hw96 = {"compute": 96, "Bandwidth": 2.5}

    def report_gpu_map(bs):
        path = (BASE / "reports" / "qwen3-235B" / "hybrid"
                / f"gqa_hybrid_merged_96T_bw1500_util96_bs{bs}.json")
        with open(path) as f:
            rep = json.load(f)
        return {e["seq"]: e["hybrid_gpu_ns"]
                for e in rep["strategies"]["hmp_reo_new"]["data"]}

    def comm_at_hop(bs, hop):
        r = calc_strategy("hmp_reo_new", hw96, bs, [SEQS_D2D[0]],
                          link_bw=LINK_BW_FIXED, hop_latency_ns=hop,
                          endpoint_delay_ns=10, model_config=mc)
        return r["comm"]["total_comm_ns"]  # comm 与 seq 无关

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=False)
    legend_handles = None
    for col, bs in enumerate([1, 16]):
        ax = axes[col]
        gpu_map = report_gpu_map(bs)
        comm_by_hop = {h: comm_at_hop(bs, h) for h in HOP_VALS}
        for seq in SEQS_D2D:
            gpu = gpu_map[seq]
            walls = [(gpu + comm_by_hop[h]) / 1e3 for h in HOP_VALS]  # μs
            ax.plot(HOP_VALS, walls, marker="o", linewidth=2.4, markersize=8,
                    color=colors_seq[seq], label=f"seq={SEQ_LABELS[seq]}")

        ax.set_yscale("log")
        ax.axvline(15, color="#7F8C8D", ls="--", lw=1.2, alpha=0.7)
        ax.set_xlabel("D2D Link Latency (ns)", fontsize=26)
        if col == 0:
            ax.set_ylabel("Latency (μs)", fontsize=26)
        ax.set_title(f"BS={bs}", fontsize=24, fontweight="normal", pad=8)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.tick_params(axis="both", labelsize=24)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    # ── 单一图例: 横置于两个 BS 子图标题之上 ──
    fig.legend(legend_handles, legend_labels, loc="upper center",
               ncol=len(SEQS_D2D), fontsize=26, frameon=True,
               bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fname = OUT / "fig6d_d2d_latency_scaling.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)

    # ==================================================================
    #  Fig 6b: Compute 缩放 (D2D=1.5)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    target_lbw = 1.5

    # 左: Wall Time vs Compute, 不同 batch
    ax = axes[0]
    for bs in bs_vals:
        pps, walls = [], []
        for pp in sa_vals:
            e = lookup.get((pp, target_lbw, bs))
            if e:
                pps.append(pp)
                walls.append(e["wall_ns"] / 1e3)
        ax.plot(pps, walls, marker="o", label=f"bs={bs}",
                color=colors_bs[bs], linewidth=2, markersize=7)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Compute Power (TFLOPS/NPU)", fontsize=13)
    ax.set_ylabel("Wall Time (μs)", fontsize=13)
    ax.set_title(f"Wall Time vs Compute (D2D={target_lbw}TB/s)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)

    # 右: Raw vs Util-Adjusted (bs=1, D2D=1.5)
    ax = axes[1]
    raw_walls, util_walls = [], []
    for pp in sa_vals:
        r = raw_lookup.get((pp, target_lbw, 1))
        u = lookup.get((pp, target_lbw, 1))
        raw_walls.append(r["wall_ns"] / 1e3 if r else 0)
        util_walls.append(u["wall_ns"] / 1e3 if u else 0)

    x = np.arange(len(sa_vals))
    w = 0.35
    ax.bar(x - w/2, raw_walls, w, label="Raw Roofline", color="#3498DB", alpha=0.8)
    ax.bar(x + w/2, util_walls, w, label="Util-Adjusted", color="#E74C3C", alpha=0.8)

    # 标注 overhead
    for i in range(len(sa_vals)):
        if raw_walls[i] > 0:
            ratio = util_walls[i] / raw_walls[i]
            ax.text(i, max(raw_walls[i], util_walls[i]) + 0.5,
                    f"{ratio:.2f}×", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}T" for v in sa_vals], fontsize=11)
    ax.set_xlabel("Compute Power (TFLOPS/NPU)", fontsize=13)
    ax.set_ylabel("Wall Time (μs)", fontsize=13)
    ax.set_title("Raw Roofline vs Util-Adjusted (bs=1)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.3)

    fig.suptitle("Compute Scaling Analysis (Qwen3-235B, seq=64K, RO_new)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = OUT / "fig6b_compute_scaling_util.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)

    # ==================================================================
    #  Fig 6c: D2D BW 缩放 (Compute=96T)
    # ==================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    target_pp = 96

    # 左: Wall Time vs D2D BW
    ax = axes[0]
    for bs in bs_vals:
        bws, walls = [], []
        for lbw in lbw_vals:
            e = lookup.get((target_pp, lbw, bs))
            if e:
                bws.append(lbw)
                walls.append(e["wall_ns"] / 1e3)
        ax.plot(bws, walls, marker="s", label=f"bs={bs}",
                color=colors_bs[bs], linewidth=2, markersize=7)

    ax.set_xlabel("D2D Link BW (TB/s)", fontsize=13)
    ax.set_ylabel("Wall Time (μs)", fontsize=13)
    ax.set_title(f"Wall Time vs D2D BW ({target_pp}T)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)

    # 右: Comm Time vs D2D BW
    ax = axes[1]
    for bs in bs_vals:
        bws, comms = [], []
        for lbw in lbw_vals:
            e = lookup.get((target_pp, lbw, bs))
            if e:
                bws.append(lbw)
                comms.append(e["comm_ns"])
        ax.plot(bws, comms, marker="s", label=f"bs={bs}",
                color=colors_bs[bs], linewidth=2, markersize=7)

    ax.set_xlabel("D2D Link BW (TB/s)", fontsize=13)
    ax.set_ylabel("Communication (ns)", fontsize=13)
    ax.set_title(f"Comm vs D2D BW ({target_pp}T)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, ls="--", alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)

    fig.suptitle("D2D Bandwidth Scaling (Qwen3-235B, seq=64K, RO_new)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = OUT / "fig6c_d2d_scaling_util.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
