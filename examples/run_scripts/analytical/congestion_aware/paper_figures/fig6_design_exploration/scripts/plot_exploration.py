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
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ── 路径 ──
DATA_UTIL = Path(__file__).resolve().parents[1] / "data" / "sweep_with_util.json"
DATA_RAW  = Path(__file__).resolve().parents[1] / "data" / "sweep_compute_bw_batch.json"
OUT = Path(__file__).resolve().parents[1] / "plots"


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
    #  Fig 6a: 热力图 (4 batch)
    # ==================================================================
    n_bs = len(bs_vals)
    fig, axes = plt.subplots(1, n_bs, figsize=(5.5 * n_bs, 5))

    for bi, bs in enumerate(bs_vals):
        ax = axes[bi]
        grid = np.zeros((len(sa_vals), len(lbw_vals)))
        for ci, pp in enumerate(sa_vals):
            for li, lbw in enumerate(lbw_vals):
                e = lookup.get((pp, lbw, bs))
                grid[ci, li] = e["wall_ns"] / 1e3 if e else 0

        # 自定义色彩: 深紫红(高值/慢) → 浅肉色(中) → 藏青(低值/快)
        custom_colors = [
            "#053061", "#134b87", "#327db7", "#6fafd2", "#c7e0ed",
            "#fbd2bc", "#feab88", "#b71c2c", "#8b0824", "#6a0624",
        ]
        custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", custom_colors, N=256)
        norm = mcolors.LogNorm(vmin=max(grid.min(), 1), vmax=grid.max())
        im = ax.imshow(grid, origin="lower", aspect="auto",
                       cmap=custom_cmap, norm=norm,
                       extent=[-0.5, len(lbw_vals)-0.5, -0.5, len(sa_vals)-0.5])

        for ci in range(len(sa_vals)):
            for li in range(len(lbw_vals)):
                val = grid[ci, li]
                if val > 0:
                    txt = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
                    # 浅色背景用黑字, 深色背景用白字
                    ratio = (np.log(val) - np.log(grid.min())) / (np.log(grid.max()) - np.log(grid.min())) if grid.max() > grid.min() else 0.5
                    ax.text(li, ci, txt, ha="center", va="center",
                            fontsize=8, fontweight="bold",
                            color="white" if ratio > 0.55 or ratio < 0.15 else "black")

        ax.set_xticks(range(len(lbw_vals)))
        ax.set_xticklabels([f"{v}" for v in lbw_vals], fontsize=11)
        ax.set_yticks(range(len(sa_vals)))
        ax.set_yticklabels([f"{v}T" for v in sa_vals], fontsize=11)
        ax.set_xlabel("D2D Link BW (TB/s)", fontsize=13)
        ax.set_title(f"bs={bs}", fontsize=14, fontweight="bold")
        cbar = plt.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("Wall Time (μs)", fontsize=11)

    axes[0].set_ylabel("Compute Power (TFLOPS/NPU)", fontsize=13)
    fig.suptitle("Design Space: Wall Time with Utilization (Qwen3-235B, seq=64K, RO_new)",
                 fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    fname = OUT / "fig6a_heatmap_util.pdf"
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
