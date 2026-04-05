#!/usr/bin/env python3
"""
Fig 2c: 瞬时功耗 Power (W)
- 与 fig2b (Token/J) 形成互补: 2b 展示能效, 2c 展示功耗代价
- 传递信息: "Ours 功耗更高, 但因速度快, 总能耗反而低"
- 布局严格对齐 fig2b: 2×4 子图, 相同配色/间距/标注风格
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1] / "plots"

# ── 策略定义与配色 (严格对齐 fig2b) ──
STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "hmp_reo_new"]
LABELS = {
    "hmp_reo_new": "Ours",
    "rubin":       "Rubin",
    "rubin_tp2":   "Rubin TP2",
    "h100":        "H100",
    "h100_tp2":    "H100 TP2",
}
COLORS = {
    "h100":        "#D4D4D4",
    "h100_tp2":    "#DBDDEF",
    "rubin":       "#92B1D9",
    "rubin_tp2":   "#C1D8E9",
    "hmp_reo_new": "#E07850",
}

# ── Batch sizes (4 列) ──
BATCH_SIZES = [1, 4, 16, 32]

# ── 模型功耗数据路径 ──
MODELS = [
    ("Qwen3-235B", {
        1:  str(BASE / "reports/qwen3-235B/power/gqa_hybrid_merged_96T_bw1500_util96_bs1_power.json"),
        4:  str(BASE / "reports/qwen3-235B/power/gqa_hybrid_merged_96T_bw1500_util96_bs4_power.json"),
        16: str(BASE / "reports/qwen3-235B/power/gqa_hybrid_merged_96T_bw1500_util96_bs16_power.json"),
        32: str(BASE / "reports/qwen3-235B/power/gqa_hybrid_merged_96T_bw1500_util96_bs32_power.json"),
    }),
    ("Llama4-Maverick", {
        1:  str(BASE / "reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_bs1_power.json"),
        4:  str(BASE / "reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_bs4_power.json"),
        16: str(BASE / "reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_bs16_power.json"),
        32: str(BASE / "reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_bs32_power.json"),
    }),
]

# ── 序列长度 ──
SEQS = [4096, 65536, 262144, 1048576]


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def load(path):
    with open(path) as f:
        return json.load(f)


def get_power_map(data, strat):
    """提取 {seq: total_power_w}"""
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["total_power_w"] for e in entries}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_models = len(MODELS)
    n_batches = len(BATCH_SIZES)
    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)

    fig, axes = plt.subplots(n_models, n_batches, figsize=(3.5 * n_batches, 2.0 * n_models),
                             sharey='row')

    for row, (model_label, bs_paths) in enumerate(MODELS):
        for col, bs in enumerate(BATCH_SIZES):
            data = load(bs_paths[bs])
            ax = axes[row][col]
            x = np.arange(n_seqs)

            total_width = 0.88
            bar_w = total_width / n_strats
            offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

            max_val = 0
            for si, sk in enumerate(STRATEGIES):
                power_map = get_power_map(data, sk)
                vals = [power_map.get(s, 0) for s in SEQS]

                ax.bar(x + offsets[si], vals, bar_w,
                       color=COLORS[sk], edgecolor="white", linewidth=0.3)

                for v in vals:
                    if v > max_val:
                        max_val = v

            ax.set_ylim(top=max_val * 1.15)

            # 在最高柱上标注功耗值 (W)
            for si, sk in enumerate(STRATEGIES):
                power_map = get_power_map(data, sk)
                vals = [power_map.get(s, 0) for s in SEQS]
                for i, v in enumerate(vals):
                    if v > 0:
                        ax.text(x[i] + offsets[si], v + max_val * 0.01,
                                f"{v:.0f}", ha="center", va="bottom",
                                fontsize=5.5, color="black", rotation=90)

            ax.set_xticks(x)
            ax.set_xticklabels([seq_label(s) for s in SEQS], fontsize=11)
            ax.grid(axis="y", ls="--", alpha=0.25)
            ax.tick_params(axis="y", labelsize=11)
            ax.set_title(f"{model_label}, BS={bs}", fontsize=10, fontweight="bold")

            if row < n_models - 1:
                ax.set_xticklabels([])

    fig.text(0.5, -0.02, "Sequence Length", va="top", ha="center", fontsize=14)
    fig.text(-0.01, 0.5, "Power (W)", va="center", ha="center",
             rotation="vertical", fontsize=14)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=10, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = OUT / "fig2c_power_W_4panel.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
