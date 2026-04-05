#!/usr/bin/env python3
"""
Fig 2b: 能量效率提升 (Token/J 相对 H100)
- 2×4 子图: 上行 Qwen3-235B, 下行 Llama4-Maverick
- 每列对应一个 batch size: 1, 4, 16, 32
- 布局严格对齐 fig1_speedup_4panel
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "plots"

# ── 策略定义与配色 (顺序: H100, H100 TP2, Rubin, Rubin TP2, Ours) ──
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
BASELINE = "h100"

# ── Batch sizes (4 columns) ──
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


def get_token_per_joule(data, strat):
    """Token/J (per token) = bs / energy_batch_nJ × 1e9"""
    bs = data.get("metadata", {}).get("batch_size", 1)
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: bs * 1e9 / e["energy_per_token_nj"] if e["energy_per_token_nj"] > 0 else 0
            for e in entries}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_models = len(MODELS)
    n_batches = len(BATCH_SIZES)
    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)

    # ── 严格对齐 fig1 尺寸 ──
    fig, axes = plt.subplots(n_models, n_batches, figsize=(3.5 * n_batches, 2.0 * n_models),
                             sharey='row')

    for row, (model_label, bs_paths) in enumerate(MODELS):
        for col, bs in enumerate(BATCH_SIZES):
            data = load(bs_paths[bs])
            baseline_map = get_token_per_joule(data, BASELINE)
            ax = axes[row][col]
            x = np.arange(n_seqs)

            total_width = 0.88
            bar_w = total_width / n_strats
            offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

            max_ratio = 0
            labels_by_group = {i: [] for i in range(n_seqs)}

            for si, sk in enumerate(STRATEGIES):
                tpj_map = get_token_per_joule(data, sk)
                ratios = []
                for s in SEQS:
                    v = tpj_map.get(s, 0)
                    bv = baseline_map.get(s, 0)
                    ratios.append(v / bv if v > 0 and bv > 0 else 0)

                ax.bar(x + offsets[si], ratios, bar_w,
                       color=COLORS[sk], edgecolor="white", linewidth=0.3)

                for i, r in enumerate(ratios):
                    if r > 0:
                        labels_by_group[i].append((x[i] + offsets[si], r, f"{r:.1f}"))
                        if r > max_ratio:
                            max_ratio = r

            ax.axhline(1.0, color="#7F8C8D", ls="--", lw=0.8, alpha=0.6)

            ax.set_ylim(top=max_ratio * 1.22)

            # 标注: 水平放置, 同组内避免重叠
            min_gap = max_ratio * 0.07
            for grp in labels_by_group.values():
                grp.sort(key=lambda t: t[1])
                y_positions = [item[1] for item in grp]
                for j in range(1, len(y_positions)):
                    if y_positions[j] - y_positions[j - 1] < min_gap:
                        y_positions[j] = y_positions[j - 1] + min_gap
                for j, (xp, _, txt) in enumerate(grp):
                    ax.text(xp, y_positions[j] + max_ratio * 0.01,
                            txt, ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold", color="black", rotation=0)

            ax.set_xticks(x)
            ax.set_xticklabels([seq_label(s) for s in SEQS], fontsize=11)
            ax.grid(axis="y", ls="--", alpha=0.25)
            ax.tick_params(axis="y", labelsize=11)

            ax.set_title(f"{model_label}, BS={bs}", fontsize=10, fontweight="bold")

            if row < n_models - 1:
                ax.set_xticklabels([])

    # ── 共用 x 轴标签, 放在底部中间 ──
    fig.text(0.5, -0.02, "Sequence Length", va="top", ha="center", fontsize=14)

    # ── 共用 y 轴标签, 放在两行中间 ──
    fig.text(-0.01, 0.5, "Energy Eff. Gain", va="center", ha="center",
             rotation="vertical", fontsize=14)

    # ── 图例横铺在顶部 ──
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=10, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = OUT / "fig2b_power_4panel.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
