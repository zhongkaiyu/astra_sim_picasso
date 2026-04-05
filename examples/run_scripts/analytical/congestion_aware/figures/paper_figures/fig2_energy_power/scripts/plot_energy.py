#!/usr/bin/env python3
"""
Fig 2: 单 Token 能耗对比 (相对 H100 的能耗节省)
- 4 张子图横排: (Qwen3 短seq, Qwen3 长seq, Llama4 短seq, Llama4 长seq)
- 分组柱状图, baseline = H100 (灰色, 1.0x)
- 纵轴为能耗比 (Energy Ratio vs H100), <1 表示节能
- 图例横铺在标题下方
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parents[4]  # congestion_aware/
OUT = Path(__file__).resolve().parents[1] / "plots"

# ── 策略定义与配色 ──
STRATEGIES = ["hmp_reo_new", "rubin", "rubin_tp2", "h100", "h100_tp2"]
LABELS = {
    "hmp_reo_new": "Ours",
    "rubin":       "Rubin",
    "rubin_tp2":   "Rubin TP2",
    "h100":        "H100",
    "h100_tp2":    "H100 TP2",
}
COLORS = {
    "hmp_reo_new": "#8E44AD",   # 紫色
    "rubin":       "#C0392B",   # 深红
    "rubin_tp2":   "#922B21",   # 暗红
    "h100":        "#7F8C8D",   # 灰色
    "h100_tp2":    "#566573",   # 深灰
}
BASELINE = "h100"  # 以 H100 单卡为基准

# ── 模型功耗数据路径 ──
MODELS = [
    ("Qwen3-235B",      str(BASE / "reports/qwen3-235B/power/gqa_hybrid_merged_96T_split4_bw1500_util96_power.json")),
    ("Llama4-Maverick",  str(BASE / "reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_power.json")),
]

# ── 序列长度分组 (精简为 4 个点, 分短/长两组) ──
SEQ_RANGES = [
    ("1K–64K",  [4096, 65536]),
    ("64K–1M",  [262144, 1048576]),
]


def seq_label(s):
    """将数值序列长度转为可读标签"""
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def load(path):
    """加载 JSON 文件"""
    with open(path) as f:
        return json.load(f)


def get_energy_map(data, strat):
    """从 power JSON 中提取 {seq: energy_per_token_nj} 映射"""
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["energy_per_token_nj"] for e in entries}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # 4 列子图: 2 模型 × 2 序列范围
    n_cols = len(MODELS) * len(SEQ_RANGES)
    n_strats = len(STRATEGIES)

    # 压扁图: 与 Fig 1 保持一致的尺寸
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 3.8))

    col = 0
    for model_label, model_path in MODELS:
        data = load(model_path)
        baseline_map = get_energy_map(data, BASELINE)  # H100 能耗作为基准

        for range_label, seqs in SEQ_RANGES:
            ax = axes[col]
            n_seqs = len(seqs)
            x = np.arange(n_seqs)

            # 柱状图间距计算
            total_width = 0.78
            bar_w = total_width / n_strats
            offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

            for si, sk in enumerate(STRATEGIES):
                energy_map = get_energy_map(data, sk)
                # 计算相对 H100 的能耗比 (<1 表示节能)
                ratios = []
                for s in seqs:
                    bv = baseline_map.get(s, 0)
                    sv = energy_map.get(s, 0)
                    ratios.append(sv / bv if bv > 0 and sv > 0 else 0)

                ax.bar(x + offsets[si], ratios, bar_w,
                       color=COLORS[sk], edgecolor="white", linewidth=0.3)

                # 在柱顶标注节能百分比 (仅对非 baseline 策略)
                for i, r in enumerate(ratios):
                    if r > 0 and abs(r - 1.0) > 0.02:
                        if r < 1.0:
                            # 节能: 显示 -xx%
                            pct = (1.0 - r) * 100
                            label_text = f"-{pct:.0f}%"
                        else:
                            # 更耗能: 显示 +xx%
                            pct = (r - 1.0) * 100
                            label_text = f"+{pct:.0f}%"
                        ax.text(x[i] + offsets[si], r + 0.02,
                                label_text, ha="center", va="bottom",
                                fontsize=9, fontweight="bold", color=COLORS[sk],
                                rotation=90 if r > 1.3 else 0)

            # H100 基准线 (1.0x)
            ax.axhline(1.0, color="#7F8C8D", ls="--", lw=1, alpha=0.6)

            ax.set_xticks(x)
            ax.set_xticklabels([seq_label(s) for s in seqs], fontsize=13)
            ax.set_xlabel("Sequence Length", fontsize=13)
            ax.set_title(f"{model_label}\n{range_label}", fontsize=14, fontweight="bold")
            ax.grid(axis="y", ls="--", alpha=0.25)
            ax.tick_params(axis="y", labelsize=12)
            col += 1

    axes[0].set_ylabel("Energy Ratio vs H100", fontsize=14)

    # ── 图例横铺在标题下方 ──
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=12, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.00))

    # ── 总标题 ──
    fig.suptitle("GQA Single-Layer Energy per Token (bs=1, FP8, baseline=H100)",
                 fontsize=17, fontweight="bold", y=1.08)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fname = OUT / "fig2_energy_4panel.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
