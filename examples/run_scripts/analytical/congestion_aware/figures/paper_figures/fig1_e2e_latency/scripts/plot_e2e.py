#!/usr/bin/env python3
"""
Fig 1: GQA 单层 Decode 加速比 (相对 H100 TP1)
- 2×4 子图: 上行 Qwen3-235B, 下行 Llama4-Maverick
- 每列对应一个 batch size: 1, 4, 16, 32
- 分组柱状图, baseline = H100
- 图例横铺在顶部
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

# ── 策略定义 (顺序: H100, H100 TP2, Rubin, Rubin TP2, Ours) ──
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

# ── 模型数据路径模板 ──
MODELS = [
    ("Qwen3-235B", {
        1:  str(BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs1.json"),
        4:  str(BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs4.json"),
        16: str(BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs16.json"),
        32: str(BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs32.json"),
    }),
    ("Llama4-Maverick", {
        1:  str(BASE / "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs1.json"),
        4:  str(BASE / "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs4.json"),
        16: str(BASE / "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs16.json"),
        32: str(BASE / "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs32.json"),
    }),
    ("DeepSeek-V3 MLA", {
        bs: str(BASE / "figures/paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json")
        for bs in [1, 4, 16, 32]
    }),
]

# DeepSeek-V3 MLA 数据格式不同, 需要特殊加载
MLA_DATA_PATH = str(BASE / "figures/paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json")
_mla_cache = None

def _load_mla():
    global _mla_cache
    if _mla_cache is None:
        with open(MLA_DATA_PATH) as f:
            _mla_cache = json.load(f)
    return _mla_cache


def get_wall_map_mla(bs):
    """从 MLA JSON 提取 {seq: wall_ns}, 仅用于 hmp_reo_new 策略"""
    mla = _load_mla()
    entries = mla.get("data", {}).get(f"bs{bs}", [])
    return {e["seq"]: e["hybrid_wall_ns"] for e in entries}

# ── 序列长度 ──
SEQS = [4096, 65536, 262144, 1048576]


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_models = len(MODELS)
    n_batches = len(BATCH_SIZES)
    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)

    fig, axes = plt.subplots(n_models, n_batches, figsize=(3.5 * n_batches, 2.0 * n_models),
                             sharey='row')

    for row, (model_label, bs_paths) in enumerate(MODELS):
        is_mla = (model_label == "DeepSeek-V3 MLA")
        for col, bs in enumerate(BATCH_SIZES):
            if is_mla:
                # MLA: 加载 Qwen3 数据作为 H100/Rubin baseline, MLA 数据作为 Ours
                qwen3_data = load(MODELS[0][1][bs])  # Qwen3 hybrid JSON
                data = qwen3_data  # baseline 策略来自 Qwen3
            else:
                data = load(bs_paths[bs])
            baseline_map = get_wall_map(data, BASELINE)
            ax = axes[row][col]
            x = np.arange(n_seqs)

            total_width = 0.88
            bar_w = total_width / n_strats
            offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

            max_sp = 0
            # labels_by_group[seq_idx] = [(x_pos, bar_top, text), ...]
            labels_by_group = {i: [] for i in range(n_seqs)}

            for si, sk in enumerate(STRATEGIES):
                if is_mla and sk == "hmp_reo_new":
                    wall_map = get_wall_map_mla(bs)
                else:
                    wall_map = get_wall_map(data, sk)
                speedups = []
                for s in SEQS:
                    bv = baseline_map.get(s, 0)
                    sv = wall_map.get(s, 0)
                    speedups.append(bv / sv if sv > 0 and bv > 0 else 0)

                ax.bar(x + offsets[si], speedups, bar_w,
                       color=COLORS[sk], edgecolor="white", linewidth=0.3)

                for i, sp in enumerate(speedups):
                    if sp > 0:
                        labels_by_group[i].append((x[i] + offsets[si], sp, f"{sp:.1f}"))
                        if sp > max_sp:
                            max_sp = sp

            ax.axhline(1.0, color="#7F8C8D", ls="--", lw=0.8, alpha=0.6)

            ax.set_ylim(top=max_sp * 1.22)

            # 标注加速比: 水平放置, 同组内避免重叠
            min_gap = max_sp * 0.07
            for grp in labels_by_group.values():
                grp.sort(key=lambda t: t[1])
                y_positions = [item[1] for item in grp]
                for j in range(1, len(y_positions)):
                    if y_positions[j] - y_positions[j - 1] < min_gap:
                        y_positions[j] = y_positions[j - 1] + min_gap
                for j, (xp, _, txt) in enumerate(grp):
                    ax.text(xp, y_positions[j] + max_sp * 0.01,
                            txt, ha="center", va="bottom",
                            fontsize=6.5, fontweight="bold", color="black", rotation=0)

            ax.set_xticks(x)
            ax.set_xticklabels([seq_label(s) for s in SEQS], fontsize=11)
            ax.grid(axis="y", ls="--", alpha=0.25)
            ax.tick_params(axis="y", labelsize=11)

            # 标题: 模型名 + batch size
            title_label = "DeepSeek-V3" if is_mla else model_label
            ax.set_title(f"{title_label}, BS={bs}", fontsize=10, fontweight="bold")

            # 只在底行显示 x 刻度
            if row < n_models - 1:
                ax.set_xticklabels([])

    # ── 共用 x 轴标签, 放在底部中间 ──
    fig.text(0.5, -0.02, "Sequence Length", va="top", ha="center", fontsize=14)

    # ── 共用 y 轴标签, 放在两行中间 ──
    fig.text(-0.01, 0.5, "Latency Speedup", va="center", ha="center",
             rotation="vertical", fontsize=14)

    # ── 图例横铺在顶部 ──
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=10, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fname = OUT / "fig1_speedup_4panel.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
