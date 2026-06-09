#!/usr/bin/env python3
"""
Fig 1: 单层 Decode 加速比 (相对 H100 TP1)
- 3×4 子图: Qwen3, Llama4, DeepSeek-V3
- 每列对应一个 batch size: 1, 4, 16, 32
- 上两行 (GQA): 6 strategies, seq=[4K, 64K, 256K, 1M], y=0-20
- 第三行 (MLA): 7 strategies (含 Ours 512T), seq=[4K, 256K, 1M], y=0-30
"""
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── 路径 ──
BASE = Path(__file__).resolve().parents[4]  # congestion_aware/
OUT = Path(__file__).resolve().parents[1] / "plots"

sys.path.insert(0, str(BASE / "backend" / "roofline"))
from roofline_gqa_calc import calc_mla_strategy

# ── GQA 策略 (上两行) ──
GQA_STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new"]
# ── MLA 策略 (第三行, 增加 Ours 512T) ──
MLA_STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new", "ours_512t"]

LABELS = {
    "hmp_reo_new": "Ours",
    "rubin":       "Rubin",
    "rubin_tp2":   "Rubin TP2",
    "h100":        "H100",
    "h100_tp2":    "H100 TP2",
    "neupims":     "NeuPims",
    "ours_512t":   "Ours 512T",
}
COLORS = {
    "h100":        "#D4D4D4",
    "h100_tp2":    "#DBDDEF",
    "rubin":       "#92B1D9",
    "rubin_tp2":   "#C1D8E9",
    "neupims":     "#F4A582",
    "hmp_reo_new": "#E07850",
    "ours_512t":   "#C04020",
}
BASELINE = "h100"

# ── NeuPims 延时模型 ──
NEUPIMS_CONFIG = {
    "Qwen3-235B":      {"qkv": 2.0, "attn": 2.0 * 16 / 9, "o": 2.0, "comm_ns": 200},
    "Llama4-Maverick": {"qkv": 2.0, "attn": 2.0 * 5 / 9,  "o": 2.0, "comm_ns": 200},
    "DeepSeek-V3 MLA": {"qkv": 2.0, "attn": 2.0 * 16 / 9, "o": 2.0, "comm_ns": 200},
}

BATCH_SIZES = [1, 4, 16, 32]
GQA_SEQS = [4096, 65536, 262144, 1048576]
MLA_SEQS = [4096, 262144, 1048576]  # 删去 64K

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

# ── 右侧标题 ──
ROW_TITLES = {
    "Qwen3-235B": "Qwen",
    "Llama4-Maverick": "Llama4",
    "DeepSeek-V3 MLA": "DeepSeek",
}

MLA_DATA_PATH = str(BASE / "figures/paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json")
_mla_cache = None

def _load_mla():
    global _mla_cache
    if _mla_cache is None:
        with open(MLA_DATA_PATH) as f:
            _mla_cache = json.load(f)
    return _mla_cache


def get_wall_map_mla(bs, strat="hmp_reo_new"):
    mla = _load_mla()
    strats = mla.get("strategies", {})
    if strat in strats and "data_by_bs" in strats[strat]:
        entries = strats[strat]["data_by_bs"].get(f"bs{bs}", [])
    else:
        entries = mla.get("data", {}).get(f"bs{bs}", [])
    return {e["seq"]: e["hybrid_wall_ns"] for e in entries}


# ── MLA Ours 512T cache ──
_mla_512t_cache = {}

def get_mla_512t_wall(bs):
    if bs not in _mla_512t_cache:
        hw = {"compute": 512, "Bandwidth": 2.5}
        result = calc_mla_strategy(hw, bs, GQA_SEQS + MLA_SEQS, link_bw=1.5,
                                   tp_h=1, tp_s=16)
        _mla_512t_cache[bs] = {e["seq"]: e["wall_total_ns"] for e in result["data"]}
    return _mla_512t_cache[bs]


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


def get_module_map_ours(data, model_label=None, is_mla=False, bs=None):
    result = {}
    if is_mla:
        mla = _load_mla()
        entries = mla.get("data", {}).get(f"bs{bs}", [])
        for e in entries:
            result[e["seq"]] = (e["hybrid_Proj_QKV_ns"], e["hybrid_attn_ns"], e["hybrid_Proj_O_ns"])
    else:
        entries = data.get("strategies", {}).get("hmp_reo_new", {}).get("data", [])
        for e in entries:
            result[e["seq"]] = (e["hybrid_Proj_QKV_ns"], e["hybrid_attn_ns"], e["hybrid_Proj_O_ns"])
    return result


def compute_neupims_wall(ours_modules, model_label):
    cfg = NEUPIMS_CONFIG.get(model_label)
    if cfg is None:
        return {}
    result = {}
    for seq, (qkv, attn, o) in ours_modules.items():
        wall = qkv * cfg["qkv"] + attn * cfg["attn"] + o * cfg["o"] + cfg["comm_ns"]
        result[seq] = wall
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_models = len(MODELS)
    n_batches = len(BATCH_SIZES)

    fig, axes = plt.subplots(n_models, n_batches, figsize=(7 * n_batches, 3.5 * n_models))

    for row, (model_label, bs_paths) in enumerate(MODELS):
        is_mla = (model_label == "DeepSeek-V3 MLA")
        strategies = MLA_STRATEGIES if is_mla else GQA_STRATEGIES
        seqs = MLA_SEQS if is_mla else GQA_SEQS
        n_strats = len(strategies)
        n_seqs = len(seqs)
        ylim_top = 30 if is_mla else 25

        for col, bs in enumerate(BATCH_SIZES):
            if is_mla:
                data = None
            else:
                data = load(bs_paths[bs])
            baseline_map = get_wall_map_mla(bs, BASELINE) if is_mla else get_wall_map(data, BASELINE)
            ax = axes[row][col]
            x = np.arange(n_seqs)

            total_width = 0.88
            bar_w = total_width / n_strats
            offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

            max_sp = 0
            labels_by_group = {i: [] for i in range(n_seqs)}

            # NeuPims
            ours_modules = get_module_map_ours(data, model_label, is_mla, bs)
            neupims_wall_map = compute_neupims_wall(ours_modules, model_label)

            for si, sk in enumerate(strategies):
                if sk == "neupims":
                    wall_map = neupims_wall_map
                elif sk == "ours_512t":
                    wall_map = get_mla_512t_wall(bs)
                elif is_mla:
                    wall_map = get_wall_map_mla(bs, sk)
                else:
                    wall_map = get_wall_map(data, sk)
                speedups = []
                for s in seqs:
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
            ax.set_ylim(0, ylim_top)

            # 标注: 增大 min_gap 和字号
            min_gap = ylim_top * 0.06
            for grp in labels_by_group.values():
                grp.sort(key=lambda t: t[1])
                y_positions = [item[1] for item in grp]
                for j in range(1, len(y_positions)):
                    if y_positions[j] - y_positions[j - 1] < min_gap:
                        y_positions[j] = y_positions[j - 1] + min_gap
                for j, (xp, _, txt) in enumerate(grp):
                    ax.text(xp, y_positions[j] + ylim_top * 0.008,
                            txt, ha="center", va="bottom",
                            fontsize=15, fontweight="bold", color="black", rotation=0)

            ax.set_xticks(x)
            ax.set_xticklabels([seq_label(s) for s in seqs], fontsize=24)
            ax.grid(axis="y", ls="--", alpha=0.25)
            ax.tick_params(axis="y", labelsize=24)

            if row == 0:
                ax.set_title(f"BS={bs}", fontsize=24, fontweight="bold", pad=8)

            if col == n_batches - 1:
                ax.annotate(ROW_TITLES.get(model_label, model_label),
                            xy=(1.03, 0.5), xycoords="axes fraction",
                            fontsize=24, fontweight="bold", rotation=-90,
                            ha="left", va="center")

            # 只在第二行和底行显示 x 刻度
            if row == 0:
                ax.set_xticklabels([])
            elif row == 1:
                ax.set_xticklabels([seq_label(s) for s in seqs], fontsize=24)

            # 只在第一列显示 y 刻度
            if col > 0:
                ax.set_yticklabels([])

    fig.text(0.5, 0.09, "Sequence Length", va="top", ha="center", fontsize=26)
    fig.text(0.030, 0.5, "Latency Speedup", va="center", ha="center",
             rotation="vertical", fontsize=26)

    # 图例: 合并 GQA + MLA 策略 (去重)
    all_strats = list(dict.fromkeys(GQA_STRATEGIES + MLA_STRATEGIES))
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in all_strats]
    legend_labels = [LABELS[sk] for sk in all_strats]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=len(all_strats),
               fontsize=26, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.00))

    fig.tight_layout(rect=[0.03, 0, 0.95, 0.93])
    fig.subplots_adjust(wspace=0.05)

    # 独立控制每行间距
    # gap_01: 第1行(Qwen)与第2行(Llama4)间距 — GQA 内部, 紧凑
    # gap_12: 第2行(Llama4)与第3行(DeepSeek)间距 — GQA→MLA, 拉开
    gap_01 = 0.02
    gap_12 = 0.04
    for col in range(n_batches):
        p0 = axes[0][col].get_position()
        p1 = axes[1][col].get_position()
        p2 = axes[2][col].get_position()
        h = p0.height

        y1 = p0.y0 - h - gap_01
        axes[1][col].set_position([p1.x0, y1, p1.width, h])

        y2 = y1 - h - gap_12
        axes[2][col].set_position([p2.x0, y2, p2.width, h])
    fname = OUT / "fig1_speedup_4panel.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
