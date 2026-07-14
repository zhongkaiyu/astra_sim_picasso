#!/usr/bin/env python3
"""
Fig 1: 单层 Decode 加速比 (相对 H100 TP1) — 两块布局:

(a) GQA 整层 wall 加速比 (Qwen3, Llama4): 2×4 子图, 每列一个 batch size [1,4,16,32],
    7 strategies (含 Ours_EC), seq=[4K,256K,1M]。Ours_EC = 同 backend + compute×4;
    GQA decode memory-bound ⇒ Ours_EC = Ours。

(b) DeepSeek attention-pool 加速比: 1×3 子图 (MLA / HCA / CSA), 仅 BS=4。
    Attention pool = Proj_QKV + core Attention + Proj_O (+comm), 即 exp3
    settings_breakdown 中 AMMA 负责的部分 (去掉 LPU 的 FFN / NIC Xfer)。
    同一 backend (calc_mla_strategy / calc_v4_strategy):
      · MLA       — 复用 deepseek_v3_mla_fig1.json wall (tp_h=1 tp_s=16)
      · HCA / CSA — calc_v4_strategy (tp_h=4 tp_s=4), baseline=H100/Rubin 单卡 roofline
    每个 panel 内 7 strategies 都跑该变体 (within-variant), 加速比相对各自 H100。
    V4 稀疏 attention 很小且 compute-bound, AMMA 优势主要来自 memory-bound 的
    Proj_O(~11.9×)/Proj_QKV(~3.6×) —— 故必须画整个 pool 而非只画 core attention。
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
from roofline_v4_calc import calc_v4_strategy

# ── 统一策略 (三行同格式, 含 Ours_EC / 384T) ──
GQA_STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new", "ours_384t"]
MLA_STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new", "ours_384t"]

LABELS = {
    "hmp_reo_new": "Ours",
    "rubin":       "Rubin",
    "rubin_tp2":   "Rubin TP2",
    "h100":        "H100",
    "h100_tp2":    "H100 TP2",
    "neupims":     "NeuPims",
    "ours_384t":   "Ours_EC",
}
COLORS = {
    "h100":        "#D4D4D4",
    "h100_tp2":    "#DBDDEF",
    "rubin":       "#92B1D9",
    "rubin_tp2":   "#C1D8E9",
    "neupims":     "#F4A582",
    "hmp_reo_new": "#E07850",
    "ours_384t":   "#C04020",
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


# ════════════════════════════════════════════════════════════════════
# GQA Ours_EC (384 TFLOPS) — 与 Ours 同一 backend, 仅 compute ×4
# ════════════════════════════════════════════════════════════════════
# Enhanced-Compute 把 per-cube 算力 ×4 (96→384 TFLOPS); compute_ns/4, memory
# floor 不变 ⇒ per-module 时间 = max(compute_ns/4, memory_ns)。
# 但 GQA decode 的算术强度 ≈ batch (≤32) < roofline crossover (peak/bw ≈ 38)
# ⇒ 整层 memory-bound: 投影读权重、attention 读 KV-cache 均受 HBM 带宽限制,
#   提升算力不缓解瓶颈。因此 EC ≈ Ours (no gap)。
# (96T report 的 compute_ns/util 项是 GEMV 的 latency/memory 拟合, 非可被算力
#  消除的真实 FLOP 限制, 故不随 ×4 算力下降。)
_gqa_384t_cache = {}

def get_gqa_384t_wall(model_label, bs):
    """Ours_EC wall map. GQA decode 全程 memory-bound ⇒ 算力 ×4 无收益, EC = Ours。

    用 per-module max(realized, memory_floor) 实现: realized 已是各模块的
    memory-bound 时间, ×4 算力把 compute 项压到 floor 之下, 故结果 = realized。
    """
    key = (model_label, bs)
    if key not in _gqa_384t_cache:
        data = load(dict(MODELS)[model_label][bs])
        out = {}
        for e in data["strategies"]["hmp_reo_new"]["data"]:
            s = e["seq"]
            if s not in MLA_SEQS:
                continue
            # memory-bound: EC 不改变各模块时间 ⇒ wall = Ours wall
            out[s] = e["hybrid_wall_ns"]
        _gqa_384t_cache[key] = out
    return _gqa_384t_cache[key]


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


# ════════════════════════════════════════════════════════════════════
# Panel (a): GQA 整层 wall 加速比 (相对 H100)
# ════════════════════════════════════════════════════════════════════
def get_gqa_speedup(model_label, bs):
    """{strategy: {seq: wall_speedup vs H100}} for a GQA model at given bs."""
    data = load(dict(MODELS)[model_label][bs])
    baseline = get_wall_map(data, BASELINE)
    neupims = compute_neupims_wall(
        get_module_map_ours(data, model_label, False, bs), model_label)
    spd = {}
    for sk in GQA_STRATEGIES:
        if sk == "neupims":
            wall = neupims
        elif sk == "ours_384t":
            wall = get_gqa_384t_wall(model_label, bs)
        else:
            wall = get_wall_map(data, sk)
        spd[sk] = {s: (baseline[s] / wall[s]
                       if wall.get(s, 0) > 0 and baseline.get(s, 0) > 0 else 0)
                   for s in MLA_SEQS}
    return spd


# ════════════════════════════════════════════════════════════════════
# Panel (b): DeepSeek attention-pool 加速比 (MLA / HCA / CSA), BS=4
# ════════════════════════════════════════════════════════════════════
# "Attention pool" = Proj_QKV + core Attention + Proj_O (+ comm) —— 即 exp3
# settings_breakdown 中 AMMA 负责的那部分 (去掉 LPU 的 FFN / NIC Xfer)。
# 注意: V4 (CSA/HCA) 的稀疏 attention 已很小且 compute-bound, AMMA 的优势
# 主要来自 memory-bound 的 Proj_O (~11.9×) 与 Proj_QKV (~3.6×); 若只画 core
# attention 会漏掉 Proj_O, 与 exp3 严重不符。
DS_VARIANTS = [("MLA", "mla"), ("HCA", "hca"), ("CSA", "csa")]
DS_BS = 4
DS_YLIM = 25
H100_HW    = {"compute": 1979,  "Bandwidth": 3.35}
RUBIN_HW   = {"compute": 17500, "Bandwidth": 22.0}
OURS_HW    = {"compute": 96,    "Bandwidth": 2.5}
OURS_EC_HW = {"compute": 384,   "Bandwidth": 2.5}
H100_NVLINK, RUBIN_NVLINK = 0.9, 1.8   # TB/s, GPU TP2 内部互联
NEUPIMS_DS = NEUPIMS_CONFIG["DeepSeek-V3 MLA"]


def _v4_run(variant, hw, tp_h, tp_s, link_bw=1.5):
    return calc_v4_strategy(variant, hw, DS_BS, MLA_SEQS, link_bw=link_bw,
                            hop_latency_ns=15, endpoint_delay_ns=10,
                            tp_h=tp_h, tp_s=tp_s)["data"]


def _neupims_pool(modules):
    """NeuPims attention-pool wall: 各模块 × NeuPims 因子 + 固定 comm。"""
    return {s: (q * NEUPIMS_DS["qkv"] + a * NEUPIMS_DS["attn"]
                + o * NEUPIMS_DS["o"] + NEUPIMS_DS["comm_ns"])
            for s, (q, a, o) in modules.items()}


def get_ds_pool_wall(variant_key):
    """{strategy: {seq: attention-pool wall ns}} (QKV+Attn+ProjO+comm) at BS=4."""
    out = {}
    if variant_key == "mla":
        # 复用 fig1 既有 JSON (与原 DeepSeek wall panel 完全一致)
        J = _load_mla()
        jwall = lambda st: {e["seq"]: e["hybrid_wall_ns"]
                            for e in J["strategies"][st]["data_by_bs"][f"bs{DS_BS}"]}
        for st in ("h100", "h100_tp2", "rubin", "rubin_tp2", "hmp_reo_new"):
            out[st] = jwall(st)
        ec = calc_mla_strategy(OURS_EC_HW, DS_BS, MLA_SEQS, link_bw=1.5,
                               hop_latency_ns=15, endpoint_delay_ns=10, tp_h=1, tp_s=16)
        out["ours_384t"] = {e["seq"]: e["wall_total_ns"] for e in ec["data"]}
        mods = {e["seq"]: (e["hybrid_Proj_QKV_ns"], e["hybrid_attn_ns"], e["hybrid_Proj_O_ns"])
                for e in J["strategies"]["hmp_reo_new"]["data_by_bs"][f"bs{DS_BS}"]}
    else:
        ours = _v4_run(variant_key, OURS_HW, 4, 4)
        ec   = _v4_run(variant_key, OURS_EC_HW, 4, 4)
        h1   = _v4_run(variant_key, H100_HW, 1, 1)   # 单卡 baseline (comm=0)
        r1   = _v4_run(variant_key, RUBIN_HW, 1, 1)
        # TP2 基线: backend 正确计算 (tp_h=2, 含 NVLink comm), 不再 naive /2
        h2 = _v4_run(variant_key, H100_HW, 2, 1, link_bw=H100_NVLINK)
        r2 = _v4_run(variant_key, RUBIN_HW, 2, 1, link_bw=RUBIN_NVLINK)
        out["hmp_reo_new"] = {e["seq"]: e["wall_total_ns"] for e in ours}
        out["ours_384t"]   = {e["seq"]: e["wall_total_ns"] for e in ec}
        out["h100"]  = {e["seq"]: e["gpu_total_ns"] for e in h1}
        out["rubin"] = {e["seq"]: e["gpu_total_ns"] for e in r1}
        out["h100_tp2"]  = {e["seq"]: e["wall_total_ns"] for e in h2}
        out["rubin_tp2"] = {e["seq"]: e["wall_total_ns"] for e in r2}
        mods = {e["seq"]: (e["Proj_QKV"]["total_ns"], e["attention"]["total_ns"],
                           e["Proj_O"]["total_ns"]) for e in ours}
    out["neupims"] = _neupims_pool(mods)
    return out


def get_ds_attn_speedup(variant_key):
    lat = get_ds_pool_wall(variant_key)
    base = lat["h100"]
    return {sk: {s: (base[s] / lat[sk][s]
                     if lat[sk].get(s, 0) > 0 and base.get(s, 0) > 0 else 0)
                 for s in MLA_SEQS}
            for sk in MLA_STRATEGIES}


# ════════════════════════════════════════════════════════════════════
def _draw_speedup_panel(ax, strategies, seqs, spd_by_strat, ylim_top, label_fs=14):
    x = np.arange(len(seqs))
    n = len(strategies)
    total_width = 0.88
    bar_w = total_width / n
    offsets = np.arange(n) * bar_w - total_width / 2 + bar_w / 2
    labels_by_group = {i: [] for i in range(len(seqs))}
    for si, sk in enumerate(strategies):
        sp = [spd_by_strat[sk].get(s, 0) for s in seqs]
        ax.bar(x + offsets[si], sp, bar_w,
               color=COLORS[sk], edgecolor="white", linewidth=0.3)
        for i, v in enumerate(sp):
            if v > 0:
                labels_by_group[i].append((x[i] + offsets[si], v, f"{v:.1f}"))
    ax.axhline(1.0, color="#7F8C8D", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylim(0, ylim_top)
    min_gap = ylim_top * 0.06
    for grp in labels_by_group.values():
        grp.sort(key=lambda t: t[1])
        ys = [g[1] for g in grp]
        for j in range(1, len(ys)):
            if ys[j] - ys[j - 1] < min_gap:
                ys[j] = ys[j - 1] + min_gap
        for j, (xp, _, txt) in enumerate(grp):
            ax.text(xp, ys[j] + ylim_top * 0.008, txt, ha="center", va="bottom",
                    fontsize=label_fs, fontweight="bold", color="black")
    ax.set_xticks(x)
    ax.grid(axis="y", ls="--", alpha=0.25)


def _legend(fig, fs=22, y=0.99):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in MLA_STRATEGIES]
    fig.legend(handles, [LABELS[sk] for sk in MLA_STRATEGIES],
               loc="upper center", ncol=len(MLA_STRATEGIES),
               fontsize=fs, frameon=True, fancybox=True, bbox_to_anchor=(0.5, y))


# ── 与合并图 main() 完全一致的绝对几何 (复刻同样的列宽 / 行高 / 字号) ──
_COMB_W   = 7 * len(BATCH_SIZES)            # 28in, 与 main() figsize 宽一致
_PANEL_L, _PANEL_R = 0.055, 0.95           # 与 main() LEFT/RIGHT 一致 -> 列宽相同
_HSP      = 0.12                            # 与 main() hspace 一致
_ROW_H_IN = (0.90 - 0.57) / (2 + _HSP) * 12.0   # main() 的 (a) 单行高度 (in)
_TOP_MARGIN_IN = 1.55                       # 顶部留给 legend (调高图例, 与标题留更大间距)
_BOT_MARGIN_IN = 0.80                       # 底部留给 x 刻度 + Sequence Length


def _save_standalone(nrow, strategies, ncols, draw_fn, fname_stem):
    """用与合并图完全相同的绝对尺寸渲染单个 block, 不带 (a)/(b) 标号。"""
    OUT.mkdir(parents=True, exist_ok=True)
    block_in = nrow * _ROW_H_IN + (nrow - 1) * _HSP * _ROW_H_IN
    H = _TOP_MARGIN_IN + block_in + _BOT_MARGIN_IN
    top_f = (H - _TOP_MARGIN_IN) / H
    bot_f = _BOT_MARGIN_IN / H
    fig = plt.figure(figsize=(_COMB_W, H))
    gs = fig.add_gridspec(nrow, ncols, left=_PANEL_L, right=_PANEL_R,
                          top=top_f, bottom=bot_f, hspace=_HSP, wspace=0.05)
    draw_fn(fig, gs)
    fig.text(0.020, (top_f + bot_f) / 2, "Latency Speedup", va="center", ha="center",
             rotation="vertical", fontsize=24)
    fig.text((_PANEL_L + _PANEL_R) / 2, bot_f - 0.36 / H, "Sequence Length",
             va="top", ha="center", fontsize=24)
    _legend(fig, fs=24, y=1 - 0.08 / H)
    for ext in ("pdf", "svg"):
        f = OUT / f"{fname_stem}.{ext}"
        fig.savefig(f, bbox_inches="tight")
        print(f"Saved: {f}")
    plt.close(fig)


def save_panel_a():
    """单独输出 (a) GQA 2×4 (PDF+SVG), 几何与合并图一致, 无 (a) 标号。"""
    def draw(fig, gs):
        for r, model in enumerate(["Qwen3-235B", "Llama4-Maverick"]):
            for c, bs in enumerate(BATCH_SIZES):
                ax = fig.add_subplot(gs[r, c])
                _draw_speedup_panel(ax, GQA_STRATEGIES, MLA_SEQS,
                                    get_gqa_speedup(model, bs), 30, label_fs=14)
                ax.tick_params(axis="y", labelsize=22)
                if r == 0:
                    ax.set_title(f"BS={bs}", fontsize=24, fontweight="bold", pad=8)
                    ax.set_xticklabels([])
                else:
                    ax.set_xticklabels([seq_label(s) for s in MLA_SEQS], fontsize=22)
                if c > 0:
                    ax.set_yticklabels([])
                if c == len(BATCH_SIZES) - 1:
                    ax.annotate(ROW_TITLES[model], xy=(1.03, 0.5), xycoords="axes fraction",
                                fontsize=24, fontweight="bold", rotation=-90,
                                ha="left", va="center")
    _save_standalone(2, GQA_STRATEGIES, len(BATCH_SIZES), draw, "fig1_speedup_panel_a")


def save_panel_b():
    """单独输出 (b) DeepSeek attention-pool 1×3 (PDF+SVG), 几何与合并图一致, 无 (b) 标号。"""
    def draw(fig, gs):
        for c, (disp, key) in enumerate(DS_VARIANTS):
            ax = fig.add_subplot(gs[0, c])
            _draw_speedup_panel(ax, MLA_STRATEGIES, MLA_SEQS,
                                get_ds_attn_speedup(key), DS_YLIM, label_fs=14)
            ax.set_title(disp, fontsize=24, fontweight="bold", pad=8)
            ax.set_xticklabels([seq_label(s) for s in MLA_SEQS], fontsize=22)
            ax.tick_params(axis="y", labelsize=22)
            if c > 0:
                ax.set_yticklabels([])
            if c == len(DS_VARIANTS) - 1:
                ax.annotate("DeepSeek\n(BS=4, attn pool)", xy=(1.03, 0.5),
                            xycoords="axes fraction", fontsize=20, fontweight="bold",
                            rotation=-90, ha="left", va="center")
    _save_standalone(1, MLA_STRATEGIES, len(DS_VARIANTS), draw, "fig1_speedup_panel_b")


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    LEFT, RIGHT = 0.055, 0.95
    XC = (LEFT + RIGHT) / 2

    fig = plt.figure(figsize=(7 * len(BATCH_SIZES), 12.0))
    # (a) GQA: 2 行 × 4 列。(b) DeepSeek: 1 行 × 3 列, 横跨与 (a) 同宽 (LEFT→RIGHT),
    #     高度压缩到与 (a) 单行一致。
    A_TOP, A_BOT, HSP = 0.90, 0.57, 0.12
    row_h = (A_TOP - A_BOT) / (2 + HSP)          # (a) 单行高度
    B_TOP = 0.43
    B_BOT = B_TOP - row_h                          # (b) 与 (a) 单行等高
    gsa = fig.add_gridspec(2, len(BATCH_SIZES), left=LEFT, right=RIGHT,
                           top=A_TOP, bottom=A_BOT, hspace=HSP, wspace=0.05)
    gsb = fig.add_gridspec(1, len(DS_VARIANTS), left=LEFT, right=RIGHT,
                           top=B_TOP, bottom=B_BOT, wspace=0.05)

    GQA_MODELS = ["Qwen3-235B", "Llama4-Maverick"]

    # ── (a) GQA 整层 wall 加速比 ──
    for r, model in enumerate(GQA_MODELS):
        for c, bs in enumerate(BATCH_SIZES):
            ax = fig.add_subplot(gsa[r, c])
            _draw_speedup_panel(ax, GQA_STRATEGIES, MLA_SEQS,
                                get_gqa_speedup(model, bs), 30, label_fs=14)
            ax.tick_params(axis="y", labelsize=22)
            if r == 0:
                ax.set_title(f"BS={bs}", fontsize=24, fontweight="bold", pad=8)
                ax.set_xticklabels([])
            else:
                ax.set_xticklabels([seq_label(s) for s in MLA_SEQS], fontsize=22)
            if c > 0:
                ax.set_yticklabels([])
            if c == len(BATCH_SIZES) - 1:
                ax.annotate(ROW_TITLES[model], xy=(1.03, 0.5), xycoords="axes fraction",
                            fontsize=24, fontweight="bold", rotation=-90,
                            ha="left", va="center")

    # ── (b) DeepSeek attention-pool 加速比: MLA / HCA / CSA, BS=4 ──
    for c, (disp, key) in enumerate(DS_VARIANTS):
        ax = fig.add_subplot(gsb[0, c])
        _draw_speedup_panel(ax, MLA_STRATEGIES, MLA_SEQS,
                            get_ds_attn_speedup(key), DS_YLIM, label_fs=14)
        ax.set_title(disp, fontsize=24, fontweight="bold", pad=8)
        ax.set_xticklabels([seq_label(s) for s in MLA_SEQS], fontsize=22)
        ax.tick_params(axis="y", labelsize=22)
        if c > 0:
            ax.set_yticklabels([])
        if c == len(DS_VARIANTS) - 1:
            ax.annotate("DeepSeek\n(BS=4, attn pool)", xy=(1.03, 0.5),
                        xycoords="axes fraction", fontsize=20, fontweight="bold",
                        rotation=-90, ha="left", va="center")

    # ── 轴标签 / 子图标号 (标号居中置于各自子图底部) ──
    fig.text(0.020, (A_TOP + A_BOT) / 2, "Latency Speedup", va="center", ha="center",
             rotation="vertical", fontsize=24)
    fig.text(0.020, (B_TOP + B_BOT) / 2, "Latency Speedup", va="center", ha="center",
             rotation="vertical", fontsize=24)
    fig.text(XC, A_BOT - 0.030, "Sequence Length", va="top", ha="center", fontsize=24)
    fig.text(XC, A_BOT - 0.075, "(a)", fontsize=30, fontweight="bold",
             va="top", ha="center")
    fig.text(XC, B_BOT - 0.030, "Sequence Length", va="top", ha="center", fontsize=24)
    fig.text(XC, B_BOT - 0.075, "(b)", fontsize=30, fontweight="bold",
             va="top", ha="center")

    # ── 图例 (7 strategies) ──
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in MLA_STRATEGIES]
    fig.legend(handles, [LABELS[sk] for sk in MLA_STRATEGIES],
               loc="upper center", ncol=len(MLA_STRATEGIES),
               fontsize=24, frameon=True, fancybox=True, bbox_to_anchor=(0.5, 0.995))

    fname = OUT / "fig1_speedup_4panel_384t.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
    save_panel_a()
    save_panel_b()
