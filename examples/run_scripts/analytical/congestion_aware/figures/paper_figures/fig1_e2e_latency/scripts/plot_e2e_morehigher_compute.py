#!/usr/bin/env python3
"""
Fig 1 Higher Compute: Ours at 256T and 512T (vs same H100/Rubin baselines).
Same style as fig1_speedup_4panel.pdf.
"""
import json, sys, bisect, math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[4]          # congestion_aware/
OUT  = Path(__file__).resolve().parents[1] / "plots"

sys.path.insert(0, str(BASE / "backend" / "roofline"))
from roofline_gqa_calc import roofline_qkv, roofline_attention, roofline_output
from roofline_gqa_calc import calc_mla_strategy

# ── Plot config ──
STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "ours_768t", "ours_1024t"]
LABELS = {
    "h100":        "H100",
    "h100_tp2":    "H100 TP2",
    "rubin":       "Rubin",
    "rubin_tp2":   "Rubin TP2",
    "ours_768t":   "Ours 768T",
    "ours_1024t":  "Ours 1024T",
}
COLORS = {
    "h100":        "#D4D4D4",
    "h100_tp2":    "#DBDDEF",
    "rubin":       "#92B1D9",
    "rubin_tp2":   "#C1D8E9",
    "ours_768t":   "#F4A582",
    "ours_1024t":  "#E07850",
}
BASELINE   = "h100"
BATCH_SIZES = [1, 4, 16, 32]
SEQS        = [4096, 65536, 262144, 1048576]

# ── GQA per-NPU configs (hmp_reo_new: tp_h=4, tp_hd=4, tp_s=4) ──
GQA_NPU_CFGS = {
    "Qwen3-235B": {
        "d_model": 4096, "num_attention_heads": 16, "num_kv_heads": 1,
        "d_head": 32, "d_head_full": 128, "tp_s": 4, "tp_hd": 4,
        "num_layers": 1, "fused_attention": False,
    },
    "Llama4-Maverick": {
        "d_model": 5120, "num_attention_heads": 10, "num_kv_heads": 2,
        "d_head": 32, "d_head_full": 128, "tp_s": 4, "tp_hd": 4,
        "num_layers": 1, "fused_attention": False,
    },
}
GQA_UTIL_MODELS = {"Qwen3-235B": "qwen3", "Llama4-Maverick": "llama4"}

HBM_BW   = 2.5   # TB/s per NPU
LINK_BW  = 1.5   # TB/s D2D

# ── GQA 96T data paths (for baselines & comm) ──
GQA_96T_PATHS = {
    "Qwen3-235B":     "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{bs}.json",
    "Llama4-Maverick": "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{bs}.json",
}

# ── MLA data path (for baselines) ──
MLA_DATA_PATH = str(BASE / "figures/paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json")


# ── Utilization helpers ──
def _load_util_profile(model_key, num_sa):
    path = BASE / "backend" / "roofline" / "utilization_profiles" / model_key / f"util_{num_sa}.json"
    with open(path) as f:
        return json.load(f)


def _lookup_util(util_map, key):
    parsed = {}
    for k, v in util_map.items():
        if k.startswith("_"):
            continue
        parsed[int(k)] = v["utilization"] if isinstance(v, dict) else float(v)
    if key in parsed:
        return parsed[key]
    skeys = sorted(parsed)
    if key <= skeys[0]:
        return parsed[skeys[0]]
    if key >= skeys[-1]:
        return parsed[skeys[-1]]
    idx = bisect.bisect_right(skeys, key) - 1
    lo, hi = skeys[idx], skeys[idx + 1]
    t = (math.log2(key) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return parsed[lo] + t * (parsed[hi] - parsed[lo])


# ── GQA Ours compute at arbitrary peak_perf ──
def gqa_ours_wall(model_label, peak_perf, bs, seq):
    """Compute Ours wall time for GQA model at given peak_perf (TFLOPS per NPU)."""
    cfg = GQA_NPU_CFGS[model_label]
    num_sa = int(peak_perf)
    util_key = GQA_UTIL_MODELS[model_label]
    util_data = _load_util_profile(util_key, num_sa)

    # Per-NPU roofline
    qkv = roofline_qkv(peak_perf, HBM_BW, cfg, bs)
    attn = roofline_attention(peak_perf, HBM_BW, cfg, bs, seq)
    out = roofline_output(peak_perf, HBM_BW, cfg, bs)

    # Apply SA utilization: adj = max(comp/util, mem)
    qkv_util = _lookup_util(util_data["proj_qkv"], bs)
    qkv_comp_adj = qkv["compute_ns"] / qkv_util if qkv_util > 0 else qkv["compute_ns"]
    qkv_ns = max(qkv_comp_adj, qkv["memory_ns"])

    # Attention: score + attn_v utilization
    dk = cfg.get("d_head_full", cfg["d_head"])
    cache_seq = seq // cfg.get("tp_s", 1)
    effective_seq = cache_seq * dk // 128

    if "sub_ops" in attn:
        score = attn["sub_ops"]["score_gemv"]
        attn_v = attn["sub_ops"]["attn_v_gemv"]
        score_util = _lookup_util(util_data["score"], effective_seq)
        attn_v_util = _lookup_util(util_data["attention"], effective_seq)
        sc_adj = max(score["compute_ns"] / score_util if score_util > 0 else score["compute_ns"],
                     score["memory_ns"])
        av_adj = max(attn_v["compute_ns"] / attn_v_util if attn_v_util > 0 else attn_v["compute_ns"],
                     attn_v["memory_ns"])
        attn_ns = sc_adj + av_adj
    else:
        # fused path
        fused_util = _lookup_util(util_data.get("attn_fused", util_data.get("score", {})), effective_seq)
        comp_adj = attn.get("einsum_comp_ns", 0) / fused_util if fused_util > 0 else attn.get("einsum_comp_ns", 0)
        attn_ns = max(comp_adj, attn.get("kv_mem_ns", attn["total_ns"]))

    out_util = _lookup_util(util_data["proj_o"], bs)
    out_comp_adj = out["compute_ns"] / out_util if out_util > 0 else out["compute_ns"]
    out_ns = max(out_comp_adj, out["memory_ns"])

    # Comm from 96T data (topology unchanged)
    comm_ns = _get_gqa_comm(model_label, bs, seq)

    gpu_ns = qkv_ns + attn_ns + out_ns
    wall_ns = gpu_ns + comm_ns
    return {
        "hybrid_wall_ns": wall_ns,
        "hybrid_Proj_QKV_ns": qkv_ns,
        "hybrid_attn_ns": attn_ns,
        "hybrid_Proj_O_ns": out_ns,
        "comm_total_ns": comm_ns,
    }


_gqa_96t_cache = {}

def _get_gqa_comm(model_label, bs, seq):
    key = (model_label, bs)
    if key not in _gqa_96t_cache:
        path = str(BASE / GQA_96T_PATHS[model_label].format(bs=bs))
        with open(path) as f:
            data = json.load(f)
        entries = data["strategies"]["hmp_reo_new"]["data"]
        _gqa_96t_cache[key] = {e["seq"]: e["comm_total_ns"] for e in entries}
    return _gqa_96t_cache[key].get(seq, 263.0)


# ── Data loading helpers ──
def _load_json(path):
    with open(path) as f:
        return json.load(f)

_mla_cache = None
def _load_mla():
    global _mla_cache
    if _mla_cache is None:
        _mla_cache = _load_json(MLA_DATA_PATH)
    return _mla_cache


def get_wall_map_gqa(model_label, bs, strat):
    path = str(BASE / GQA_96T_PATHS[model_label].format(bs=bs))
    data = _load_json(path)
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["hybrid_wall_ns"] for e in entries}


def get_wall_map_mla(bs, strat):
    mla = _load_mla()
    strats = mla.get("strategies", {})
    if strat in strats and "data_by_bs" in strats[strat]:
        entries = strats[strat]["data_by_bs"].get(f"bs{bs}", [])
    else:
        entries = mla.get("data", {}).get(f"bs{bs}", [])
    return {e["seq"]: e["hybrid_wall_ns"] for e in entries}


# ── Model list ──
MODELS = ["Qwen3-235B", "Llama4-Maverick", "DeepSeek-V3 MLA"]


def get_baseline_map(model_label, bs, strat):
    if model_label == "DeepSeek-V3 MLA":
        return get_wall_map_mla(bs, strat)
    else:
        return get_wall_map_gqa(model_label, bs, strat)


_mla_wall_cache = {}

def get_ours_wall(model_label, peak_perf, bs, seq):
    if model_label == "DeepSeek-V3 MLA":
        key = (peak_perf, bs)
        if key not in _mla_wall_cache:
            hw = {"compute": peak_perf, "Bandwidth": HBM_BW}
            result = calc_mla_strategy(hw, bs, SEQS, link_bw=LINK_BW,
                                       tp_h=1, tp_s=16)
            _mla_wall_cache[key] = {e["seq"]: e["wall_total_ns"] for e in result["data"]}
        return _mla_wall_cache[key].get(seq, 0)
    else:
        e = gqa_ours_wall(model_label, peak_perf, bs, seq)
        return e["hybrid_wall_ns"]


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_models  = len(MODELS)
    n_batches = len(BATCH_SIZES)
    n_strats  = len(STRATEGIES)
    n_seqs    = len(SEQS)

    fig, axes = plt.subplots(n_models, n_batches,
                             figsize=(3.5 * n_batches, 1.5 * n_models),
                             sharey=True)

    for row, model_label in enumerate(MODELS):
        is_mla = (model_label == "DeepSeek-V3 MLA")
        for col, bs in enumerate(BATCH_SIZES):
            baseline_map = get_baseline_map(model_label, bs, BASELINE)
            ax = axes[row][col]
            x = np.arange(n_seqs)

            total_width = 0.88
            bar_w = total_width / n_strats
            offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

            max_sp = 0
            labels_by_group = {i: [] for i in range(n_seqs)}

            for si, sk in enumerate(STRATEGIES):
                if sk.startswith("ours_"):
                    pp = int(sk.split("_")[1].replace("t", ""))
                    wall_map = {s: get_ours_wall(model_label, pp, bs, s) for s in SEQS}
                else:
                    wall_map = get_baseline_map(model_label, bs, sk)

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
            ax.set_ylim(0, 30)

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

            if row == 0:
                ax.set_title(f"BS={bs}", fontsize=10, fontweight="bold", pad=4)
            if col == n_batches - 1:
                title = "DeepSeek-V3" if is_mla else model_label
                ax.annotate(title, xy=(1.02, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", rotation=-90,
                            ha="left", va="center")
            if row < n_models - 1:
                ax.set_xticklabels([])

    fig.text(0.5, -0.01, "Sequence Length", va="top", ha="center", fontsize=14)
    fig.text(-0.01, 0.5, "Latency Speedup", va="center", ha="center",
             rotation="vertical", fontsize=14)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=10, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.02))

    fig.tight_layout(rect=[0, 0, 0.93, 0.95])
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    fname = OUT / "fig1_speedup_4panel_morehighercmp.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
