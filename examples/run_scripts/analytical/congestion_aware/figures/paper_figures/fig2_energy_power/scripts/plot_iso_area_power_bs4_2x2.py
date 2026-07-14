#!/usr/bin/env python3
"""
Fig 2f: 2x2 layout — BS=4 only — ISO-AREA version of fig2e
  Top row:    Iso-Area Power Eff. Gain  ((Token/J)/mm² vs H100)  [Qwen3-235B | Llama4-Maverick]
  Bottom row: Iso-Area Power (W/mm²)                             [Qwen3-235B | Llama4-Maverick]

Same data/style as plot_power_bs4_2x2.py (fig2e), but every quantity is
normalized by the per-setting package area (mm²):
  - Top:    energy/power efficiency (Token/J, = tokens per watt-second) divided
            by package area, then taken as a ratio vs H100.
  - Bottom: instantaneous power divided by package area (power density).

Package areas (mm²). TP2 variants run on 2 chips → 2× base silicon.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1] / "plots"

STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new"]
LABELS = {
    "hmp_reo_new": "Ours", "rubin": "Rubin", "rubin_tp2": "Rubin TP2",
    "h100": "H100", "h100_tp2": "H100 TP2", "neupims": "NeuPIMs",
}
COLORS = {
    "h100": "#D4D4D4", "h100_tp2": "#DBDDEF",
    "rubin": "#92B1D9", "rubin_tp2": "#C1D8E9",
    "neupims": "#F4A582", "hmp_reo_new": "#E07850",
}
BASELINE = "h100"

# Per-setting package area (mm²). TP2 = 2 chips → 2× base.
PACKAGE_AREA = {
    "h100":        1516.0,
    "h100_tp2":    1516.0 * 2,   # 3032
    "rubin":       2568.0,
    "rubin_tp2":   2568.0 * 2,   # 5136
    "neupims":     1516.0,
    "hmp_reo_new": 1936.0,       # Ours (16 Cubes)
}

BS = 4

NEUPIMS_CONFIG = {
    "Qwen3-235B":      {"qkv": 2.0, "attn": 2.0 * 16 / 9, "o": 2.0, "comm_ns": 200},
    "Llama4-Maverick": {"qkv": 2.0, "attn": 2.0 * 5 / 9,  "o": 2.0, "comm_ns": 200},
}

MODELS = [
    ("Qwen3-235B", str(BASE / f"reports/qwen3-235B/power/gqa_hybrid_merged_96T_bw1500_util96_bs{BS}_power.json")),
    ("Llama4-Maverick", str(BASE / f"reports/llama4/power/gqa_hybrid_merged_96T_bw1500_util96_bs{BS}_power.json")),
]

HYBRID_MODELS = {
    "Qwen3-235B": str(BASE / f"reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{BS}.json"),
    "Llama4-Maverick": str(BASE / f"reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{BS}.json"),
}

SEQS = [4096, 262144, 1048576]


def seq_label(s):
    if s >= 1048576: return f"{s // 1048576}M"
    if s >= 1024:    return f"{s // 1024}K"
    return str(s)


def load(path):
    with open(path) as f:
        return json.load(f)


def get_power_map(data, strat):
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: e["total_power_w"] for e in entries}


def get_token_per_joule(data, strat):
    bs = data.get("metadata", {}).get("batch_size", 1)
    entries = data.get("strategies", {}).get(strat, {}).get("data", [])
    return {e["seq"]: bs * 1e9 / e["energy_per_token_nj"] if e["energy_per_token_nj"] > 0 else 0
            for e in entries}


def compute_neupims_token_per_joule(power_data, hybrid_data, model_label):
    cfg = NEUPIMS_CONFIG.get(model_label)
    if cfg is None:
        return {}

    bs = power_data.get("metadata", {}).get("batch_size", 1)
    ours_power_entries = power_data.get("strategies", {}).get("hmp_reo_new", {}).get("data", [])
    ours_hybrid_entries = hybrid_data.get("strategies", {}).get("hmp_reo_new", {}).get("data", [])

    ours_power_map = {e["seq"]: e["total_power_w"] for e in ours_power_entries}
    ours_module_map = {e["seq"]: (e["hybrid_Proj_QKV_ns"], e["hybrid_attn_ns"], e["hybrid_Proj_O_ns"])
                       for e in ours_hybrid_entries}

    result = {}
    for seq in SEQS:
        if seq not in ours_power_map or seq not in ours_module_map:
            continue
        qkv, attn, o = ours_module_map[seq]
        neupims_wall_ns = qkv * cfg["qkv"] + attn * cfg["attn"] + o * cfg["o"] + cfg["comm_ns"]
        neupims_power = ours_power_map[seq] + 760
        neupims_wall_s = neupims_wall_ns * 1e-9
        if neupims_power > 0 and neupims_wall_s > 0:
            result[seq] = bs / (neupims_power * neupims_wall_s)
        else:
            result[seq] = 0
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    n_strats = len(STRATEGIES)
    n_seqs = len(SEQS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharey='row')

    total_width = 0.88
    bar_w = total_width / n_strats
    offsets = np.arange(n_strats) * bar_w - total_width / 2 + bar_w / 2

    bot_max_global = 0  # for shared bottom-row y limit

    for col, (model_label, path) in enumerate(MODELS):
        data = load(path)
        hybrid_data = load(HYBRID_MODELS[model_label])

        # ── Top row: Iso-Area Power Eff. Gain ((Token/J)/mm² vs H100) ──
        ax_top = axes[0][col]
        x = np.arange(n_seqs)

        baseline_map = get_token_per_joule(data, BASELINE)
        baseline_area = PACKAGE_AREA[BASELINE]
        neupims_tpj = compute_neupims_token_per_joule(data, hybrid_data, model_label)

        def iso_area_eff_gain(sk):
            """ (Token/J / area_strat) / (Token/J_H100 / area_H100) per seq """
            tpj_map = neupims_tpj if sk == "neupims" else get_token_per_joule(data, sk)
            area = PACKAGE_AREA[sk]
            ratios = []
            for s in SEQS:
                v = tpj_map.get(s, 0)
                bv = baseline_map.get(s, 0)
                if v > 0 and bv > 0:
                    ratios.append((v / area) / (bv / baseline_area))
                else:
                    ratios.append(0)
            return ratios

        max_ratio = 0
        for si, sk in enumerate(STRATEGIES):
            ratios = iso_area_eff_gain(sk)
            ax_top.bar(x + offsets[si], ratios, bar_w,
                       color=COLORS[sk], edgecolor="white", linewidth=0.3)
            for r in ratios:
                if r > max_ratio:
                    max_ratio = r

        ax_top.axhline(1.0, color="#7F8C8D", ls="--", lw=0.8, alpha=0.6)
        ax_top.set_ylim(top=max_ratio * 1.35)

        labels_by_group = {i: [] for i in range(n_seqs)}
        for si, sk in enumerate(STRATEGIES):
            ratios = iso_area_eff_gain(sk)
            for i, r in enumerate(ratios):
                if r > 0:
                    labels_by_group[i].append((x[i] + offsets[si], r, f"{r:.1f}"))

        min_gap = max_ratio * 0.09
        for grp in labels_by_group.values():
            grp.sort(key=lambda t: t[1])
            y_positions = [item[1] for item in grp]
            for j in range(1, len(y_positions)):
                if y_positions[j] - y_positions[j - 1] < min_gap:
                    y_positions[j] = y_positions[j - 1] + min_gap
            for j, (xp, _, txt) in enumerate(grp):
                ax_top.text(xp, y_positions[j] + max_ratio * 0.01,
                            txt, ha="center", va="bottom",
                            fontsize=15, fontweight="bold",
                            color="black", rotation=0)
        ax_top.set_xticks(x)
        ax_top.set_xticklabels([seq_label(s) for s in SEQS], fontsize=24, fontweight="normal")
        ax_top.grid(axis="y", ls="--", alpha=0.25)
        ax_top.tick_params(axis="y", labelsize=24)
        ax_top.set_title(f"{model_label}, BS={BS}", fontsize=24, fontweight="normal")

        # ── Bottom row: Iso-Area Power (W/mm²) ──
        ax_bot = axes[1][col]

        ours_power_map = get_power_map(data, "hmp_reo_new")
        neupims_power_map = {s: v + 760 for s, v in ours_power_map.items()}

        def iso_area_power(sk):
            pmap = neupims_power_map if sk == "neupims" else get_power_map(data, sk)
            area = PACKAGE_AREA[sk]
            return [pmap.get(s, 0) / area for s in SEQS]

        max_val = 0
        for si, sk in enumerate(STRATEGIES):
            vals = iso_area_power(sk)
            ax_bot.bar(x + offsets[si], vals, bar_w,
                       color=COLORS[sk], edgecolor="white", linewidth=0.3)
            for v in vals:
                if v > max_val:
                    max_val = v
        if max_val > bot_max_global:
            bot_max_global = max_val

        for si, sk in enumerate(STRATEGIES):
            vals = iso_area_power(sk)
            for i, v in enumerate(vals):
                if v > 0:
                    if sk in ["h100", "h100_tp2"]:
                        ax_bot.text(x[i] + offsets[si], v + max_val * 0.01,
                                    f"{v:.2f}", ha="center", va="bottom",
                                    fontsize=15, fontweight="bold",
                                    color="black", rotation=90)
                    else:
                        ax_bot.text(x[i] + offsets[si], v - max_val * 0.03,
                                    f"{v:.2f}", ha="center", va="top",
                                    fontsize=15, fontweight="bold",
                                    color="black", rotation=90)

        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([seq_label(s) for s in SEQS], fontsize=24, fontweight="normal")
        ax_bot.grid(axis="y", ls="--", alpha=0.25)
        ax_bot.tick_params(axis="y", labelsize=24)

    # ── Bottom row: shared auto y-axis with headroom for rotated labels ──
    axes[1][0].set_ylim(0, bot_max_global * 1.30)

    # ── Y-axis labels (left column only, per row) ──
    axes[0][0].set_ylabel("Iso-Area Power\nEff. Gain", fontsize=26, fontweight="normal")
    axes[1][0].set_ylabel("Iso-Area Power\n(W/mm²)", fontsize=26, fontweight="normal")

    # ── Shared x-axis label ──
    fig.text(0.5, 0.02, "Sequence Length", va="top", ha="center",
             fontsize=26, fontweight="normal")

    # ── Legend ──
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    legend_labels = [LABELS[sk] for sk in STRATEGIES]
    fig.legend(handles, legend_labels,
               loc="upper center", ncol=n_strats,
               fontsize=26, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, 1.02),
               columnspacing = 0.6,
               handletextpad=0.3,
               handlelength = 1.0)

    # ── 间距独立控制 ──
    GAP_LEGEND_ROW1 = 0.08   # legend 与第一行图之间 (越大越远)
    GAP_ROW1_ROW2   = 0.35   # 第一行图与第二行图之间

    fig.tight_layout(rect=[0, 0, 1, 1.0 - GAP_LEGEND_ROW1])
    fig.subplots_adjust(hspace=GAP_ROW1_ROW2)
    fname = OUT / "fig2f_iso_area_power_bs4_2x2.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
