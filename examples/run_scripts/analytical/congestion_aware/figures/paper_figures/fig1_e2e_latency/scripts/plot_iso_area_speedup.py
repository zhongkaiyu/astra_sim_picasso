#!/usr/bin/env python3
"""
Fig 1c / 1d: ISO-AREA latency speedup (相对 H100), GQA models only.

Source: fig1_speedup_4panel_384t (GQA panel) latency speedup = H100_wall/strat_wall.
This is the *performance* per device; the iso-area version normalizes by package
area, i.e.
    iso_area_speedup = (H100_wall / strat_wall) * (area_H100 / area_strat)
                     = speedup * (1516 / area_strat)
consistent with fig2f's top-row iso-area efficiency gain.

Outputs:
  fig1c — 2x2: rows {Qwen, Llama4} × cols {BS=1, BS=4}
  fig1d — 1x2: BS=4 only {Qwen | Llama4}

Font / style kept consistent with fig2f_iso_area_power_bs4_2x2.py.

Package areas (mm²). TP2 variants run on 2 chips → 2× base silicon.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[4]   # congestion_aware/
OUT = Path(__file__).resolve().parents[1] / "plots"

STRATEGIES = ["h100", "h100_tp2", "rubin", "rubin_tp2", "neupims", "hmp_reo_new", "ours_384t"]
LABELS = {
    "hmp_reo_new": "Ours", "rubin": "Rubin", "rubin_tp2": "Rubin TP2",
    "h100": "H100", "h100_tp2": "H100 TP2", "neupims": "NeuPims",
    "ours_384t": "Ours_EC",
}
COLORS = {
    "h100": "#D4D4D4", "h100_tp2": "#DBDDEF",
    "rubin": "#92B1D9", "rubin_tp2": "#C1D8E9",
    "neupims": "#F4A582", "hmp_reo_new": "#E07850", "ours_384t": "#C04020",
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
    "ours_384t":   1936.0,       # Ours_EC (16 Cubes)
}

NEUPIMS_CONFIG = {
    "Qwen3-235B":      {"qkv": 2.0, "attn": 2.0 * 16 / 9, "o": 2.0, "comm_ns": 200},
    "Llama4-Maverick": {"qkv": 2.0, "attn": 2.0 * 5 / 9,  "o": 2.0, "comm_ns": 200},
}

BATCH_SIZES = [1, 4]
SEQS = [4096, 262144, 1048576]

MODELS = {
    "Qwen3-235B": {
        1: str(BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs1.json"),
        4: str(BASE / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs4.json"),
    },
    "Llama4-Maverick": {
        1: str(BASE / "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs1.json"),
        4: str(BASE / "reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs4.json"),
    },
}
DISP = {"Qwen3-235B": "Qwen", "Llama4-Maverick": "Llama4"}


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


def get_module_map_ours(data):
    entries = data.get("strategies", {}).get("hmp_reo_new", {}).get("data", [])
    return {e["seq"]: (e["hybrid_Proj_QKV_ns"], e["hybrid_attn_ns"], e["hybrid_Proj_O_ns"])
            for e in entries}


def compute_neupims_wall(ours_modules, model_label):
    cfg = NEUPIMS_CONFIG.get(model_label)
    if cfg is None:
        return {}
    return {seq: qkv * cfg["qkv"] + attn * cfg["attn"] + o * cfg["o"] + cfg["comm_ns"]
            for seq, (qkv, attn, o) in ours_modules.items()}


def get_iso_area_speedup(model_label, bs):
    """{strategy: [iso-area speedup per seq]} vs H100.

    GQA decode is memory-bound ⇒ Ours_EC (compute×4) wall = Ours wall.
    iso-area speedup = (H100_wall/strat_wall) * (area_H100/area_strat).
    """
    data = load(MODELS[model_label][bs])
    baseline = get_wall_map(data, BASELINE)
    ours_wall = get_wall_map(data, "hmp_reo_new")
    neupims = compute_neupims_wall(get_module_map_ours(data), model_label)
    area_base = PACKAGE_AREA[BASELINE]

    spd = {}
    for sk in STRATEGIES:
        if sk == "neupims":
            wall = neupims
        elif sk == "ours_384t":
            wall = ours_wall          # memory-bound ⇒ EC == Ours
        else:
            wall = get_wall_map(data, sk)
        area_factor = area_base / PACKAGE_AREA[sk]
        vals = []
        for s in SEQS:
            w, b = wall.get(s, 0), baseline.get(s, 0)
            vals.append((b / w) * area_factor if w > 0 and b > 0 else 0)
        spd[sk] = vals
    return spd


def draw_panel(ax, spd_by_strat):
    """Draw one iso-area speedup panel with fig2f-style fonts/labels."""
    n = len(STRATEGIES)
    n_seqs = len(SEQS)
    x = np.arange(n_seqs)
    total_width = 0.88
    bar_w = total_width / n
    offsets = np.arange(n) * bar_w - total_width / 2 + bar_w / 2

    max_val = 0
    for si, sk in enumerate(STRATEGIES):
        vals = spd_by_strat[sk]
        ax.bar(x + offsets[si], vals, bar_w,
               color=COLORS[sk], edgecolor="white", linewidth=0.3)
        for v in vals:
            if v > max_val:
                max_val = v

    ax.axhline(1.0, color="#7F8C8D", ls="--", lw=0.8, alpha=0.6)
    ax.set_ylim(top=max_val * 1.35)

    # value labels with vertical de-overlap (fig2f top-row style)
    labels_by_group = {i: [] for i in range(n_seqs)}
    for si, sk in enumerate(STRATEGIES):
        for i, v in enumerate(spd_by_strat[sk]):
            if v > 0:
                labels_by_group[i].append((x[i] + offsets[si], v, f"{v:.1f}"))
    min_gap = max_val * 0.09
    for grp in labels_by_group.values():
        grp.sort(key=lambda t: t[1])
        ys = [g[1] for g in grp]
        for j in range(1, len(ys)):
            if ys[j] - ys[j - 1] < min_gap:
                ys[j] = ys[j - 1] + min_gap
        for j, (xp, _, txt) in enumerate(grp):
            ax.text(xp, ys[j] + max_val * 0.01, txt, ha="center", va="bottom",
                    fontsize=15, fontweight="bold", color="black", rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([seq_label(s) for s in SEQS], fontsize=24, fontweight="normal")
    ax.grid(axis="y", ls="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=24)


def add_legend(fig, y=1.02):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[sk]) for sk in STRATEGIES]
    fig.legend(handles, [LABELS[sk] for sk in STRATEGIES],
               loc="upper center", ncol=len(STRATEGIES),
               fontsize=22, frameon=True, fancybox=True,
               bbox_to_anchor=(0.5, y),
               columnspacing=0.6, handletextpad=0.3, handlelength=1.0)


def make_fig1c():
    """2x2: rows {Qwen, Llama4} × cols {BS=1, BS=4}."""
    model_order = ["Qwen3-235B", "Llama4-Maverick"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    for r, model in enumerate(model_order):
        for c, bs in enumerate(BATCH_SIZES):
            ax = axes[r][c]
            draw_panel(ax, get_iso_area_speedup(model, bs))
            ax.set_title(f"{DISP[model]}, BS={bs}", fontsize=24, fontweight="normal")

    fig.text(0.04, 0.5, "Throughput per mm$^2$\n(normalized to H100)",
             va="center", ha="center", rotation=90,
             fontsize=26, fontweight="normal")
    fig.text(0.5, 0.02, "Sequence Length", va="top", ha="center",
             fontsize=26, fontweight="normal")
    add_legend(fig)
    fig.tight_layout(rect=[0.06, 0, 1, 0.92])
    fig.subplots_adjust(hspace=0.35)
    fname = OUT / "fig1c_iso_area_speedup_2x2.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


def make_fig1d():
    """1x2: BS=4 only {Qwen | Llama4}."""
    model_order = ["Qwen3-235B", "Llama4-Maverick"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    for c, model in enumerate(model_order):
        ax = axes[c]
        draw_panel(ax, get_iso_area_speedup(model, 4))
        ax.set_title(f"{DISP[model]}, BS=4", fontsize=24, fontweight="normal")
        if c == 0:
            ax.set_ylabel("Throughput per mm$^2$\n(normalized to H100)",
                          fontsize=26, fontweight="normal")

    fig.text(0.5, -0.01, "Sequence Length", va="top", ha="center",
             fontsize=26, fontweight="normal")
    add_legend(fig, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fname = OUT / "fig1d_iso_area_speedup_bs4_1x2.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_fig1c()
    make_fig1d()


if __name__ == "__main__":
    main()
