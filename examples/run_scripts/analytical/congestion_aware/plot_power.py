#!/usr/bin/env python3
"""Visualize GQA single-layer power model results from power JSON reports."""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

STRAT_KEYS = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]
STRAT_LABELS = {
    "HMP_reo": "HMP_RO", "hmp_reo_new": "RO_new",
    "hmp": "HMP", "tp16": "TP16", "rubin": "Rubin",
}
STRAT_COLORS = {
    "HMP_reo":     "#E67E22",
    "hmp_reo_new": "#8E44AD",
    "hmp":         "#2E86C1",
    "tp16":        "#27AE60",
    "rubin":       "#C0392B",
}
STRAT_MARKERS = {
    "HMP_reo": "s", "hmp_reo_new": "P", "hmp": "D", "tp16": "^", "rubin": "o",
}

COMP_COLORS = {"hbm": "#3498DB", "cmpt": "#E74C3C", "d2d": "#F39C12"}
COMP_LABELS = {"hbm": "HBM", "cmpt": "Compute", "d2d": "D2D Link"}


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def seq_label(s: int) -> str:
    if s >= 1048576:
        return f"{s // 1048576}M"
    if s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def _available_strats(summary):
    return [sk for sk in STRAT_KEYS
            if any(sk in row for row in summary)]


def plot_total_power(summary, out_dir: Path, title_suffix: str):
    """Fig 1: Total power (W) vs sequence length, per strategy."""
    strats = _available_strats(summary)
    fig, ax = plt.subplots(figsize=(10, 5))

    for sk in strats:
        seqs = [r["seq"] for r in summary if sk in r]
        pows = [r[sk]["total_power_w"] for r in summary if sk in r]
        ax.plot(seqs, pows, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Total Power (W)", fontsize=12)
    ax.set_title(f"GQA Power: Total Power vs Seq{title_suffix}", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "power_total_vs_seq.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_total_vs_seq.png")


def plot_energy_per_token(summary, out_dir: Path, title_suffix: str):
    """Fig 2: Energy ratio vs Rubin baseline at each seq (Rubin=1)."""
    strats = _available_strats(summary)
    has_rubin = "rubin" in strats
    seq_to_row = {row["seq"]: row for row in summary}

    fig, ax = plt.subplots(figsize=(10, 5))

    for sk in strats:
        seqs = [r["seq"] for r in summary if sk in r]
        energies_nj = [r[sk]["energy_nj"] for r in summary if sk in r]

        if has_rubin and sk != "rubin":
            # Ratio vs Rubin at each data point: energy_rubin / energy_strat
            rubin_nj = [seq_to_row.get(s, {}).get("rubin", {}).get("energy_nj") for s in seqs]
            ratios = []
            for i, e in enumerate(energies_nj):
                r_nj = rubin_nj[i]
                if e and e > 0 and r_nj and r_nj > 0:
                    ratios.append(r_nj / e)
                else:
                    ratios.append(np.nan)
            y_vals = ratios
        else:
            # Rubin baseline: ratio = 1 everywhere
            y_vals = [1.0] * len(seqs) if sk == "rubin" else [np.nan] * len(seqs)

        ax.plot(seqs, y_vals, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2, markersize=6)

        # Annotate each point with ratio value (skip Rubin, always 1.0)
        if sk != "rubin":
            for s, y in zip(seqs, y_vals):
                if not np.isnan(y) and y > 0:
                    offset_y = 5 if y >= 1 else -8
                    ax.annotate(f"{y:.2f}", (s, y), textcoords="offset points", xytext=(0, offset_y),
                                ha="center", fontsize=7, color=STRAT_COLORS[sk])

    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Token/Energy Ratio (× Rubin, 1 = Rubin baseline)", fontsize=12)
    ax.set_title(f"GQA Power: Energy Efficiency vs Rubin{title_suffix}", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "power_energy_per_token_vs_seq.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_energy_per_token_vs_seq.png")


def plot_energy_per_token_trend(summary, out_dir: Path, title_suffix: str):
    """Energy/Token vs Seq trend: per-strategy comparison as seq scales."""
    strats = _available_strats(summary)
    fig, ax = plt.subplots(figsize=(10, 5))

    for sk in strats:
        seqs = [r["seq"] for r in summary if sk in r]
        energies = [r[sk]["energy_nj"] / 1e3 for r in summary if sk in r]  # nJ → μJ
        ax.plot(seqs, energies, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Energy per Token (μJ)", fontsize=12)
    ax.set_title(f"Energy/Token vs Seq — Strategy Comparison{title_suffix}", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "energy_per_token_trend.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: energy_per_token_trend.png")


def plot_power_breakdown_bars(summary, out_dir: Path, title_suffix: str):
    """Fig 3: Stacked bar — HBM / Compute / D2D breakdown at each seq, grouped by strategy."""
    strats = _available_strats(summary)
    seqs = [r["seq"] for r in summary]
    x_labels = [seq_label(s) for s in seqs]
    n_seq = len(seqs)
    n_strat = len(strats)

    fig, ax = plt.subplots(figsize=(max(12, n_seq * 1.2), 6))
    bar_w = 0.18
    x = np.arange(n_seq)

    for si, sk in enumerate(strats):
        hbm_vals, cmpt_vals, d2d_vals = [], [], []
        for row in summary:
            d = row.get(sk, {})
            hbm_vals.append(d.get("hbm_power_w", 0))
            cmpt_vals.append(d.get("cmpt_power_w", 0))
            d2d_vals.append(d.get("d2d_power_w", 0))

        offset = (si - n_strat / 2 + 0.5) * bar_w
        hbm_arr = np.array(hbm_vals)
        cmpt_arr = np.array(cmpt_vals)
        d2d_arr = np.array(d2d_vals)

        p1 = ax.bar(x + offset, hbm_arr, bar_w, color=COMP_COLORS["hbm"],
                     edgecolor="white", linewidth=0.5)
        p2 = ax.bar(x + offset, cmpt_arr, bar_w, bottom=hbm_arr,
                     color=COMP_COLORS["cmpt"], edgecolor="white", linewidth=0.5)
        p3 = ax.bar(x + offset, d2d_arr, bar_w, bottom=hbm_arr + cmpt_arr,
                     color=COMP_COLORS["d2d"], edgecolor="white", linewidth=0.5)

        for xi in range(n_seq):
            total = hbm_arr[xi] + cmpt_arr[xi] + d2d_arr[xi]
            if total > 0:
                ax.text(x[xi] + offset, total + 8, STRAT_LABELS[sk],
                        ha="center", va="bottom", fontsize=7, rotation=45)

    comp_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=COMP_COLORS["hbm"]),
        plt.Rectangle((0, 0), 1, 1, fc=COMP_COLORS["cmpt"]),
        plt.Rectangle((0, 0), 1, 1, fc=COMP_COLORS["d2d"]),
    ]
    ax.legend(comp_handles, [COMP_LABELS[k] for k in ["hbm", "cmpt", "d2d"]],
              fontsize=10, loc="upper left")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.set_title(f"GQA Power Breakdown: HBM / Compute / D2D{title_suffix}", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "power_breakdown_bars.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_breakdown_bars.png")


def plot_power_breakdown_single_seq(summary, strategies_data, out_dir: Path,
                                     title_suffix: str, target_seq=65536):
    """Fig 4: Horizontal bar — per-strategy power breakdown at one seq length."""
    strats = _available_strats(summary)
    row = next((r for r in summary if r["seq"] == target_seq), summary[-1])
    actual_seq = row["seq"]

    fig, ax = plt.subplots(figsize=(10, 4))
    y_pos = np.arange(len(strats))
    bar_h = 0.5

    hbm_vals = [row.get(sk, {}).get("hbm_power_w", 0) for sk in strats]
    cmpt_vals = [row.get(sk, {}).get("cmpt_power_w", 0) for sk in strats]
    d2d_vals = [row.get(sk, {}).get("d2d_power_w", 0) for sk in strats]

    hbm_arr = np.array(hbm_vals)
    cmpt_arr = np.array(cmpt_vals)
    d2d_arr = np.array(d2d_vals)

    ax.barh(y_pos, hbm_arr, bar_h, color=COMP_COLORS["hbm"], label=COMP_LABELS["hbm"])
    ax.barh(y_pos, cmpt_arr, bar_h, left=hbm_arr, color=COMP_COLORS["cmpt"],
            label=COMP_LABELS["cmpt"])
    ax.barh(y_pos, d2d_arr, bar_h, left=hbm_arr + cmpt_arr, color=COMP_COLORS["d2d"],
            label=COMP_LABELS["d2d"])

    for i, sk in enumerate(strats):
        total = hbm_arr[i] + cmpt_arr[i] + d2d_arr[i]
        ax.text(total + 5, i, f"{total:.0f}W", va="center", fontsize=10, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([STRAT_LABELS[sk] for sk in strats], fontsize=11)
    ax.set_xlabel("Power (W)", fontsize=12)
    ax.set_title(f"Power Breakdown @ seq={seq_label(actual_seq)}{title_suffix}", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / f"power_breakdown_seq{actual_seq}.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_breakdown_seq{actual_seq}.png")


def plot_utilization(strategies_data, out_dir: Path, title_suffix: str):
    """Fig 5: HBM and Compute utilization vs seq, per strategy (dual y-axis)."""
    strats = [sk for sk in STRAT_KEYS if sk in strategies_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for sk in strats:
        entries = strategies_data[sk]["data"]
        seqs = [e["seq"] for e in entries]
        hbm_u = [e.get("hbm_util_per_cube", 0) * 100 for e in entries]
        cmpt_u = [e.get("cmpt_util_per_cube", e.get("cmpt_util_per_die", 0)) * 100
                  for e in entries]

        ax1.plot(seqs, hbm_u, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                 color=STRAT_COLORS[sk], linewidth=2, markersize=5)
        ax2.plot(seqs, cmpt_u, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                 color=STRAT_COLORS[sk], linewidth=2, markersize=5)

    for ax, title in [(ax1, "HBM Utilization"), (ax2, "Compute Utilization")]:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
        ax.set_xlabel("Sequence Length", fontsize=12)
        ax.set_ylabel("Utilization (%)", fontsize=12)
        ax.set_title(f"{title}{title_suffix}", fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(out_dir / "power_utilization_vs_seq.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_utilization_vs_seq.png")


def plot_power_efficiency(summary, out_dir: Path, title_suffix: str):
    """Fig 6: Tokens per Watt per μs (throughput efficiency) vs seq."""
    strats = _available_strats(summary)
    fig, ax = plt.subplots(figsize=(10, 5))

    for sk in strats:
        seqs, eff = [], []
        for r in summary:
            if sk not in r:
                continue
            seqs.append(r["seq"])
            time_us = r[sk]["time_ns"] / 1e3
            power_w = r[sk]["total_power_w"]
            eff.append(1.0 / (time_us * power_w) * 1e6 if power_w > 0 else 0)
        ax.plot(seqs, eff, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Tokens / (W · μs)", fontsize=12)
    ax.set_title(f"GQA Power Efficiency: Tokens per Watt-μs{title_suffix}", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "power_efficiency_vs_seq.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_efficiency_vs_seq.png")


def plot_tdp_fraction(strategies_data, out_dir: Path, title_suffix: str):
    """Fig 7: Actual power as fraction of TDP vs seq."""
    strats = [sk for sk in STRAT_KEYS if sk in strategies_data]
    fig, ax = plt.subplots(figsize=(10, 5))

    for sk in strats:
        entries = strategies_data[sk]["data"]
        seqs = [e["seq"] for e in entries]
        fracs = [e["total_power_w"] / e["tdp_total_w"] * 100
                 if e.get("tdp_total_w", 0) > 0 else 0 for e in entries]
        ax.plot(seqs, fracs, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2, markersize=6)

    ax.axhline(y=100, color="gray", linestyle="--", linewidth=1, alpha=0.7, label="TDP limit")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Power / TDP (%)", fontsize=12)
    ax.set_title(f"GQA Power: Actual vs TDP{title_suffix}", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "power_tdp_fraction_vs_seq.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_tdp_fraction_vs_seq.png")


HBM_MOD_COLORS = {
    "Proj_QKV": "#1B4F72",
    "Attention": "#3498DB",
    "Proj_O":    "#A9CCE3",
}


def plot_hbm_breakdown(summary, strategies_data, out_dir: Path, title_suffix: str):
    """HBM power breakdown by module (Proj_QKV / Attention / Proj_O) per strategy across seq."""
    strats = _available_strats(summary)
    seqs = [r["seq"] for r in summary]
    x = np.arange(len(seqs))
    seq_labels = [seq_label(s) for s in seqs]
    n_strat = len(strats)

    fig, axes = plt.subplots(1, n_strat, figsize=(6 * n_strat, 8), sharey=True)
    if n_strat == 1:
        axes = [axes]

    bar_w = 0.65
    global_ymax = 0

    for idx, sk in enumerate(strats):
        qkv_vals, attn_vals, projo_vals = [], [], []
        for row in summary:
            d = row.get(sk, {})
            bd = d.get("hbm_breakdown", {})
            qkv_vals.append(bd.get("Proj_QKV_hbm_w", 0))
            attn_vals.append(bd.get("Attn_hbm_w", 0))
            projo_vals.append(bd.get("Proj_O_hbm_w", 0))

        qkv_arr = np.array(qkv_vals)
        attn_arr = np.array(attn_vals)
        projo_arr = np.array(projo_vals)
        top = qkv_arr + attn_arr + projo_arr
        if top.max() > global_ymax:
            global_ymax = top.max()

    y_limit = global_ymax * 1.12

    for idx, sk in enumerate(strats):
        ax = axes[idx]
        qkv_vals, attn_vals, projo_vals = [], [], []
        for row in summary:
            d = row.get(sk, {})
            bd = d.get("hbm_breakdown", {})
            qkv_vals.append(bd.get("Proj_QKV_hbm_w", 0))
            attn_vals.append(bd.get("Attn_hbm_w", 0))
            projo_vals.append(bd.get("Proj_O_hbm_w", 0))

        qkv_arr = np.array(qkv_vals)
        attn_arr = np.array(attn_vals)
        projo_arr = np.array(projo_vals)

        ax.bar(x, qkv_arr, bar_w, color=HBM_MOD_COLORS["Proj_QKV"])
        ax.bar(x, attn_arr, bar_w, bottom=qkv_arr, color=HBM_MOD_COLORS["Attention"])
        ax.bar(x, projo_arr, bar_w, bottom=qkv_arr + attn_arr, color=HBM_MOD_COLORS["Proj_O"])

        ax.set_ylim(0, y_limit)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels, rotation=45, fontsize=14)
        ax.set_xlabel("Seq", fontsize=16)
        ax.set_title(STRAT_LABELS[sk], fontsize=18, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.tick_params(axis="y", labelsize=14)

    axes[0].set_ylabel("HBM Power (W)", fontsize=16)

    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=HBM_MOD_COLORS["Proj_QKV"]),
        plt.Rectangle((0, 0), 1, 1, fc=HBM_MOD_COLORS["Attention"]),
        plt.Rectangle((0, 0), 1, 1, fc=HBM_MOD_COLORS["Proj_O"]),
    ]
    labels = ["Proj_QKV (weight)", "Attention (KV cache)", "Proj_O (weight)"]

    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=14,
               bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(f"HBM Power Breakdown by Module{title_suffix}",
                 fontsize=20, fontweight="bold", y=1.10)
    fig.tight_layout()
    fig.savefig(out_dir / "power_hbm_breakdown.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: power_hbm_breakdown.png")


def _static_power_w(sk: str, meta: dict) -> float:
    """Static power (W) from config. D2D has no static component."""
    if sk == "rubin":
        cfg = meta.get("rubin_config", {})
        sr = cfg.get("static_ratio", 0.1)
        return (cfg.get("n_hbm_cubes", 8) * cfg.get("hbm_tdp_per_cube", 75)
                + cfg.get("n_cmpt_dies", 2) * cfg.get("cmpt_tdp_per_die", 800)) * sr
    cfg = meta.get("ours_config", {})
    sr = cfg.get("static_ratio", 0.1)
    n = cfg.get("n_cubes", 16)
    return n * (cfg.get("hbm_tdp_per_cube", 75) + cfg.get("cmpt_tdp_per_cube", 15)) * sr


def plot_static_dynamic_energy(data: dict, out_dir: Path, title_suffix: str):
    """Static and dynamic energy (nJ) vs seq, per strategy."""
    strategies_data = data.get("strategies", {})
    meta = data.get("metadata", {})
    strats = [sk for sk in STRAT_KEYS if sk in strategies_data]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Top: Stacked area or grouped bars - static vs dynamic energy
    ax1 = axes[0]
    for sk in strats:
        entries = strategies_data[sk]["data"]
        seqs = [e["seq"] for e in entries]
        time_s = [e["time_ns"] * 1e-9 for e in entries]
        p_static = _static_power_w(sk, meta)
        p_dynamic = [e["total_power_w"] - p_static for e in entries]
        e_static = [p_static * t * 1e9 for t in time_s]
        e_dynamic = [pd * t * 1e9 for pd, t in zip(p_dynamic, time_s)]
        ax1.plot(seqs, e_static, marker=STRAT_MARKERS[sk], label=f"{STRAT_LABELS[sk]} static",
                 color=STRAT_COLORS[sk], linewidth=2, markersize=5, linestyle="--")
        ax1.plot(seqs, e_dynamic, marker=STRAT_MARKERS[sk], label=f"{STRAT_LABELS[sk]} dynamic",
                 color=STRAT_COLORS[sk], linewidth=2, markersize=5)

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_ylabel("Energy (nJ)", fontsize=12)
    ax1.set_title(f"Static vs Dynamic Energy per Token{title_suffix}", fontsize=13)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax1.legend(fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Bottom: Fraction of dynamic energy
    ax2 = axes[1]
    for sk in strats:
        entries = strategies_data[sk]["data"]
        seqs = [e["seq"] for e in entries]
        time_s = [e["time_ns"] * 1e-9 for e in entries]
        p_static = _static_power_w(sk, meta)
        p_dynamic = [e["total_power_w"] - p_static for e in entries]
        e_total = [e["energy_per_token_nj"] for e in entries]
        e_dynamic = [pd * t * 1e9 for pd, t in zip(p_dynamic, time_s)]
        frac_dynamic = [ed / et * 100 if et > 0 else 0 for ed, et in zip(e_dynamic, e_total)]
        ax2.plot(seqs, frac_dynamic, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                 color=STRAT_COLORS[sk], linewidth=2, markersize=5)

    ax2.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Sequence Length", fontsize=12)
    ax2.set_ylabel("Dynamic Energy (%)", fontsize=12)
    ax2.set_title("Dynamic Energy Fraction vs Seq", fontsize=13)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    fig.tight_layout()
    fig.savefig(out_dir / "power_static_dynamic_energy_vs_seq.png", dpi=180)
    plt.close(fig)
    print(f"  Saved: power_static_dynamic_energy_vs_seq.png")


def _build_static_dynamic_rows(data: dict):
    """Build list of (seq, strat, p_static, p_total, p_dyn, e_total, e_static, e_dynamic, dyn_pct)."""
    strategies_data = data.get("strategies", {})
    meta = data.get("metadata", {})
    rows = []
    for sk in STRAT_KEYS:
        if sk not in strategies_data:
            continue
        p_static = _static_power_w(sk, meta)
        for e in strategies_data[sk]["data"]:
            seq = e["seq"]
            p_total = e["total_power_w"]
            p_dyn = p_total - p_static
            time_s = e["time_ns"] * 1e-9
            e_total = e["energy_per_token_nj"]
            e_static = p_static * time_s * 1e9
            e_dynamic = p_dyn * time_s * 1e9
            dyn_pct = e_dynamic / e_total * 100 if e_total > 0 else 0
            rows.append((seq, sk, p_static, p_total, p_dyn, e_total, e_static, e_dynamic, dyn_pct))
    return rows


def export_static_dynamic_table(data: dict, out_dir: Path):
    """Export static/dynamic data to CSV and print wide-format table."""
    import csv
    rows = _build_static_dynamic_rows(data)
    strats = [sk for sk in STRAT_KEYS if any(r[1] == sk for r in rows)]
    seqs = sorted(set(r[0] for r in rows))

    # CSV: long format
    csv_path = out_dir / "power_static_dynamic_energy.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seq", "strategy", "P_static_W", "P_total_W", "P_dynamic_W",
                    "E_total_nJ", "E_static_nJ", "E_dynamic_nJ", "Dyn_pct"])
        for r in rows:
            w.writerow([r[0], r[1], f"{r[2]:.2f}", f"{r[3]:.2f}", f"{r[4]:.2f}",
                        f"{r[5]:,.0f}", f"{r[6]:,.0f}", f"{r[7]:,.0f}", f"{r[8]:.1f}"])

    print(f"  Saved: power_static_dynamic_energy.csv")

    # Wide table: seq × (strategy: E_static, E_dynamic, E_total, Dyn%)
    lookup = {(r[0], r[1]): r for r in rows}
    col_w = 14

    print("\n" + "=" * 160)
    print("  Static vs Dynamic Energy per Token — Full Data Table")
    print("  E_static = P_static × time, E_dynamic = (P_total - P_static) × time")
    print("=" * 160)

    # Header
    hdr = f"{'Seq':>8}"
    for sk in strats:
        hdr += f" | {STRAT_LABELS[sk] + '_E_static':>{col_w}} {STRAT_LABELS[sk] + '_E_dyn':>{col_w}} {STRAT_LABELS[sk] + '_E_total':>{col_w}} {STRAT_LABELS[sk] + '_Dyn%':>7}"
    print(hdr)
    print("-" * len(hdr))

    for seq in seqs:
        line = f"{seq_label(seq):>8}"
        for sk in strats:
            r = lookup.get((seq, sk))
            if r:
                line += f" | {r[6]:>{col_w},.0f} {r[7]:>{col_w},.0f} {r[5]:>{col_w},.0f} {r[8]:>6.1f}%"
            else:
                line += f" | {'—':>{col_w}} {'—':>{col_w}} {'—':>{col_w}} {'—':>7}"
        print(line)

    # Power table
    print("\n" + "-" * 160)
    print("  Power (W) — P_static, P_total, P_dynamic")
    print("-" * 160)
    hdr2 = f"{'Seq':>8}"
    for sk in strats:
        hdr2 += f" | {STRAT_LABELS[sk] + '_P_sta':>9} {STRAT_LABELS[sk] + '_P_tot':>9} {STRAT_LABELS[sk] + '_P_dyn':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    for seq in seqs:
        line = f"{seq_label(seq):>8}"
        for sk in strats:
            r = lookup.get((seq, sk))
            if r:
                line += f" | {r[2]:>9.1f} {r[3]:>9.1f} {r[4]:>9.1f}"
            else:
                line += f" | {'—':>9} {'—':>9} {'—':>9}"
        print(line)
    print()


def print_static_dynamic_table(data: dict):
    """Print static/dynamic power and energy table to stdout (per-strategy view)."""
    strategies_data = data.get("strategies", {})
    meta = data.get("metadata", {})
    strats = [sk for sk in STRAT_KEYS if sk in strategies_data]

    print("\n" + "=" * 100)
    print("  Static vs Dynamic Power & Energy per Token (per strategy)")
    print("  P_static = static_ratio × TDP (HBM+Compute), P_dynamic = P_total - P_static, E = P × time")
    print("=" * 100)

    for sk in strats:
        p_static = _static_power_w(sk, meta)
        entries = strategies_data[sk]["data"]
        print(f"\n--- {STRAT_LABELS[sk]} (P_static={p_static:.1f} W) ---")
        print(f"{'Seq':>8} | {'P_total':>10} | {'P_dynamic':>10} | {'E_total(nJ)':>14} | {'E_static(nJ)':>14} | {'E_dynamic(nJ)':>14} | {'Dyn%':>6}")
        print("-" * 90)
        for e in entries:
            seq = e["seq"]
            p_total = e["total_power_w"]
            p_dyn = p_total - p_static
            time_s = e["time_ns"] * 1e-9
            e_total = e["energy_per_token_nj"]
            e_static = p_static * time_s * 1e9
            e_dynamic = p_dyn * time_s * 1e9
            dyn_pct = e_dynamic / e_total * 100 if e_total > 0 else 0
            print(f"{seq_label(seq):>8} | {p_total:>10.2f} | {p_dyn:>10.2f} | {e_total:>14,.0f} | {e_static:>14,.0f} | {e_dynamic:>14,.0f} | {dyn_pct:>5.1f}%")


def plot_hbm_fraction_pie(summary, out_dir: Path, title_suffix: str,
                          target_seq: int = 65536):
    """Pie chart showing HBM power fraction per module at a given seq."""
    strats = _available_strats(summary)
    row = next((r for r in summary if r["seq"] == target_seq), summary[-1])
    actual_seq = row["seq"]
    n = len(strats)

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1:
        axes = [axes]

    colors = [HBM_MOD_COLORS["Proj_QKV"], HBM_MOD_COLORS["Attention"],
              HBM_MOD_COLORS["Proj_O"]]
    mod_labels = ["Proj_QKV", "Attention", "Proj_O"]

    for idx, sk in enumerate(strats):
        ax = axes[idx]
        bd = row.get(sk, {}).get("hbm_breakdown", {})
        fracs = [bd.get("Proj_QKV_frac", 0), bd.get("Attn_frac", 0),
                 bd.get("Proj_O_frac", 0)]
        total_hbm = row.get(sk, {}).get("hbm_power_w", 0)

        if sum(fracs) > 0:
            ax.pie(fracs, labels=mod_labels, autopct="%1.1f%%", colors=colors,
                   textprops={"fontsize": 10}, startangle=90)
        ax.set_title(f"{STRAT_LABELS[sk]}\n({total_hbm:.0f}W HBM)", fontsize=13,
                     fontweight="bold")

    fig.suptitle(f"HBM Power Module Fraction @ seq={seq_label(actual_seq)}{title_suffix}",
                 fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"power_hbm_fraction_seq{actual_seq}.png", dpi=180,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: power_hbm_fraction_seq{actual_seq}.png")


def main():
    parser = argparse.ArgumentParser(description="Visualize GQA power model results.")
    parser.add_argument("--data", type=str,
                        default="reports/power/gqa_hybrid_merged_80T_split4_bw1500_power.json",
                        help="Path to power JSON (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=str, default="",
                        help="Output directory for PNGs (default: same as data)")
    parser.add_argument("--target-seq", type=int, default=65536,
                        help="Target seq for single-seq breakdown (default: %(default)s)")
    args = parser.parse_args()

    data = load_data(args.data)
    summary = data.get("summary_table", [])
    strategies_data = data.get("strategies", {})

    if not summary:
        print("ERROR: no summary_table in data", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(args.data).resolve().parent

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.data).stem
    title_suffix = f" ({stem.replace('_power', '')})"

    print(f"Plotting power results from: {args.data}")
    print(f"Output: {out_dir}/")

    plot_total_power(summary, out_dir, title_suffix)
    plot_energy_per_token(summary, out_dir, title_suffix)
    plot_energy_per_token_trend(summary, out_dir, title_suffix)
    plot_power_breakdown_bars(summary, out_dir, title_suffix)
    plot_power_breakdown_single_seq(summary, strategies_data, out_dir,
                                     title_suffix, args.target_seq)
    plot_utilization(strategies_data, out_dir, title_suffix)
    plot_power_efficiency(summary, out_dir, title_suffix)
    plot_tdp_fraction(strategies_data, out_dir, title_suffix)
    plot_static_dynamic_energy(data, out_dir, title_suffix)
    plot_hbm_breakdown(summary, strategies_data, out_dir, title_suffix)
    plot_hbm_fraction_pie(summary, out_dir, title_suffix, args.target_seq)

    export_static_dynamic_table(data, out_dir)
    print(f"\nDone — 11 figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
