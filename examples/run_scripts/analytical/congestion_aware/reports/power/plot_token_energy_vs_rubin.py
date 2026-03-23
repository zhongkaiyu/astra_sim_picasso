#!/usr/bin/python3
"""Visualize Token/Energy efficiency: Ours strategies vs Rubin baseline.

Two-panel figure:
  Top:    Absolute energy per token (μJ) for all strategies
  Bottom: Energy saving % vs Rubin (positive = Ours saves energy)

Key sequence points (1K, 8K, 64K, 256K, 1M) are annotated with savings.

Usage:
  python3 plot_token_energy_vs_rubin.py [--data <power_json>] [-o <output_png>]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# --- Style constants (consistent with plot_power.py) ---
OURS_STRATS = ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]
ALL_STRATS = OURS_STRATS + ["rubin"]

STRAT_LABELS = {
    "HMP_reo": "HMP_RO",
    "hmp_reo_new": "RO_new",
    "hmp": "HMP",
    "tp16": "TP16",
    "rubin": "Rubin",
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
STRAT_LINESTYLES = {
    "HMP_reo": "-", "hmp_reo_new": "-", "hmp": "--", "tp16": "-.", "rubin": "-",
}

KEY_SEQS = [1024, 8192, 65536, 262144, 1048576]


def seq_label(s):
    if s >= 1048576:
        return f"{s // 1048576}M"
    if s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def load_power_json(path):
    with open(path) as f:
        return json.load(f)


def extract_summary(data):
    """Return {strat: [(seq, energy_nj, power_w, time_ns), ...]}."""
    out = {}
    for strat in ALL_STRATS:
        sd = data.get("strategies", {}).get(strat, {}).get("data", [])
        entries = []
        for e in sd:
            entries.append((
                e["seq"],
                e["energy_per_token_nj"],
                e["total_power_w"],
                e["time_ns"],
            ))
        entries.sort(key=lambda x: x[0])
        out[strat] = entries
    return out


def _compute_savings(summary, rubin_map):
    """Return {strat: [(seq, saving_pct), ...]} for all OURS strategies."""
    out = {}
    for sk in OURS_STRATS:
        savings = []
        for s, e_nj, _, _ in summary[sk]:
            r_nj = rubin_map.get(s)
            if r_nj and r_nj > 0:
                savings.append((s, (1 - e_nj / r_nj) * 100))
            else:
                savings.append((s, np.nan))
        out[sk] = savings
    return out


def _draw_saving_panel(ax, strats_to_draw, savings_map, summary,
                        annotate_key=True, show_shade_text=True):
    """Draw energy saving curves + Rubin baseline on an axis."""
    rubin_seqs = [s for s, _, _, _ in summary["rubin"]]
    ax.axhline(y=0, color=STRAT_COLORS["rubin"], linestyle="-",
               linewidth=2.5, alpha=0.7, label=f"{STRAT_LABELS['rubin']} (baseline)")
    # Shade
    ylim = ax.get_ylim()
    ax.axhspan(0, 1e4, color="#2ecc71", alpha=0.05, zorder=0)
    ax.axhspan(-1e4, 0, color="#e74c3c", alpha=0.05, zorder=0)
    if show_shade_text:
        ax.text(rubin_seqs[0], 0.8, " ↑ Ours saves energy", fontsize=7.5,
                color="#27ae60", alpha=0.7, va="bottom")
        ax.text(rubin_seqs[0], -0.8, " ↓ Ours wastes energy", fontsize=7.5,
                color="#c0392b", alpha=0.7, va="top")

    for si, sk in enumerate(strats_to_draw):
        seqs = [s for s, _ in savings_map[sk]]
        vals = [v for _, v in savings_map[sk]]
        ax.plot(seqs, vals,
                marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2.5, markersize=7,
                linestyle=STRAT_LINESTYLES[sk], zorder=5)

        if annotate_key:
            n_strats = len(strats_to_draw)
            for s, sv in zip(seqs, vals):
                if s in KEY_SEQS and not np.isnan(sv):
                    base_y = 14 if sv >= 0 else -16
                    stagger = (si - n_strats // 2) * 15
                    offset_y = base_y + stagger
                    bbox_color = "#d5f5e3" if sv >= 0 else "#fadbd8"
                    sign = "+" if sv >= 0 else ""
                    ax.annotate(
                        f"{STRAT_LABELS[sk]} {sign}{sv:.1f}%",
                        xy=(s, sv), xycoords="data",
                        xytext=(0, offset_y), textcoords="offset points",
                        ha="center",
                        va="bottom" if offset_y > 0 else "top",
                        fontsize=8, fontweight="bold",
                        color=STRAT_COLORS[sk],
                        bbox=dict(boxstyle="round,pad=0.15", fc=bbox_color,
                                  ec=STRAT_COLORS[sk], alpha=0.85, lw=0.8),
                        arrowprops=dict(arrowstyle="-",
                                       color=STRAT_COLORS[sk],
                                       lw=0.6, alpha=0.5),
                    )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)


def make_figure(summary, out_path, title_suffix=""):
    """Three-panel figure:
      1) Absolute energy per token (all strategies)
      2) Energy saving % for RO_new / HMP / TP16 (fine scale)
      3) Energy saving % for HMP_RO (large scale, separate)
    """
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(14, 13),
        gridspec_kw={"height_ratios": [1, 1.2, 0.8]},
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.15)

    rubin_map = {s: e for s, e, _, _ in summary.get("rubin", [])}
    savings_map = _compute_savings(summary, rubin_map)

    # ================================================================
    # Panel 1: Absolute energy per token (μJ)
    # ================================================================
    for sk in ALL_STRATS:
        seqs = [s for s, _, _, _ in summary[sk]]
        energy_uj = [e / 1e3 for _, e, _, _ in summary[sk]]
        ax1.plot(seqs, energy_uj,
                 marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                 color=STRAT_COLORS[sk], linewidth=2, markersize=6,
                 linestyle=STRAT_LINESTYLES[sk])

    ax1.set_ylabel("Energy per Token (μJ)", fontsize=12)
    ax1.set_title(f"Token Energy: Ours vs Rubin{title_suffix}", fontsize=14,
                  fontweight="bold")
    ax1.set_yscale("log")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x >= 1 else f"{x:.1f}"))

    # ================================================================
    # Panel 2: RO_new / HMP / TP16  (fine-grained saving %)
    # ================================================================
    fine_strats = ["hmp_reo_new", "hmp", "tp16"]
    _draw_saving_panel(ax2, fine_strats, savings_map, summary,
                       annotate_key=True, show_shade_text=True)
    ax2.set_ylabel("Energy Saving vs Rubin (%)", fontsize=12)
    ax2.set_title("Energy Saving: RO_new / HMP / TP16", fontsize=12)

    # Set Y range based on fine strats only
    fine_vals = [v for sk in fine_strats for _, v in savings_map[sk]
                 if not np.isnan(v)]
    if fine_vals:
        ymax = max(abs(min(fine_vals)), abs(max(fine_vals))) * 1.6
        ax2.set_ylim(-ymax, ymax)

    # TDP annotation
    tdp_text = ("TDP: Ours 1440W (65%) vs Rubin 2200W\n"
                "Ours: 16×(75W HBM+15W Cmpt)\n"
                "Rubin: 8×75W HBM + 2×800W Cmpt")
    ax2.text(0.02, 0.02, tdp_text, transform=ax2.transAxes,
             fontsize=8, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", fc="white",
                       ec="gray", alpha=0.9))

    # ================================================================
    # Panel 3: HMP_RO  (large scale — duplicate Wo penalty)
    # ================================================================
    _draw_saving_panel(ax3, ["HMP_reo"], savings_map, summary,
                       annotate_key=True, show_shade_text=False)
    ax3.set_ylabel("Energy Saving (%)", fontsize=12)
    ax3.set_xlabel("Sequence Length", fontsize=12)
    ax3.set_title("Energy Saving: HMP_RO (duplicate Wo — high HBM traffic)",
                  fontsize=12)

    hmpro_vals = [v for _, v in savings_map["HMP_reo"] if not np.isnan(v)]
    if hmpro_vals:
        ymin = min(hmpro_vals) * 1.3
        ax3.set_ylim(ymin, max(hmpro_vals) + 10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "gqa_hybrid_merged_80T_split4_bw1500_power.json"),
                        help="Power JSON path")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output PNG path (default: auto)")
    args = parser.parse_args()

    data = load_power_json(args.data)
    summary = extract_summary(data)

    if args.output:
        out_path = Path(args.output)
    else:
        stem = Path(args.data).stem.replace("_power", "")
        out_path = Path(args.data).resolve().parent / f"{stem}_token_energy_vs_rubin.png"

    # Derive title suffix from filename
    name = Path(args.data).stem
    parts = []
    if "80T" in name:
        parts.append("80T")
    if "split4" in name:
        parts.append("split4")
    if "bw1500" in name:
        parts.append("BW1.5T")
    title_suffix = f" ({', '.join(parts)})" if parts else ""

    make_figure(summary, out_path, title_suffix)


if __name__ == "__main__":
    main()
