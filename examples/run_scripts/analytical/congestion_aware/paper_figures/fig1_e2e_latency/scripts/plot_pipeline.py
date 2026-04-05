#!/usr/bin/env python3
"""
Fig1 数据收集流程图: 从模型配置到最终绘图的完整 pipeline。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "plots"


def rounded_box(ax, xy, w, h, text, fc, ec="#333333", fontsize=8,
                fontweight="normal", text_color="black", lw=1.2, radius=0.02):
    """Draw a rounded rectangle with centered text."""
    box = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2)
    ax.add_patch(box)
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, zorder=3)


def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.2, style="-|>"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Colors ──
    C_SCRIPT = "#E8F0FE"     # light blue — scripts
    C_CONFIG = "#FFF3E0"     # light orange — config files
    C_DATA   = "#E8F5E9"     # light green — intermediate data
    C_OUTPUT = "#FCE4EC"     # light pink — final output
    C_HW     = "#F3E5F5"     # light purple — hardware/model configs
    C_TITLE  = "#1A237E"     # dark blue — section titles

    BW = 2.6   # box width
    BH = 0.55  # box height

    # ════════════════════════════════════════════════════════
    # Title
    # ════════════════════════════════════════════════════════
    ax.text(7, 8.7, "Fig1 Data Collection Pipeline", ha="center", va="center",
            fontsize=16, fontweight="bold", color=C_TITLE)

    # ════════════════════════════════════════════════════════
    # Row 0 (y~8.0): Hardware & Model Configs (inputs)
    # ════════════════════════════════════════════════════════
    y0 = 7.85
    ax.text(7, y0 + 0.5, "Model & Hardware Configs", ha="center",
            fontsize=10, fontweight="bold", color="#555")

    configs = [
        (0.4,  "attention.py\nMODEL_CONFIGS\n(d, Hq, Hkv, dk)"),
        (3.3,  "hardware_config.py\nH100 / Rubin / NPU\n(FLOPS, BW, D2D)"),
        (6.2,  "utilization_96.json\nGPU SM Utilization\n(per module, per batch)"),
        (9.1,  "h100_rubin_util.json\nH100 BW Utilization\n(proj_qkv, attn, proj_o)"),
        (11.5, "h100_tp2_profile.json\nTP2 BW Utilization\n(proj_qkv, attn, proj_o)"),
    ]
    for xc, txt in configs:
        rounded_box(ax, (xc, y0), BW, BH * 1.5, txt, C_HW, fontsize=6.5)

    # ════════════════════════════════════════════════════════
    # Row 1 (y~6.5): Step 1 & 2 — Roofline + AstraSim (parallel)
    # ════════════════════════════════════════════════════════
    y1 = 6.3
    ax.text(7, y1 + 0.65, "Step 1 & 2: Analytical + Simulation (parallel)", ha="center",
            fontsize=10, fontweight="bold", color="#555")

    # Step 1: Roofline
    rx = 1.2
    rounded_box(ax, (rx, y1), BW, BH, "roofline_gqa_calc.py", C_SCRIPT, fontsize=8, fontweight="bold")

    # Step 2: AstraSim
    sx = 5.4
    rounded_box(ax, (sx, y1), BW, BH, "collect_gqa_data.py", C_SCRIPT, fontsize=8, fontweight="bold")
    # AstraSim simulation box
    rounded_box(ax, (sx + 0.1, y1 - 0.7), BW - 0.2, BH * 0.9,
                "AstraSim Simulator\n(mesh2D congestion model)", C_CONFIG, fontsize=6.5)

    # Roofline output
    ro_x, ro_y = 0.6, 5.2
    rounded_box(ax, (ro_x, ro_y), BW + 0.3, BH,
                "roofline/gqa_roofline_bs{B}_96T.json\n(6 strategies x 11 seq lengths)", C_DATA, fontsize=6)

    # AstraSim output
    ao_x, ao_y = 5.0, 5.2
    rounded_box(ax, (ao_x, ao_y), BW + 0.3, BH,
                "astrasim/gqa_seq_scaling_bs{B}_data.json\n(4 strategies, comm breakdown)", C_DATA, fontsize=6)

    # Arrows: configs → scripts
    for xc, _ in configs[:2]:
        arrow(ax, xc + BW/2, y0, rx + BW/2, y1 + BH)
    arrow(ax, configs[0][0] + BW/2, y0, sx + BW/2, y1 + BH)

    # Arrows: scripts → outputs
    arrow(ax, rx + BW/2, y1, ro_x + (BW+0.3)/2, ro_y + BH)
    arrow(ax, sx + BW/2, y1 - 0.7, ao_x + (BW+0.3)/2, ao_y + BH)
    arrow(ax, sx + BW/2, y1, sx + BW/2, y1 - 0.7 + BH * 0.9)

    # ════════════════════════════════════════════════════════
    # Row 2 (y~4.0): Step 3 — Merge
    # ════════════════════════════════════════════════════════
    y2 = 3.8
    ax.text(7, y2 + 0.7, "Step 3: Hybrid Merge", ha="center",
            fontsize=10, fontweight="bold", color="#555")

    mx = 3.5
    rounded_box(ax, (mx, y2), BW + 1.0, BH,
                "merge_gqa_results.py\nmax(AstraSim compute, Roofline mem) + Utilization",
                C_SCRIPT, fontsize=7.5, fontweight="bold")

    # Arrows: roofline & astrasim → merge
    arrow(ax, ro_x + (BW+0.3)/2, ro_y, mx + (BW+1.0)/4, y2 + BH)
    arrow(ax, ao_x + (BW+0.3)/2, ro_y, mx + 3*(BW+1.0)/4, y2 + BH)

    # Arrows: utilization configs → merge
    arrow(ax, configs[2][0] + BW/2, y0, mx + (BW+1.0)/3, y2 + BH, color="#999")
    arrow(ax, configs[3][0] + BW/2, y0, mx + 2*(BW+1.0)/3, y2 + BH, color="#999")

    # Merge output
    mo_x, mo_y = 2.5, 2.8
    rounded_box(ax, (mo_x, mo_y), BW + 2.0, BH,
                "hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{B}.json\n"
                "(HMP_reo, hmp_reo_new, hmp, tp16, rubin, h100)",
                C_DATA, fontsize=6.5)
    arrow(ax, mx + (BW+1.0)/2, y2, mo_x + (BW+2.0)/2, mo_y + BH)

    # ════════════════════════════════════════════════════════
    # Row 3 (y~2.0): Step 4 — Add TP2
    # ════════════════════════════════════════════════════════
    y3 = 1.8
    ax.text(9.0, y3 + 0.65, "Step 4: Add TP2", ha="center",
            fontsize=10, fontweight="bold", color="#555")

    tx = 8.0
    rounded_box(ax, (tx, y3), BW, BH,
                "add_rubin_tp2.py", C_SCRIPT, fontsize=8, fontweight="bold")

    # Arrow: merge output → add_tp2
    arrow(ax, mo_x + (BW+2.0) - 0.2, mo_y, tx + BW/2, y3 + BH)
    # Arrow: tp2 profile → add_tp2
    arrow(ax, configs[4][0] + BW/2, y0, tx + BW/2, y3 + BH, color="#999")

    # TP2 output
    to_x, to_y = 7.2, 0.9
    rounded_box(ax, (to_x, to_y), BW + 1.2, BH,
                "hybrid merged + rubin_tp2 + h100_tp2\n(all 5 strategies complete)",
                C_DATA, fontsize=6.5)
    arrow(ax, tx + BW/2, y3, to_x + (BW+1.2)/2, to_y + BH)

    # ════════════════════════════════════════════════════════
    # Row 4 (y~0.0): Step 5 — Plot
    # ════════════════════════════════════════════════════════
    px = 1.0
    py = 0.15
    rounded_box(ax, (px, py), BW, BH,
                "plot_e2e.py", C_SCRIPT, fontsize=8, fontweight="bold")

    # Final output
    fx = 4.0
    rounded_box(ax, (fx, py), BW + 0.4, BH,
                "fig1_speedup_4panel.pdf\n2x4 grid (2 models x 4 BS)", C_OUTPUT,
                fontsize=7, fontweight="bold", text_color="#B71C1C")

    # Arrow: merge output → plot
    arrow(ax, mo_x + (BW+2.0)/4, mo_y, px + BW/2, py + BH)
    # Arrow: tp2 output → plot
    arrow(ax, to_x, to_y, px + BW, py + BH)
    # Arrow: plot → output
    arrow(ax, px + BW, py + BH/2, fx, py + BH/2)

    # ════════════════════════════════════════════════════════
    # Legend
    # ════════════════════════════════════════════════════════
    leg_y = 0.15
    leg_x = 10.5
    legend_items = [
        (C_HW,     "Config / Profile"),
        (C_SCRIPT, "Python Script"),
        (C_DATA,   "Intermediate Data"),
        (C_OUTPUT, "Final Output"),
    ]
    for i, (c, label) in enumerate(legend_items):
        yi = leg_y + (len(legend_items) - 1 - i) * 0.42
        rounded_box(ax, (leg_x, yi), 0.35, 0.3, "", c, lw=0.8)
        ax.text(leg_x + 0.5, yi + 0.15, label, fontsize=7, va="center")

    # ════════════════════════════════════════════════════════
    # Batch size annotation
    # ════════════════════════════════════════════════════════
    ax.text(0.4, 4.55, "x4 runs per model\n(BS = 1, 4, 16, 32)",
            fontsize=7, fontstyle="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="#CCC", lw=0.8))

    # Model annotation
    ax.text(9.5, 4.55, "x2 models\nQwen3-235B\nLlama4-Maverick",
            fontsize=7, fontstyle="italic", color="#666",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFFDE7", ec="#CCC", lw=0.8))

    fig.tight_layout()
    fname = OUT / "fig1_pipeline.pdf"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    fig.savefig(fname.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"Saved: {fname}")
    print(f"Saved: {fname.with_suffix('.png')}")
    plt.close(fig)


if __name__ == "__main__":
    main()
