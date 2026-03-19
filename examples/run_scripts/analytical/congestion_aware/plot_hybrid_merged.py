#!/usr/bin/env python3
"""Plot GQA hybrid merged results: wall time, speedup vs Rubin, module breakdown
with communication sub-operation split."""
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

COMM_COLORS = {
    "qkv_allgather":  "#F9E79F",
    "attn_comm":      "#E74C3C",
    "attn_rs":        "#AF7AC5",
    "output_ar_comm": "#7B241C",
    "output_ag_comm": "#E67E22",
    "output_comm":    "#D35400",
    "final_reduce":   "#641E16",
}
COMM_LABELS = {
    "qkv_allgather":  "QKV AllGather",
    "attn_comm":      "Attn AR (score)",
    "attn_rs":        "Attn RS (score)",
    "output_ar_comm": "Output AR (tp_h)",
    "output_ag_comm": "Output AG (tp_s)",
    "output_comm":    "Output 2×AR",
    "final_reduce":   "Final Reduce",
}


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def seq_label(s: int) -> str:
    if s >= 1048576:
        return f"{s // 1048576}M"
    if s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def extract_comm_ops(strategies_data: dict, strat_key: str) -> dict:
    """Extract per-seq comm breakdown from the detailed strategy data."""
    strat_data = strategies_data.get(strat_key, {}).get("data", [])
    result = {}
    for entry in strat_data:
        seq = entry["seq"]
        ops = {}
        for op in entry.get("comm_breakdown", {}).get("ops", []):
            ops[op["name"]] = op["time_ns"]
        result[seq] = ops
    return result


def plot_wall_time(summary, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    for sk in STRAT_KEYS:
        seqs = [r["seq"] for r in summary if sk in r]
        walls = [r[sk]["wall_ns"] / 1e3 for r in summary if sk in r]
        ax.plot(seqs, walls, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2, markersize=6)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Wall Time (μs)", fontsize=12)
    ax.set_title("GQA Hybrid: Wall Time vs Seq (B=1, 80T, Mesh2D 4×4)", fontsize=13)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.legend(fontsize=11)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "hybrid_bs1_80T_wall_time.png", dpi=180)
    print(f"Saved: {out_dir / 'hybrid_bs1_80T_wall_time.png'}")
    plt.close(fig)


def plot_speedup(summary, out_dir: Path):
    fig, ax = plt.subplots(figsize=(11, 5.5))

    compare_keys = ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]

    ax.axhspan(0, 1.0, color="#FFCCCC", alpha=0.25)
    ax.axhspan(1.0, 5.0, color="#CCFFCC", alpha=0.25)
    ax.axhline(1.0, color="red", ls="--", lw=1.2, label="Rubin = 1.0×")

    for sk in compare_keys:
        seqs = [r["seq"] for r in summary if sk in r and "rubin" in r]
        speedups = [r["rubin"]["wall_ns"] / r[sk]["wall_ns"] for r in summary
                    if sk in r and "rubin" in r]
        ax.plot(seqs, speedups, marker=STRAT_MARKERS[sk], label=STRAT_LABELS[sk],
                color=STRAT_COLORS[sk], linewidth=2.2, markersize=7)
        for s, sp in zip(seqs, speedups):
            ax.annotate(f"{sp:.2f}×", (s, sp), textcoords="offset points",
                        xytext=(0, 10), fontsize=7.5, ha="center",
                        color=STRAT_COLORS[sk], fontweight="bold")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: seq_label(int(x))))
    ax.set_xlabel("Sequence Length", fontsize=12)
    ax.set_ylabel("Speedup vs Rubin (single GPU)", fontsize=12)
    ax.set_title("GQA Hybrid Estimation: Speedup over Rubin (B=1, 80T, Mesh2D 4×4)",
                 fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, which="both", ls="--", alpha=0.4)

    ymax = max(r["rubin"]["wall_ns"] / r[sk]["wall_ns"]
               for r in summary for sk in compare_keys
               if sk in r and "rubin" in r) * 1.25
    ax.set_ylim(0, max(ymax, 1.8))

    last = summary[-1]
    seq_last = seq_label(last["seq"])
    annot_offsets = {
        "HMP_reo": (15, 25), "hmp_reo_new": (60, 15),
        "hmp": (-60, 30), "tp16": (-60, -35),
    }
    for sk in compare_keys:
        if sk in last and "rubin" in last:
            sp = last["rubin"]["wall_ns"] / last[sk]["wall_ns"]
            label = STRAT_LABELS[sk]
            ox, oy = annot_offsets.get(sk, (40, 30))
            ax.annotate(f"{label} = {sp:.2f}× Rubin\n@ seq={seq_last}",
                        xy=(last["seq"], sp),
                        xytext=(ox, oy),
                        textcoords="offset points", fontsize=9,
                        ha="center",
                        arrowprops=dict(arrowstyle="->", color=STRAT_COLORS[sk]),
                        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                                  ec=STRAT_COLORS[sk], alpha=0.9))

    fig.tight_layout()
    fig.savefig(out_dir / "hybrid_bs1_80T_speedup_vs_rubin.png", dpi=180)
    print(f"Saved: {out_dir / 'hybrid_bs1_80T_speedup_vs_rubin.png'}")
    plt.close(fig)


def plot_module_breakdown(full_data, out_dir: Path):
    """Module breakdown with comm sub-operations split out."""
    strategies = full_data["strategies"]
    summary = full_data["summary_table"]
    seqs = [r["seq"] for r in summary]
    x = np.arange(len(seqs))
    bar_w = 0.72
    seq_labels = [seq_label(s) for s in seqs]

    compute_colors = {
        "Proj_QKV":  "#1B4F72",
        "Attention": "#3498DB",
        "Proj_O":    "#A9CCE3",
    }

    comm_ops_order = {
        "HMP_reo":     ["qkv_allgather", "final_reduce"],
        "hmp_reo_new": ["qkv_allgather", "attn_rs", "final_reduce"],
        "hmp":         ["qkv_allgather", "attn_comm", "output_ar_comm", "output_ag_comm"],
        "tp16":        ["attn_comm", "output_comm"],
        "rubin":       [],
    }

    fig, axes = plt.subplots(1, len(STRAT_KEYS), figsize=(30, 8.5), sharey=True)

    all_comm_ops_seen = set()
    global_ymax = 0

    all_tops = {}
    for idx, sk in enumerate(STRAT_KEYS):
        comm_by_seq = extract_comm_ops(strategies, sk)
        for i, r in enumerate(summary):
            top = 0
            if sk in r:
                d = r[sk]
                top = (d["Proj_QKV_ns"] + d["attn_ns"] + d["Proj_O_ns"]) / 1e3
            for s in seqs:
                ops = comm_by_seq.get(s, {})
                for op_name in comm_ops_order.get(sk, []):
                    pass
            seq = r["seq"]
            ops = comm_by_seq.get(seq, {})
            for op_name in comm_ops_order.get(sk, []):
                top += ops.get(op_name, 0) / 1e3
            if top > global_ymax:
                global_ymax = top

    y_limit = global_ymax * 1.08

    for idx, sk in enumerate(STRAT_KEYS):
        ax = axes[idx]
        comm_by_seq = extract_comm_ops(strategies, sk)

        proj_qkv = []
        attn_vals = []
        proj_o = []
        for r in summary:
            if sk in r:
                d = r[sk]
                proj_qkv.append(d["Proj_QKV_ns"] / 1e3)
                attn_vals.append(d["attn_ns"] / 1e3)
                proj_o.append(d["Proj_O_ns"] / 1e3)
            else:
                proj_qkv.append(0)
                attn_vals.append(0)
                proj_o.append(0)

        proj_qkv = np.array(proj_qkv)
        attn_arr = np.array(attn_vals)
        proj_o_arr = np.array(proj_o)

        ax.bar(x, proj_qkv, bar_w, color=compute_colors["Proj_QKV"])
        ax.bar(x, attn_arr, bar_w, bottom=proj_qkv, color=compute_colors["Attention"])
        ax.bar(x, proj_o_arr, bar_w, bottom=proj_qkv + attn_arr,
               color=compute_colors["Proj_O"])

        bottom = proj_qkv + attn_arr + proj_o_arr
        for op_name in comm_ops_order.get(sk, []):
            op_vals = []
            for s in seqs:
                ops = comm_by_seq.get(s, {})
                op_vals.append(ops.get(op_name, 0) / 1e3)
            op_arr = np.array(op_vals)
            color = COMM_COLORS.get(op_name, "#999999")
            ax.bar(x, op_arr, bar_w, bottom=bottom, color=color)
            bottom = bottom + op_arr
            all_comm_ops_seen.add(op_name)

        ax.set_ylim(0, y_limit)
        ax.set_xticks(x)
        ax.set_xticklabels(seq_labels, rotation=45, fontsize=18)
        ax.set_xlabel("Seq", fontsize=22)
        ax.set_title(STRAT_LABELS[sk], fontsize=24, fontweight="bold")
        ax.grid(axis="y", ls="--", alpha=0.3)
        ax.tick_params(axis="y", labelsize=18)

    axes[0].set_ylabel("Latency (μs)", fontsize=22)

    compute_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=compute_colors["Proj_QKV"]),
        plt.Rectangle((0, 0), 1, 1, fc=compute_colors["Attention"]),
        plt.Rectangle((0, 0), 1, 1, fc=compute_colors["Proj_O"]),
    ]
    compute_labels = ["Proj_QKV (weight)", "Attention (KV cache)", "Proj_O (weight)"]

    comm_order_all = ["qkv_allgather", "attn_comm", "attn_rs", "output_ar_comm",
                      "output_ag_comm", "output_comm", "final_reduce"]
    comm_handles = []
    comm_label_list = []
    for op in comm_order_all:
        if op in all_comm_ops_seen:
            comm_handles.append(plt.Rectangle((0, 0), 1, 1, fc=COMM_COLORS[op]))
            comm_label_list.append(COMM_LABELS[op])

    all_handles = compute_handles + comm_handles
    all_labels = compute_labels + comm_label_list

    fig.legend(all_handles, all_labels,
               loc="upper center", ncol=min(len(all_labels), 5), fontsize=16,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("GQA Hybrid: Module Breakdown with Comm Split (B=1, 80T)",
                 fontsize=26, fontweight="bold", y=1.14)
    fig.tight_layout()
    fig.savefig(out_dir / "hybrid_bs1_80T_module_breakdown.png", dpi=180,
                bbox_inches="tight")
    print(f"Saved: {out_dir / 'hybrid_bs1_80T_module_breakdown.png'}")
    plt.close(fig)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="reports/hybrid/gqa_hybrid_merged_80T.json")
    parser.add_argument("--out-dir", default="reports/plots")
    args = parser.parse_args()

    data = load_data(args.data)
    summary = data["summary_table"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_wall_time(summary, out_dir)
    plot_speedup(summary, out_dir)
    plot_module_breakdown(data, out_dir)


if __name__ == "__main__":
    main()
