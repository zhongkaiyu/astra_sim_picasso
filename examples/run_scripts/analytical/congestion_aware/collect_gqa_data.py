#!/usr/bin/env python3
"""
Collect per-module latency data from GQA seq-scaling simulation logs.

Parses logs for Baseline (block2x2 Mesh2D), HMP (block2x2 Mesh2D), and
TP16 (Torus) strategies across multiple sequence lengths.  Outputs a JSON
file suitable for downstream plotting.

Usage:
  python3 collect_gqa_data.py                       # defaults
  python3 collect_gqa_data.py -o reports/custom.json # custom output
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from analyze_comm import parse_log

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[3]

SEQS = [1024, 2048, 4096, 8192, 16384, 32768, 65536,
        131072, 262144, 524288, 1048576]

def _make_strategies(batch: int, suffix: str = "", model_tag: str = "qwen",
                     rank_remap: str = "") -> dict:
    """Return strategy dict with log/config patterns matching the given batch/model.

    suffix: appended to output dir and log file names (e.g. '_split4').
    model_tag: "qwen" for qwen3, "llama4" for llama4, etc.
    rank_remap: rank remap tag (e.g. 'block2x2'), applied to all strategies.
    """
    sfx = suffix
    device_tag = "" if batch == 64 and not sfx and model_tag == "qwen" else "_80t"
    sfx_label = f", {sfx.strip('_')}" if sfx else ""
    remap_tag = f"_{rank_remap}" if rank_remap else ""

    # For legacy qwen3 bs=64 paths (no bs in dir name, tp16 without remap)
    if batch == 64 and not sfx and model_tag == "qwen":
        return {
            "hmp": {
                "label": "HMP (block2x2 Mesh2D)",
                "log_pattern": "output_qwen/gqa_seq_scaling/baseline_block2x2_fwd_sl{seq}/logs/simulation_log_gqa_baseline_block2x2_fwd_sl{seq}.txt",
                "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_baseline_fwd_block2x2_bs64_sl{{seq}}.json",
                "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/baseline/qwen_gqa_baseline_fwd_tph4_tps4_sl{{seq}}_block2x2.json",
            },
            "HMP_reo": {
                "label": "HMP_reo (block2x2 Mesh2D)",
                "log_pattern": "output_qwen/gqa_seq_scaling/hmp_block2x2_fwd_sl{seq}/logs/simulation_log_gqa_hmp_block2x2_fwd_sl{seq}.txt",
                "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_hmp_fwd_block2x2_bs64_sl{{seq}}.json",
                "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/hmp/qwen_gqa_hmp_fwd_tph4_tps4_sl{{seq}}_block2x2.json",
            },
            "tp16": {
                "label": "TP16 (Torus OneRing)",
                "log_pattern": "output_qwen/gqa_seq_scaling/tp16_fwd_sl{seq}/logs/simulation_log_gqa_tp16_fwd_sl{seq}.txt",
                "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4_ring_mesh2d_4x4_onering_qwen_gqa_tp16_fwd_bs64_sl{{seq}}.json",
                "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/tp16/qwen_gqa_tp16_fwd_tph4_tps4_sl{{seq}}.json",
            },
        }

    return {
        "hmp": {
            "label": f"HMP (block2x2 Mesh2D, bs={batch}{sfx_label})",
            "log_pattern": f"output_{model_tag}/gqa_seq_scaling/baseline{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}/logs/simulation_log_gqa_baseline{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}.txt",
            "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4{device_tag}_mesh2d_4x4_onering_{model_tag}_gqa_baseline_fwd{remap_tag}_bs{batch}_sl{{seq}}{sfx}.json",
            "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/baseline/{model_tag}_gqa_baseline_fwd_tph4_tps4_bs{batch}_sl{{seq}}{remap_tag}.json",
        },
        "HMP_reo": {
            "label": f"HMP_reo (block2x2 Mesh2D, bs={batch}{sfx_label})",
            "log_pattern": f"output_{model_tag}/gqa_seq_scaling/hmp{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}/logs/simulation_log_gqa_hmp{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}.txt",
            "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4{device_tag}_mesh2d_4x4_onering_{model_tag}_gqa_hmp_fwd{remap_tag}_bs{batch}_sl{{seq}}{sfx}.json",
            "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/hmp/{model_tag}_gqa_hmp_fwd_tph4_tps4_bs{batch}_sl{{seq}}{remap_tag}.json",
        },
        "hmp_reo_new": {
            "label": f"HMP_reo_new (block2x2 Mesh2D, bs={batch}{sfx_label}, RS+TreeReduce)",
            "log_pattern": f"output_{model_tag}/gqa_seq_scaling/baseline{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}/logs/simulation_log_gqa_baseline{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}.txt",
            "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4{device_tag}_mesh2d_4x4_onering_{model_tag}_gqa_baseline_fwd{remap_tag}_bs{batch}_sl{{seq}}{sfx}.json",
            "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/baseline/{model_tag}_gqa_baseline_fwd_tph4_tps4_bs{batch}_sl{{seq}}{remap_tag}.json",
        },
        "tp16": {
            "label": f"TP16 (Mesh2D, bs={batch}{sfx_label})",
            "log_pattern": f"output_{model_tag}/gqa_seq_scaling/tp16{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}/logs/simulation_log_gqa_tp16{remap_tag}_fwd_bs{batch}_sl{{seq}}{sfx}.txt",
            "sim_config_pattern": f"examples/run_scripts/analytical/congestion_aware/configs/gqa_seq_scaling/tp16_hbm4{device_tag}_mesh2d_4x4_onering_{model_tag}_gqa_tp16_fwd{remap_tag}_bs{batch}_sl{{seq}}{sfx}.json",
            "trace_config_pattern": f"examples/run_scripts/analytical/congestion_aware/trace_configs/gqa/tp16/{model_tag}_gqa_tp16_fwd_tph4_tps4_bs{batch}_sl{{seq}}{remap_tag}.json",
        },
    }


def collect_module_times(summary) -> dict[str, float]:
    """Aggregate Layer-0 compute and comm events by module (in ns).

    output_comm is further split into output_ar_comm (AllReduce on g17)
    and output_ag_comm (AllGather on g21) for finer hybrid correction.
    Comm message sizes are recorded as {module}_bytes keys.
    Comm group info is recorded as {module}_group_id keys.
    """
    modules: dict[str, float] = defaultdict(float)
    for e in summary.comp_events:
        if e.layer_name == "transformer.0":
            modules[e.module] += e.runtime_ns
    for e in summary.events:
        if e.layer_name == "transformer.0":
            modules[e.module] += e.latency_ns
            modules[e.module + "_bytes"] += e.size_bytes
            if not modules.get(e.module + "_group_id"):
                modules[e.module + "_group_id"] = e.group_id
            if e.module == "output_comm":
                ct = e.comm_type.upper().replace("_", "")
                if ct == "ALLREDUCE":
                    modules["output_ar_comm"] += e.latency_ns
                    modules["output_ar_comm_bytes"] += e.size_bytes
                    modules["output_ar_comm_group_id"] = e.group_id
                elif ct == "ALLGATHER":
                    modules["output_ag_comm"] += e.latency_ns
                    modules["output_ag_comm_bytes"] += e.size_bytes
                    modules["output_ag_comm_group_id"] = e.group_id
    return dict(modules)


def collect_all(seqs: list[int], batch: int = 64, suffix: str = "",
                model_tag: str = "qwen", rank_remap: str = "") -> dict:
    strategies = _make_strategies(batch, suffix=suffix, model_tag=model_tag,
                                  rank_remap=rank_remap)
    result = {}
    for key, cfg in strategies.items():
        entries = []
        for seq in seqs:
            log_path = PROJECT_DIR / cfg["log_pattern"].format(seq=seq)
            if not log_path.is_file():
                print(f"[WARN] Missing log: {log_path}", file=sys.stderr)
                continue
            summary = parse_log(str(log_path), npu=0)
            modules = collect_module_times(summary)
            entries.append({
                "seq": seq,
                "wall_ns": summary.wall_time,
                "gpu_ns": summary.gpu_time,
                "comm_ns": summary.comm_time,
                "modules": modules,
            })
        result[key] = {
            "label": cfg["label"],
            "log_pattern": cfg["log_pattern"],
            "sim_config_pattern": cfg["sim_config_pattern"],
            "trace_config_pattern": cfg["trace_config_pattern"],
            "data": entries,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect GQA seq-scaling data")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--seq", type=int, nargs="+", default=SEQS)
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size used in simulations (default: 64)")
    parser.add_argument("--suffix", type=str, default="",
                        help="Suffix appended to output dir/log names (e.g. '_split4')")
    parser.add_argument("--model", type=str, default="qwen3",
                        choices=["qwen3", "llama4"],
                        help="Model name for path resolution (default: qwen3)")
    parser.add_argument("--rank-remap", type=str, default="",
                        choices=["", "block2x2"],
                        help="Rank remap tag used in simulation (e.g. block2x2)")
    args = parser.parse_args()

    model_tag = "qwen" if args.model == "qwen3" else args.model

    # model key -> reports subfolder
    _report_dirs = {"qwen3": "qwen3-235B", "llama4": "llama4"}
    report_dir = _report_dirs.get(args.model, args.model)

    sfx_tag = args.suffix.strip("_")
    if not args.output:
        sfx_part = f"_{sfx_tag}" if sfx_tag else ""
        args.output = str(SCRIPT_DIR / "reports" / report_dir / "astrasim" / f"gqa_seq_scaling_bs{args.batch}{sfx_part}_data.json")

    data = collect_all(args.seq, batch=args.batch, suffix=args.suffix,
                       model_tag=model_tag, rank_remap=args.rank_remap)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Data saved to {out_path}")

    for key in data:
        n = len(data[key]["data"])
        print(f"  {key}: {n} seq points")

    return 0


if __name__ == "__main__":
    sys.exit(main())
