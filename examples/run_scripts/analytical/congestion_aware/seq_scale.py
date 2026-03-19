#!/usr/bin/env python3
"""
Batch generate trace_configs and simulation configs for GQA sequence scaling,
then optionally execute via bulk_run_configs.py.

Examples:
  # Generate configs only (dry run)
  python3 seq_scale.py --seq 1024 2048 4096 65536 --dry-run

  # Generate and run all (HMP + Baseline + TP16)
  python3 seq_scale.py --seq 1024 2048 4096 65536

  # Only HMP strategy
  python3 seq_scale.py --seq 1024 2048 --strategies hmp

  # Only TP16 strategy
  python3 seq_scale.py --seq 1024 2048 4096 65536 --strategies tp16

  # block2x2 rank remap for HMP + Baseline
  python3 seq_scale.py --seq 1024 2048 4096 65536 --strategies hmp baseline --rank-remap block2x2

  # Custom batch and device
  python3 seq_scale.py --seq 1024 2048 --batch 128 --device H100 --topology Mesh2D_16gpus_4x4_H100
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[4]

STRATEGY_DEFAULTS = {
    "hmp": {
        "model_type": "qwen_gqa_hmp_fwd",
        "trace_prefix": "qwen_gqa_hmp_fwd",
        "dp": 1, "tp": 1, "tp_h": 4, "tp_s": 4,
    },
    "baseline": {
        "model_type": "qwen_gqa_baseline_fwd",
        "trace_prefix": "qwen_gqa_baseline_fwd",
        "dp": 1, "tp": 1, "tp_h": 4, "tp_s": 4,
    },
    "tp16": {
        "model_type": "qwen_gqa_tp16_fwd",
        "trace_prefix": "qwen_gqa_tp16_fwd",
        "dp": 1, "tp": 1, "tp_h": 4, "tp_s": 4,
    },
}

MODEL_DEFAULTS = {
    "kvhead": 4,
    "head": 64,
    "num_stacks": 1,
    "dmodel": 4096,
    "head_dim": 128,
    "tpsp": True,
}

DEVICE_CONFIGS = {
    "HBM4": {
        "system": "onering16_HBM4.json",
        "network": "Mesh2D_16gpus_4x4_HBM4.yml",
    },
    "HBM4_60T": {
        "system": "onering16_HBM4_60T.json",
        "network": "Mesh2D_16gpus_4x4_HBM4.yml",
    },
    "HBM4_80T": {
        "system": "onering16_HBM4_80T.json",
        "network": "Mesh2D_16gpus_4x4_HBM4.yml",
    },
    "HBM4_Ring": {
        "system": "onering16_HBM4.json",
        "network": "Torus2D_16gpus_4x4_HBM4.yml",
    },
    "H100": {
        "system": "onering16_H100.json",
        "network": "Mesh2D_16gpus_4x4_H100.yml",
    },
}


def make_trace_config(strategy, seq, batch, tp_h, tp_s, rank_remap=""):
    sdef = STRATEGY_DEFAULTS[strategy]
    remap_tag = f"_{rank_remap}" if rank_remap else ""
    suffix = f"tph{tp_h}_tps{tp_s}_bs{batch}_sl{seq}{remap_tag}"
    tc = {
        "output_dir": f"{{PROJECT_DIR}}/symbolic_tensor_graph_picasso/et_trace/gqa/{strategy}_fwd_{suffix}",
        "output_name": f"{strategy}_fwd_{suffix}",
        "model_type": sdef["model_type"],
        "dp": sdef["dp"],
        "tp": sdef["tp"],
        "tp_h": tp_h,
        "tp_s": tp_s,
        "kvhead": MODEL_DEFAULTS["kvhead"],
        "head": MODEL_DEFAULTS["head"],
        "num_stacks": MODEL_DEFAULTS["num_stacks"],
        "dmodel": MODEL_DEFAULTS["dmodel"],
        "batch": batch,
        "seq": seq,
        "tpsp": MODEL_DEFAULTS["tpsp"],
        "head_dim": MODEL_DEFAULTS["head_dim"],
    }
    if rank_remap:
        tc["args"] = f"--rank_remap {rank_remap}"
    return tc


def make_sim_config(strategy, seq, batch, tp_h, tp_s, device, rank_remap=""):
    dcfg = DEVICE_CONFIGS[device]
    remap_tag = f"_{rank_remap}" if rank_remap else ""
    suffix = f"tph{tp_h}_tps{tp_s}_bs{batch}_sl{seq}{remap_tag}"
    strat_label = f"{strategy}{remap_tag}"
    return {
        "output_dir": f"{{PROJECT_DIR}}/output_qwen/gqa_seq_scaling/{strat_label}_fwd_bs{batch}_sl{seq}",
        "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
        "system": f"{{EXAMPLE_DIR}}/system/native_collectives/{dcfg['system']}",
        "network": f"{{EXAMPLE_DIR}}/network/analytical/{dcfg['network']}",
        "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
        "workload_dir": f"{{PROJECT_DIR}}/symbolic_tensor_graph_picasso/et_trace/gqa/{strategy}_fwd_{suffix}",
        "workload_base": f"{strategy}_fwd_{suffix}",
        "log_file": f"simulation_log_gqa_{strat_label}_fwd_bs{batch}_sl{seq}.txt",
    }


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate and run GQA seq-scaling configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seq", type=int, nargs="+", required=True,
                        help="Sequence lengths to test")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--strategies", nargs="+",
                        default=list(STRATEGY_DEFAULTS.keys()),
                        choices=list(STRATEGY_DEFAULTS.keys()))
    parser.add_argument("--tp-h", type=int, default=4)
    parser.add_argument("--tp-s", type=int, default=4)
    parser.add_argument("--device", default="HBM4",
                        choices=list(DEVICE_CONFIGS.keys()))
    parser.add_argument("--sim-configs-dir", type=str, default="",
                        help="Simulation configs subdir under configs/ (default: gqa_seq_scaling)")
    parser.add_argument("--cache-db", type=str, default="",
                        help="Cache DB directory (default: cache_db_seq_scaling)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only generate configs, do not run")
    parser.add_argument("--force-run", action="store_true",
                        help="Force re-run even if cached")
    parser.add_argument("--trace-force", action="store_true",
                        help="Force regenerate traces")
    parser.add_argument("--rank-remap", type=str, default="",
                        choices=["", "block2x2"],
                        help="Rank remap strategy (e.g. block2x2)")
    args = parser.parse_args()

    sim_subdir = args.sim_configs_dir or "gqa_seq_scaling"
    cache_subdir = args.cache_db or "cache_db_seq_scaling"

    trace_configs_dir = SCRIPT_DIR / "trace_configs" / "gqa"
    sim_configs_dir = SCRIPT_DIR / "configs" / sim_subdir
    cache_dir = SCRIPT_DIR / cache_subdir

    generated_trace_paths = []
    generated_sim_paths = []

    remap = args.rank_remap
    remap_tag = f"_{remap}" if remap else ""

    for strategy in args.strategies:
        for seq in args.seq:
            tc = make_trace_config(strategy, seq, args.batch,
                                   args.tp_h, args.tp_s, remap)
            tc_filename = f"qwen_gqa_{strategy}_fwd_tph{args.tp_h}_tps{args.tp_s}_bs{args.batch}_sl{seq}{remap_tag}.json"
            tc_path = trace_configs_dir / strategy / tc_filename
            write_json(str(tc_path), tc)
            generated_trace_paths.append(tc_path)

            sc = make_sim_config(strategy, seq, args.batch,
                                 args.tp_h, args.tp_s, args.device, remap)
            sc_filename = (
                f"tp16_{args.device.lower()}_mesh2d_4x4_onering_"
                f"qwen_gqa_{strategy}_fwd{remap_tag}_bs{args.batch}_sl{seq}.json"
            )
            sc_path = sim_configs_dir / sc_filename
            write_json(str(sc_path), sc)
            generated_sim_paths.append(sc_path)

    print(f"[INFO] Generated {len(generated_trace_paths)} trace configs in {trace_configs_dir}")
    print(f"[INFO] Generated {len(generated_sim_paths)} sim configs in {sim_configs_dir}")

    for p in generated_trace_paths:
        print(f"  trace: {p.relative_to(SCRIPT_DIR)}")
    for p in generated_sim_paths:
        print(f"  sim:   {p.relative_to(SCRIPT_DIR)}")

    if args.dry_run:
        print("[DRY-RUN] Skipping execution.")
        return 0

    os.makedirs(cache_dir, exist_ok=True)
    db_path = cache_dir / "results.sqlite"
    csv_path = cache_dir / "results_cache.csv"

    trace_pattern = "gqa_seq_scaling/*.json"
    if sim_subdir != "gqa_seq_scaling":
        trace_pattern = f"{sim_subdir}/*.json"

    cmd = [
        "python3", str(SCRIPT_DIR / "bulk_run_configs.py"),
        "--configs-dir", str(sim_configs_dir),
        "--pattern", "*.json",
        "--gen-trace",
        "--trace-configs-dir", str(trace_configs_dir),
        "--trace-pattern", "**/*_sl*.json",
        "--db", str(db_path),
        "--csv", str(csv_path),
    ]
    if args.force_run:
        cmd.append("--force-run")
    if args.trace_force:
        cmd.append("--trace-force")

    for strat in args.strategies:
        for seq in args.seq:
            sl_tag = f"sl{seq}"
            if sl_tag not in " ".join(str(p) for p in generated_trace_paths):
                continue

    print(f"\n[RUN] {' '.join(cmd)}\n")
    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print(f"\n[DONE] Results cached in {db_path}")
        print(f"       CSV export: {csv_path}")
    else:
        print(f"\n[FAIL] Exit code: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
