#!/usr/bin/env python3
"""
Collect per-module latency data from GQA batch-scaling simulation logs.

Parses logs for Baseline (block2x2 Mesh2D), HMP (block2x2 Mesh2D), and
TP16 (Torus) strategies across multiple batch sizes at a fixed seq length.

Usage:
  python3 collect_gqa_batch_data.py                         # defaults (seq=65536)
  python3 collect_gqa_batch_data.py --seq 65536 --batch 1 2 4 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from collect_gqa_data import collect_module_times
from analyze_comm import parse_log

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[3]

BATCHES = [1, 2, 4, 8, 16, 32, 64]

STRATEGIES = {
    "baseline": {
        "label": "Baseline (block2x2 Mesh2D)",
        "log_pattern": "output_qwen/gqa_seq_scaling/baseline_block2x2_fwd_bs{batch}_sl{seq}/logs/simulation_log_gqa_baseline_block2x2_fwd_bs{batch}_sl{seq}.txt",
    },
    "hmp": {
        "label": "HMP (block2x2 Mesh2D)",
        "log_pattern": "output_qwen/gqa_seq_scaling/hmp_block2x2_fwd_bs{batch}_sl{seq}/logs/simulation_log_gqa_hmp_block2x2_fwd_bs{batch}_sl{seq}.txt",
    },
    "tp16": {
        "label": "TP16 (Torus OneRing)",
        "log_pattern": "output_qwen/gqa_seq_scaling/tp16_fwd_bs{batch}_sl{seq}/logs/simulation_log_gqa_tp16_fwd_bs{batch}_sl{seq}.txt",
    },
}


def collect_all(seq: int, batches: list[int]) -> dict:
    result = {}
    for key, cfg in STRATEGIES.items():
        entries = []
        for batch in batches:
            log_path = PROJECT_DIR / cfg["log_pattern"].format(seq=seq, batch=batch)
            if not log_path.is_file():
                print(f"[WARN] Missing log: {log_path}", file=sys.stderr)
                continue
            summary = parse_log(str(log_path), npu=0)
            modules = collect_module_times(summary)
            entries.append({
                "batch": batch,
                "wall_ns": summary.wall_time,
                "gpu_ns": summary.gpu_time,
                "comm_ns": summary.comm_time,
                "modules": modules,
            })
        result[key] = {
            "label": cfg["label"],
            "seq": seq,
            "data": entries,
        }
    return result


def main():
    parser = argparse.ArgumentParser(description="Collect GQA batch-scaling data")
    parser.add_argument("-o", "--output", default="")
    parser.add_argument("--seq", type=int, default=65536)
    parser.add_argument("--batch", type=int, nargs="+", default=BATCHES)
    args = parser.parse_args()

    out_path = Path(args.output) if args.output else (
        SCRIPT_DIR / "reports" / f"gqa_batch_scaling_sl{args.seq}_data.json"
    )

    data = collect_all(args.seq, args.batch)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Data saved to {out_path}")

    for key in data:
        n = len(data[key]["data"])
        print(f"  {key}: {n} batch points")

    return 0


if __name__ == "__main__":
    sys.exit(main())
