#!/usr/bin/env python3
"""Analyze MoE FFN decode ncu reports (companion to ncu_moe_decode_profile.py).

The dense analyzer (analyze_ncu_report.py) keys GEMM stages off fixed DRAM-read
signatures of the big dense MLP. MoE expert GEMVs are many, tiny, and same-sized,
so here we instead group GEMM kernels by their DRAM-read size (which maps 1:1 to
a weight matrix: gating / expert_gate_up / expert_down / shared_*), and report the
median HBM-BW% (memory util) and TensorCore% (compute util) per group.

Output is exactly the two numbers the Rubin FFN model needs per MoE stage:
    mem_util  = dram__bytes_read.sum.pct_of_peak_sustained_elapsed   (HBM BW%)
    cmpt_util = sm__pipe_tensor_cycles_active...pct_of_peak           (TensorCore%)

Usage:
    NCU_PATH=/usr/local/cuda-13.0/bin/ncu python analyze_moe_ncu.py \
        H100_results/moe_decode_dsv3.ncu-rep --model deepseek3
"""
import argparse
import csv
import io
import os
import statistics
import subprocess
import sys

NCU = os.environ.get("NCU_PATH", "/usr/local/cuda-12.4/bin/ncu")
METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
]

# Per-stage weight DRAM-read size (MB) for the supported MoE models, fp8 (1B) and
# fp16 (2B). tp shards the intermediate dim. The analyzer matches each GEMM kernel
# to the nearest signature within 30%.
MOE_DIMS = {
    "deepseek3":    dict(d=7168, ne=256, di=2048, ns=1),
    "qwen3-235b":   dict(d=4096, ne=128, di=1536, ns=0),
    "gpt-oss-120b": dict(d=2880, ne=128, di=2880, ns=0),
}


def stage_sizes_mb(model, tp, wbytes):
    m = MOE_DIMS[model]
    d, ne, di, ns = m["d"], m["ne"], m["di"] // tp, m["ns"]
    sigs = {
        "gating":         d * ne * wbytes / 1e6,
        "expert_gate_up": d * (2 * di) * wbytes / 1e6,
        "expert_down":    di * d * wbytes / 1e6,
    }
    if ns > 0:
        sigs["shared_gate_up"] = d * (2 * di * ns) * wbytes / 1e6
        sigs["shared_down"] = (di * ns) * d * wbytes / 1e6
    return sigs


def classify(name):
    nl = name.lower()
    if "xmma_gemm" in nl or "sm90_xmma" in nl:
        return "GEMM"
    if nl.strip().endswith("params)") and "flash" not in nl:
        return "GEMM"      # cuBLAS
    return "other"


def extract_rows(rep, ncu_bin):
    cmd = [ncu_bin, "--import", rep, "--csv", "--page", "raw",
           "--metrics", ",".join(METRICS)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if r.returncode != 0:
        print(f"ERROR: ncu export failed: {r.stderr[:400]}", file=sys.stderr)
        sys.exit(1)
    reader = csv.reader(io.StringIO(r.stdout))
    header = next(reader); next(reader)
    col = {h.strip('" '): i for i, h in enumerate(header)}

    def sf(row, name):
        i = col.get(name)
        try:
            return float(row[i])
        except (TypeError, ValueError, IndexError):
            return 0.0
    rows = []
    for row in reader:
        if len(row) < len(col) // 2:
            continue
        rows.append({
            "kernel": row[col.get("Kernel Name", 4)],
            "dur": sf(row, "gpu__time_duration.sum"),
            "rd_mb": sf(row, "dram__bytes_read.sum") / 1e6,
            "mem_util": sf(row, "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
            "tc_util": sf(row, "pmsampling:sm__pipe_tensor_cycles_active_realtime"
                               ".avg.pct_of_peak_sustained_elapsed"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("report")
    ap.add_argument("--model", default="deepseek3", choices=list(MOE_DIMS))
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--wbytes", type=int, default=1, help="weight bytes (fp8=1, fp16=2)")
    ap.add_argument("--ncu", default=NCU)
    args = ap.parse_args()

    rows = extract_rows(args.report, args.ncu)
    gemms = [r for r in rows if classify(r["kernel"]) == "GEMM"]
    sigs = stage_sizes_mb(args.model, args.tp, args.wbytes)

    buckets = {s: [] for s in sigs}
    for g in gemms:
        best, bd = None, 1e18
        for s, mb in sigs.items():
            diff = abs(g["rd_mb"] - mb)
            if diff < mb * 0.30 and diff < bd:
                best, bd = s, diff
        if best:
            buckets[best].append(g)

    print(f"\n== MoE FFN utilization — {os.path.basename(args.report)} "
          f"| model={args.model} tp={args.tp} wbytes={args.wbytes} ==")
    print(f"{'stage':<16} | {'n':>3} | {'rd(MB)':>8} | {'mem_util':>8} | {'cmpt_util':>9}")
    print("-" * 60)
    out = {}
    for s, mb in sigs.items():
        b = buckets[s]
        if not b:
            print(f"{s:<16} | {'0':>3} | {mb:>8.1f} | {'--':>8} | {'--':>9}  (no match)")
            continue
        mu = statistics.median(x["mem_util"] for x in b)
        cu = statistics.median(x["tc_util"] for x in b)
        out[s] = {"mem_util": round(mu / 100, 4), "compute_util": round(cu / 100, 4)}
        print(f"{s:<16} | {len(b):>3} | {mb:>8.1f} | {mu:>7.1f}% | {cu:>8.1f}%")
    print("\nJSON (mem_util/compute_util as fractions):")
    import json
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
