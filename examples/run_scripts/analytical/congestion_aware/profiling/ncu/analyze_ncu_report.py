#!/usr/bin/env python3
"""Analyze ncu decode profiling reports.

Extracts per-stage, per-seq 3-dimensional utilization:
  - HBM bandwidth utilization (%)
  - SM instruction throughput (%)
  - Tensor Core pipe utilization (%)

Usage:
    python analyze_ncu_report.py H100_results/single_gpu_decode.ncu-rep
    python analyze_ncu_report.py H100_results/tp2_decode.ncu-rep
    python analyze_ncu_report.py H100_results/*.ncu-rep          # compare multiple
"""
import argparse
import csv
import io
import os
import subprocess
import sys

# ── ncu metric names ─────────────────────────────────────────────
METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "dram__bytes_read.sum.pct_of_peak_sustained_elapsed",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
    "pmsampling:sm__pipe_tensor_cycles_active_realtime.avg.pct_of_peak_sustained_elapsed",
]

# ncu binary
NCU = os.environ.get("NCU_PATH", "/usr/local/cuda-12.4/bin/ncu")

# ── Expected DRAM read sizes (bytes) for stage identification ────
# Qwen3-235B GQA: D=4096, Q=64*128=8192, KV=4*128=512, INTER=14336
# Sizes in MB for fp16 weights:
WEIGHT_SIZES = {
    "qkv_proj":     4096 * (8192 + 512 + 512) * 2 / 1e6,   # ~75.5 MB
    "o_proj":       8192 * 4096 * 2 / 1e6,                   # ~67.1 MB
    "gate_up_proj": 4096 * (2 * 14336) * 2 / 1e6,           # ~235.0 MB
    "down_proj":    14336 * 4096 * 2 / 1e6,                  # ~117.4 MB
}

# For TP=2, weight sizes are halved for column-parallel (qkv, gate_up)
# and row-parallel (o, down) — row-parallel has same weight read but
# the DRAM read size changes differently. We match with tolerance.
WEIGHT_SIZES_TP2 = {
    "qkv_proj":     4096 * (4096 + 256 + 256) * 2 / 1e6,    # ~37.7 MB
    "o_proj":       4096 * 4096 * 2 / 1e6,                    # ~33.6 MB
    "gate_up_proj": 4096 * (2 * 7168) * 2 / 1e6,             # ~117.4 MB
    "down_proj":    7168 * 4096 * 2 / 1e6,                    # ~58.7 MB
}

# All known stage DRAM signatures (for matching)
ALL_WEIGHT_SIGS = {
    # (min_MB, max_MB) → stage name
    # TP=1
    (220, 250): "gate_up_proj",
    (105, 130): "down_proj",
    (65, 90):   "qkv_proj",
    (55, 75):   "o_proj",
    # TP=2
    (30, 42):   "qkv_proj",
    (28, 38):   "o_proj",
}


# ── Kernel classification ────────────────────────────────────────

def classify_kernel(name):
    nl = name.lower()
    if "xmma_gemm" in nl or "sm90_xmma" in nl:
        return "GEMM_TC"
    if "flash_fwd_splitkv_k" in nl or ("flash_fwd_kernel" in nl and "combine" not in nl):
        return "FA_compute"
    if "flash_fwd_splitkv_c" in nl or "flash_fwd_combine" in nl:
        return "FA_combine"
    # cuBLAS GEMM (shows as generic "Params)" ending)
    if nl.strip().endswith("params)") and "flash" not in nl:
        return "cuBLAS_GEMM"
    if "ncclkernel" in nl or "nccl" in nl or "allreduce" in nl:
        return "NCCL"
    if "rsqrt" in nl:
        return "RSQRT"
    return "small"


# ── Extract CSV from ncu report ──────────────────────────────────

def extract_csv(ncu_rep_path, ncu_bin=NCU):
    """Run ncu --import to export metrics as CSV."""
    cmd = [
        ncu_bin, "--import", ncu_rep_path,
        "--csv", "--page", "raw",
        "--metrics", ",".join(METRICS),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"ERROR: ncu export failed: {result.stderr[:500]}", file=sys.stderr)
        sys.exit(1)
    return result.stdout


def parse_csv(csv_text):
    """Parse ncu CSV output into list of kernel dicts.

    Uses header names (not fixed indices) so it works regardless of
    which metrics ncu actually exports.
    """
    reader = csv.reader(io.StringIO(csv_text))
    header = next(reader)
    units = next(reader)

    # Build column index map by header name (strip quotes)
    col = {}
    for i, h in enumerate(header):
        h = h.strip('" ')
        col[h] = i

    def sf(r, name):
        idx = col.get(name)
        if idx is None or idx >= len(r):
            return 0.0
        try:
            return float(r[idx])
        except (ValueError, TypeError):
            return 0.0

    rows = []
    for r in reader:
        if len(r) < 11:  # at least base columns
            continue
        try:
            kid = int(r[col.get("ID", 0)])
        except (ValueError, TypeError):
            continue

        rows.append({
            "id":          kid,
            "kernel":      r[col.get("Kernel Name", 4)],
            "device":      r[col.get("Device", 9)] if "Device" in col else "0",
            "dram_rd":     sf(r, "dram__bytes_read.sum"),
            "dram_rd_pct": sf(r, "dram__bytes_read.sum.pct_of_peak_sustained_elapsed"),
            "dram_wr":     sf(r, "dram__bytes_write.sum"),
            "dur_us":      sf(r, "gpu__time_duration.sum"),
            "sm_inst_pct": sf(r, "sm__inst_executed.avg.pct_of_peak_sustained_elapsed"),
            "tensor_pct":  sf(r, "pmsampling:sm__pipe_tensor_cycles_active_realtime"
                                 ".avg.pct_of_peak_sustained_elapsed"),
        })
    return rows


# ── Group kernels into per-seq decode layers ─────────────────────

def find_seq_boundaries(rows):
    """Split kernel list into per-seq groups using RSQRT (rmsnorm) markers."""
    # RSQRT = start of rmsnorm. Pattern per layer:
    #   RSQRT(rmsnorm_attn) ... RSQRT(rmsnorm_mlp) ...
    # Two RSQRTs per seq. Use the first one of each pair as seq boundary.
    rsqrt_ids = [k["id"] for k in rows if classify_kernel(k["kernel"]) == "RSQRT"]

    # Group pairs: (rmsnorm_attn, rmsnorm_mlp)
    attn_starts = rsqrt_ids[0::2]
    mlp_starts = rsqrt_ids[1::2]

    return attn_starts, mlp_starts


def identify_gemm_stage(kernel, tp):
    """Identify GEMM stage by DRAM read size."""
    rd = kernel["dram_rd"]
    sigs = WEIGHT_SIZES_TP2 if tp == 2 else WEIGHT_SIZES
    best_name, best_diff = None, 999999
    for name, expected_mb in sigs.items():
        diff = abs(rd - expected_mb)
        # Tolerance: 30% of expected
        if diff < expected_mb * 0.3 and diff < best_diff:
            best_diff = diff
            best_name = name
    return best_name


def analyze_seq(rows, attn_start, mlp_start, seq_end, tp=1):
    """Analyze one seq's decode layer. Returns dict of stage results."""
    attn_half = [k for k in rows if attn_start <= k["id"] < mlp_start]
    mlp_half = [k for k in rows if mlp_start <= k["id"] < seq_end]
    all_kernels = attn_half + mlp_half

    stages = {}

    # ── GEMMs: identify by DRAM read size ────────────────────────
    # Attention half: qkv_proj (GEMM_TC or cuBLAS), then o_proj
    # MLP half: gate_up_proj (GEMM_TC or cuBLAS), then down_proj
    for half_name, half in [("attn", attn_half), ("mlp", mlp_half)]:
        gemms = [k for k in half
                 if classify_kernel(k["kernel"]) in ("GEMM_TC", "cuBLAS_GEMM")]
        for g in gemms:
            stage_name = identify_gemm_stage(g, tp)
            if stage_name and stage_name not in stages:
                stages[stage_name] = {
                    "dur_us": g["dur_us"],
                    "dram_rd": g["dram_rd"],
                    "dram_wr": g["dram_wr"],
                    "dram_rd_pct": g["dram_rd_pct"],
                    "sm_inst_pct": g["sm_inst_pct"],
                    "tensor_pct": g["tensor_pct"],
                }

    # ── FlashAttention ────────────────────────────────────────────
    fa_kernels = [k for k in attn_half
                  if classify_kernel(k["kernel"]) in ("FA_compute", "FA_combine")]
    if fa_kernels:
        main = next((k for k in fa_kernels
                     if classify_kernel(k["kernel"]) == "FA_compute"), fa_kernels[0])
        stages["attn"] = {
            "dur_us":      sum(k["dur_us"] for k in fa_kernels),
            "dram_rd":     sum(k["dram_rd"] for k in fa_kernels),
            "dram_wr":     sum(k["dram_wr"] for k in fa_kernels),
            "dram_rd_pct": main["dram_rd_pct"],
            "sm_inst_pct": main["sm_inst_pct"],
            "tensor_pct":  main["tensor_pct"],
        }

    # ── NCCL all-reduce (TP only) ────────────────────────────────
    nccl_attn = [k for k in attn_half
                 if classify_kernel(k["kernel"]) == "NCCL"]
    nccl_mlp = [k for k in mlp_half
                if classify_kernel(k["kernel"]) == "NCCL"]
    if nccl_attn:
        stages["allreduce_attn"] = {
            "dur_us": sum(k["dur_us"] for k in nccl_attn),
            "dram_rd": 0, "dram_wr": 0,
            "dram_rd_pct": 0, "sm_inst_pct": 0, "tensor_pct": 0,
        }
    if nccl_mlp:
        stages["allreduce_mlp"] = {
            "dur_us": sum(k["dur_us"] for k in nccl_mlp),
            "dram_rd": 0, "dram_wr": 0,
            "dram_rd_pct": 0, "sm_inst_pct": 0, "tensor_pct": 0,
        }

    # ── Totals ────────────────────────────────────────────────────
    total_dur = sum(k["dur_us"] for k in all_kernels)
    heavy_dur = sum(stages[s]["dur_us"] for s in stages)
    small_dur = total_dur - heavy_dur

    return stages, total_dur, small_dur


# ── Main analysis ────────────────────────────────────────────────

STAGE_ORDER = [
    "qkv_proj", "attn", "o_proj", "allreduce_attn",
    "gate_up_proj", "down_proj", "allreduce_mlp",
]

HBM_PEAK_GBS = 3350.0  # H100 HBM3


def bottleneck_tag(dram_pct, sm_pct, tensor_pct):
    if dram_pct > 30:
        return "MEM"
    if tensor_pct > 5:
        return "COMPUTE"
    if sm_pct > dram_pct and sm_pct > 5:
        return "SM"
    return "LATENCY"


def analyze_report(ncu_rep_path, seq_lens=None, tp=1, ncu_bin=NCU):
    """Full analysis of one ncu report. Returns structured results."""
    print(f"Importing {ncu_rep_path} ...")
    csv_text = extract_csv(ncu_rep_path, ncu_bin)
    rows = parse_csv(csv_text)
    print(f"  {len(rows)} kernels loaded")

    # Detect TP from NCCL kernels
    has_nccl = any(classify_kernel(k["kernel"]) == "NCCL" for k in rows)
    if has_nccl and tp == 1:
        tp = 2
        print(f"  Detected NCCL kernels → TP={tp}")

    # If TP=2, we may have kernels from 2 devices. Group by device.
    devices = sorted(set(k["device"] for k in rows))
    print(f"  Devices: {devices}")

    all_results = {}  # device → [(seq, stages, total, small), ...]

    for dev in devices:
        dev_rows = [k for k in rows if k["device"] == dev]
        attn_starts, mlp_starts = find_seq_boundaries(dev_rows)

        if not attn_starts:
            continue

        results = []
        for si in range(min(len(attn_starts), len(mlp_starts))):
            a_s = attn_starts[si]
            m_s = mlp_starts[si]
            a_e = attn_starts[si + 1] if si + 1 < len(attn_starts) \
                else max(k["id"] for k in dev_rows)

            stages, total, small = analyze_seq(dev_rows, a_s, m_s, a_e, tp)
            results.append((si, stages, total, small))

        all_results[dev] = results

    return all_results, tp, devices


def print_report(ncu_rep_path, all_results, tp, devices, seq_lens):
    """Print formatted report."""
    basename = os.path.basename(ncu_rep_path).replace(".ncu-rep", "")

    for dev in devices:
        results = all_results.get(dev, [])
        if not results:
            continue

        print()
        print("=" * 130)
        print(f"  {basename} | device={dev} | TP={tp}")
        print("=" * 130)
        print(f"{'Seq':>8} | {'Stage':<18} | {'Time(us)':>9} | "
              f"{'HBM BW%':>7} | {'BW(GB/s)':>9} | "
              f"{'SM Inst%':>8} | {'TensorC%':>8} | {'Bottleneck':<10}")
        print("-" * 130)

        for si, stages, total, small in results:
            S = seq_lens[si] if si < len(seq_lens) else f"seq_{si}"

            for sname in STAGE_ORDER:
                if sname not in stages:
                    continue
                d = stages[sname]
                dur = d["dur_us"]
                rd, wr = d["dram_rd"], d["dram_wr"]
                bw = (rd + wr) / (dur / 1e6) / 1000 if dur > 0 else 0
                dp = d["dram_rd_pct"]
                sp = d["sm_inst_pct"]
                tp_val = d["tensor_pct"]
                bn = bottleneck_tag(dp, sp, tp_val)

                print(f"{S:>8} | {sname:<18} | {dur:>9.1f} | "
                      f"{dp:>6.1f}% | {bw:>9.1f} | "
                      f"{sp:>7.1f}% | {tp_val:>7.1f}% | {bn:<10}")

            print(f"{S:>8} | {'small_ops':<18} | {small:>9.1f} |"
                  f"         |           |          |          | LATENCY")
            print(f"{S:>8} | {'LAYER TOTAL':<18} | {total:>9.1f} |")
            print("-" * 130)

    # Summary table
    print()
    print("=" * 100)
    print(f"  3-Dimensional Utilization Summary — {basename}")
    print("=" * 100)
    print(f"{'Seq':>8} | {'Stage':<18} | {'HBM BW%':>8} | {'SM Inst%':>9} | "
          f"{'TensorC%':>9} | {'Bottleneck':<10}")
    print("-" * 100)

    # Use first device for summary
    if not devices:
        print("  (no data)")
        return
    dev0 = devices[0]
    for si, stages, total, small in all_results.get(dev0, []):
        S = seq_lens[si] if si < len(seq_lens) else f"seq_{si}"
        for sname in STAGE_ORDER:
            if sname not in stages:
                continue
            d = stages[sname]
            dp = d["dram_rd_pct"]
            sp = d["sm_inst_pct"]
            tp_val = d["tensor_pct"]
            bn = bottleneck_tag(dp, sp, tp_val)
            print(f"{S:>8} | {sname:<18} | {dp:>7.1f}% | {sp:>8.1f}% | "
                  f"{tp_val:>8.1f}% | {bn:<10}")
        print("-" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ncu decode profiling reports")
    parser.add_argument("reports", nargs="+",
                        help="Path(s) to .ncu-rep files")
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536],
                        help="Sequence lengths used in profiling (in order)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel degree (auto-detected from NCCL)")
    parser.add_argument("--ncu", type=str, default=NCU,
                        help="Path to ncu binary")
    args = parser.parse_args()

    ncu_bin = args.ncu

    for rep_path in args.reports:
        if not os.path.exists(rep_path):
            print(f"WARNING: {rep_path} not found, skipping")
            continue
        all_results, tp, devices = analyze_report(
            rep_path, args.seq, args.tp, ncu_bin=ncu_bin)
        print_report(rep_path, all_results, tp, devices, args.seq)
        print()


if __name__ == "__main__":
    main()
