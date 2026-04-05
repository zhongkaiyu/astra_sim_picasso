#!/usr/bin/env python3
"""Add rubin_tp2 strategy to hybrid merged JSON.

Rubin TP2 model:
  1. Compute per GPU = Rubin TP1 roofline / 2 (model halved across 2 GPUs)
  2. Apply H100 TP2 BW utilization for compute adjustment
  3. Communication = roofline AllReduce (NVLink analytical model)

AllReduce model (ring, 2 GPUs):
  msg = bs * d_model * bytes_per_elem
  steps = 2 * (n-1) = 2
  chunk = msg / n
  time = steps * (nvlink_latency + chunk / nvlink_bw)

Usage:
  python3 add_rubin_tp2.py --data hybrid.json --tp2-profile h100_tp2_profile.json
  python3 add_rubin_tp2.py --data hybrid.json --nvlink-bw 1.8 --nvlink-latency 5000 --d-model 4096
"""
import argparse
import bisect
import json
import math
import sys
from pathlib import Path


def _lookup(mapping: dict, key: int) -> float:
    """Lookup with log2-linear interpolation."""
    parsed = {}
    for k, v in mapping.items():
        if k.startswith("_"):
            continue
        parsed[int(k)] = v["utilization"] if isinstance(v, dict) else float(v)
    if key in parsed:
        return parsed[key]
    sorted_keys = sorted(parsed)
    if key <= sorted_keys[0]:
        return parsed[sorted_keys[0]]
    if key >= sorted_keys[-1]:
        return parsed[sorted_keys[-1]]
    idx = bisect.bisect_right(sorted_keys, key) - 1
    lo, hi = sorted_keys[idx], sorted_keys[idx + 1]
    t = (math.log2(key) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return parsed[lo] + t * (parsed[hi] - parsed[lo])


def ring_allreduce_ns(msg_bytes: int, n_gpus: int,
                      nvlink_bw_tbs: float, latency_ns: float) -> dict:
    """Analytical ring AllReduce time for n GPUs.

    Ring AR = 2*(n-1) steps, each step sends msg/n bytes.
    Per-step: latency + chunk / bw.
    """
    steps = 2 * (n_gpus - 1)
    chunk = msg_bytes / n_gpus
    bw_ns = chunk / (nvlink_bw_tbs * 1e3)  # TB/s -> GB/ns -> bytes/ns
    per_step = latency_ns + bw_ns
    total = steps * per_step
    return {
        "time_ns": round(total, 2),
        "msg_bytes": msg_bytes,
        "n_gpus": n_gpus,
        "steps": steps,
        "chunk_bytes": round(chunk, 2),
        "latency_ns": latency_ns,
        "bw_per_step_ns": round(bw_ns, 4),
        "nvlink_bw_tbs": nvlink_bw_tbs,
    }


def main():
    parser = argparse.ArgumentParser(description="Add rubin_tp2 to hybrid JSON")
    parser.add_argument("--data", required=True, help="Hybrid merged JSON")
    parser.add_argument("--tp2-profile", default="h100_tp2_profile.json",
                        help="H100 TP2 BW utilization profile JSON")
    parser.add_argument("--nvlink-bw", type=float, default=1.8,
                        help="NVLink bandwidth in TB/s per direction (default: 1.8)")
    parser.add_argument("--nvlink-latency", type=float, default=900,
                        help="NVLink AllReduce per-step latency in ns (default: 900)")
    parser.add_argument("--d-model", type=int, default=4096,
                        help="Model hidden dimension for AR message size (default: 4096)")
    parser.add_argument("--bytes-per-elem", type=int, default=1,
                        help="Bytes per element, FP8=1, FP16=2 (default: 1)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("-o", "--output", default="",
                        help="Output path (default: overwrite input)")
    args = parser.parse_args()

    with open(args.data) as f:
        hybrid = json.load(f)
    with open(args.tp2_profile) as f:
        tp2 = json.load(f)

    bw_util = tp2["bw_util"]

    rubin_data = hybrid.get("strategies", {}).get("rubin", {}).get("data", [])
    if not rubin_data:
        print("ERROR: no rubin strategy in hybrid data", file=sys.stderr)
        return 1

    # AllReduce message: bs * d_model * bytes_per_elem
    ar_msg = args.batch * args.d_model * args.bytes_per_elem
    ar = ring_allreduce_ns(ar_msg, n_gpus=2,
                           nvlink_bw_tbs=args.nvlink_bw,
                           latency_ns=args.nvlink_latency)
    comm_ns = ar["time_ns"]

    tp2_entries = []
    for entry in rubin_data:
        seq = entry["seq"]

        # Rubin TP1 roofline times (before BW util adjustment)
        qkv_raw = entry.get("Proj_QKV_roofline_ns", entry["hybrid_Proj_QKV_ns"])
        attn_raw = entry.get("attn_roofline_ns", entry["hybrid_attn_ns"])
        out_raw = entry.get("Proj_O_roofline_ns", entry["hybrid_Proj_O_ns"])

        # TP2: each GPU holds half → roofline / 2
        qkv_tp2_rf = qkv_raw / 2
        attn_tp2_rf = attn_raw / 2
        out_tp2_rf = out_raw / 2

        # Apply H100 TP2 BW utilization
        qkv_u = _lookup(bw_util["proj_qkv"], 1)
        attn_u = _lookup(bw_util["attn_fused"], seq)
        out_u = _lookup(bw_util["proj_o"], 1)

        qkv_ns = qkv_tp2_rf / qkv_u if qkv_u > 0 else qkv_tp2_rf
        attn_ns = attn_tp2_rf / attn_u if attn_u > 0 else attn_tp2_rf
        out_ns = out_tp2_rf / out_u if out_u > 0 else out_tp2_rf
        gpu_ns = qkv_ns + attn_ns + out_ns

        wall_ns = gpu_ns + comm_ns

        # Aggregate metrics
        agg = entry.get("aggregate_metrics", {})
        hbm_per_gpu = agg.get("hbm_read_bytes_per_cube", 0) // 2
        macs_per_gpu = agg.get("compute_macs_per_cube", 0) // 2

        tp2_entry = {
            "seq": seq,
            "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
            "hybrid_attn_ns": round(attn_ns, 2),
            "hybrid_Proj_O_ns": round(out_ns, 2),
            "hybrid_gpu_ns": round(gpu_ns, 2),
            "comm_total_ns": round(comm_ns, 2),
            "hybrid_wall_ns": round(wall_ns, 2),
            "comm_breakdown": {
                "ops": [{
                    "name": "allreduce_attn",
                    "source": "roofline_nvlink",
                    "time_ns": round(comm_ns, 2),
                    **ar,
                }],
                "total_comm_ns": round(comm_ns, 2),
            },
            "aggregate_metrics": {
                "hbm_read_bytes_per_cube": hbm_per_gpu,
                "hbm_read_bytes_all_cubes": hbm_per_gpu * 2,
                "compute_macs_per_cube": macs_per_gpu,
                "compute_flops_per_cube": macs_per_gpu * 2,
                "compute_flops_all_cubes": macs_per_gpu * 2 * 2,
                "n_cubes": 2,
                "total_time_ns": round(wall_ns, 2),
            },
            "source": "rubin_tp2 (2 GPUs, TP2 BW util + NVLink roofline AR)",
            "utilization_applied": True,
            "Proj_QKV_bw_util": round(qkv_u, 4),
            "Proj_QKV_roofline_ns": round(qkv_tp2_rf, 2),
            "attn_bw_util": round(attn_u, 4),
            "attn_roofline_ns": round(attn_tp2_rf, 2),
            "Proj_O_bw_util": round(out_u, 4),
            "Proj_O_roofline_ns": round(out_tp2_rf, 2),
        }
        tp2_entries.append(tp2_entry)

    hybrid["strategies"]["rubin_tp2"] = {"data": tp2_entries}

    # --- H100 TP2 (same pattern, from h100 strategy) ---
    h100_data = hybrid.get("strategies", {}).get("h100", {}).get("data", [])
    h100_tp2_entries = []
    if h100_data:
        h100_nvlink_bw = 0.9  # H100 NVLink: 900 GB/s per direction
        h100_ar = ring_allreduce_ns(ar_msg, n_gpus=2,
                                     nvlink_bw_tbs=h100_nvlink_bw,
                                     latency_ns=args.nvlink_latency)
        h100_comm_ns = h100_ar["time_ns"]

        for entry in h100_data:
            seq = entry["seq"]
            qkv_raw = entry.get("Proj_QKV_roofline_ns", entry["hybrid_Proj_QKV_ns"])
            attn_raw = entry.get("attn_roofline_ns", entry["hybrid_attn_ns"])
            out_raw = entry.get("Proj_O_roofline_ns", entry["hybrid_Proj_O_ns"])

            qkv_tp2_rf = qkv_raw / 2
            attn_tp2_rf = attn_raw / 2
            out_tp2_rf = out_raw / 2

            qkv_u = _lookup(bw_util["proj_qkv"], 1)
            attn_u = _lookup(bw_util["attn_fused"], seq)
            out_u = _lookup(bw_util["proj_o"], 1)

            qkv_ns = qkv_tp2_rf / qkv_u if qkv_u > 0 else qkv_tp2_rf
            attn_ns = attn_tp2_rf / attn_u if attn_u > 0 else attn_tp2_rf
            out_ns = out_tp2_rf / out_u if out_u > 0 else out_tp2_rf
            gpu_ns = qkv_ns + attn_ns + out_ns
            wall_ns = gpu_ns + h100_comm_ns

            agg = entry.get("aggregate_metrics", {})
            hbm_per_gpu = agg.get("hbm_read_bytes_per_cube", 0) // 2
            macs_per_gpu = agg.get("compute_macs_per_cube", 0) // 2

            h100_tp2_entries.append({
                "seq": seq,
                "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
                "hybrid_attn_ns": round(attn_ns, 2),
                "hybrid_Proj_O_ns": round(out_ns, 2),
                "hybrid_gpu_ns": round(gpu_ns, 2),
                "comm_total_ns": round(h100_comm_ns, 2),
                "hybrid_wall_ns": round(wall_ns, 2),
                "comm_breakdown": {
                    "ops": [{"name": "allreduce_attn", "source": "roofline_nvlink_h100",
                             "time_ns": round(h100_comm_ns, 2), **h100_ar}],
                    "total_comm_ns": round(h100_comm_ns, 2),
                },
                "aggregate_metrics": {
                    "hbm_read_bytes_per_cube": hbm_per_gpu,
                    "hbm_read_bytes_all_cubes": hbm_per_gpu * 2,
                    "compute_macs_per_cube": macs_per_gpu,
                    "compute_flops_per_cube": macs_per_gpu * 2,
                    "compute_flops_all_cubes": macs_per_gpu * 2 * 2,
                    "n_cubes": 2,
                    "total_time_ns": round(wall_ns, 2),
                },
                "source": "h100_tp2 (2 GPUs, TP2 BW util + H100 NVLink roofline AR)",
                "utilization_applied": True,
                "Proj_QKV_bw_util": round(qkv_u, 4),
                "Proj_QKV_roofline_ns": round(qkv_tp2_rf, 2),
                "attn_bw_util": round(attn_u, 4),
                "attn_roofline_ns": round(attn_tp2_rf, 2),
                "Proj_O_bw_util": round(out_u, 4),
                "Proj_O_roofline_ns": round(out_tp2_rf, 2),
            })
        hybrid["strategies"]["h100_tp2"] = {"data": h100_tp2_entries}

    # Update summary table
    for strat_name, entries in [("rubin_tp2", tp2_entries), ("h100_tp2", h100_tp2_entries)]:
        lookup = {e["seq"]: e for e in entries}
        for row in hybrid.get("summary_table", []):
            seq = row["seq"]
            if seq in lookup:
                e = lookup[seq]
                row[strat_name] = {
                    "Proj_QKV_ns": round(e["hybrid_Proj_QKV_ns"], 1),
                    "attn_ns": round(e["hybrid_attn_ns"], 1),
                    "Proj_O_ns": round(e["hybrid_Proj_O_ns"], 1),
                    "gpu_ns": round(e["hybrid_gpu_ns"], 1),
                    "comm_ns": round(e["comm_total_ns"], 1),
                    "wall_ns": round(e["hybrid_wall_ns"], 1),
                }

    # Update metadata
    hybrid["metadata"]["rubin_tp2_model"] = (
        f"Rubin TP1 roofline / 2 per GPU, "
        f"H100 TP2 BW util, "
        f"NVLink roofline AR (bw={args.nvlink_bw}TB/s, lat={args.nvlink_latency}ns, "
        f"msg={ar_msg}B)"
    )
    if h100_tp2_entries:
        hybrid["metadata"]["h100_tp2_model"] = (
            f"H100 TP1 roofline / 2 per GPU, "
            f"H100 TP2 BW util, "
            f"H100 NVLink AR (bw=0.9TB/s, lat={args.nvlink_latency}ns, "
            f"msg={ar_msg}B)"
        )

    out_path = args.output or args.data
    with open(out_path, "w") as f:
        json.dump(hybrid, f, indent=2, ensure_ascii=False)

    # Print summary
    for label, entries in [("Rubin TP2", tp2_entries), ("H100 TP2", h100_tp2_entries)]:
        if not entries:
            continue
        c = entries[0]["comm_total_ns"]
        print(f"\n=== {label} added ({len(entries)} entries, AR={c:.0f}ns) ===")
        print(f"{'seq':>8} | {'QKV':>7} {'Attn':>7} {'ProjO':>7} {'GPU':>7} {'Comm':>7} {'Wall':>7}")
        print("-" * 70)
        for e in entries:
            print(f"{e['seq']//1024:>6}K | {e['hybrid_Proj_QKV_ns']:>7.0f} {e['hybrid_attn_ns']:>7.0f} "
                  f"{e['hybrid_Proj_O_ns']:>7.0f} {e['hybrid_gpu_ns']:>7.0f} "
                  f"{e['comm_total_ns']:>7.0f} {e['hybrid_wall_ns']:>7.0f}")

    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
