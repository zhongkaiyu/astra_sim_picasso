#!/usr/bin/env python3
"""
Merge AstraSim simulation data with roofline weight-loading estimates.

AstraSim provides: communication latency (topology + congestion aware),
                   attention compute (includes KV cache loading).
Roofline provides: weight loading time for QKV / Output projections,
                   analytical comm estimates (QKV AllGather, final reduce,
                   Baseline output AllGather).

For each operation the hybrid estimate is:
  QKV    = max(AstraSim qkv_comp, roofline qkv.memory_ns)
  Attn   = max(AstraSim attn_comp, roofline attn fused)
  Output = max(AstraSim output_comp, roofline output.memory_ns)

Communication breakdown (per strategy):
  HMP:      qkv_allgather (roofline) + final_reduce (roofline)
  Baseline: qkv_allgather (roofline) + attn_comm (AstraSim)
            + output_ar (AstraSim) + output_ag (roofline, corrected)
  TP16:     attn_comm (AstraSim) + output_comm (AstraSim)

Aggregate per-entry metrics:
  d2d_link_transfer_bytes, hbm_read_bytes, compute_ops_flops, total_time_ns

Usage:
  python3 merge_gqa_results.py \\
    --astrasim-data reports/qwen3/astrasim/gqa_seq_scaling_bs1_80t_split4_data.json \\
    --roofline-data reports/qwen3/roofline/gqa_roofline_with_weights_bs1_80T.json \\
    -o reports/qwen3/hybrid/gqa_hybrid_merged_80T_split4.json
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Utilization helpers
# ---------------------------------------------------------------------------

def _lookup_utilization(util_map: dict, key: int) -> float:
    """Look up utilization for a given key (bs or seq_len).

    util_map formats:
      proj_qkv/proj_o: {"1": 0.028, "2": 0.056, ...}
      score/attention:  {"16": {"utilization": 0.008, "k_split": 4}, ...}

    Exact match → return directly.
    Missing key → log2-linear interpolation between nearest neighbors.
    Out of range → clamp to boundary.
    """
    parsed: dict[int, float] = {}
    for k, v in util_map.items():
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
    lo_val, hi_val = parsed[lo], parsed[hi]
    if lo == hi:
        return lo_val
    t = (math.log2(key) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return lo_val + t * (hi_val - lo_val)


def apply_utilization(rf_entry: dict, util_data: dict,
                      batch: int, cache_seq: int,
                      dk: int = 128, dk_ref: int = 128) -> dict:
    """Compute utilization-adjusted roofline times for one entry.

    adj_compute_ns = compute_ns / utilization
    adj_total_ns   = max(adj_compute_ns, memory_ns)

    For attention sub-ops, the utilization lookup uses
    effective_seq = cache_seq * dk / dk_ref to normalize across strategies
    with different (cache_seq, dk) combinations but identical total MACs.
    """
    result: dict = {}

    # --- Proj_QKV ---
    qkv = rf_entry.get("Proj_QKV", {})
    qkv_util = _lookup_utilization(util_data["proj_qkv"], batch)
    qkv_comp = qkv.get("compute_ns", 0)
    qkv_comp_adj = qkv_comp / qkv_util if qkv_util > 0 else qkv_comp
    qkv_mem = qkv.get("memory_ns", 0)
    result["qkv_utilization"] = round(qkv_util, 6)
    result["qkv_adj_compute_ns"] = round(qkv_comp_adj, 2)
    result["qkv_adj_total_ns"] = round(max(qkv_comp_adj, qkv_mem), 2)

    # --- Attention sub-ops ---
    attn = rf_entry.get("attention", {})
    # Normalize: effective_seq = cache_seq * dk / dk_ref so strategies with
    # same total MACs (cache_seq * dk) get the same utilization lookup.
    effective_seq = cache_seq * dk // dk_ref if dk_ref > 0 else cache_seq

    if "sub_ops" in attn:
        score = attn["sub_ops"]["score_gemv"]
        attn_v = attn["sub_ops"]["attn_v_gemv"]

        score_util = _lookup_utilization(util_data["score"], effective_seq)
        attn_v_util = _lookup_utilization(util_data["attention"], effective_seq)

        sc_comp = score["compute_ns"]
        sc_comp_adj = sc_comp / score_util if score_util > 0 else sc_comp
        sc_fused_adj = max(sc_comp_adj, score["memory_ns"])

        av_comp = attn_v["compute_ns"]
        av_comp_adj = av_comp / attn_v_util if attn_v_util > 0 else av_comp
        av_fused_adj = max(av_comp_adj, attn_v["memory_ns"])

        result["score_utilization"] = round(score_util, 6)
        result["score_adj_compute_ns"] = round(sc_comp_adj, 2)
        result["score_adj_fused_ns"] = round(sc_fused_adj, 2)
        result["attn_v_utilization"] = round(attn_v_util, 6)
        result["attn_v_adj_compute_ns"] = round(av_comp_adj, 2)
        result["attn_v_adj_fused_ns"] = round(av_fused_adj, 2)
        result["attn_adj_total_fused_ns"] = round(sc_fused_adj + av_fused_adj, 2)

    elif "einsum_comp_ns" in attn:
        # Fused attention (Rubin): geometric mean of score and attn_v utils
        score_util = _lookup_utilization(util_data["score"], effective_seq)
        attn_v_util = _lookup_utilization(util_data["attention"], effective_seq)
        fused_util = (math.sqrt(score_util * attn_v_util)
                      if score_util > 0 and attn_v_util > 0
                      else max(score_util, attn_v_util))
        einsum_comp = attn["einsum_comp_ns"]
        einsum_comp_adj = einsum_comp / fused_util if fused_util > 0 else einsum_comp
        kv_mem = attn.get("kv_mem_ns", 0)
        result["fused_utilization"] = round(fused_util, 6)
        result["fused_adj_compute_ns"] = round(einsum_comp_adj, 2)
        result["attn_adj_total_fused_ns"] = round(max(einsum_comp_adj, kv_mem), 2)

    # --- Proj_O ---
    out = rf_entry.get("Proj_O", {})
    out_util = _lookup_utilization(util_data["proj_o"], batch)
    out_comp = out.get("compute_ns", 0)
    out_comp_adj = out_comp / out_util if out_util > 0 else out_comp
    out_mem = out.get("memory_ns", 0)
    result["output_utilization"] = round(out_util, 6)
    result["output_adj_compute_ns"] = round(out_comp_adj, 2)
    result["output_adj_total_ns"] = round(max(out_comp_adj, out_mem), 2)

    return result


def build_roofline_lookup(roofline: dict) -> dict[str, dict[int, dict]]:
    """Build {strategy: {seq: entry}} from roofline JSON."""
    lookup: dict[str, dict[int, dict]] = {}
    for strat_name, strat_data in roofline.get("strategies", {}).items():
        seq_map: dict[int, dict] = {}
        for entry in strat_data.get("data", []):
            seq_map[entry["seq"]] = entry
        lookup[strat_name] = seq_map
    return lookup


def build_astrasim_lookup(astrasim: dict) -> dict[str, dict[int, dict]]:
    """Build {strategy: {seq: entry}} from collect_gqa_data.py JSON."""
    lookup: dict[str, dict[int, dict]] = {}
    for strat_name, strat_data in astrasim.items():
        if strat_name.startswith("_"):
            continue
        if not isinstance(strat_data, dict) or "data" not in strat_data:
            continue
        seq_map: dict[int, dict] = {}
        for entry in strat_data["data"]:
            seq_map[entry["seq"]] = entry
        lookup[strat_name] = seq_map
    return lookup


def _ring_ar_d2d(n_gpus: int, msg_bytes: int) -> int:
    """D2D bytes for ring AllReduce: 2*(n-1)*M."""
    return 2 * (n_gpus - 1) * msg_bytes


def _ring_ag_d2d(n_gpus: int, chunk_bytes_per_gpu: int) -> int:
    """D2D bytes for ring AllGather: n*(n-1)*chunk."""
    return n_gpus * (n_gpus - 1) * chunk_bytes_per_gpu


def _ring_rs_d2d(n_gpus: int, total_bytes: int) -> int:
    """D2D bytes for ring ReduceScatter: (n-1)*total/n * n_groups handled by caller."""
    return (n_gpus - 1) * (total_bytes // n_gpus)


def _tree_reduce_d2d(n_gpus: int, msg_bytes: int) -> int:
    """D2D bytes for tree-reduce: (n-1)*msg."""
    return (n_gpus - 1) * msg_bytes


N_GPUS_PER_GROUP = 4   # Mesh2D 4×4: all comm groups have 4 GPUs
N_GROUPS = 4           # 4 parallel groups per dimension (tp_h=4, tp_s=4)


def build_comm_breakdown(strategy: str, mods: dict, rf_entry: dict,
                         rf_strat: dict, comm_batch_scale: int = 1) -> dict:
    """Build per-operation communication breakdown.

    comm_batch_scale: scale factor for astrasim comm times when roofline
    batch > astrasim batch.  Roofline comm already contains the correct
    batch; only astrasim-sourced ops need linear scaling.
    """
    rf_comm = rf_strat.get("comm", {})
    qkv_ag_rf = rf_comm.get("qkv_allgather", {})
    attn_rs_rf = rf_comm.get("attn_rs", {})
    final_red_rf = rf_comm.get("final_reduce", {})
    output_ar_rf = rf_comm.get("output_ar", {})
    output_ag_rf = rf_comm.get("output_ag", {})

    g = N_GPUS_PER_GROUP
    ng = N_GROUPS
    ops = []

    # 1. QKV AllGather (roofline, HMP/Baseline only — d_head split by tp_s)
    qkv_ag_ns = qkv_ag_rf.get("time_ns", 0)
    if qkv_ag_ns > 0:
        chunk = qkv_ag_rf.get("per_cube_chunk_elems", 0)
        d2d = _ring_ag_d2d(g, chunk) * ng
        ops.append({
            "name": "qkv_allgather",
            "source": "roofline",
            "time_ns": round(qkv_ag_ns, 2),
            "msg_bytes_per_cube": chunk,
            "d2d_bytes_all_cubes": d2d,
            "group": f"tp_s={g}, {ng} groups (ring AG)",
        })

    # 2. Attn comm (AstraSim — score/attn reduction)
    # hmp_reo_new uses analytical attn_rs instead of AstraSim attn_comm
    as_attn_comm = mods.get("attn_comm", 0)
    if as_attn_comm > 0 and strategy != "hmp_reo_new":
        scaled_time = as_attn_comm * comm_batch_scale
        attn_bytes = int(mods.get("attn_comm_bytes", 0)) * comm_batch_scale
        d2d = _ring_ar_d2d(g, attn_bytes) * ng
        gid = int(mods.get("attn_comm_group_id", 0))
        src = "astrasim" if comm_batch_scale == 1 else f"astrasim×{comm_batch_scale}"
        ops.append({
            "name": "attn_comm",
            "source": src,
            "time_ns": round(scaled_time, 2),
            "msg_bytes": attn_bytes,
            "d2d_bytes_all_cubes": d2d,
            "group": f"g{gid} ({g} GPUs, {ng} groups, ring AR)",
        })

    # 2b. Attn RS (roofline, hmp_reo_new only — ReduceScatter after attention)
    attn_rs_ns = attn_rs_rf.get("time_ns", 0)
    if strategy == "hmp_reo_new" and attn_rs_ns > 0:
        total_elems = attn_rs_rf.get("total_elems", 0)
        d2d = _ring_rs_d2d(g, total_elems) * ng
        ops.append({
            "name": "attn_rs",
            "source": "roofline",
            "time_ns": round(attn_rs_ns, 2),
            "total_elems": total_elems,
            "d2d_bytes_all_cubes": d2d,
            "group": f"tp_s={g}, {ng} groups (ring RS, block2x2 1-hop)",
        })

    # 3. Output comm
    if strategy == "hmp":
        # 3a. Output AR on g17 (roofline — block2x2 棋盘 2-hop, no congestion)
        output_ar_ns = output_ar_rf.get("time_ns", 0)
        as_output_ar = mods.get("output_ar_comm", 0)
        if output_ar_ns > 0:
            ar_msg = int(output_ar_rf.get("msg_elems", 0))
            d2d = _ring_ar_d2d(g, ar_msg) * ng
            ops.append({
                "name": "output_ar_comm",
                "source": "roofline",
                "time_ns": round(output_ar_ns, 2),
                "time_ns_astrasim_original": round(as_output_ar, 2),
                "msg_bytes": ar_msg,
                "d2d_bytes_all_cubes": d2d,
                "group": f"tp_s={g}, {ng} groups (ring AR, block2x2 1-hop, no congestion)",
            })
        # 3b. Output AG on g21 (roofline, corrected — no congestion)
        output_ag_ns = output_ag_rf.get("time_ns", 0)
        as_output_ag = mods.get("output_ag_comm", 0)
        chunk = output_ag_rf.get("per_cube_chunk_elems", 0)
        d2d = _ring_ag_d2d(g, chunk) * ng if chunk > 0 else 0
        ops.append({
            "name": "output_ag_comm",
            "source": "roofline (corrected)",
            "time_ns": round(output_ag_ns, 4),
            "time_ns_astrasim_original": round(as_output_ag, 2),
            "msg_bytes_per_cube": chunk,
            "d2d_bytes_all_cubes": d2d,
            "group": f"tp_s={g}, {ng} groups (ring AG, no congestion)",
        })
    elif strategy == "tp16":
        # TP16: output has 2 ARs (g17 + g21), aggregated as output_comm
        as_output_comm = mods.get("output_comm", 0)
        if as_output_comm > 0:
            scaled_time = as_output_comm * comm_batch_scale
            out_bytes = int(mods.get("output_comm_bytes", 0)) * comm_batch_scale
            d2d = _ring_ar_d2d(g, out_bytes) * ng
            src = "astrasim" if comm_batch_scale == 1 else f"astrasim×{comm_batch_scale}"
            ops.append({
                "name": "output_comm",
                "source": src,
                "time_ns": round(scaled_time, 2),
                "msg_bytes_total": out_bytes,
                "d2d_bytes_all_cubes": d2d,
                "group": f"2×AR on g17+g21 ({g} GPUs, {ng} groups each)",
            })

    # 4. Final reduce (roofline, HMP_reo / hmp_reo_new)
    final_red_ns = final_red_rf.get("time_ns", 0)
    if strategy in ("HMP_reo", "hmp_reo_new") and final_red_ns > 0:
        msg = final_red_rf.get("msg_elems_per_step", 0)
        n_total = g * ng
        d2d = _tree_reduce_d2d(n_total, msg)
        ops.append({
            "name": "final_reduce",
            "source": "roofline",
            "time_ns": round(final_red_ns, 2),
            "msg_bytes_per_step": msg,
            "d2d_bytes_all_cubes": d2d,
            "group": f"tree-reduce {n_total} NPUs",
        })

    total_comm_ns = sum(op["time_ns"] for op in ops)
    total_d2d = sum(op.get("d2d_bytes_all_cubes", 0) for op in ops)

    return {
        "ops": ops,
        "total_comm_ns": round(total_comm_ns, 2),
        "total_d2d_bytes_all_cubes": total_d2d,
    }


def build_aggregate_metrics(strategy: str, rf_entry: dict, rf_strat: dict,
                            comm_bd: dict, hybrid_gpu_ns: float,
                            hybrid_wall_ns: float) -> dict:
    """Build aggregate metrics: D2D, HBM read, compute ops, total time."""
    cfg = rf_strat.get("config", {})
    tp_h = cfg.get("tp_h", 1)
    tp_s = cfg.get("tp_s", 1)
    n_total = tp_h * tp_s

    qkv = rf_entry.get("Proj_QKV", {})
    attn = rf_entry.get("attention", {})
    output = rf_entry.get("Proj_O", {})

    hbm_per_cube = (qkv.get("weight_elems", 0) + qkv.get("activation_elems", 0)
                    + output.get("weight_elems", 0) + output.get("activation_elems", 0))

    if "sub_ops" in attn:
        hbm_per_cube += attn["sub_ops"].get("concat_k", {}).get("elems", 0)
        hbm_per_cube += attn["sub_ops"].get("concat_v", {}).get("elems", 0)
    elif "kv_cache_elems" in attn:
        hbm_per_cube += attn["kv_cache_elems"]

    compute_macs_per_cube = qkv.get("total_macs", 0) + output.get("total_macs", 0)
    if "sub_ops" in attn:
        compute_macs_per_cube += attn["sub_ops"].get("score_gemv", {}).get("macs", 0)
        compute_macs_per_cube += attn["sub_ops"].get("attn_v_gemv", {}).get("macs", 0)
    elif "einsum_macs" in attn:
        compute_macs_per_cube += attn["einsum_macs"]

    return {
        "d2d_link_transfer_bytes_all_cubes": comm_bd.get("total_d2d_bytes_all_cubes", 0),
        "hbm_read_bytes_per_cube": hbm_per_cube,
        "hbm_read_bytes_all_cubes": hbm_per_cube * n_total,
        "compute_macs_per_cube": compute_macs_per_cube,
        "compute_flops_per_cube": compute_macs_per_cube * 2,
        "compute_flops_all_cubes": compute_macs_per_cube * 2 * n_total,
        "n_cubes": n_total,
        "total_time_ns": round(hybrid_wall_ns, 2),
    }


def merge_entry(strategy: str, as_entry: dict, rf_entry: dict,
                rf_strat: dict, util_data: dict | None = None) -> dict:
    """Merge a single (strategy, seq) pair."""
    mods = as_entry.get("modules", {})
    as_qkv = mods.get("qkv_comp", 0)
    as_attn = mods.get("attn_comp", 0)
    as_output = mods.get("output_comp", 0)

    rf_qkv_mem = rf_entry.get("Proj_QKV", {}).get("memory_ns", 0)
    rf_out_mem = rf_entry.get("Proj_O", {}).get("memory_ns", 0)
    rf_attn = rf_entry.get("attention", {})
    rf_attn_fused = rf_attn.get("total_ns_fused", rf_attn.get("total_ns", 0))

    # Utilization-adjusted roofline values
    util_adj = None
    if util_data is not None:
        batch = rf_entry.get("batch", 1)
        seq = as_entry["seq"]
        cache_seq = rf_attn.get("cache_seq", seq)
        # dk used in this strategy's attention GEMM (d_head_full after AG, or d_head)
        rf_cfg = rf_strat.get("config", {})
        dk = rf_cfg.get("d_head_full", rf_cfg.get("d_head", 128))
        # dk_ref: the d_head that utilization.json was profiled with (full dk=128)
        dk_ref = 128
        util_adj = apply_utilization(rf_entry, util_data, batch, cache_seq,
                                     dk=dk, dk_ref=dk_ref)
        rf_qkv_effective = util_adj["qkv_adj_total_ns"]
        rf_attn_fused_effective = util_adj["attn_adj_total_fused_ns"]
        rf_out_effective = util_adj["output_adj_total_ns"]
    else:
        rf_qkv_effective = rf_qkv_mem
        rf_attn_fused_effective = rf_attn_fused
        rf_out_effective = rf_out_mem

    hybrid_qkv = max(as_qkv, rf_qkv_effective)
    hybrid_attn = max(as_attn, rf_attn_fused_effective)
    hybrid_output = max(as_output, rf_out_effective)
    hybrid_gpu = hybrid_qkv + hybrid_attn + hybrid_output

    rf_batch = rf_entry.get("batch", 1)
    comm_bd = build_comm_breakdown(strategy, mods, rf_entry, rf_strat,
                                   comm_batch_scale=rf_batch)
    comm_total = comm_bd["total_comm_ns"]
    hybrid_wall = hybrid_gpu + comm_total

    agg = build_aggregate_metrics(strategy, rf_entry, rf_strat,
                                  comm_bd, hybrid_gpu, hybrid_wall)

    compute_bd = {
        "Proj_QKV_source": ("roofline_util_adj" if util_adj and rf_qkv_effective >= as_qkv
                            else "roofline_weight_load" if rf_qkv_effective >= as_qkv
                            else "astrasim_compute"),
        "Proj_QKV_astrasim_ns": round(as_qkv, 2),
        "Proj_QKV_roofline_mem_ns": round(rf_qkv_mem, 2),
        "attn_source": ("roofline_util_adj" if util_adj and rf_attn_fused_effective >= as_attn
                        else "roofline_kv_cache_fused" if rf_attn_fused_effective >= as_attn
                        else "astrasim_compute"),
        "attn_astrasim_ns": round(as_attn, 2),
        "attn_roofline_fused_ns": round(rf_attn_fused, 2),
        "Proj_O_source": ("roofline_util_adj" if util_adj and rf_out_effective >= as_output
                          else "roofline_weight_load" if rf_out_effective >= as_output
                          else "astrasim_compute"),
        "Proj_O_astrasim_ns": round(as_output, 2),
        "Proj_O_roofline_mem_ns": round(rf_out_mem, 2),
    }

    if util_adj is not None:
        compute_bd["utilization_applied"] = True
        compute_bd["Proj_QKV_utilization"] = util_adj["qkv_utilization"]
        compute_bd["Proj_QKV_roofline_util_adj_ns"] = util_adj["qkv_adj_total_ns"]
        compute_bd["attn_roofline_util_adj_ns"] = util_adj["attn_adj_total_fused_ns"]
        if "score_utilization" in util_adj:
            compute_bd["score_utilization"] = util_adj["score_utilization"]
            compute_bd["attn_v_utilization"] = util_adj["attn_v_utilization"]
        elif "fused_utilization" in util_adj:
            compute_bd["fused_attn_utilization"] = util_adj["fused_utilization"]
        compute_bd["Proj_O_utilization"] = util_adj["output_utilization"]
        compute_bd["Proj_O_roofline_util_adj_ns"] = util_adj["output_adj_total_ns"]
    else:
        compute_bd["utilization_applied"] = False

    return {
        "seq": as_entry["seq"],
        "hybrid_Proj_QKV_ns": round(hybrid_qkv, 2),
        "hybrid_attn_ns": round(hybrid_attn, 2),
        "hybrid_Proj_O_ns": round(hybrid_output, 2),
        "hybrid_gpu_ns": round(hybrid_gpu, 2),
        "comm_total_ns": round(comm_total, 2),
        "hybrid_wall_ns": round(hybrid_wall, 2),
        "comm_breakdown": comm_bd,
        "aggregate_metrics": agg,
        "compute_breakdown": compute_bd,
        "original_astrasim": {
            "wall_ns": as_entry.get("wall_ns", 0),
            "gpu_ns": as_entry.get("gpu_ns", 0),
            "comm_ns": as_entry.get("comm_ns", 0),
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Merge AstraSim + roofline data into hybrid GQA estimate."
    )
    parser.add_argument("--astrasim-data", required=True,
                        help="collect_gqa_data.py output JSON")
    parser.add_argument("--roofline-data", required=True,
                        help="roofline_gqa_calc.py output JSON")
    parser.add_argument("-o", "--output", default="reports/hybrid/gqa_hybrid_merged.json",
                        help="Output path for merged JSON")
    parser.add_argument("--utilization", default=None,
                        help="Path to utilization.json (model-specific GPU SM "
                             "utilization data). When provided, roofline "
                             "compute_ns is adjusted by compute_ns/utilization "
                             "before hybrid max().")
    parser.add_argument("--tp16-utilization", default=None,
                        help="Path to TP16-specific utilization JSON. "
                             "When provided, tp16 strategy uses this instead "
                             "of --utilization (different QKV GEMM dimensions).")
    parser.add_argument("--rubin-utilization", default=None,
                        help="Path to H100 rubin utilization JSON. Applied to "
                             "Rubin total roofline time per module as "
                             "time_ns / utilization (captures real GPU overhead).")
    args = parser.parse_args()

    astrasim = load_json(args.astrasim_data)
    roofline = load_json(args.roofline_data)

    def _load_util(path_str, label):
        if not path_str:
            return None
        p = Path(path_str)
        if not p.exists():
            print(f"[ERROR] {label} file not found: {p}", file=sys.stderr)
            sys.exit(1)
        data = load_json(str(p))
        print(f"Loaded {label} from: {p}")
        return data

    util_data = _load_util(args.utilization, "utilization")
    tp16_util_data = _load_util(args.tp16_utilization, "TP16 utilization")
    rubin_util_data = _load_util(args.rubin_utilization, "Rubin utilization")

    as_lookup = build_astrasim_lookup(astrasim)
    rf_lookup = build_roofline_lookup(roofline)
    rf_strategies = roofline.get("strategies", {})

    rf_meta = roofline.get("metadata", {})
    device_hw = rf_meta.get("device_hw", {})

    merged_strategies = ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]

    if util_data:
        sources = {
            "Proj_QKV": "max(AstraSim compute, max(roofline compute_ns/util, roofline memory_ns))",
            "attn": "max(AstraSim compute, sum(max(comp_ns/util, mem_ns) per sub-op))",
            "Proj_O": "max(AstraSim compute, max(roofline compute_ns/util, roofline memory_ns))",
        }
    else:
        sources = {
            "Proj_QKV": "max(AstraSim compute, roofline weight_load)",
            "attn": "max(AstraSim compute, roofline KV_cache_fused)",
            "Proj_O": "max(AstraSim compute, roofline weight_load)",
        }
    sources.update({
        "comm_HMP_reo": "roofline: qkv_allgather + final_reduce",
        "comm_hmp_reo_new": "roofline: qkv_allgather + attn_rs + final_reduce",
        "comm_hmp": "roofline: qkv_ag + output_ar + output_ag; astrasim: attn_comm",
        "comm_tp16": "astrasim: attn_comm + output_comm",
    })

    meta = {
        "_sources": sources,
        "astrasim_file": args.astrasim_data,
        "roofline_file": args.roofline_data,
        "device_hw": device_hw,
        "batch_size": rf_meta.get("batch_size", 1),
    }
    if util_data:
        meta["utilization_file"] = args.utilization
        meta["utilization_model"] = (
            "compute_ns / gpu_utilization per module; "
            "proj_qkv/proj_o indexed by batch_size; "
            "score/attn_v indexed by effective_seq=cache_seq*dk/dk_ref"
        )
    if tp16_util_data:
        meta["tp16_utilization_file"] = args.tp16_utilization
    if rubin_util_data:
        meta["rubin_utilization_file"] = args.rubin_utilization
        meta["rubin_utilization_model"] = (
            "H100 HBM BW utilization from e2e profiling; "
            "applied to Rubin roofline total_ns as total_ns / bw_util"
        )

    result = {
        "metadata": meta,
        "strategies": {},
    }

    for strat in merged_strategies:
        if strat not in as_lookup:
            print(f"[WARN] Strategy '{strat}' not in AstraSim data, skipping",
                  file=sys.stderr)
            continue
        if strat not in rf_lookup:
            print(f"[WARN] Strategy '{strat}' not in roofline data, skipping",
                  file=sys.stderr)
            continue

        rf_strat = rf_strategies.get(strat, {})
        merged_entries = []
        for seq in sorted(as_lookup[strat]):
            if seq not in rf_lookup[strat]:
                print(f"[WARN] seq={seq} not in roofline for {strat}, skipping",
                      file=sys.stderr)
                continue
            # TP16 uses its own utilization file if provided
            strat_util = (tp16_util_data if strat == "tp16" and tp16_util_data
                          else util_data)
            entry = merge_entry(strat, as_lookup[strat][seq],
                                rf_lookup[strat][seq], rf_strat,
                                util_data=strat_util)
            merged_entries.append(entry)

        result["strategies"][strat] = {"data": merged_entries}

    if "rubin" in rf_lookup:
        rubin_strat = rf_strategies.get("rubin", {})
        rubin_cfg = rubin_strat.get("config", {})
        rubin_entries = []
        for seq in sorted(rf_lookup["rubin"]):
            rr = rf_lookup["rubin"][seq]
            batch = rr.get("batch", 1)
            qkv_ns_raw = rr.get("Proj_QKV", {}).get("total_ns", 0)
            attn_data = rr.get("attention", {})
            attn_ns_raw = attn_data.get("total_ns", attn_data.get("total_ns_fused", 0))
            out_ns_raw = rr.get("Proj_O", {}).get("total_ns", 0)

            # Apply H100 BW utilization to Rubin roofline
            rubin_util_info = {}
            if rubin_util_data:
                qkv_u = _lookup_utilization(rubin_util_data["proj_qkv"], batch)
                attn_u = _lookup_utilization(rubin_util_data["attn_fused"], seq)
                out_u = _lookup_utilization(rubin_util_data["proj_o"], batch)
                qkv_ns = qkv_ns_raw / qkv_u if qkv_u > 0 else qkv_ns_raw
                attn_ns = attn_ns_raw / attn_u if attn_u > 0 else attn_ns_raw
                out_ns = out_ns_raw / out_u if out_u > 0 else out_ns_raw
                rubin_util_info = {
                    "utilization_applied": True,
                    "Proj_QKV_bw_util": round(qkv_u, 4),
                    "Proj_QKV_roofline_ns": round(qkv_ns_raw, 2),
                    "attn_bw_util": round(attn_u, 4),
                    "attn_roofline_ns": round(attn_ns_raw, 2),
                    "Proj_O_bw_util": round(out_u, 4),
                    "Proj_O_roofline_ns": round(out_ns_raw, 2),
                }
            else:
                qkv_ns, attn_ns, out_ns = qkv_ns_raw, attn_ns_raw, out_ns_raw
                rubin_util_info = {"utilization_applied": False}

            gpu_ns = qkv_ns + attn_ns + out_ns

            qkv_d = rr.get("Proj_QKV", {})
            out_d = rr.get("Proj_O", {})
            hbm = (qkv_d.get("weight_elems", 0) + qkv_d.get("activation_elems", 0)
                   + out_d.get("weight_elems", 0) + out_d.get("activation_elems", 0))
            if "kv_cache_elems" in attn_data:
                hbm += attn_data["kv_cache_elems"]
            macs = qkv_d.get("total_macs", 0) + out_d.get("total_macs", 0)
            if "einsum_macs" in attn_data:
                macs += attn_data["einsum_macs"]

            entry = {
                "seq": seq,
                "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
                "hybrid_attn_ns": round(attn_ns, 2),
                "hybrid_Proj_O_ns": round(out_ns, 2),
                "hybrid_gpu_ns": round(gpu_ns, 2),
                "comm_total_ns": 0,
                "hybrid_wall_ns": round(gpu_ns, 2),
                "comm_breakdown": {"ops": [], "total_comm_ns": 0,
                                   "total_d2d_bytes_all_cubes": 0},
                "aggregate_metrics": {
                    "d2d_link_transfer_bytes_all_cubes": 0,
                    "hbm_read_bytes_per_cube": hbm,
                    "hbm_read_bytes_all_cubes": hbm,
                    "compute_macs_per_cube": macs,
                    "compute_flops_per_cube": macs * 2,
                    "compute_flops_all_cubes": macs * 2,
                    "n_cubes": 1,
                    "total_time_ns": round(gpu_ns, 2),
                },
                "source": "roofline (single GPU, no communication)",
            }
            entry.update(rubin_util_info)
            rubin_entries.append(entry)
        result["strategies"]["rubin"] = {"data": rubin_entries}

    # H100 single GPU (same pattern as Rubin, same H100 BW util)
    if "h100" in rf_lookup:
        h100_entries = []
        for seq in sorted(rf_lookup["h100"]):
            rr = rf_lookup["h100"][seq]
            batch = rr.get("batch", 1)
            qkv_ns_raw = rr.get("Proj_QKV", {}).get("total_ns", 0)
            attn_data = rr.get("attention", {})
            attn_ns_raw = attn_data.get("total_ns", attn_data.get("total_ns_fused", 0))
            out_ns_raw = rr.get("Proj_O", {}).get("total_ns", 0)

            h100_util_info = {}
            if rubin_util_data:
                qkv_u = _lookup_utilization(rubin_util_data["proj_qkv"], batch)
                attn_u = _lookup_utilization(rubin_util_data["attn_fused"], seq)
                out_u = _lookup_utilization(rubin_util_data["proj_o"], batch)
                qkv_ns = qkv_ns_raw / qkv_u if qkv_u > 0 else qkv_ns_raw
                attn_ns = attn_ns_raw / attn_u if attn_u > 0 else attn_ns_raw
                out_ns = out_ns_raw / out_u if out_u > 0 else out_ns_raw
                h100_util_info = {
                    "utilization_applied": True,
                    "Proj_QKV_bw_util": round(qkv_u, 4),
                    "Proj_QKV_roofline_ns": round(qkv_ns_raw, 2),
                    "attn_bw_util": round(attn_u, 4),
                    "attn_roofline_ns": round(attn_ns_raw, 2),
                    "Proj_O_bw_util": round(out_u, 4),
                    "Proj_O_roofline_ns": round(out_ns_raw, 2),
                }
            else:
                qkv_ns, attn_ns, out_ns = qkv_ns_raw, attn_ns_raw, out_ns_raw
                h100_util_info = {"utilization_applied": False}

            gpu_ns = qkv_ns + attn_ns + out_ns
            qkv_d = rr.get("Proj_QKV", {})
            out_d = rr.get("Proj_O", {})
            hbm = (qkv_d.get("weight_elems", 0) + qkv_d.get("activation_elems", 0)
                   + out_d.get("weight_elems", 0) + out_d.get("activation_elems", 0))
            if "kv_cache_elems" in attn_data:
                hbm += attn_data["kv_cache_elems"]
            macs = qkv_d.get("total_macs", 0) + out_d.get("total_macs", 0)
            if "einsum_macs" in attn_data:
                macs += attn_data["einsum_macs"]

            entry = {
                "seq": seq,
                "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
                "hybrid_attn_ns": round(attn_ns, 2),
                "hybrid_Proj_O_ns": round(out_ns, 2),
                "hybrid_gpu_ns": round(gpu_ns, 2),
                "comm_total_ns": 0,
                "hybrid_wall_ns": round(gpu_ns, 2),
                "comm_breakdown": {"ops": [], "total_comm_ns": 0,
                                   "total_d2d_bytes_all_cubes": 0},
                "aggregate_metrics": {
                    "d2d_link_transfer_bytes_all_cubes": 0,
                    "hbm_read_bytes_per_cube": hbm,
                    "hbm_read_bytes_all_cubes": hbm,
                    "compute_macs_per_cube": macs,
                    "compute_flops_per_cube": macs * 2,
                    "compute_flops_all_cubes": macs * 2,
                    "n_cubes": 1,
                    "total_time_ns": round(gpu_ns, 2),
                },
                "source": "h100 (single GPU, no communication)",
            }
            entry.update(h100_util_info)
            h100_entries.append(entry)
        result["strategies"]["h100"] = {"data": h100_entries}

    # --- Summary table ---
    all_strategies = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin", "h100"]
    all_seqs = set()
    for strat in all_strategies:
        for e in result.get("strategies", {}).get(strat, {}).get("data", []):
            all_seqs.add(e["seq"])

    summary_table = []
    for seq in sorted(all_seqs):
        row = {"seq": seq}
        for strat in all_strategies:
            sd = result.get("strategies", {}).get(strat, {}).get("data", [])
            match = [e for e in sd if e["seq"] == seq]
            if match:
                e = match[0]
                row[strat] = {
                    "Proj_QKV_ns": round(e["hybrid_Proj_QKV_ns"], 1),
                    "attn_ns": round(e["hybrid_attn_ns"], 1),
                    "Proj_O_ns": round(e["hybrid_Proj_O_ns"], 1),
                    "gpu_ns": round(e["hybrid_gpu_ns"], 1),
                    "comm_ns": round(e.get("comm_total_ns", 0), 1),
                    "wall_ns": round(e["hybrid_wall_ns"], 1),
                }
        summary_table.append(row)
    result["summary_table"] = summary_table

    # --- Print ---
    print("\n=== Hybrid GQA Estimation (with comm breakdown) ===\n")
    col = f"{'ProjQKV':>7} {'Attn':>7} {'ProjO':>7} {'GPU':>7} {'Comm':>7} {'Wall':>7}"
    print(f"{'seq':>8} | {col} | {col} | {col} | {col} | {col}")
    print(f"{'':>8} | {'--- HMP_reo ---':^43} | {'--- reo_new ---':^43} | {'--- HMP ---':^43}"
          f" | {'--- TP16 ---':^43} | {'--- Rubin ---':^43}")
    print("-" * 240)

    for row in summary_table:
        seq = row["seq"]
        parts = []
        for strat in all_strategies:
            if strat in row:
                s = row[strat]
                parts.append(f"{s['Proj_QKV_ns']:>7.0f} {s['attn_ns']:>7.0f} {s['Proj_O_ns']:>7.0f}"
                             f" {s['gpu_ns']:>7.0f} {s['comm_ns']:>7.0f} {s['wall_ns']:>7.0f}")
            else:
                parts.append(f"{'N/A':>43}")
        print(f"{seq:>8} | {' | '.join(parts)}")

    # --- Comm breakdown detail ---
    print("\n--- Comm breakdown per strategy ---")
    for strat in merged_strategies:
        sd = result.get("strategies", {}).get(strat, {}).get("data", [])
        if sd:
            e = sd[-1]
            print(f"\n[{strat.upper()}] seq={e['seq']}:")
            for op in e.get("comm_breakdown", {}).get("ops", []):
                print(f"  {op['name']:>20}: {op['time_ns']:>10.2f} ns"
                      f"  D2D={op.get('d2d_bytes_all_cubes', 0):>10,} bytes"
                      f"  [{op['source']}]")
            cb = e.get("comm_breakdown", {})
            print(f"  {'TOTAL':>20}: {cb.get('total_comm_ns', 0):>10.2f} ns"
                  f"  D2D={cb.get('total_d2d_bytes_all_cubes', 0):>10,} bytes")

    # --- Aggregate metrics ---
    print("\n--- Aggregate metrics (last seq) ---")
    for strat in all_strategies:
        sd = result.get("strategies", {}).get(strat, {}).get("data", [])
        if sd:
            e = sd[-1]
            agg = e.get("aggregate_metrics", {})
            print(f"[{strat.upper():>8}] D2D={agg.get('d2d_link_transfer_bytes_all_cubes', 0):>12,}B"
                  f"  HBM_read={agg.get('hbm_read_bytes_all_cubes', 0):>14,}B"
                  f"  FLOPs={agg.get('compute_flops_all_cubes', 0):>16,}"
                  f"  wall={agg.get('total_time_ns', 0):>10.1f}ns")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
