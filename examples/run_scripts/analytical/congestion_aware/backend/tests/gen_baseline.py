#!/usr/bin/env python3
"""Baseline test-data generator for the 13-function utilization/power API.

Captures snapshots of the CURRENT (pre-refactor) implementation so the new
`utilization_lib.py` can be regression-tested.

Source-of-truth modules used here:
  - backend/roofline/roofline_gqa_calc.py
      roofline_mla_qkv / roofline_mla_attention / roofline_mla_output
      make_mla_npu_config
  - backend/roofline/utilization_profiles/deepseek3/util_{N}.json   (AMMA SA util)
  - backend/roofline/utilization_profiles/h100/h100_mla_profile.json (Rubin BW util)
  - backend/hybrid_merge/power_model.py:POWER_COEFFS

What this dataset COVERS (1-7 of the requested 13 functions):
  - compute_utilization_qkv / memory_utilization_qkv / power_qkv
  - compute_utilization_o / memory_utilization_o / power_o
  - compute_utilization_attention / memory_utilization_attention / power_attention

What this dataset does NOT cover (8-13):
  - gating / combine / moe_per_expert
  No baseline exists; these must be designed from scratch and added to
  `baseline_dataset.json` after the new implementation is in place.
"""
import sys
import json
import math
import bisect
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "roofline"))
sys.path.insert(0, str(BACKEND / "hybrid_merge"))

from roofline_gqa_calc import (
    roofline_mla_qkv, roofline_mla_attention, roofline_mla_output,
    make_mla_npu_config,
)
from power_model import POWER_COEFFS


# ---------------------------------------------------------------------------
# Hardware test configurations (subset; B200 omitted, no POWER_COEFFS yet)
# ---------------------------------------------------------------------------

HW_CFGS = {
    "amma": {
        "arch": "amma",
        "peak_per_npu_tflops": 48,
        "bw_per_npu_tbs": 2.5,
        "TP": 16,
        "tp_h": 4, "tp_s": 4,
        "num_sa_for_util": 64,
        "power_key": "ours",
    },
    "rubin": {
        "arch": "rubin",
        "peak_per_npu_tflops": 17500,
        "bw_per_npu_tbs": 22,
        "TP": 1,
        "tp_h": 1, "tp_s": 1,
        "power_key": "rubin",
    },
}

UTIL_DIR = BACKEND / "roofline" / "utilization_profiles"


# ---------------------------------------------------------------------------
# Utility profile lookup (log2-linear interp; matches deepseek_v3_mla_roofline)
# ---------------------------------------------------------------------------

def _lookup(profile, key):
    parsed = {}
    for k, v in profile.items():
        if str(k).startswith("_"):
            continue
        parsed[int(k)] = v["utilization"] if isinstance(v, dict) else float(v)
    if key in parsed:
        return parsed[key]
    ks = sorted(parsed)
    if key <= ks[0]:
        return parsed[ks[0]]
    if key >= ks[-1]:
        return parsed[ks[-1]]
    i = bisect.bisect_right(ks, key) - 1
    lo, hi = ks[i], ks[i + 1]
    t = (math.log2(key) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return parsed[lo] + t * (parsed[hi] - parsed[lo])


def amma_utils(num_sa, bs, cache_seq):
    path = UTIL_DIR / "deepseek3" / f"util_{num_sa}.json"
    data = json.loads(path.read_text())
    return {
        "proj_qkv": _lookup(data["proj_qkv"], bs),
        "score":    _lookup(data["score"], cache_seq),
        "attention": _lookup(data["attention"], cache_seq),
        "proj_o":   _lookup(data["proj_o"], bs),
    }


def rubin_utils(bs, seq):
    path = UTIL_DIR / "h100" / "h100_mla_profile.json"
    data = json.loads(path.read_text())["bw_util"]
    return {
        "proj_qkv":  _lookup(data["proj_qkv"], bs),
        "attention": _lookup(data["attn_absorbed"], seq),
        "proj_o":    _lookup(data["proj_o"], bs),
    }


# ---------------------------------------------------------------------------
# Case generator
# ---------------------------------------------------------------------------

def gen_case(hw_name, hw, bs, seq):
    cfg = make_mla_npu_config(tp_h=hw["tp_h"], tp_s=hw["tp_s"])
    pp = hw["peak_per_npu_tflops"]
    bw = hw["bw_per_npu_tbs"]
    arch = hw["arch"]
    TP = hw["TP"]
    cache_seq = seq // TP

    qkv = roofline_mla_qkv(pp, bw, cfg, bs)
    attn = roofline_mla_attention(pp, bw, cfg, bs, seq)
    o = roofline_mla_output(pp, bw, cfg, bs)

    qkv_compute = qkv["compute_ns"]
    qkv_mem = qkv["memory_ns"]
    qkv_flops = 2 * qkv["total_macs"]
    qkv_bytes = qkv["weight_elems"] + qkv["activation_elems"]

    o_compute = o["compute_ns"]
    o_mem = o["memory_ns"]
    o_flops = 2 * o["total_macs"]
    o_bytes = o["weight_elems"] + o["activation_elems"]

    score_comp = attn["sub_ops"]["score_gemv"]["compute_ns"]
    score_mem = attn["sub_ops"]["score_gemv"]["memory_ns"]
    attnv_comp = attn["sub_ops"]["attn_v_gemv"]["compute_ns"]
    attn_total_compute = score_comp + attnv_comp
    attn_total_mem = score_mem  # KV unified, value reuses cache (attn_v_mem = 0)

    if arch == "amma":
        u = amma_utils(hw["num_sa_for_util"], bs, cache_seq)
        qkv_profile = u["proj_qkv"]
        qkv_realized = max(qkv_compute / qkv_profile, qkv_mem)
        attn_profile = (u["score"] + u["attention"]) / 2
        attn_realized = max(attn_total_compute / attn_profile, attn_total_mem)
        o_profile = u["proj_o"]
        o_realized = max(o_compute / o_profile, o_mem)
        # AMMA convention: compute_util = SA profile (efficiency cap);
        #                  memory_util  = realized bytes / time / peak_bw
        qkv_cu, qkv_mu = qkv_profile, qkv_mem / qkv_realized
        attn_cu, attn_mu = attn_profile, (attn_total_mem / attn_realized if attn_realized > 0 else 0)
        o_cu, o_mu = o_profile, o_mem / o_realized
    else:  # rubin
        u = rubin_utils(bs, seq)
        # Rubin convention: BW-bound decode, realized_time = ideal_total_ns / bw_profile
        qkv_profile = u["proj_qkv"]
        attn_profile = u["attention"]
        o_profile = u["proj_o"]
        qkv_realized = qkv["total_ns"] / qkv_profile
        attn_realized = attn["total_ns"] / attn_profile
        o_realized = o["total_ns"] / o_profile
        qkv_cu, qkv_mu = 0.0, qkv_profile
        attn_cu, attn_mu = 0.0, attn_profile
        o_cu, o_mu = 0.0, o_profile

    pc = POWER_COEFFS[hw["power_key"]]
    def _power(cu, mu):
        return pc["static_w"] + pc["cmpt_coeff"] * cu + pc["mem_coeff"] * mu

    return {
        "id": f"{hw_name}_bs{bs}_seq{seq}",
        "inputs": {
            "arch": arch,
            "bs": bs,
            "seq_len": seq,
            "TP": TP,
            "peak_per_npu_tflops": pp,
            "bw_per_npu_tbs": bw,
            "cache_seq": cache_seq,
        },
        "qkv": {
            "flops": qkv_flops,
            "bytes": qkv_bytes,
            "compute_ns_ideal": round(qkv_compute, 6),
            "memory_ns": round(qkv_mem, 6),
            "profile_util": round(qkv_profile, 6),
            "expected_realized_time_ns": round(qkv_realized, 6),
            "expected_compute_util": round(qkv_cu, 6),
            "expected_memory_util": round(qkv_mu, 6),
            "expected_power_w": round(_power(qkv_cu, qkv_mu), 4),
        },
        "attention": {
            "kv_cache_elems": attn["kv_cache_elems"],
            "compute_ns_ideal": round(attn_total_compute, 6),
            "memory_ns": round(attn_total_mem, 6),
            "profile_util": round(attn_profile, 6),
            "expected_realized_time_ns": round(attn_realized, 6),
            "expected_compute_util": round(attn_cu, 6),
            "expected_memory_util": round(attn_mu, 6),
            "expected_power_w": round(_power(attn_cu, attn_mu), 4),
        },
        "proj_o": {
            "flops": o_flops,
            "bytes": o_bytes,
            "compute_ns_ideal": round(o_compute, 6),
            "memory_ns": round(o_mem, 6),
            "profile_util": round(o_profile, 6),
            "expected_realized_time_ns": round(o_realized, 6),
            "expected_compute_util": round(o_cu, 6),
            "expected_memory_util": round(o_mu, 6),
            "expected_power_w": round(_power(o_cu, o_mu), 4),
        },
    }


def main():
    bs_list = [1, 4, 16, 32]
    seq_list = [1024, 8192, 65536, 131072]

    cases = []
    for hw_name, hw in HW_CFGS.items():
        for bs in bs_list:
            for seq in seq_list:
                cases.append(gen_case(hw_name, hw, bs, seq))

    out = {
        "_description": "Baseline dataset for the 13-function utilization/power API regression test.",
        "_purpose": "Verify new utilization_lib.py reproduces these numbers within 1e-4 relative tolerance.",
        "_source_code": {
            "roofline": "backend/roofline/roofline_gqa_calc.py",
            "power_coeffs": "backend/hybrid_merge/power_model.py:POWER_COEFFS",
            "amma_util": "backend/roofline/utilization_profiles/deepseek3/util_{num_sa}.json",
            "rubin_util": "backend/roofline/utilization_profiles/h100/h100_mla_profile.json",
        },
        "_conventions": {
            "amma": "compute_util = SA profile; memory_util = realized = mem_ns / realized_time_ns;"
                    " realized_time = max(compute_ns / profile_util, mem_ns)",
            "rubin": "compute_util = 0 (decode bs=1 BW-bound); memory_util = BW profile;"
                     " realized_time = ideal_total_ns / bw_profile",
            "power": "P_kernel_w = static_w + cmpt_coeff × U_cmpt + mem_coeff × U_mem"
                     "  (D2D not included in this baseline)",
        },
        "_grid": {
            "batch_sizes": bs_list,
            "seq_lengths": seq_list,
            "hardware": list(HW_CFGS.keys()),
        },
        "_coverage": {
            "qkv":            "OK",
            "attention":      "OK",
            "proj_o":         "OK",
            "gating":         "NOT COVERED — no existing impl, design from scratch",
            "combine":        "NOT COVERED — no existing impl, design from scratch",
            "moe_per_expert": "NOT COVERED — no existing impl, design from scratch",
        },
        "_unresolved": [
            "AMMA D2D power omitted (per-kernel comm bytes not part of single roofline call).",
            "roofline_mla_qkv: num_attention_heads/n_npus may double-count tp_s vs n_npus —"
            " baseline preserves the existing arithmetic as-is.",
            "B200: no POWER_COEFFS and no BW util profile yet — excluded from baseline.",
            "Rubin compute_util fixed at 0 (decode BW-bound assumption); does not generalise"
            " to prefill/large-bs use cases.",
        ],
        "hw_configs": HW_CFGS,
        "test_cases": cases,
    }

    out_path = Path(__file__).resolve().parent / "baseline_dataset.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {len(cases)} cases to {out_path}")


if __name__ == "__main__":
    main()
