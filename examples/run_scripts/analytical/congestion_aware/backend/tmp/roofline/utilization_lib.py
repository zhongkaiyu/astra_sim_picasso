#!/usr/bin/env python3
"""Unified compute / memory / power utilization API for Rubin / AMMA / B200.

13-function API (see backend/plan.md §2)
==========================================
Attention block:
  compute_utilization_qkv(config, bs, hw, TP)         memory_utilization_qkv(config, bs, hw, TP)
  compute_utilization_o(config, bs, hw, TP)           memory_utilization_o(config, bs, hw, TP)
  compute_utilization_attention(config, seq, hw, TP)  memory_utilization_attention(config, seq, hw, TP)
  power_qkv / power_attention / power_o (mem_util, compute_util, hw)
MoE block:
  compute_utilization_gating(config, bs, hw)          memory_utilization_gating(config, bs, hw)
  compute_utilization_combine(config, bs, hw)         memory_utilization_combine(config, bs, hw)
  compute_utilization_moe_per_expert(config, bs, hw)  memory_utilization_moe_per_expert(config, bs, hw)
  power_gating / power_combine / power_moe_per_expert (mem_util, compute_util, hw)

Conventions
-----------
  arch="amma":  compute_util = SA-profile util (efficiency cap on this kernel shape);
                memory_util  = realized bytes / (realized_time × peak_bw)
  arch="rubin"/"b200":
                compute_util = 0 (decode bs=1 is BW-bound);
                memory_util  = measured BW-utilization profile

Power
  P_kernel_w = static_w + cmpt_coeff × U_cmpt + mem_coeff × U_mem
"""
import bisect
import json
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "hybrid_merge"))

import roofline_gqa_calc as rgc
from power_model import POWER_COEFFS


# ---------------------------------------------------------------------------
# Model defaults — DeepSeek-V3 MLA + MoE
# ---------------------------------------------------------------------------

DEEPSEEK_V3_DEFAULTS = {
    "name": "deepseek_v3",
    # MLA
    "d_model": 7168,
    "n_h": 128,
    "d_c": 512,
    "d_c_prime": 1536,
    "d_r": 64,
    "d_score": 576,
    "d_value": 512,
    # MoE
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "top_k": 8,
    "d_moe_inter": 2048,
}


# ---------------------------------------------------------------------------
# Hardware helpers
# ---------------------------------------------------------------------------

_ARCH_POWER_KEY = {"amma": "ours", "rubin": "rubin", "b200": "h100"}
_ARCH_TP_H = {"amma": 4, "rubin": 1, "b200": 1}


def _arch(hw):           return hw.get("arch", "amma")
def _peak_tflops(hw):    return hw["peak_per_npu_tflops"]
def _peak_bw_tbs(hw):    return hw["bw_per_npu_tbs"]


def _power_coeffs(hw):
    key = hw.get("power_key") or _ARCH_POWER_KEY[_arch(hw)]
    return POWER_COEFFS[key]


def _tp_factor(hw, TP):
    """Decompose TP -> (tp_h, tp_s). Prefer explicit hw.tp_h / hw.tp_s, otherwise
    use arch default head-group factor."""
    if "tp_h" in hw and "tp_s" in hw:
        return hw["tp_h"], hw["tp_s"]
    th = min(_ARCH_TP_H.get(_arch(hw), 1), TP)
    return th, max(1, TP // th)


# ---------------------------------------------------------------------------
# Profile loading + log2-linear interpolation
# ---------------------------------------------------------------------------

_UTIL_CACHE = {}
_PROFILE_DIR = _HERE / "utilization_profiles"


def _interp(profile, key):
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


def _amma_profile(num_sa):
    k = f"amma_{num_sa}"
    if k not in _UTIL_CACHE:
        _UTIL_CACHE[k] = json.loads(
            (_PROFILE_DIR / "deepseek3" / f"util_{num_sa}.json").read_text()
        )
    return _UTIL_CACHE[k]


def _rubin_profile():
    if "rubin" not in _UTIL_CACHE:
        _UTIL_CACHE["rubin"] = json.loads(
            (_PROFILE_DIR / "h100" / "h100_mla_profile.json").read_text()
        )["bw_util"]
    return _UTIL_CACHE["rubin"]


def _b200_profile():
    # TODO: real B200 profile; fall back to Rubin BW util.
    return _rubin_profile()


def _non_amma_profile(arch):
    return _b200_profile() if arch == "b200" else _rubin_profile()


# ---------------------------------------------------------------------------
# Realized-time helper (AMMA convention)
# ---------------------------------------------------------------------------

def _amma_realized_ns(compute_ns, memory_ns, profile_util):
    if profile_util <= 0:
        return max(compute_ns, memory_ns)
    return max(compute_ns / profile_util, memory_ns)


# ===========================================================================
# 1-2, 7a:  QKV
# ===========================================================================

def _qkv_roofline(hw, TP, bs):
    th, ts = _tp_factor(hw, TP)
    cfg = rgc.make_mla_npu_config(tp_h=th, tp_s=ts)
    return rgc.roofline_mla_qkv(_peak_tflops(hw), _peak_bw_tbs(hw), cfg, bs)


def compute_utilization_qkv(config, bs, hw, TP):
    if _arch(hw) == "amma":
        return _interp(_amma_profile(hw["num_sa_for_util"])["proj_qkv"], bs)
    return 0.0


def memory_utilization_qkv(config, bs, hw, TP):
    arch = _arch(hw)
    r = _qkv_roofline(hw, TP, bs)
    if arch == "amma":
        sa = _interp(_amma_profile(hw["num_sa_for_util"])["proj_qkv"], bs)
        realized = _amma_realized_ns(r["compute_ns"], r["memory_ns"], sa)
        return r["memory_ns"] / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["proj_qkv"], bs)


def power_qkv(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ===========================================================================
# 3-4, 7b:  ProjO
# ===========================================================================

def _o_roofline(hw, TP, bs):
    th, ts = _tp_factor(hw, TP)
    cfg = rgc.make_mla_npu_config(tp_h=th, tp_s=ts)
    return rgc.roofline_mla_output(_peak_tflops(hw), _peak_bw_tbs(hw), cfg, bs)


def compute_utilization_o(config, bs, hw, TP):
    if _arch(hw) == "amma":
        return _interp(_amma_profile(hw["num_sa_for_util"])["proj_o"], bs)
    return 0.0


def memory_utilization_o(config, bs, hw, TP):
    arch = _arch(hw)
    r = _o_roofline(hw, TP, bs)
    if arch == "amma":
        sa = _interp(_amma_profile(hw["num_sa_for_util"])["proj_o"], bs)
        realized = _amma_realized_ns(r["compute_ns"], r["memory_ns"], sa)
        return r["memory_ns"] / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["proj_o"], bs)


def power_o(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ===========================================================================
# 5-6, 7c:  Attention
# ===========================================================================

def _attn_roofline(hw, TP, seq, bs=1):
    th, ts = _tp_factor(hw, TP)
    cfg = rgc.make_mla_npu_config(tp_h=th, tp_s=ts)
    return rgc.roofline_mla_attention(_peak_tflops(hw), _peak_bw_tbs(hw), cfg, bs, seq)


def compute_utilization_attention(config, seq, hw, TP):
    arch = _arch(hw)
    th, ts = _tp_factor(hw, TP)
    cache_seq = seq // (th * ts)
    if arch == "amma":
        prof = _amma_profile(hw["num_sa_for_util"])
        return (_interp(prof["score"], cache_seq) + _interp(prof["attention"], cache_seq)) / 2
    return 0.0


def memory_utilization_attention(config, seq, hw, TP):
    arch = _arch(hw)
    # mu(seq) is bs-independent (both compute_ns and mem_ns scale linearly).
    r = _attn_roofline(hw, TP, seq, bs=1)
    score_c = r["sub_ops"]["score_gemv"]["compute_ns"]
    attnv_c = r["sub_ops"]["attn_v_gemv"]["compute_ns"]
    score_m = r["sub_ops"]["score_gemv"]["memory_ns"]
    total_c = score_c + attnv_c
    if arch == "amma":
        sa = compute_utilization_attention(config, seq, hw, TP)
        realized = _amma_realized_ns(total_c, score_m, sa)
        return score_m / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["attn_absorbed"], seq)


def power_attention(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ===========================================================================
# 8-9, 7d:  MoE Gating (router GEMM)
# ===========================================================================
#   Input:  x[bs, d_model]
#   Weight: W_gate[d_model, n_routed_experts]
#   Output: logits[bs, n_routed_experts]
#   No TP sharding in our model (gating is small enough to replicate).

def _gating_flops_bytes(config, bs):
    d = config["d_model"]
    E = config["n_routed_experts"]
    flops = 2 * bs * d * E
    bytes_ = bs * d + d * E + bs * E   # in + W + out  (FP8 = 1 B/elem)
    return flops, bytes_


def _gating_roofline(config, bs, hw):
    flops, bytes_ = _gating_flops_bytes(config, bs)
    compute_ns = flops / (_peak_tflops(hw) * 1e12) * 1e9
    memory_ns = bytes_ / (_peak_bw_tbs(hw) * 1e12) * 1e9
    return compute_ns, memory_ns, flops, bytes_


def compute_utilization_gating(config, bs, hw):
    # Gating ≈ small GEMM; reuse proj_qkv SA profile at `bs`.
    if _arch(hw) == "amma":
        return _interp(_amma_profile(hw["num_sa_for_util"])["proj_qkv"], bs)
    return 0.0


def memory_utilization_gating(config, bs, hw):
    arch = _arch(hw)
    comp_ns, mem_ns, _, _ = _gating_roofline(config, bs, hw)
    if arch == "amma":
        sa = _interp(_amma_profile(hw["num_sa_for_util"])["proj_qkv"], bs)
        realized = _amma_realized_ns(comp_ns, mem_ns, sa)
        return mem_ns / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["proj_qkv"], bs)


def power_gating(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ===========================================================================
# 10-11, 7e:  MoE Combine (weighted sum of top_k expert outputs)
# ===========================================================================
#   Memory-bound scatter-add — compute util ~ 0 across all archs.

def _combine_flops_bytes(config, bs):
    d = config["d_model"]
    k = config["top_k"]
    flops = 2 * bs * k * d        # mul + add per element
    bytes_ = bs * k * d + bs * k + bs * d   # k inputs + k weights + 1 output
    return flops, bytes_


def _combine_roofline(config, bs, hw):
    flops, bytes_ = _combine_flops_bytes(config, bs)
    compute_ns = flops / (_peak_tflops(hw) * 1e12) * 1e9
    memory_ns = bytes_ / (_peak_bw_tbs(hw) * 1e12) * 1e9
    return compute_ns, memory_ns, flops, bytes_


def compute_utilization_combine(config, bs, hw):
    return 0.0   # pure memory-bound op


def memory_utilization_combine(config, bs, hw):
    comp_ns, mem_ns, _, _ = _combine_roofline(config, bs, hw)
    realized = max(comp_ns, mem_ns)
    return min(1.0, mem_ns / realized) if realized > 0 else 0.0


def power_combine(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ===========================================================================
# 12-13, 7f:  MoE per-expert (SwiGLU FFN: gate / up / down)
# ===========================================================================
#   AMMA = OUR strategy: expert weights tensor-sharded across 16 NPUs.
#   Rubin / B200: full weights on a single device (EP-style).
#   tokens_per_expert = bs * top_k / n_routed_experts  (fractional in decode bs=1).

def _moe_tp(hw):
    return 16 if _arch(hw) == "amma" else 1


def _moe_per_expert_flops_bytes(config, bs, TP):
    d = config["d_model"]
    di = config["d_moe_inter"]
    k = config["top_k"]
    E = config["n_routed_experts"]
    tokens = bs * k / E
    # 3 GEMMs (gate, up — col-parallel; down — row-parallel)
    flops = 3 * 2 * tokens * d * (di / TP)
    weight_bytes = 3 * d * (di / TP)
    act_bytes = tokens * d + tokens * di + tokens * d
    return flops, weight_bytes + act_bytes


def _moe_per_expert_roofline(config, bs, hw):
    TP = _moe_tp(hw)
    flops, bytes_ = _moe_per_expert_flops_bytes(config, bs, TP)
    compute_ns = flops / (_peak_tflops(hw) * 1e12) * 1e9
    memory_ns = bytes_ / (_peak_bw_tbs(hw) * 1e12) * 1e9
    return compute_ns, memory_ns, flops, bytes_


def compute_utilization_moe_per_expert(config, bs, hw):
    if _arch(hw) == "amma":
        tokens = max(
            1, int(math.ceil(bs * config["top_k"] / config["n_routed_experts"]))
        )
        return _interp(_amma_profile(hw["num_sa_for_util"])["proj_qkv"], tokens)
    return 0.0


def memory_utilization_moe_per_expert(config, bs, hw):
    arch = _arch(hw)
    comp_ns, mem_ns, _, _ = _moe_per_expert_roofline(config, bs, hw)
    if arch == "amma":
        sa = compute_utilization_moe_per_expert(config, bs, hw)
        realized = _amma_realized_ns(comp_ns, mem_ns, sa)
        return mem_ns / realized if realized > 0 else 0.0
    # Rubin / B200: assume the per-expert FFN is BW-bound; use proj_qkv BW util.
    return _interp(_non_amma_profile(arch)["proj_qkv"], 1)


def power_moe_per_expert(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ---------------------------------------------------------------------------

# ===========================================================================
# GQA API (Qwen3 / Llama4) — mirrors the MLA QKV / Attention / O block.
# ===========================================================================
# Profile lives in utilization_profiles/<model>/util_<num_sa>.json with the
# same shape as the MLA profile: proj_qkv / proj_o keyed by bs, score /
# attention keyed by effective_seq.  Per-NPU GQA roofline configs come from
# roofline_gqa_calc.make_strategy_config(strategy, tp_h, tp_hd, model_config).

QWEN3_DEFAULTS = {
    "name": "qwen3",
    "d_model": 4096,
    "num_attention_heads": 64,
    "num_kv_heads": 4,
    "d_head": 128,
    "num_layers": 94,
}

LLAMA4_DEFAULTS = {
    "name": "llama4",
    "d_model": 5120,
    "num_attention_heads": 40,
    "num_kv_heads": 8,
    "d_head": 128,
    "num_layers": 48,
}


def _gqa_profile(model_name, num_sa):
    k = f"gqa_{model_name}_{num_sa}"
    if k not in _UTIL_CACHE:
        _UTIL_CACHE[k] = json.loads(
            (_PROFILE_DIR / model_name / f"util_{num_sa}.json").read_text()
        )
    return _UTIL_CACHE[k]


def _gqa_cfg(config, hw, TP, strategy):
    th, ts = _tp_factor(hw, TP)
    model_cfg = {
        "d_model": config["d_model"],
        "num_attention_heads": config["num_attention_heads"],
        "num_kv_heads": config["num_kv_heads"],
        "d_head": config["d_head"],
        "num_layers": config.get("num_layers", 1),
    }
    return rgc.make_strategy_config(strategy, tp_h=th, tp_hd=ts, model_config=model_cfg)


def _gqa_effective_seq(config, cfg, seq):
    # effective_seq = cache_seq * dk_local / dk_ref — normalizes the SA-util
    # lookup across strategies whose (cache_seq, dk) differ but whose total
    # MACs are equal (TP16 uses dk=32, HMP/HMP_reo use dk_full=128).
    tp_s = cfg.get("tp_s", 1)
    cache_seq = seq // tp_s if tp_s > 1 else seq
    dk_used = cfg.get("d_head_full", cfg["d_head"])
    dk_ref = config["d_head"]
    return max(1, cache_seq * dk_used // dk_ref)


# --- GQA QKV ---------------------------------------------------------------

def _gqa_qkv_roofline(config, hw, TP, strategy, bs):
    cfg = _gqa_cfg(config, hw, TP, strategy)
    return rgc.roofline_qkv(_peak_tflops(hw), _peak_bw_tbs(hw), cfg, bs)


def compute_utilization_qkv_gqa(config, bs, hw, TP, strategy="HMP_reo"):
    if _arch(hw) == "amma":
        return _interp(_gqa_profile(config["name"], hw["num_sa_for_util"])["proj_qkv"], bs)
    return 0.0


def memory_utilization_qkv_gqa(config, bs, hw, TP, strategy="HMP_reo"):
    arch = _arch(hw)
    r = _gqa_qkv_roofline(config, hw, TP, strategy, bs)
    if arch == "amma":
        sa = _interp(_gqa_profile(config["name"], hw["num_sa_for_util"])["proj_qkv"], bs)
        realized = _amma_realized_ns(r["compute_ns"], r["memory_ns"], sa)
        return r["memory_ns"] / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["proj_qkv"], bs)


def power_qkv_gqa(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# --- GQA ProjO -------------------------------------------------------------

def _gqa_o_roofline(config, hw, TP, strategy, bs):
    cfg = _gqa_cfg(config, hw, TP, strategy)
    return rgc.roofline_output(_peak_tflops(hw), _peak_bw_tbs(hw), cfg, bs)


def compute_utilization_o_gqa(config, bs, hw, TP, strategy="HMP_reo"):
    if _arch(hw) == "amma":
        return _interp(_gqa_profile(config["name"], hw["num_sa_for_util"])["proj_o"], bs)
    return 0.0


def memory_utilization_o_gqa(config, bs, hw, TP, strategy="HMP_reo"):
    arch = _arch(hw)
    r = _gqa_o_roofline(config, hw, TP, strategy, bs)
    if arch == "amma":
        sa = _interp(_gqa_profile(config["name"], hw["num_sa_for_util"])["proj_o"], bs)
        realized = _amma_realized_ns(r["compute_ns"], r["memory_ns"], sa)
        return r["memory_ns"] / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["proj_o"], bs)


def power_o_gqa(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# --- GQA Attention ---------------------------------------------------------

def _gqa_attn_roofline(config, hw, TP, strategy, seq, bs=1):
    cfg = _gqa_cfg(config, hw, TP, strategy)
    return rgc.roofline_attention(_peak_tflops(hw), _peak_bw_tbs(hw), cfg, bs, seq)


def compute_utilization_attention_gqa(config, seq, hw, TP, strategy="HMP_reo"):
    if _arch(hw) != "amma":
        return 0.0
    cfg = _gqa_cfg(config, hw, TP, strategy)
    eff_seq = _gqa_effective_seq(config, cfg, seq)
    prof = _gqa_profile(config["name"], hw["num_sa_for_util"])
    return (_interp(prof["score"], eff_seq) + _interp(prof["attention"], eff_seq)) / 2


def memory_utilization_attention_gqa(config, seq, hw, TP, strategy="HMP_reo"):
    arch = _arch(hw)
    r = _gqa_attn_roofline(config, hw, TP, strategy, seq, bs=1)
    if "sub_ops" in r:
        sc = r["sub_ops"]["score_gemv"]
        av = r["sub_ops"]["attn_v_gemv"]
        total_c = sc["compute_ns"] + av["compute_ns"]
        total_m = sc["memory_ns"] + av["memory_ns"]
    else:
        total_c = r["einsum_comp_ns"]
        total_m = r["kv_mem_ns"]
    if arch == "amma":
        sa = compute_utilization_attention_gqa(config, seq, hw, TP, strategy)
        realized = _amma_realized_ns(total_c, total_m, sa)
        return total_m / realized if realized > 0 else 0.0
    return _interp(_non_amma_profile(arch)["attn_absorbed"], seq)


def power_attention_gqa(mem_util, compute_util, hw):
    pc = _power_coeffs(hw)
    return pc["static_w"] + pc["cmpt_coeff"] * compute_util + pc["mem_coeff"] * mem_util


# ---------------------------------------------------------------------------

__all__ = [
    "DEEPSEEK_V3_DEFAULTS",
    "compute_utilization_qkv", "memory_utilization_qkv", "power_qkv",
    "compute_utilization_o", "memory_utilization_o", "power_o",
    "compute_utilization_attention", "memory_utilization_attention", "power_attention",
    "compute_utilization_gating", "memory_utilization_gating", "power_gating",
    "compute_utilization_combine", "memory_utilization_combine", "power_combine",
    "compute_utilization_moe_per_expert", "memory_utilization_moe_per_expert", "power_moe_per_expert",
    # GQA (Qwen3 / Llama4)
    "QWEN3_DEFAULTS", "LLAMA4_DEFAULTS",
    "compute_utilization_qkv_gqa", "memory_utilization_qkv_gqa", "power_qkv_gqa",
    "compute_utilization_o_gqa", "memory_utilization_o_gqa", "power_o_gqa",
    "compute_utilization_attention_gqa", "memory_utilization_attention_gqa", "power_attention_gqa",
]
