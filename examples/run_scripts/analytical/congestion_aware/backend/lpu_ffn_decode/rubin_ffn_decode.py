#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rubin_ffn_decode.py — FFN decode latency on an **NVIDIA GPU (Rubin)**, the
GPU-side counterpart of ffn_decode.py (which targets the LPU).

This is the "FFN-on-GPU" piece of the two decode baselines:
    - GPU+GPU : attention on Rubin  + FFN on Rubin  (this file)
    - GPU+LPU : attention on Rubin  + FFN on LPU    (ffn_decode.py)

==========================================================================
Modeling
--------------------------------------------------------------------------
Same roofline decomposition as ffn_decode.py (identical FLOPs / weight-byte
accounting, identical MoE expert OI=1 / TP sharding), with two differences:

  1) Hardware = Rubin GPU: peak FP8 compute 17500 TFLOPS, HBM 22 TB/s
     (roofline/hardware_config.py:rubin_single_layer_config).

  2) **Real-card utilization derating** (the whole point of this baseline).
     A pure roofline assumes 100% efficiency. Real GPUs realize only a
     fraction of peak on tiny decode GEMVs. We divide each roofline edge by
     the measured fraction (data/gpu_ffn_utilization.json, from H100 ncu):
         t_mem     = weight_bytes / (peak_BW   * TP * mem_util)
         t_compute = flops        / (peak_FLOP * TP * compute_util)
         t_stage   = max(t_mem, t_compute)
     This mirrors exactly how the attention side derates the Rubin roofline
     with h100_rubin_utilization.json ("total_ns / bw_util"). The assumption
     (kernel efficiency transfers H100 -> Rubin) is the same in both places.

  3) TP AllReduce over NVLink (FFN down-proj output), small-message latency
     dominated at low batch (parametric, see GPUSpec).
==========================================================================
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple

from lpu_config import FFNModelSpec, RunSpec   # reuse the model library + RunSpec

# trace 测得的「真实 distinct 激活专家数」曲线（exp5 用；由 expert-selection trace
# 蒙特卡洛采样得到，见 contbatch/data/expert_activation.json）。比解析上界
# min(B·top_k, n_experts) 低很多（专家选择有重合/偏斜）。
_EXPERT_ACT = None
def _trace_weight_reads(model_name: str, B: int, n_experts: int, top_k: int) -> float:
    global _EXPERT_ACT
    if _EXPERT_ACT is None:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "contbatch", "data", "expert_activation.json")
        try:
            _EXPERT_ACT = json.load(open(p)).get("distinct_experts", {})
        except Exception:
            _EXPERT_ACT = {}
    curve = _EXPERT_ACT.get(model_name, {})
    if str(B) in curve:
        return float(curve[str(B)])
    return float(min(B * top_k, n_experts))   # 回退：解析上界


# ---------------------------------------------------------------------------
# Rubin GPU hardware spec (mirrors roofline/hardware_config.py:rubin_single_layer)
# ---------------------------------------------------------------------------
@dataclass
class GPUSpec:
    name: str = "rubin"
    compute_TFLOPs: float = 17500.0    # FP8 dense Tensor Core peak (Rubin spec)
    hbm_bw_TBs: float = 22.0           # HBM bandwidth (= T-elements/s at FP8)
    capacity_GB: float = 288.0
    power_W: float = 2200.0
    # NVLink C2C for TP AllReduce
    nvlink_bw_GBs: float = 1800.0      # per-direction device link BW (1.8 TB/s)
    allreduce_lat_ns: float = 900.0    # fixed small-message AllReduce latency (NVLink C2C)
    # data types
    w_bytes: int = 1                   # FP8 weights
    act_bytes: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


RUBIN = GPUSpec()


# ---------------------------------------------------------------------------
# Utilization profile loader (data/gpu_ffn_utilization.json)
# ---------------------------------------------------------------------------
_UTIL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "gpu_ffn_utilization.json")


class FFNUtil:
    """Maps an FFN stage name -> (mem_util, compute_util) from the real-card JSON.

    MoE util is per-model (depends on per-expert weight size): prefer
    moe_by_model[model_name], fall back to the generic 'moe' block.
    """

    # dense stage-name -> key in profile["dense"]
    _DENSE = {"gate": "gate_up_proj", "up": "gate_up_proj", "down": "down_proj"}
    # MoE stage-name -> key in the moe block
    _MOE = {
        "gating": "gating",
        "expert_gate": "expert_gate_up", "expert_up": "expert_gate_up",
        "expert_down": "expert_down",
        "shared_gate": "shared_gate_up", "shared_up": "shared_gate_up",
        "shared_down": "shared_down",
    }

    def __init__(self, path: str = _UTIL_PATH, model_name: str = None):
        with open(path) as f:
            self.prof = json.load(f)
        self.path = path
        self.model_name = model_name
        # pick the MoE util block for this model (per-model measured, else generic)
        by_model = self.prof.get("moe_by_model", {})
        self.moe_block = by_model.get(model_name) if model_name else None
        if self.moe_block is None:
            self.moe_block = self.prof["moe"]

    def get(self, stage: str) -> Tuple[float, float]:
        if stage in self._DENSE:
            e = self.prof["dense"][self._DENSE[stage]]
        else:
            key = self._MOE.get(stage, "expert_gate_up")
            e = self.moe_block.get(key) or self.prof["moe"][key]
        return float(e["mem_util"]), float(e["compute_util"])


# ---------------------------------------------------------------------------
# roofline primitive (with utilization derating)
# ---------------------------------------------------------------------------
def _roofline_stage(name: str, flops: float, weight_bytes: float,
                    act_bytes: float, gpu: GPUSpec, tp: int,
                    mem_util: float, compute_util: float) -> Dict[str, Any]:
    """Single GEMV/GEMM stage roofline on Rubin, derated by real-card utilization."""
    peak_flops_s = gpu.compute_TFLOPs * 1e12 * tp * max(compute_util, 1e-9)
    mem_bw_Bs = gpu.hbm_bw_TBs * 1e12 * tp * max(mem_util, 1e-9)

    t_compute = flops / peak_flops_s if peak_flops_s > 0 else 0.0
    bytes_total = weight_bytes + act_bytes
    t_mem = bytes_total / mem_bw_Bs if mem_bw_Bs > 0 else 0.0

    t = max(t_compute, t_mem)
    return {
        "name": name,
        "flops": flops,
        "weight_bytes": weight_bytes,
        "act_bytes": act_bytes,
        "mem_util": mem_util,
        "compute_util": compute_util,
        "t_compute_ns": t_compute * 1e9,
        "t_mem_ns": t_mem * 1e9,
        "t_ns": t * 1e9,
        "bound": "mem" if t_mem >= t_compute else "compute",
    }


def _allreduce_nvlink(payload_bytes: float, gpu: GPUSpec, run: RunSpec) -> Dict[str, Any]:
    """FFN down-proj TP AllReduce over NVLink: ring model, fixed-latency dominated
    at low batch. t = lat + 2(TP-1)/TP * payload / bw."""
    tp = run.TP
    if tp <= 1:
        return {"name": "AllReduce", "payload_bytes": 0.0, "t_ns": 0.0}
    lat = gpu.allreduce_lat_ns * 1e-9
    bw_term = (2.0 * (tp - 1) / tp) * payload_bytes / (gpu.nvlink_bw_GBs * 1e9)
    t = lat + bw_term
    return {"name": "AllReduce", "payload_bytes": payload_bytes,
            "eff_bw_GBs": gpu.nvlink_bw_GBs, "t_ns": t * 1e9}


# ---------------------------------------------------------------------------
# Dense FFN (SwiGLU)
# ---------------------------------------------------------------------------
def dense_ffn_stages(model: FFNModelSpec, gpu: GPUSpec, run: RunSpec,
                     util: FFNUtil) -> List[Dict[str, Any]]:
    B, d, d_ff = run.batch, model.d_model, model.d_ff
    w, a = gpu.w_bytes, gpu.act_bytes
    n_up = 2 if model.gated else 1
    stages: List[Dict[str, Any]] = []

    for i in range(n_up):
        nm = "gate" if (model.gated and i == 0) else "up"
        mu, cu = util.get(nm)
        flops = 2.0 * B * d * d_ff
        wbytes = d * d_ff * w
        act = B * d * a + B * d_ff * a
        stages.append(_roofline_stage(nm, flops, wbytes, act, gpu, run.TP, mu, cu))

    mu, cu = util.get("down")
    flops = 2.0 * B * d_ff * d
    wbytes = d_ff * d * w
    act = B * d_ff * a + B * d * a
    stages.append(_roofline_stage("down", flops, wbytes, act, gpu, run.TP, mu, cu))

    stages.append(_allreduce_nvlink(B * d * a, gpu, run))
    return stages


# ---------------------------------------------------------------------------
# MoE FFN (expert OI=1, same accounting as ffn_decode.moe_ffn_stages)
# ---------------------------------------------------------------------------
def moe_ffn_stages(model: FFNModelSpec, gpu: GPUSpec, run: RunSpec,
                   util: FFNUtil) -> List[Dict[str, Any]]:
    B, d, di = run.batch, model.d_model, model.d_moe_inter
    w, a = gpu.w_bytes, gpu.act_bytes
    top_k = model.top_k
    n_up = 2 if model.gated else 1
    stages: List[Dict[str, Any]] = []

    # 1) router / gating
    mu, cu = util.get("gating")
    flops = 2.0 * B * d * model.n_experts
    wbytes = d * model.n_experts * w
    act = B * d * a + B * model.n_experts * a
    stages.append(_roofline_stage("gating", flops, wbytes, act, gpu, run.TP, mu, cu))

    # 2) weight-reuse factor (OI=1 per_token vs batched upper bound)
    #   batched_membound: 同 batched 的权重封顶(min(B·top_k, n_experts))，并在 decode 区
    #   强制 expert/shared 走 memory-bound(见函数末尾)。主流观测：MoE decode 全程
    #   weight-load 主导，compute-bound 需 ~万级 token；本模型的 GEMV-util compute 项会在
    #   B~数十就假性 compute-bound，故在该模式下剔除。
    if run.expert_mode == "per_token":
        weight_reads = B * top_k
        compute_tokens = B * top_k
    elif run.expert_mode in ("batched", "batched_membound"):
        weight_reads = min(B * top_k, model.n_experts)
        compute_tokens = B * top_k
    elif run.expert_mode == "trace":
        # 真实 distinct 激活专家数（trace 标定）；FLOPs 仍按 B·top_k 个 token-expert 对
        weight_reads = _trace_weight_reads(model.name, B, model.n_experts, top_k)
        compute_tokens = B * top_k
    else:
        raise ValueError(f"unknown expert_mode: {run.expert_mode}")

    # 3) routed experts gate/up/down
    for i in range(n_up):
        nm = "expert_gate" if (model.gated and i == 0) else "expert_up"
        mu, cu = util.get(nm)
        flops = 2.0 * compute_tokens * d * di
        wbytes = weight_reads * (d * di) * w
        act = compute_tokens * (d + di) * a
        stages.append(_roofline_stage(nm, flops, wbytes, act, gpu, run.TP, mu, cu))

    mu, cu = util.get("expert_down")
    flops = 2.0 * compute_tokens * di * d
    wbytes = weight_reads * (di * d) * w
    act = compute_tokens * (di + d) * a
    stages.append(_roofline_stage("expert_down", flops, wbytes, act, gpu, run.TP, mu, cu))

    # 4) shared expert (dense small FFN, all tokens reuse the weight once)
    if model.n_shared_experts > 0:
        ns = model.n_shared_experts
        for i in range(n_up):
            nm = "shared_gate" if (model.gated and i == 0) else "shared_up"
            mu, cu = util.get(nm)
            flops = 2.0 * B * d * (di * ns)
            wbytes = (d * di * ns) * w
            act = B * (d + di * ns) * a
            stages.append(_roofline_stage(nm, flops, wbytes, act, gpu, run.TP, mu, cu))
        mu, cu = util.get("shared_down")
        flops = 2.0 * B * (di * ns) * d
        wbytes = (di * ns * d) * w
        act = B * (di * ns + d) * a
        stages.append(_roofline_stage("shared_down", flops, wbytes, act, gpu, run.TP, mu, cu))

    # 5) TP AllReduce
    stages.append(_allreduce_nvlink(B * d * a, gpu, run))

    # batched_membound / trace: decode 区 MoE expert/shared 强制 memory-bound(权重读为瓶颈)
    if run.expert_mode in ("batched_membound", "trace"):
        for s in stages:
            if s["name"].startswith(("expert_", "shared_")):
                s["t_ns"] = s["t_mem_ns"]
                s["bound"] = "mem"
    return stages


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def simulate_layer(model: FFNModelSpec, gpu: GPUSpec, run: RunSpec,
                   util: FFNUtil = None) -> Dict[str, Any]:
    """One FFN layer on Rubin. Returns {stages, layer_total_ns, bound_breakdown_ns}."""
    if util is None:
        util = FFNUtil()
    stages = (moe_ffn_stages if model.is_moe else dense_ffn_stages)(model, gpu, run, util)
    layer_total = sum(s["t_ns"] for s in stages)

    breakdown = {"compute": 0.0, "mem": 0.0, "comm": 0.0}
    for s in stages:
        if s["name"] == "AllReduce":
            breakdown["comm"] += s["t_ns"]
        elif s.get("bound") == "compute":
            breakdown["compute"] += s["t_ns"]
        else:
            breakdown["mem"] += s["t_ns"]

    return {"stages": stages, "layer_total_ns": layer_total,
            "bound_breakdown_ns": breakdown}


def simulate_decode(model: FFNModelSpec, gpu: GPUSpec, run: RunSpec,
                    util: FFNUtil = None) -> Dict[str, Any]:
    layer = simulate_layer(model, gpu, run, util)
    n = model.num_layers
    ffn_tpot_ns = layer["layer_total_ns"] * n
    return {
        "per_layer": layer,
        "decode": {
            "num_layers": n,
            "per_layer_ffn_ns": layer["layer_total_ns"],
            "ffn_tpot_ns": ffn_tpot_ns,
            "ffn_tpot_us": ffn_tpot_ns / 1e3,
            "bound_breakdown_per_layer_ns": layer["bound_breakdown_ns"],
        },
    }
