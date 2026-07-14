#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ffn_decode.py — 用论文 SHIP/LPU 的方法，估算单层 FFN 的 decode 延迟（TPOT 分量）。

==========================================================================
⚠️ 利用率口径（与 rubin_ffn_decode.py 对比时务必注意）
--------------------------------------------------------------------------
本文件是【纯 roofline，效率隐含 = 100%】——LPU 在小 MoE GEMV 上被当作满带宽。
而 GPU 侧 rubin_ffn_decode.py 用【实测 util ~0.43-0.66】derate。二者口径不对称，
会高估 LPU FFN 优势约 1.8×（同 398.5MB 权重：Rubin 2018ns@util0.56 vs LPU 166ns@util1.0
= 12.2× = 6.8×[真实 SRAM/HBM 带宽] × 1.78×[util 不对称]）。且默认 LPX 150 TB/s 是
peak-derived（机架 40PB/s÷256），取 100% 本就乐观（LPUv1 18.4 TB/s 才是 sustained 值）。
→ 当前保留 100% 作为【LPU 乐观上界】（见 reports/baseline_gpu_decode_plan.md），结论对此 robust。

==========================================================================
论文 modeling 方式（本文件实现的核心）
--------------------------------------------------------------------------
1) Decode 是 memory-bound 的 roofline（论文 §2.2 / §3.2）：
       t_stage = max( FLOPs / peak_compute , bytes_loaded / mem_BW )
   FFN 不含 KV，故 bytes 主要是【权重读取】，decode 下几乎总是落在 mem 边。

2) SRAM 带宽主导（论文 §3.2 / Table 2）：
   LPU 把权重放片上 SRAM，带宽比 HBM 高 >10×，所以 weight-bound 时间
   = weight_bytes / (sram_bw_per_lpu × TP)。TP 把权重和带宽都切 TP 份。

3) MoE expert 的 OI=1（论文 §6 "Expert Imbalance"）：
   小 batch 下每个 token 独立执行其 top_k 个 expert，expert 权重被逐 token
   重新读取，算术强度 OI≈1。于是 MoE-FFN 的权重字节随 (batch × top_k) 线性增长。
   expert_mode="batched" 则给出 batch 内复用权重的乐观上界。

4) TP collective（论文 §4.2 / Fig.6）：
   FFN down-proj 之后需要一次 AllReduce。用 LPU 同步 C2C 模型估算：
       t_allreduce = diameter × hop_latency + 2(TP-1)/TP × payload / eff_bw
   其中 eff_bw 按论文 Fig.6a 的饱和曲线（32KiB→50%, 80KiB→90%）随 payload 调整。

5) PP（论文 §3 / §6）：逐层 FFN 串联，整模 FFN decode 延迟
       ffn_tpot = num_layers × per_layer_ffn_latency
   （PP 只改延迟的归属，不改单层关键路径，这里只做聚合报告。）
==========================================================================

输出：所有函数返回结构化 dict / list[dict]，便于直接 json.dump，字段含义见 run.py。
"""

from __future__ import annotations
import json
import os
from typing import Dict, Any, List

from lpu_config import LPUSpec, FFNModelSpec, RunSpec

# trace 测得的真实 distinct 激活专家数曲线（见 rubin_ffn_decode._trace_weight_reads）。
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
    return float(min(B * top_k, n_experts))


# ---------------------------------------------------------------------------
# 基础 roofline 原语
# ---------------------------------------------------------------------------
def _roofline_stage(name: str, flops: float, weight_bytes: float,
                    act_bytes: float, lpu: LPUSpec, tp: int) -> Dict[str, Any]:
    """
    单个 matmul stage 的 roofline 估算（已按 TP 切分聚合带宽/算力）。

    参数:
      flops        : 该 stage 的浮点运算数（2×MAC），已是「整个 TP 组」的总量。
      weight_bytes : 需从 SRAM 读取的权重字节（整个 TP 组总量）。
      act_bytes    : 激活读写字节（整个 TP 组总量，量级小，一并计入 mem）。
      tp           : TP 切分数 —— 算力和 SRAM 带宽都放大 tp 倍。
    返回: 含 compute/mem 两条边、取 max 的 t_ns，以及 bound 标记。
    """
    # 聚合资源：TP 颗 LPU 并行 → 峰值算力与带宽各乘 TP
    peak_flops_s = lpu.compute_TFLOPs * 1e12 * tp
    mem_bw_Bs = lpu.sram_bw_TBs * 1e12 * tp

    t_compute = flops / peak_flops_s if peak_flops_s > 0 else 0.0
    bytes_total = weight_bytes + act_bytes
    t_mem = bytes_total / mem_bw_Bs if mem_bw_Bs > 0 else 0.0

    t = max(t_compute, t_mem)
    return {
        "name": name,
        "flops": flops,
        "weight_bytes": weight_bytes,
        "act_bytes": act_bytes,
        "t_compute_ns": t_compute * 1e9,
        "t_mem_ns": t_mem * 1e9,
        "t_ns": t * 1e9,
        "bound": "mem" if t_mem >= t_compute else "compute",
    }


def _coll_eff_bw(payload_bytes: float, lpu: LPUSpec) -> float:
    """
    按论文 Fig.6a 的饱和曲线，估算 AllReduce 在给定 payload 下的【有效带宽】(B/s)。
    用 (32KiB,50%) 与 (80KiB,90%) 两拐点做分段线性插值，基准带宽用二分带宽。
    """
    kib = payload_bytes / 1024.0
    (k0, k1) = lpu.coll_knee_kib
    (f0, f1) = lpu.coll_knee_frac
    if kib <= k0:
        frac = f0 * (kib / k0) if k0 > 0 else f0      # 极小张量：线性退化到 0 附近
        frac = max(frac, 0.05)                         # 给个下限，避免除零式爆炸
    elif kib >= k1:
        frac = f1                                       # 大张量：饱和到 90%
    else:
        # 两拐点间线性插值
        frac = f0 + (f1 - f0) * (kib - k0) / (k1 - k0)
    return lpu.bisection_bw_GBs * 1e9 * frac


def _allreduce_c2c(payload_bytes: float, lpu: LPUSpec, run: RunSpec) -> Dict[str, Any]:
    """
    FFN down-proj 后的 TP AllReduce，用 LPU 同步 C2C 模型估算。

    t = diameter × hop_latency  +  2(TP-1)/TP × payload / eff_bw   （ring-allreduce）
        ^^^^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        固定跳延迟（论文 300ns/跳）   带宽项（有效带宽随 payload 饱和）
    TP=1 时无通信，返回 0。
    """
    tp = run.TP
    if tp <= 1:
        return {"name": "AllReduce", "payload_bytes": 0.0, "hops": 0,
                "eff_bw_GBs": 0.0, "t_ns": 0.0}

    eff_bw = _coll_eff_bw(payload_bytes, lpu)
    lat_hops = run.tp_diameter_hops * lpu.hop_latency_ns * 1e-9    # 固定延迟项 (s)
    bw_term = (2.0 * (tp - 1) / tp) * payload_bytes / eff_bw       # 带宽项 (s)
    t = lat_hops + bw_term
    return {
        "name": "AllReduce",
        "payload_bytes": payload_bytes,
        "hops": run.tp_diameter_hops,
        "eff_bw_GBs": eff_bw / 1e9,
        "t_ns": t * 1e9,
    }


# ---------------------------------------------------------------------------
# Dense FFN（SwiGLU 门控）
# ---------------------------------------------------------------------------
def dense_ffn_stages(model: FFNModelSpec, lpu: LPUSpec,
                     run: RunSpec) -> List[Dict[str, Any]]:
    """
    Dense FFN 单层的各 stage。门控 SwiGLU = gate + up + down 三个矩阵，
    沿中间维 d_ff 做 TP 切分（gate/up 列并行，down 行并行 → 末尾 AllReduce）。
    """
    B = run.batch
    d = model.d_model
    d_ff = model.d_ff
    w = lpu.w_bytes
    a = lpu.act_bytes

    # 门控有 gate+up 两个上投影；非门控只有 up 一个
    n_up = 2 if model.gated else 1
    stages: List[Dict[str, Any]] = []

    # up/gate 投影: (B, d) @ (d, d_ff) —— 权重 d×d_ff，整 TP 组总量
    for i in range(n_up):
        nm = "gate" if (model.gated and i == 0) else "up"
        flops = 2.0 * B * d * d_ff
        wbytes = d * d_ff * w
        act = B * d * a + B * d_ff * a            # 读输入 + 写中间激活
        stages.append(_roofline_stage(nm, flops, wbytes, act, lpu, run.TP))

    # down 投影: (B, d_ff) @ (d_ff, d)
    flops = 2.0 * B * d_ff * d
    wbytes = d_ff * d * w
    act = B * d_ff * a + B * d * a
    stages.append(_roofline_stage("down", flops, wbytes, act, lpu, run.TP))

    # TP AllReduce：payload = 输出激活 (B × d)
    stages.append(_allreduce_c2c(B * d * a, lpu, run))
    return stages


# ---------------------------------------------------------------------------
# MoE FFN
# ---------------------------------------------------------------------------
def moe_ffn_stages(model: FFNModelSpec, lpu: LPUSpec,
                   run: RunSpec) -> List[Dict[str, Any]]:
    """
    MoE FFN 单层的各 stage：router → (shared expert) → routed experts → AllReduce。

    关键：expert 权重读取量取决于 expert_mode（论文 §6, expert OI=1）：
      - per_token：每 token 独立读其 top_k expert 权重 → 权重字节 × (B × top_k)；
      - batched ：假设这些 token 复用权重 → 权重字节只算被激活的 expert 一次（上界乐观）。
    """
    B = run.batch
    d = model.d_model
    di = model.d_moe_inter
    w = lpu.w_bytes
    a = lpu.act_bytes
    top_k = model.top_k
    n_up = 2 if model.gated else 1
    stages: List[Dict[str, Any]] = []

    # --- 1) Router / gating: (B, d) @ (d, n_experts) ---
    flops = 2.0 * B * d * model.n_experts
    wbytes = d * model.n_experts * w
    act = B * d * a + B * model.n_experts * a
    stages.append(_roofline_stage("gating", flops, wbytes, act, lpu, run.TP))

    # --- 2) 权重重用因子：决定 expert 权重被读几遍 ---
    if run.expert_mode == "per_token":
        # 每 token 各读自己的 top_k expert 一次 → OI=1
        weight_reads = B * top_k
        compute_tokens = B * top_k
    elif run.expert_mode in ("batched", "batched_membound"):
        # batch 内复用：权重按「被激活的 distinct expert 数」上界 = min(B*top_k, n_experts) 读一次
        # batched_membound 额外在 decode 区强制 expert/shared 走 memory-bound（见函数末尾）
        weight_reads = min(B * top_k, model.n_experts)
        compute_tokens = B * top_k
    elif run.expert_mode == "trace":
        # 真实 distinct 激活专家数（trace 标定），FLOPs 仍按 B·top_k 个 token-expert 对
        weight_reads = _trace_weight_reads(model.name, B, model.n_experts, top_k)
        compute_tokens = B * top_k
    else:
        raise ValueError(f"未知 expert_mode: {run.expert_mode}")

    # --- 3) Routed experts 的 gate/up/down ---
    # 单个 expert 单 token 的一次矩阵: (1, d)@(d, di) 或 (1, di)@(di, d)
    for i in range(n_up):
        nm = "expert_gate" if (model.gated and i == 0) else "expert_up"
        flops = 2.0 * compute_tokens * d * di
        wbytes = weight_reads * (d * di) * w     # 逐 token 重读体现在 weight_reads
        act = compute_tokens * (d + di) * a
        stages.append(_roofline_stage(nm, flops, wbytes, act, lpu, run.TP))

    flops = 2.0 * compute_tokens * di * d
    wbytes = weight_reads * (di * d) * w
    act = compute_tokens * (di + d) * a
    stages.append(_roofline_stage("expert_down", flops, wbytes, act, lpu, run.TP))

    # --- 4) Shared expert（每 token 必走，等价 dense 小 FFN）---
    if model.n_shared_experts > 0:
        ns = model.n_shared_experts
        for i in range(n_up):
            nm = "shared_gate" if (model.gated and i == 0) else "shared_up"
            flops = 2.0 * B * d * (di * ns)
            wbytes = (d * di * ns) * w           # 共享权重所有 token 复用，只读一次
            act = B * (d + di * ns) * a
            stages.append(_roofline_stage(nm, flops, wbytes, act, lpu, run.TP))
        flops = 2.0 * B * (di * ns) * d
        wbytes = (di * ns * d) * w
        act = B * (di * ns + d) * a
        stages.append(_roofline_stage("shared_down", flops, wbytes, act, lpu, run.TP))

    # --- 5) TP AllReduce：payload = 输出激活 (B × d) ---
    stages.append(_allreduce_c2c(B * d * a, lpu, run))

    # batched_membound / trace: decode 区 MoE expert/shared 强制 memory-bound（权重读为瓶颈）
    if run.expert_mode in ("batched_membound", "trace"):
        for s in stages:
            if s["name"].startswith(("expert_", "shared_")):
                s["t_ns"] = s["t_mem_ns"]
                s["bound"] = "mem"
    return stages


# ---------------------------------------------------------------------------
# 顶层入口：单层 + 整模聚合
# ---------------------------------------------------------------------------
def simulate_layer(model: FFNModelSpec, lpu: LPUSpec,
                   run: RunSpec) -> Dict[str, Any]:
    """跑单层 FFN，返回 {stages, layer_total_ns, bound_breakdown}。"""
    stages = (moe_ffn_stages if model.is_moe else dense_ffn_stages)(model, lpu, run)
    layer_total = sum(s["t_ns"] for s in stages)

    # 按 compute / mem / comm 三类汇总时间，便于看瓶颈在哪
    breakdown = {"compute": 0.0, "mem": 0.0, "comm": 0.0}
    for s in stages:
        if s["name"] == "AllReduce":
            breakdown["comm"] += s["t_ns"]
        elif s.get("bound") == "compute":
            breakdown["compute"] += s["t_ns"]
        else:
            breakdown["mem"] += s["t_ns"]

    return {
        "stages": stages,
        "layer_total_ns": layer_total,
        "bound_breakdown_ns": breakdown,
    }


def simulate_decode(model: FFNModelSpec, lpu: LPUSpec,
                    run: RunSpec) -> Dict[str, Any]:
    """
    整模 FFN decode：单层 × num_layers，得到每 token 的 FFN 延迟分量（TPOT 的 FFN 部分）。
    返回结构化 dict（顶层结果），meta 由 run.py 负责拼装。
    """
    layer = simulate_layer(model, lpu, run)
    n = model.num_layers
    ffn_tpot_ns = layer["layer_total_ns"] * n

    return {
        "per_layer": layer,
        "decode": {
            "num_layers": n,
            "per_layer_ffn_ns": layer["layer_total_ns"],
            "ffn_tpot_ns": ffn_tpot_ns,           # 整模所有层 FFN 的 decode 延迟和
            "ffn_tpot_us": ffn_tpot_ns / 1e3,
            "bound_breakdown_per_layer_ns": layer["bound_breakdown_ns"],
        },
    }
