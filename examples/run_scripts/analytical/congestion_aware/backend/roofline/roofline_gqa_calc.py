#!/usr/bin/env python3
"""
Full roofline GQA single-layer calculation WITH weight loading.
Uses make_strategy_config() from attention.py for per-NPU model configs.
Outputs JSON with per-module breakdown: QKV, Attention (concat+score+attn_v), Output.
"""
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hardware_config as hw
from attention import make_strategy_config, FULL_CONFIG, MODEL_CONFIGS, MODEL_REPORT_DIRS


def roofline_qkv(peak_perf, bw, cfg, bs):
    """QKV projection: roofline = max(compute, memory_with_weights).

    HMP/Baseline: d_head = d_head_full // tp_s (aligned with TP16).
    Each cube computes Q[B, Hq, dk_local], K[B, Hkv, dk_local], V[B, Hkv, dk_local]
    where dk_local = d_head.  AllGather(tp_s) within cube group reconstructs
    full d_head_full for attention.
    """
    d = cfg["d_model"]
    Hq = cfg["num_attention_heads"]
    Hkv = cfg["num_kv_heads"]
    dk = cfg["d_head"]

    q_mac = bs * Hq * dk * d
    kv_mac = bs * 2 * Hkv * dk * d
    total_mac = q_mac + kv_mac

    w_q = d * Hq * dk
    w_k = d * Hkv * dk
    w_v = d * Hkv * dk
    weight_elems = w_q + w_k + w_v
    act_in = bs * d
    act_out = bs * (Hq * dk + 2 * Hkv * dk)
    total_mem = weight_elems + act_in + act_out

    flops = 2 * total_mac
    comp_ns = flops / (peak_perf * 1e12) * 1e9
    mem_ns = total_mem / (bw * 1e12) * 1e9

    result = {
        "total_ns": max(comp_ns, mem_ns),
        "compute_ns": round(comp_ns, 2),
        "memory_ns": round(mem_ns, 2),
        "bound": "memory" if mem_ns > comp_ns else "compute",
        "weight_elems": weight_elems,
        "activation_elems": act_in + act_out,
        "total_macs": total_mac,
        "oi": round(flops / total_mem, 2),
    }

    tp_hd = cfg.get("tp_hd", 1)
    d_head_full = cfg.get("d_head_full")
    if tp_hd > 1 and d_head_full is not None:
        ag_per_cube = bs * (Hq + 2 * Hkv) * dk
        result["qkv_allgather"] = {
            "tp_hd": tp_hd,
            "per_cube_elems": ag_per_cube,
            "total_elems": ag_per_cube * tp_hd,
        }
    return result


def roofline_attention(peak_perf, bw, cfg, bs, seq):
    """Attention: concat(K/V) + score + attn_v, with sub-op breakdown.
    Uses d_head_full (after AllGather) when available; falls back to d_head."""
    Hq = cfg["num_attention_heads"]
    Hkv = cfg["num_kv_heads"]
    dk = cfg.get("d_head_full", cfg["d_head"])
    tp_s = cfg.get("tp_s", 1)
    cache_seq = seq // tp_s if tp_s > 1 else seq

    if cfg.get("fused_attention", False):
        kv_elems = bs * 2 * Hkv * cache_seq * dk
        mem_ns = kv_elems / (bw * 1e12) * 1e9
        einsum_mac = 2 * bs * Hq * cache_seq * dk
        einsum_flops = 2 * einsum_mac
        comp_ns = einsum_flops / (peak_perf * 1e12) * 1e9
        total_ns = max(mem_ns, comp_ns)
        return {
            "total_ns": total_ns,
            "model": "fused_roofline",
            "bound": "memory" if mem_ns > comp_ns else "compute",
            "kv_cache_elems": kv_elems,
            "einsum_macs": einsum_mac,
            "kv_mem_ns": round(mem_ns, 2),
            "einsum_comp_ns": round(comp_ns, 2),
            "cache_seq": cache_seq,
            "gqa_oi": round(einsum_flops / kv_elems, 2),
        }

    k_cache = bs * Hkv * cache_seq * dk
    v_cache = k_cache
    concat_k_ns = k_cache / (bw * 1e12) * 1e9
    concat_v_ns = v_cache / (bw * 1e12) * 1e9

    score_mac = bs * Hq * cache_seq * dk
    score_mem = k_cache
    score_comp_ns = (2 * score_mac) / (peak_perf * 1e12) * 1e9
    score_mem_ns = score_mem / (bw * 1e12) * 1e9

    attn_v_mac = bs * Hq * cache_seq * dk
    attn_v_mem = v_cache
    attn_v_comp_ns = (2 * attn_v_mac) / (peak_perf * 1e12) * 1e9
    attn_v_mem_ns = attn_v_mem / (bw * 1e12) * 1e9

    gqa_oi = (2 * score_mac) / score_mem if score_mem > 0 else 0

    serial_total = concat_k_ns + score_comp_ns + concat_v_ns + attn_v_comp_ns
    fused_score = max(score_mem_ns, score_comp_ns)
    fused_attn_v = max(attn_v_mem_ns, attn_v_comp_ns)
    fused_total = fused_score + fused_attn_v

    return {
        "total_ns_serial": round(serial_total, 2),
        "total_ns_fused": round(fused_total, 2),
        "model": "serial(astrasim) + fused(gemv_roofline)",
        "cache_seq": cache_seq,
        "gqa_oi": round(gqa_oi, 2),
        "sub_ops": {
            "concat_k": {
                "time_ns": round(concat_k_ns, 2),
                "elems": k_cache,
                "bound": "memory",
            },
            "score_gemv": {
                "compute_ns": round(score_comp_ns, 2),
                "memory_ns": round(score_mem_ns, 2),
                "fused_ns": round(fused_score, 2),
                "macs": score_mac,
                "bound": "memory" if score_mem_ns > score_comp_ns else "compute",
            },
            "concat_v": {
                "time_ns": round(concat_v_ns, 2),
                "elems": v_cache,
                "bound": "memory",
            },
            "attn_v_gemv": {
                "compute_ns": round(attn_v_comp_ns, 2),
                "memory_ns": round(attn_v_mem_ns, 2),
                "fused_ns": round(fused_attn_v, 2),
                "macs": attn_v_mac,
                "bound": "memory" if attn_v_mem_ns > attn_v_comp_ns else "compute",
            },
        },
    }


def roofline_output(peak_perf, bw, cfg, bs):
    """Output projection: roofline = max(compute, memory_with_weights).
    Uses d_head_full (after AllGather) when available; falls back to d_head."""
    d = cfg["d_model"]
    Hq = cfg["num_attention_heads"]
    dk = cfg.get("d_head_full", cfg["d_head"])
    tp_s = cfg.get("tp_s", 1)
    wo_mode = cfg.get("wo_mode", "duplicate")

    if wo_mode == "duplicate":
        Dq = Hq * dk
        mac = bs * Dq * d
        w_elems = Dq * d
    elif wo_mode == "sharded":
        Dq = Hq * dk
        mac = bs * (Dq // tp_s) * d
        w_elems = (Dq // tp_s) * d
    elif wo_mode == "row_sharded":
        Dq = Hq * dk
        mac = bs * Hq * (dk // tp_s) * d
        w_elems = Hq * (dk // tp_s) * d
    elif wo_mode == "full":
        Dq = Hq * dk
        mac = bs * Dq * d
        w_elems = Dq * d
    elif wo_mode == "none":
        Dq = Hq * dk
        mac = bs * Dq * d
        w_elems = Dq * d
    else:
        Dq = Hq * dk
        mac = bs * Dq * d
        w_elems = Dq * d

    act_elems = bs * (Hq * dk + d)
    total_mem = w_elems + act_elems

    flops = 2 * mac
    comp_ns = flops / (peak_perf * 1e12) * 1e9
    mem_ns = total_mem / (bw * 1e12) * 1e9

    return {
        "total_ns": max(comp_ns, mem_ns),
        "compute_ns": round(comp_ns, 2),
        "memory_ns": round(mem_ns, 2),
        "bound": "memory" if mem_ns > comp_ns else "compute",
        "weight_elems": w_elems,
        "activation_elems": act_elems,
        "total_macs": mac,
        "wo_mode": wo_mode,
        "oi": round(flops / total_mem, 2) if total_mem > 0 else 0,
    }


def roofline_qkv_allgather(link_bw_tbs, cfg, bs,
                            hop_latency_ns=100, endpoint_delay_ns=10,
                            hops_per_step=1):
    """Comm time for AllGather(tp_hd) within cube group after QKV projection.

    Each cube has Q[B,Hq,dk_local]+K[B,Hkv,dk_local]+V[B,Hkv,dk_local].
    Ring AllGather: (tp_hd-1) steps, each step transfers per-cube chunk.
    Per-step time = hops * hop_latency + endpoint_delay + chunk/link_bw.
    After AG: Q shared within group, K/V per-group (different KV heads).
    """
    tp_hd = cfg.get("tp_hd", 1)
    d_head_full = cfg.get("d_head_full")
    if tp_hd <= 1 or d_head_full is None:
        return {"time_ns": 0, "description": "no d_head split"}

    Hq = cfg["num_attention_heads"]
    Hkv = cfg["num_kv_heads"]
    dk_local = cfg["d_head"]
    chunk_elems = bs * (Hq + 2 * Hkv) * dk_local
    n_steps = tp_hd - 1
    bw_per_step_ns = chunk_elems / (link_bw_tbs * 1e12) * 1e9
    fixed_per_step_ns = hops_per_step * hop_latency_ns + endpoint_delay_ns
    time_ns = n_steps * (fixed_per_step_ns + bw_per_step_ns)
    return {
        "time_ns": round(time_ns, 2),
        "tp_hd": tp_hd,
        "per_cube_chunk_elems": chunk_elems,
        "total_ag_elems": chunk_elems * tp_hd,
        "steps": n_steps,
        "fixed_per_step_ns": fixed_per_step_ns,
        "bw_per_step_ns": round(bw_per_step_ns, 4),
    }


def roofline_attn_rs(link_bw_tbs, cfg, bs,
                     hop_latency_ns=100, endpoint_delay_ns=10,
                     hops_per_step=1):
    """ReduceScatter(tp_s) on attention output for hmp_reo_new.

    After attention kernel, each cube holds [B, Hq/tp_h, d_head_full]
    as partial sum across tp_s. Ring RS scatters along HeadDim, giving
    each cube [B, Hq/tp_h, d_head_full/tp_s].
    """
    tp_s = cfg.get("tp_s", 1)
    if tp_s <= 1:
        return {"time_ns": 0, "description": "no tp_s split"}
    Hq = cfg["num_attention_heads"]
    dk = cfg.get("d_head_full", cfg["d_head"])
    total_elems = bs * Hq * dk
    chunk_elems = total_elems // tp_s
    n_steps = tp_s - 1
    bw_per_step_ns = chunk_elems / (link_bw_tbs * 1e12) * 1e9
    fixed_per_step_ns = hops_per_step * hop_latency_ns + endpoint_delay_ns
    time_ns = n_steps * (fixed_per_step_ns + bw_per_step_ns)
    return {
        "time_ns": round(time_ns, 2),
        "tp_s": tp_s,
        "total_elems": total_elems,
        "chunk_elems_per_step": chunk_elems,
        "steps": n_steps,
        "fixed_per_step_ns": fixed_per_step_ns,
        "bw_per_step_ns": round(bw_per_step_ns, 4),
    }


def roofline_hmp_reduce(link_bw_tbs, cfg, bs,
                         hop_latency_ns=100, endpoint_delay_ns=10,
                         hops_per_step=1):
    """Tree-reduce across all tp_h × tp_s NPUs so one cube gets the complete
    GQA output [B, d_model].

    log2(N) steps, each step transfers B × d_model elements.
    Per-step time = hops * hop_latency + endpoint_delay + msg/link_bw.
    """
    tp_h = cfg.get("tp_h", 1)
    tp_s = cfg.get("tp_s", 1)
    n_total = tp_h * tp_s
    if n_total <= 1:
        return {"time_ns": 0, "description": "single NPU"}

    num_steps = int(math.log2(n_total))
    msg_elems = bs * cfg["d_model"]
    bw_per_step_ns = msg_elems / (link_bw_tbs * 1e12) * 1e9
    fixed_per_step_ns = hops_per_step * hop_latency_ns + endpoint_delay_ns
    time_ns = num_steps * (fixed_per_step_ns + bw_per_step_ns)
    return {
        "time_ns": round(time_ns, 2),
        "n_total": n_total,
        "num_steps": num_steps,
        "msg_elems_per_step": msg_elems,
        "fixed_per_step_ns": fixed_per_step_ns,
        "bw_per_step_ns": round(bw_per_step_ns, 4),
    }


def roofline_output_ar(link_bw_tbs, cfg, bs,
                       hop_latency_ns=100, endpoint_delay_ns=10,
                       hops_per_step=1,
                       dataset_splits=1, active_chunks=1):
    """Ring AllReduce(tp_s) for output projection partial sums.

    Wo is row-sharded across tp_s cubes: each cube holds
    Wo_shard = [Dq/tp_s, d_model].  After matmul, each cube produces a
    partial sum of the FULL output [B, 1, d_model].
    Ring AR across tp_s cubes sums the partials.
    block2x2 layout: g21 groups are 2×2 block, all 1-hop, no congestion.

    msg_elems = B * d_model  (full output dim, NOT divided by tp_s).

    With dataset_splits > 1 (e.g. preferred-dataset-splits=16), the message
    is split into that many sub-messages.  Each sub-message goes through
    a full ring AR of 2*(n-1) steps.  active_chunks controls how many
    sub-messages are pipelined concurrently (active-chunks-per-dimension).

    Total steps = dataset_splits * 2*(n-1) / active_chunks.
    Chunk per step = msg / (dataset_splits * n).
    Per-step time = hops * hop_latency + endpoint_delay + chunk/link_bw.
    """
    tp_s = cfg.get("tp_s", 1)
    wo_mode = cfg.get("wo_mode", "duplicate")
    if tp_s <= 1 or wo_mode != "sharded":
        return {"time_ns": 0, "description": "no sharded wo / no tp_s split"}

    msg_elems = bs * cfg["d_model"]
    n = tp_s
    steps_per_split = 2 * (n - 1)
    n_steps = dataset_splits * steps_per_split // active_chunks
    chunk_per_step = msg_elems / (dataset_splits * n)
    bw_per_step_ns = chunk_per_step / (link_bw_tbs * 1e12) * 1e9
    fixed_per_step_ns = hops_per_step * hop_latency_ns + endpoint_delay_ns
    time_ns = n_steps * (fixed_per_step_ns + bw_per_step_ns)
    return {
        "time_ns": round(time_ns, 2),
        "tp_s": tp_s,
        "msg_elems": msg_elems,
        "dataset_splits": dataset_splits,
        "active_chunks": active_chunks,
        "chunk_per_step": round(chunk_per_step, 4),
        "n_steps": n_steps,
        "fixed_per_step_ns": fixed_per_step_ns,
        "bw_per_step_ns": round(bw_per_step_ns, 4),
        "hops_per_step": hops_per_step,
    }


def roofline_baseline_output_ag(link_bw_tbs, cfg, bs,
                                hop_latency_ns=100, endpoint_delay_ns=10,
                                hops_per_step=1):
    """AllGather(tp_s) for Baseline output (wo sharded along Dmodel by tp_s).

    After AllReduce on tp_h each cube holds partial [B, D/tp_s].
    Ring AllGather across tp_s cubes reconstructs full [B, D].
    block2x2 layout: g21 groups are all 1-hop, no congestion.

    Per-step time = hops * hop_latency + endpoint_delay + chunk/link_bw.
    """
    tp_s = cfg.get("tp_s", 1)
    wo_mode = cfg.get("wo_mode", "duplicate")
    if tp_s <= 1 or wo_mode != "sharded":
        return {"time_ns": 0, "description": "no sharded wo / no tp_s split"}

    n_steps = tp_s - 1
    chunk_elems = bs * cfg["d_model"] // tp_s
    bw_per_step_ns = chunk_elems / (link_bw_tbs * 1e12) * 1e9
    fixed_per_step_ns = hops_per_step * hop_latency_ns + endpoint_delay_ns
    time_ns = n_steps * (fixed_per_step_ns + bw_per_step_ns)
    return {
        "time_ns": round(time_ns, 2),
        "tp_s": tp_s,
        "per_cube_chunk_elems": chunk_elems,
        "total_ag_elems": chunk_elems * tp_s,
        "steps": n_steps,
        "fixed_per_step_ns": fixed_per_step_ns,
        "bw_per_step_ns": round(bw_per_step_ns, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MLA (Multi-head Latent Attention) Roofline Functions
#
#  DeepSeek-V3 MLA 使用吸收后 (absorbed) 权重, 详见 Deppseek-v3.md 附录 A/C.
#  与 GQA 的关键区别:
#    - Score 维度 (d_score = d_c + d_r = 576) ≠ Value 维度 (d_value = d_c = 512)
#    - KV cache 存压缩表示 c_KV[d_c] + k_R[d_r], 共 d_score 元素/tok
#    - QKV 包含下投影 (shared) + 吸收后 Q 投影 (per-head)
#    - ProjO 使用吸收后 W_O_abs [d_model × d_value] per head
#
#  Per-NPU 配置 (tp_h=1, tp_s=16, 16 NPUs):
#    heads_per_npu = n_h / tp_h = 128
#    cache_seq = seq / tp_s = seq / 16
# ═══════════════════════════════════════════════════════════════════════════

# MLA 模型参数 (DeepSeek-V3, Deppseek-v3.md 附录 A)
MLA_CONFIG = {
    "d_model": 7168,
    "n_h": 128,            # 总注意力头数
    "d_c": 512,            # KV 压缩维度
    "d_c_prime": 1536,     # Query 压缩维度
    "d_r": 64,             # RoPE 维度
    "d_score": 576,        # d_c + d_r (Score 点积维度)
    "d_value": 512,        # d_c (Value 聚合维度, 即吸收后 O 的输入维度)
}


def make_mla_npu_config(tp_h=4, tp_s=4):
    """生成 MLA 的 per-NPU 配置 (GQA 等效视角, 附录 C.3).

    与 GQA make_strategy_config 对应:
      - num_attention_heads = n_h // tp_h  (与 GQA 一致: tp_s 组内共享全部 heads)
      - num_kv_heads = 1 (共享压缩 KV, 无法被 tp_h 切分)
      - d_score (576) ≠ d_value (512): Score 用 d_c+d_r, Value 用 d_c

    GQA 对比:
      GQA Qwen3: Hq_npu = 64/tp_h = 16, Hkv_npu = 4/tp_h = 1
      MLA:       Hq_npu = 128/tp_h = 32, Hkv_npu = 1 (不切分)
      → AI_score = 2 × Hq_npu / Hkv_npu = 64 (GQA=32)

    QKV 使用 Appendix A factored 形式:
      (a) 下投影: shared (不按 head 分)
      (b) 吸收后 Q: per-head, Hq_npu 个 heads per NPU
    """
    mc = MLA_CONFIG
    Hq_npu = mc["n_h"] // tp_h     # 128/4=32, 与 GQA 一致: tp_s 内共享 heads

    return {
        "d_model": mc["d_model"],
        "num_attention_heads": Hq_npu,         # 32 heads per NPU (共享于 tp_s 组)
        "num_kv_heads": 1,                     # 共享压缩 KV (不按 tp_h 切分)
        "d_score": mc["d_score"],              # d_c + d_r = 576 (Score 点积维度)
        "d_value": mc["d_value"],              # d_c = 512 (Value 聚合 + Output 投影维度)
        "d_c": mc["d_c"],
        "d_c_prime": mc["d_c_prime"],
        "d_r": mc["d_r"],
        "tp_s": tp_s,
        "tp_h": tp_h,
        "n_h_total": mc["n_h"],
        "num_layers": 1,
        "mla": True,
    }


def roofline_mla_qkv(peak_perf, bw, cfg, bs):
    """MLA QKV projection roofline (absorbed form, Appendix A Steps 1-2).

    两组子操作:
      (a) Down projections: W_DKV[d_c × d], W_KR[d_r × d], W_DQ[d_c' × d]
          → 共享权重, 每 NPU 加载完整 copy
      (b) Absorbed Q: W_Q_abs[d_c × d_c'] + W_QR[d_r × d_c'] per head
          → 按 N_NPUS 切分: 每 NPU 只加载 n_h/N 个 heads
          → QKV 后 AllGather 重建全部 heads 的 Q 向量供 Attention 使用

    与 GQA 的 tp_hd 切分类似: 权重分摊, AllGather 重建.
    """
    d = cfg["d_model"]
    Hq = cfg["num_attention_heads"]       # attention 用全部 heads
    d_c = cfg["d_c"]
    d_c_prime = cfg["d_c_prime"]
    d_r = cfg["d_r"]
    d_score = cfg["d_score"]              # d_c + d_r
    tp_s = cfg.get("tp_s", 1)
    tp_h = cfg.get("tp_h", 1)
    n_npus = tp_h * tp_s

    # 每 NPU 负责计算的 heads 数 (权重切分)。
    # BUGFIX: Hq=cfg["num_attention_heads"] 已经是 n_h//tp_h(=32)；per-head 权重应按
    # n_h_total // n_npus 摊 (== Hq // tp_s = 8)。旧版用 Hq // n_npus(=2) 又多除了一次
    # tp_h，使 absorbed-Q 权重 4× 偏小、ProjQKV 过度乐观。
    heads_per_npu = cfg["n_h_total"] // n_npus    # 128/16 = 8

    # (a) Down projections: shared, each NPU loads full copy
    down_weight = (d_c + d_r + d_c_prime) * d
    down_mac = bs * (d_c + d_r + d_c_prime) * d

    # (b) Absorbed per-head Q: 只加载 heads_per_npu 个 heads 的权重
    abs_weight = heads_per_npu * d_score * d_c_prime
    abs_mac = bs * heads_per_npu * d_score * d_c_prime

    weight_elems = down_weight + abs_weight
    total_mac = down_mac + abs_mac
    act_in = bs * d
    act_out = bs * (d_c + d_r + heads_per_npu * d_score)
    total_mem = weight_elems + act_in + act_out

    flops = 2 * total_mac
    comp_ns = flops / (peak_perf * 1e12) * 1e9
    mem_ns = total_mem / (bw * 1e12) * 1e9

    return {
        "total_ns": max(comp_ns, mem_ns),
        "compute_ns": round(comp_ns, 2),
        "memory_ns": round(mem_ns, 2),
        "bound": "memory" if mem_ns > comp_ns else "compute",
        "weight_elems": weight_elems,
        "activation_elems": act_in + act_out,
        "total_macs": total_mac,
        "oi": round(flops / total_mem, 2),
        "down_weight_elems": down_weight,
        "abs_weight_elems": abs_weight,
        "down_macs": down_mac,
        "abs_macs": abs_mac,
        "heads_per_npu_weight": heads_per_npu,
    }


def roofline_mla_attention(peak_perf, bw, cfg, bs, seq):
    """MLA Attention roofline — 仿照 GQA roofline_attention, 维度换为附录 C.3.

    MLA ≈ GQA with Hkv=1, Hq=32(per NPU), dk=d_score=576.
    唯一区别: MLA 的 K cache 和 V cache 合一 (c_KV[d_c] + k_R[d_r]):
      - GQA: kv_mem = 2 × Hkv × cache_seq × dk  (K/V 分开)
      - MLA: kv_mem = 1 × cache_seq × d_score    (K/V 合一, 只读一次)

    Score: Q[Hq, d_score] · Cache[cache_seq, d_score]^T → 读完整 cache
    Value: α[Hq, cache_seq] · Cache[cache_seq, d_value] → c_KV 已加载, 不额外读 HBM
    """
    Hq = cfg["num_attention_heads"]       # per NPU = 32
    Hkv = cfg["num_kv_heads"]             # = 1
    d_score = cfg["d_score"]              # 576 = d_c + d_r
    d_value = cfg["d_value"]              # 512 = d_c
    tp_s = cfg.get("tp_s", 1)
    cache_seq = seq // tp_s if tp_s > 1 else seq

    # KV cache: 合一存储, c_KV[d_c] + k_R[d_r] = d_score per position
    # 与 GQA 的区别: GQA 是 2 × Hkv × dk, MLA 是 1 × d_score (不乘 2)
    kv_cache = bs * Hkv * cache_seq * d_score

    # ── Score: Q · Cache^T (d_score 维点积) ──
    score_mac = bs * Hq * cache_seq * d_score
    score_mem = kv_cache                   # 读完整 KV cache
    score_comp_ns = (2 * score_mac) / (peak_perf * 1e12) * 1e9
    score_mem_ns = score_mem / (bw * 1e12) * 1e9

    # ── Value: α · Cache (d_value 维聚合) ──
    # KV 合一: c_KV 数据已在 Score 阶段从 HBM 加载, Value 不额外读取
    attn_v_mac = bs * Hq * cache_seq * d_value
    attn_v_mem = 0                         # 合一 KV, 复用 Score 的读取
    attn_v_comp_ns = (2 * attn_v_mac) / (peak_perf * 1e12) * 1e9
    attn_v_mem_ns = 0.0

    # OI: 与 GQA 对比
    # GQA: gqa_oi = 2 × score_mac / score_mem = 2 × Hq × dk / (Hkv × dk) = 2R
    # MLA: score 读 576, value 不额外读 → total_flops / kv_cache
    gqa_oi = (2 * score_mac) / score_mem if score_mem > 0 else 0

    fused_score = max(score_comp_ns, score_mem_ns)
    fused_attn_v = max(attn_v_comp_ns, attn_v_mem_ns)  # 纯 compute (mem=0)
    fused_total = fused_score + fused_attn_v

    return {
        "total_ns_serial": round(score_comp_ns + attn_v_comp_ns, 2),
        "total_ns_fused": round(fused_total, 2),
        "total_ns": round(fused_total, 2),
        "model": "mla_unified_kv (GQA-equivalent, Appendix C.3)",
        "cache_seq": cache_seq,
        "kv_cache_elems": kv_cache,
        "gqa_oi": round(gqa_oi, 2),
        "sub_ops": {
            "score_gemv": {
                "compute_ns": round(score_comp_ns, 2),
                "memory_ns": round(score_mem_ns, 2),
                "fused_ns": round(fused_score, 2),
                "macs": score_mac,
                "bound": "memory" if score_mem_ns > score_comp_ns else "compute",
            },
            "attn_v_gemv": {
                "compute_ns": round(attn_v_comp_ns, 2),
                "memory_ns": 0.0,           # KV 合一, 复用 Score 读取
                "fused_ns": round(fused_attn_v, 2),
                "macs": attn_v_mac,
                "bound": "compute",          # 永远 compute-bound (无额外 HBM 读取)
            },
        },
    }


def roofline_mla_output(peak_perf, bw, cfg, bs):
    """MLA Output projection roofline (absorbed form, Appendix A Step 6).

    W_O_abs[i] = W_O[i] · W_UV[i] ∈ [d_model × d_value] per head.
    按 N_NPUS 切分 d_value: 每 NPU 加载 n_h × d_model × (d_value / N) 的权重.
    每 NPU 计算部分输出 → tree-reduce 求和得最终 o_t ∈ [d_model].
    与 GQA 的 row_sharded Wo 等价.
    """
    d = cfg["d_model"]
    n_h = cfg["n_h_total"]              # 全部 heads (128)；W_O_abs 是 per-head 权重
    d_value = cfg["d_value"]            # d_c = 512
    tp_s = cfg.get("tp_s", 1)
    tp_h = cfg.get("tp_h", 1)
    n_npus = tp_h * tp_s

    # 按 N_NPUS 切分 d_value: 每 NPU 处理 d_value_shard 维度
    d_value_shard = d_value // n_npus    # 512/16 = 32

    # Per-NPU weight: n_h × d_model × (d_value/N) = 总权重的 1/N 切片 (docstring 口径)。
    # BUGFIX: 旧版用 Hq(=n_h//tp_h=32) 而非 n_h(128)，使每 NPU 权重 = 总量/64 < 16 芯片
    # 物理下限(总量/16)，ProjO 4× 偏小、过度乐观。
    w_elems = n_h * d * d_value_shard
    mac = bs * n_h * d * d_value_shard
    act_elems = bs * (n_h * d_value_shard + d)
    total_mem = w_elems + act_elems

    flops = 2 * mac
    comp_ns = flops / (peak_perf * 1e12) * 1e9
    mem_ns = total_mem / (bw * 1e12) * 1e9

    return {
        "total_ns": max(comp_ns, mem_ns),
        "compute_ns": round(comp_ns, 2),
        "memory_ns": round(mem_ns, 2),
        "bound": "memory" if mem_ns > comp_ns else "compute",
        "weight_elems": w_elems,
        "activation_elems": act_elems,
        "total_macs": mac,
        "oi": round(flops / total_mem, 2) if total_mem > 0 else 0,
        "d_value_shard": d_value_shard,
    }


def calc_mla_strategy(hw_cfg, bs, seq_list, link_bw=1.5,
                       hop_latency_ns=15, endpoint_delay_ns=10,
                       tp_h=4, tp_s=4):
    """计算 MLA Ours 策略的 roofline (per-NPU compute + analytical comm).

    通信模型与 deepseek_v3_mla_roofline.py 的 Option B 一致:
      (1) Score/lse AllReduce (tp_s)
      (2) o_comp AllReduce (tp_s)
      (3) Final head-Reduce (tp_h × tp_s)
    """
    cfg = make_mla_npu_config(tp_h=tp_h, tp_s=tp_s)
    pp = hw_cfg["compute"]
    bw = hw_cfg["Bandwidth"]
    crossover = pp / bw

    Hq = cfg["num_attention_heads"]
    d_c = cfg["d_c"]
    d_model = cfg["d_model"]
    n_npus = tp_h * tp_s
    fixed_per_step = hop_latency_ns + endpoint_delay_ns

    # 每 NPU 负责计算的 heads 数 (权重切分)。BUGFIX: 同 roofline_mla_qkv，用 n_h_total//n_npus。
    heads_per_npu_weight = cfg["n_h_total"] // n_npus   # 128/16 = 8

    # ── Communication ──
    # (0) QKV AllGather: 每 NPU 算 8 heads 的 Q, 需要 AllGather 重建 128 heads
    #     msg per step = heads_per_npu_weight × d_score × bs (FP8)
    qkv_ag_chunk = bs * heads_per_npu_weight * cfg["d_score"]   # 8 × 576 = 4608
    qkv_ag_steps = n_npus - 1   # ring AllGather
    qkv_ag_ns = qkv_ag_steps * (fixed_per_step + qkv_ag_chunk / (link_bw * 1e3))

    # (1) Score/lse AllReduce within tp_s (seq-split 后合并 attention scores)
    score_ar_msg = bs * Hq * 2 * 4     # 2 FP32 floats per head
    score_ar_steps = 2 * int(math.log2(tp_s))
    score_ar_ns = score_ar_steps * (fixed_per_step + score_ar_msg / (link_bw * 1e3))

    # (2) o_comp AllReduce within tp_s (seq-split 后合并 value 聚合)
    ocomp_ar_msg = bs * Hq * d_c       # FP8
    ocomp_ar_steps = 2 * int(math.log2(tp_s))
    ocomp_ar_ns = ocomp_ar_steps * (fixed_per_step + ocomp_ar_msg / (link_bw * 1e3))

    # (3) ProjO tree-Reduce: W_O 按 N_NPUS 切分, 每 NPU 产生 partial output
    #     msg = bs × d_model, log2(N) steps
    proj_o_red_msg = bs * d_model
    proj_o_red_steps = int(math.log2(n_npus))
    proj_o_red_ns = proj_o_red_steps * (fixed_per_step + proj_o_red_msg / (link_bw * 1e3))

    comm_total_ns = qkv_ag_ns + score_ar_ns + ocomp_ar_ns + proj_o_red_ns

    results = []
    for seq in seq_list:
        qkv = roofline_mla_qkv(pp, bw, cfg, bs)
        attn = roofline_mla_attention(pp, bw, cfg, bs, seq)
        out = roofline_mla_output(pp, bw, cfg, bs)

        gpu_ns = qkv["total_ns"] + attn["total_ns"] + out["total_ns"]
        wall_ns = gpu_ns + comm_total_ns

        entry = {
            "seq": seq, "batch": bs,
            "Proj_QKV": {k: round(v, 2) if isinstance(v, float) else v
                         for k, v in qkv.items()},
            "attention": {k: round(v, 2) if isinstance(v, float) else v
                          for k, v in attn.items()},
            "Proj_O": {k: round(v, 2) if isinstance(v, float) else v
                       for k, v in out.items()},
            "gpu_total_ns": round(gpu_ns, 2),
            "comm_total_ns": round(comm_total_ns, 2),
            "wall_total_ns": round(wall_ns, 2),
        }
        results.append(entry)

    return {
        "strategy": "mla_ours",
        "config": cfg,
        "hardware": {
            "peak_perf_tflops": pp,
            "bandwidth_tb_s": bw,
            "roofline_crossover_oi": round(crossover, 2),
            "link_bw_tb_s": link_bw,
        },
        "comm": {
            "qkv_ag_ns": round(qkv_ag_ns, 2),
            "score_ar_ns": round(score_ar_ns, 2),
            "ocomp_ar_ns": round(ocomp_ar_ns, 2),
            "proj_o_reduce_ns": round(proj_o_red_ns, 2),
            "total_comm_ns": round(comm_total_ns, 2),
        },
        "data": results,
    }


def calc_strategy(strategy, hw_cfg, bs, seq_list, link_bw=2.0,
                  dataset_splits=1, active_chunks=1,
                  hop_latency_ns=100, endpoint_delay_ns=10,
                  model_config=None):
    cfg = make_strategy_config(strategy, model_config=model_config)
    pp = hw_cfg["compute"]
    bw = hw_cfg["Bandwidth"]
    crossover = pp / bw

    qkv_ag = roofline_qkv_allgather(link_bw, cfg, bs,
                                     hop_latency_ns=hop_latency_ns,
                                     endpoint_delay_ns=endpoint_delay_ns)
    final_reduce = roofline_hmp_reduce(link_bw, cfg, bs,
                                       hop_latency_ns=hop_latency_ns,
                                       endpoint_delay_ns=endpoint_delay_ns)
    attn_rs = roofline_attn_rs(link_bw, cfg, bs,
                                hop_latency_ns=hop_latency_ns,
                                endpoint_delay_ns=endpoint_delay_ns)
    output_ar = roofline_output_ar(link_bw, cfg, bs,
                                   hop_latency_ns=hop_latency_ns,
                                   endpoint_delay_ns=endpoint_delay_ns,
                                   dataset_splits=dataset_splits,
                                   active_chunks=active_chunks)
    output_ag = roofline_baseline_output_ag(link_bw, cfg, bs,
                                            hop_latency_ns=hop_latency_ns,
                                            endpoint_delay_ns=endpoint_delay_ns)

    if strategy == "hmp_reo_new":
        comm_total_ns = qkv_ag["time_ns"] + attn_rs["time_ns"] + final_reduce["time_ns"]
    else:
        comm_total_ns = (qkv_ag["time_ns"] + final_reduce["time_ns"]
                         + output_ar["time_ns"] + output_ag["time_ns"])

    results = []
    for seq in seq_list:
        qkv = roofline_qkv(pp, bw, cfg, bs)
        attn = roofline_attention(pp, bw, cfg, bs, seq)
        out = roofline_output(pp, bw, cfg, bs)

        if cfg.get("fused_attention", False):
            gpu_total = qkv["total_ns"] + attn["total_ns"] + out["total_ns"]
        else:
            gpu_serial = qkv["total_ns"] + attn["total_ns_serial"] + out["total_ns"]
            gpu_fused = qkv["total_ns"] + attn["total_ns_fused"] + out["total_ns"]

        entry = {"seq": seq, "batch": bs}
        entry["Proj_QKV"] = {k: round(v, 2) if isinstance(v, float) else v
                             for k, v in qkv.items()}
        entry["attention"] = {k: round(v, 2) if isinstance(v, float) else v
                              for k, v in attn.items()}
        entry["Proj_O"] = {k: round(v, 2) if isinstance(v, float) else v
                           for k, v in out.items()}

        entry["qkv_allgather_comm"] = qkv_ag
        entry["attn_rs_comm"] = attn_rs
        entry["final_reduce_comm"] = final_reduce
        entry["output_ar_comm"] = output_ar
        entry["output_ag_comm"] = output_ag

        if cfg.get("fused_attention", False):
            entry["gpu_total_ns"] = round(gpu_total, 2)
            entry["wall_total_ns"] = round(gpu_total + comm_total_ns, 2)
        else:
            entry["gpu_total_serial_ns"] = round(gpu_serial, 2)
            entry["gpu_total_fused_ns"] = round(gpu_fused, 2)
            entry["wall_total_serial_ns"] = round(gpu_serial + comm_total_ns, 2)
            entry["wall_total_fused_ns"] = round(gpu_fused + comm_total_ns, 2)

        results.append(entry)

    return {
        "strategy": strategy,
        "config": cfg,
        "hardware": {
            "peak_perf_tflops": pp,
            "bandwidth_tb_s": bw,
            "roofline_crossover_oi": round(crossover, 2),
            "link_bw_tb_s": link_bw,
        },
        "comm": {
            "qkv_allgather": qkv_ag,
            "attn_rs": attn_rs,
            "final_reduce": final_reduce,
            "output_ar": output_ar,
            "output_ag": output_ag,
            "total_comm_ns": round(comm_total_ns, 2),
        },
        "data": results,
    }


def generate_mla_fig1_data(output_path, tp_h=1, tp_s=16):
    """生成 fig1 所需的 MLA 数据 JSON (Ours + H100/Rubin baselines).

    Ours: 使用 calc_mla_strategy (roofline_gqa_calc MLA 函数)
    Baselines: 使用 mla_single_gpu_baseline (absorbed form, GQA-style roofline)

    输出格式与 plot_e2e.py 兼容:
      strategies.{hmp_reo_new,h100,h100_tp2,rubin,rubin_tp2}.data_by_bs.bs{N}[]
      每条 entry 包含 hybrid_wall_ns, hybrid_Proj_QKV_ns 等字段
    """
    import bisect
    seqs = [1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576]
    batches = [1, 4, 16, 32]

    ours_hw = {"compute": 96, "Bandwidth": 2.5}

    # H100 / Rubin baseline 参数 (与 deepseek_v3_mla_roofline.py 一致)
    H100_PEAK, H100_HBM = 1979, 3.35
    RUBIN_PEAK, RUBIN_HBM = 17500, 22.0
    H100_NVLINK, RUBIN_NVLINK = 0.9, 1.8
    NVLINK_LATENCY = 800  # ns

    # ── H100/Rubin 单卡 baseline (absorbed form) ──
    # 复用 MLA_CONFIG 和 h100_mla_profile 的 BW utilization
    here = Path(__file__).resolve().parent
    with open(here / "utilization_profiles" / "h100" / "h100_mla_profile.json") as f:
        h100_prof = json.load(f)
    bw_util = h100_prof["bw_util"]

    def _mla_baseline(peak_tflops, hbm_tbs, bs, seq):
        """单卡 MLA baseline: max(comp, mem) / bw_util per stage (absorbed form)."""
        mc = MLA_CONFIG
        d, n_h = mc["d_model"], mc["n_h"]
        d_c, d_c_prime, d_r = mc["d_c"], mc["d_c_prime"], mc["d_r"]
        d_score = mc["d_score"]
        bw_bps = hbm_tbs * 1e12
        peak_flops = peak_tflops * 1e12
        bpe = 2  # BF16

        # QKV (absorbed)
        w_qkv = ((d_c + d_r + d_c_prime) * d + n_h * d_score * d_c_prime)
        act_qkv = bs * (d + d_c + d_r + d_c_prime + n_h * d_score)
        qkv_mac = bs * ((d_c + d_r + d_c_prime) * d + n_h * d_score * d_c_prime)
        qkv_rf = max(2 * qkv_mac / peak_flops * 1e9,
                     (w_qkv + act_qkv) * bpe / bw_bps * 1e9)
        u_qkv = bw_util["proj_qkv"]["1"]
        qkv_ns = qkv_rf / u_qkv

        # Attention (unified KV cache)
        kv_bytes = bs * seq * d_score * bpe
        attn_mac = bs * n_h * seq * (d_score + mc["d_value"])
        attn_keys = sorted(int(k) for k in bw_util["attn_absorbed"])
        if seq <= attn_keys[0]:
            u_attn = bw_util["attn_absorbed"][str(attn_keys[0])]
        elif seq >= attn_keys[-1]:
            u_attn = bw_util["attn_absorbed"][str(attn_keys[-1])]
        else:
            idx = bisect.bisect_right(attn_keys, seq) - 1
            lo, hi = attn_keys[idx], attn_keys[idx + 1]
            t = (math.log2(seq) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
            u_attn = bw_util["attn_absorbed"][str(lo)] + t * (bw_util["attn_absorbed"][str(hi)] - bw_util["attn_absorbed"][str(lo)])
        attn_rf = max(2 * attn_mac / peak_flops * 1e9, kv_bytes / bw_bps * 1e9)
        attn_ns = attn_rf / u_attn

        # ProjO (absorbed)
        w_o = n_h * d * mc["d_value"]
        act_o = bs * (n_h * mc["d_value"] + d)
        o_mac = bs * n_h * d * mc["d_value"]
        u_o = bw_util["proj_o"]["1"]
        o_rf = max(2 * o_mac / peak_flops * 1e9, (w_o + act_o) * bpe / bw_bps * 1e9)
        o_ns = o_rf / u_o

        wall = qkv_ns + attn_ns + o_ns
        return {
            "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
            "hybrid_attn_ns": round(attn_ns, 2),
            "hybrid_Proj_O_ns": round(o_ns, 2),
            "hybrid_wall_ns": round(wall, 2),
            "comm_total_ns": 0.0,
            "Proj_QKV_roofline_ns": round(qkv_rf, 2),
            "Proj_QKV_bw_util": round(u_qkv, 4),
            "attn_roofline_ns": round(attn_rf, 2),
            "attn_bw_util": round(u_attn, 4),
            "Proj_O_roofline_ns": round(o_rf, 2),
            "Proj_O_bw_util": round(u_o, 4),
            "constant_overhead_ns": 0.0,
        }

    def _mla_tp2(single, bs, nvlink_bw, nvlink_lat, d=7168):
        """TP2: 权重/KV 对半 + NVLink AllReduce."""
        qkv_ns = single["Proj_QKV_roofline_ns"] / 2 / single["Proj_QKV_bw_util"]
        attn_ns = single["attn_roofline_ns"] / 2 / single["attn_bw_util"]
        o_ns = single["Proj_O_roofline_ns"] / 2 / single["Proj_O_bw_util"]
        gpu = qkv_ns + attn_ns + o_ns
        comm = (bs * d * 2) / (nvlink_bw * 1e3) + 2 * nvlink_lat
        return {
            "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
            "hybrid_attn_ns": round(attn_ns, 2),
            "hybrid_Proj_O_ns": round(o_ns, 2),
            "hybrid_wall_ns": round(gpu + comm, 2),
            "comm_total_ns": round(comm, 2),
        }

    # ── 收集数据 ──
    all_data = {"hmp_reo_new": {}, "h100": {}, "h100_tp2": {},
                "rubin": {}, "rubin_tp2": {}}

    for bs in batches:
        # Ours
        result = calc_mla_strategy(ours_hw, bs, seqs, link_bw=1.5,
                                   hop_latency_ns=15, endpoint_delay_ns=10,
                                   tp_h=tp_h, tp_s=tp_s)
        ours_entries = []
        for e in result["data"]:
            ours_entries.append({
                "seq": e["seq"], "batch": e["batch"],
                "hybrid_Proj_QKV_ns": e["Proj_QKV"]["total_ns"],
                "hybrid_attn_ns": e["attention"]["total_ns"],
                "hybrid_Proj_O_ns": e["Proj_O"]["total_ns"],
                "hybrid_gpu_ns": e["gpu_total_ns"],
                "comm_total_ns": e["comm_total_ns"],
                "hybrid_wall_ns": e["wall_total_ns"],
            })
        all_data["hmp_reo_new"][f"bs{bs}"] = ours_entries

        # Baselines
        h100_list, rubin_list, h100tp2_list, rubintp2_list = [], [], [], []
        for seq in seqs:
            h = _mla_baseline(H100_PEAK, H100_HBM, bs, seq)
            h["seq"] = seq; h["batch"] = bs; h100_list.append(h)
            r = _mla_baseline(RUBIN_PEAK, RUBIN_HBM, bs, seq)
            r["seq"] = seq; r["batch"] = bs; rubin_list.append(r)
            ht = _mla_tp2(h, bs, H100_NVLINK, NVLINK_LATENCY)
            ht["seq"] = seq; ht["batch"] = bs; h100tp2_list.append(ht)
            rt = _mla_tp2(r, bs, RUBIN_NVLINK, NVLINK_LATENCY)
            rt["seq"] = seq; rt["batch"] = bs; rubintp2_list.append(rt)
        all_data["h100"][f"bs{bs}"] = h100_list
        all_data["rubin"][f"bs{bs}"] = rubin_list
        all_data["h100_tp2"][f"bs{bs}"] = h100tp2_list
        all_data["rubin_tp2"][f"bs{bs}"] = rubintp2_list

    output = {
        "_description": f"DeepSeek-V3 MLA (roofline_gqa_calc, tp_h={tp_h} tp_s={tp_s})",
        "_model": MLA_CONFIG,
        "_architecture": f"16 NPUs, tp_h={tp_h}, tp_s={tp_s}",
        "strategies": {k: {"data_by_bs": v} for k, v in all_data.items()},
        "data": all_data["hmp_reo_new"],  # back-compat
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved MLA fig1 data: {output_path}")


def main():
    import argparse

    default_seqs = [1024, 2048, 4096, 8192, 16384, 32768, 65536,
                     131072, 262144, 524288, 1048576]

    parser = argparse.ArgumentParser(
        description="GQA single-layer roofline WITH weight loading."
    )
    parser.add_argument("--peak-perf", type=float, default=60,
                        help="NPU peak performance in TFLOPS (default: 60)")
    parser.add_argument("--bandwidth", type=float, default=2.5,
                        help="HBM bandwidth in TB/s (default: 2.5)")
    parser.add_argument("--link-bw", type=float, default=2.0,
                        help="D2D link bandwidth in TB/s (default: 2.0)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--seq", type=int, nargs="+", default=default_seqs,
                        help="Sequence lengths to evaluate")
    parser.add_argument("--dataset-splits", type=int, default=1,
                        help="preferred-dataset-splits for output_ar chunking (1=no extra split, 16=original)")
    parser.add_argument("--active-chunks", type=int, default=1,
                        help="active-chunks-per-dimension for output_ar pipelining (1=no pipeline)")
    parser.add_argument("--hop-latency", type=float, default=100,
                        help="D2D hop latency in ns (default: 100)")
    parser.add_argument("--endpoint-delay", type=float, default=10,
                        help="Endpoint delay in ns (default: 10)")
    parser.add_argument("--model", type=str, default="qwen3",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model config to use (default: qwen3)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output JSON path (default: auto-generated)")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]

    seq_list = args.seq
    bs = args.batch
    pp = args.peak_perf
    bw_val = args.bandwidth
    link_bw_val = args.link_bw
    ds_splits = args.dataset_splits
    act_chunks = args.active_chunks
    hop_lat = args.hop_latency
    ep_delay = args.endpoint_delay

    device_hw = {"compute": pp, "Bandwidth": bw_val, "device_link_bw": link_bw_val}
    rubin_hw = {
        "compute": hw.rubin_single_layer_config["compute"],
        "Bandwidth": hw.rubin_single_layer_config["Bandwidth"],
        "device_link_bw": hw.rubin_single_layer_config.get("device_link_bw", 1.8),
    }
    h100_hw = {
        "compute": hw.H100_fp8_config["compute"],
        "Bandwidth": hw.H100_fp8_config["Bandwidth"],
        "device_link_bw": hw.H100_fp8_config.get("device_link_bw", 0.9),
    }
    crossover = pp / bw_val

    model_desc = (f"{args.model} (d={model_cfg['d_model']}, "
                  f"Hq={model_cfg['num_attention_heads']}, "
                  f"Hkv={model_cfg['num_kv_heads']}, dk={model_cfg['d_head']})")

    output = {
        "metadata": {
            "_source": "roofline_gqa_calc.py (roofline_qkv + roofline_attention + roofline_output + comm)",
            "description": f"GQA single-layer roofline WITH weight loading + comm, bs={bs}, peak-perf={pp}T",
            "model": model_desc,
            "batch_size": bs,
            "strategies": ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin", "h100"],
            "device_hw": {
                "peak_perf_tflops": pp,
                "bandwidth_tb_s": bw_val,
                "link_bw_tb_s": link_bw_val,
                "crossover_oi": round(crossover, 2),
            },
            "rubin_hw": {
                "peak_perf_tflops": rubin_hw["compute"],
                "bandwidth_tb_s": rubin_hw["Bandwidth"],
                "crossover_oi": round(rubin_hw["compute"] / rubin_hw["Bandwidth"], 2),
            },
            "h100_hw": {
                "peak_perf_tflops": h100_hw["compute"],
                "bandwidth_tb_s": h100_hw["Bandwidth"],
                "crossover_oi": round(h100_hw["compute"] / h100_hw["Bandwidth"], 2),
            },
            "comm_model": "QKV AllGather(tp_s) ring + final tree-reduce log2(tp_h*tp_s) steps",
            "output_ar_chunking": {
                "dataset_splits": ds_splits,
                "active_chunks": act_chunks,
            },
            "d2d_latency": {
                "hop_latency_ns": hop_lat,
                "endpoint_delay_ns": ep_delay,
            },
        },
        "strategies": {},
    }

    for strat in ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]:
        output["strategies"][strat] = calc_strategy(strat, device_hw, bs, seq_list,
                                                     link_bw=link_bw_val,
                                                     dataset_splits=ds_splits,
                                                     active_chunks=act_chunks,
                                                     hop_latency_ns=hop_lat,
                                                     endpoint_delay_ns=ep_delay,
                                                     model_config=model_cfg)
    output["strategies"]["rubin"] = calc_strategy("rubin", rubin_hw, bs, seq_list,
                                                   link_bw=rubin_hw["device_link_bw"],
                                                   dataset_splits=ds_splits,
                                                   active_chunks=act_chunks,
                                                   model_config=model_cfg)
    output["strategies"]["h100"] = calc_strategy("h100", h100_hw, bs, seq_list,
                                                  link_bw=h100_hw["device_link_bw"],
                                                  dataset_splits=ds_splits,
                                                  active_chunks=act_chunks,
                                                  model_config=model_cfg)

    # --- GPU compute table ---
    summary_rows = []
    print(f"{'seq':>8} | {'HMP_reo serial':>14} {'HMP_reo fused':>14} | {'reo_new serial':>14} {'reo_new fused':>14}"
          f" | {'HMP serial':>12} {'HMP fused':>12} | {'TP16 serial':>12} {'TP16 fused':>12} | {'Rubin':>12}")
    print("-" * 160)
    for i, seq in enumerate(seq_list):
        h = output["strategies"]["HMP_reo"]["data"][i]
        n = output["strategies"]["hmp_reo_new"]["data"][i]
        b = output["strategies"]["hmp"]["data"][i]
        t = output["strategies"]["tp16"]["data"][i]
        r = output["strategies"]["rubin"]["data"][i]
        print(f"{seq:>8} | {h['gpu_total_serial_ns']:>12.0f}ns {h['gpu_total_fused_ns']:>12.0f}ns"
              f" | {n['gpu_total_serial_ns']:>12.0f}ns {n['gpu_total_fused_ns']:>12.0f}ns"
              f" | {b['gpu_total_serial_ns']:>10.0f}ns {b['gpu_total_fused_ns']:>10.0f}ns"
              f" | {t['gpu_total_serial_ns']:>10.0f}ns {t['gpu_total_fused_ns']:>10.0f}ns"
              f" | {r['gpu_total_ns']:>10.0f}ns")

        summary_rows.append({
            "seq": seq,
            "HMP_reo_serial_ns": round(h["gpu_total_serial_ns"]),
            "HMP_reo_fused_ns": round(h["gpu_total_fused_ns"]),
            "HMP_reo_wall_serial_ns": round(h["wall_total_serial_ns"]),
            "HMP_reo_wall_fused_ns": round(h["wall_total_fused_ns"]),
            "hmp_reo_new_serial_ns": round(n["gpu_total_serial_ns"]),
            "hmp_reo_new_fused_ns": round(n["gpu_total_fused_ns"]),
            "hmp_reo_new_wall_serial_ns": round(n["wall_total_serial_ns"]),
            "hmp_reo_new_wall_fused_ns": round(n["wall_total_fused_ns"]),
            "hmp_serial_ns": round(b["gpu_total_serial_ns"]),
            "hmp_fused_ns": round(b["gpu_total_fused_ns"]),
            "hmp_wall_serial_ns": round(b["wall_total_serial_ns"]),
            "hmp_wall_fused_ns": round(b["wall_total_fused_ns"]),
            "tp16_serial_ns": round(t["gpu_total_serial_ns"]),
            "tp16_fused_ns": round(t["gpu_total_fused_ns"]),
            "tp16_wall_serial_ns": round(t["wall_total_serial_ns"]),
            "tp16_wall_fused_ns": round(t["wall_total_fused_ns"]),
            "rubin_ns": round(r["gpu_total_ns"]),
        })

    output["summary_comparison"] = summary_rows

    # --- Comm breakdown ---
    print(f"\n--- Communication breakdown (link_bw={link_bw_val} TB/s) ---")
    for strat in ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]:
        comm = output["strategies"][strat].get("comm", {})
        ag = comm.get("qkv_allgather", {})
        ars = comm.get("attn_rs", {})
        red = comm.get("final_reduce", {})
        out_ar = comm.get("output_ar", {})
        out_ag = comm.get("output_ag", {})
        total_comm = comm.get("total_comm_ns", 0)
        ag_ns = ag.get("time_ns", 0)
        ars_ns = ars.get("time_ns", 0)
        red_ns = red.get("time_ns", 0)
        out_ar_ns = out_ar.get("time_ns", 0)
        out_ag_ns = out_ag.get("time_ns", 0)
        print(f"  [{strat.upper():>12}]  QKV_AG={ag_ns:.1f}ns  Attn_RS={ars_ns:.1f}ns  Reduce={red_ns:.1f}ns"
              f"  Out_AR={out_ar_ns:.2f}ns  Out_AG={out_ag_ns:.2f}ns  total_comm={total_comm:.1f}ns")

    # --- Attention breakdown ---
    print(f"\n--- Attention breakdown at seq={seq_list[-1]} ---")
    for strat in ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]:
        d = output["strategies"][strat]["data"][-1]
        a = d["attention"]
        print(f"\n[{strat.upper()}]")
        if "sub_ops" in a:
            for name, info in a["sub_ops"].items():
                if "fused_ns" in info:
                    print(f"  {name}: comp={info['compute_ns']:.1f}ns  mem={info['memory_ns']:.1f}ns  fused={info['fused_ns']:.1f}ns  [{info['bound']}]")
                else:
                    print(f"  {name}: {info['time_ns']:.1f}ns  [{info['bound']}]")
            print(f"  GQA OI = {a['gqa_oi']}")
            print(f"  Total serial={a['total_ns_serial']:.1f}ns  fused={a['total_ns_fused']:.1f}ns")
        else:
            print(f"  Fused: kv_mem={a['kv_mem_ns']:.1f}ns  einsum_comp={a['einsum_comp_ns']:.1f}ns  [{a['bound']}]")
            print(f"  GQA OI = {a['gqa_oi']}")
            print(f"  Total = {a['total_ns']:.1f}ns")

    # --- Wall time (gpu + comm) table ---
    print(f"\n--- Wall time (gpu + comm) at seq={seq_list[-1]} ---")
    for strat in ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]:
        d = output["strategies"][strat]["data"][-1]
        comm_ns = output["strategies"][strat]["comm"]["total_comm_ns"]
        print(f"  [{strat.upper():>8}]  gpu_serial={d['gpu_total_serial_ns']:.0f}ns"
              f"  comm={comm_ns:.1f}ns  wall_serial={d['wall_total_serial_ns']:.0f}ns")
    r = output["strategies"]["rubin"]["data"][-1]
    print(f"  [   RUBIN]  gpu={r['gpu_total_ns']:.0f}ns  comm=0ns  wall={r['gpu_total_ns']:.0f}ns")

    if args.output:
        out_path = Path(args.output)
    else:
        pp_tag = f"{int(pp)}T" if pp == int(pp) else f"{pp}T"
        report_dir = MODEL_REPORT_DIRS.get(args.model, args.model)
        out_path = Path(__file__).resolve().parents[2] / "reports" / report_dir / "roofline" / f"gqa_roofline_with_weights_bs{bs}_{pp_tag}.json"
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
