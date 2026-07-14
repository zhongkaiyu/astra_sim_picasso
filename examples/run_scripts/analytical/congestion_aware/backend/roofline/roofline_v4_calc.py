#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
roofline_v4_calc.py — DeepSeek-V4 CSA / HCA attention roofline (decode), AMMA "ours" 切分.

与 roofline_gqa_calc.py 的 MLA 函数完全同构（同样的 per-NPU compute + 解析 comm，
返回结构对齐 calc_mla_strategy），但换成 DeepSeek-V4 的两种混合注意力：

  CSA = Compressed Sparse Attention   (报告 §2.3.1)
        每 m 个 token 压成 1 条 KV entry；lightning indexer 给所有 ceil(L/m) 条
        compressed block 打分，top-k 选 k 条进入 core；再加 sliding-window n_win 条。
        core 是 shared-KV MQA（1 条 entry 同时当 K/V，被全部 n_h 个 head 共享）。
  HCA = Heavily Compressed Attention  (报告 §2.3.2)
        每 m'(≫m) 个 token 压成 1 条 entry，**dense**（不做 top-k），core 同为 shared-KV MQA。

关键 roofline 事实（推导见 figures/.../exp3_MLA_breakdown/arithmetic_intensity.md）：
  - core MQA 计算强度 AI = 4*n_h / (KV_bytes/elem)，与 S_eff（attend 的 entry 数）无关：
      Pro(n_h=128, KV 混精 576B/512elem=1.125B) -> AI≈455 FLOP/B；纯 FP8 口径 512。
  - CSA core 的 S_eff = k + n_win，**与 L 无关**（稀疏）；HCA 的 S_eff = ceil(L/m') + n_win。
  - CSA 另有 indexer：扫 ceil(L/m) 条 compressed block，每条 c_I 维点积、n_I_h 个头，FP4。
  - 在 AMMA 上 AI≫crossover(38) -> attention compute-bound；但 S_eff 被砍后绝对 FLOP 很小，
    decode attention 不再像 MLA 那样随 L 线性暴涨成为瓶颈（CSA 几乎恒定 + indexer 缓增；HCA ∝L/m'）。

KV 存储（报告 §2.3.3，混合精度）：每条 entry c=512 维 = 64 维 RoPE(BF16,2B) + 448 维(FP8,1B)=576 B。
indexer keys 走 FP4（0.5 B/elem）。
"""
from __future__ import annotations
import math

# DeepSeek-V4-Pro 注意力超参（报告 §4.2.1）。Flash 见 V4_CONFIGS["flash"]。
V4_CONFIG = {
    "d_model": 7168,
    "n_h": 128,            # query heads
    "c": 512,              # head dim（K=V 合一，shared-KV MQA）
    "d_c_prime": 1536,     # query 压缩维度 d_c
    "rope_dims": 64,       # partial RoPE：最后 64 维 BF16，其余 FP8
    # CSA
    "m": 4,                # CSA 压缩率
    "top_k": 1024,         # CSA 稀疏 top-k（compressed entries）
    "n_I_h": 64,           # lightning indexer query heads
    "c_I": 128,            # indexer head dim
    # HCA
    "m_prime": 128,        # HCA 压缩率（≫m）
    # 公共
    "n_win": 128,          # sliding window
    "g": 16,               # output projection groups
    "d_g": 1024,           # 每组 intermediate dim
    "num_layers": 61,
}

V4_CONFIGS = {
    "pro": V4_CONFIG,
    "flash": {**V4_CONFIG, "d_model": 4096, "n_h": 64, "d_c_prime": 1024,
              "top_k": 512, "g": 8, "num_layers": 43},
}

_BPE_FP8 = 1.0          # 权重/激活字节（FP8）
_BPE_FP4 = 0.5          # indexer FP4


def _kv_bytes_per_entry(cfg) -> float:
    """一条 compressed KV entry 的字节：rope 维 BF16(2B) + 其余 FP8(1B)。"""
    rope = cfg["rope_dims"]
    return rope * 2 + (cfg["c"] - rope) * 1   # 64*2 + 448*1 = 576


def make_v4_npu_config(variant: str, tp_h: int = 4, tp_s: int = 4,
                       model: str = "pro") -> dict:
    """V4 per-NPU 配置（与 make_mla_npu_config 对应）。

    variant ∈ {"csa","hca"}。tp_h 切 query head，tp_s 切 KV 序列/entry 维。
    num_kv_heads=1（shared-KV，不按 tp_h 切）。core 维度 d_score=d_value=c（partial RoPE 就地）。
    """
    if variant not in ("csa", "hca"):
        raise ValueError(f"variant 必须是 csa/hca，收到 {variant!r}")
    mc = V4_CONFIGS[model]
    Hq_npu = mc["n_h"] // tp_h
    return {
        "variant": variant,
        "d_model": mc["d_model"],
        "num_attention_heads": Hq_npu,        # per-NPU query heads
        "num_kv_heads": 1,
        "c": mc["c"],
        "d_score": mc["c"],                   # = c（无 MLA 的 576/512 不对称）
        "d_value": mc["c"],
        "d_c_prime": mc["d_c_prime"],
        "rope_dims": mc["rope_dims"],
        "kv_bytes_per_entry": _kv_bytes_per_entry(mc),
        "m": mc["m"], "m_prime": mc["m_prime"],
        "top_k": mc["top_k"], "n_win": mc["n_win"],
        "n_I_h": mc["n_I_h"], "c_I": mc["c_I"],
        "g": mc["g"], "d_g": mc["d_g"],
        "tp_s": tp_s, "tp_h": tp_h,
        "n_h_total": mc["n_h"],
        "num_layers": mc["num_layers"],
        "v4": True,
    }


def roofline_v4_qkv(peak_perf, bw, cfg, bs):
    """V4 QKV / 压缩 / indexer-query 投影（absorbed 形式，与 roofline_mla_qkv 同构）。

    (a) down/compress（shared，每 NPU 整份）：KV-compress W_KV[c×d] + 权重 W_Z[c×d]
        + query down W_DQ[d_c'×d]。
    (b) absorbed per-head query up W_UQ[c×d_c']，按 N_NPUS 切 head。
    """
    d = cfg["d_model"]
    Hq = cfg["num_attention_heads"]
    c = cfg["c"]
    d_c_prime = cfg["d_c_prime"]
    tp_s = cfg.get("tp_s", 1)
    tp_h = cfg.get("tp_h", 1)
    n_npus = tp_h * tp_s
    # per-head 权重按 n_h_total/n_npus 摊 (== Hq//tp_s)；不要再用 Hq//n_npus（多除一次 tp_h）。
    heads_per_npu = max(1, cfg["n_h_total"] // n_npus)

    # down/compress 投影 (W_KV+W_Z+W_DQ) 是 shared latent 权重。旧模型让每个 NPU 各读
    # 整份 (replicated)，在多个小 NPU 上按单卡带宽重复读取——这是 V4 decode 的主导项,
    # 使 Ours 的 QKV 在低单卡带宽下严重吃亏。改为沿 N_NPUS 张量并行切分: 每 NPU 只读
    # 1/N 的 down 权重, 计算后做一次 latent AllReduce 重建 (comm 在 calc_v4_strategy 里计)。
    down_weight = ((c + c + d_c_prime) * d) // n_npus      # W_KV + W_Z + W_DQ, 切到 N_NPUS
    down_mac = (bs * (c + c + d_c_prime) * d) // n_npus
    abs_weight = heads_per_npu * c * d_c_prime      # W_UQ per head（按 NPU 切）
    abs_mac = bs * heads_per_npu * c * d_c_prime

    weight_elems = down_weight + abs_weight
    total_mac = down_mac + abs_mac
    act_in = bs * d
    act_out = bs * ((c + c + d_c_prime) + heads_per_npu * c)
    total_mem = (weight_elems + act_in + act_out) * _BPE_FP8

    flops = 2 * total_mac
    comp_ns = flops / (peak_perf * 1e12) * 1e9
    mem_ns = total_mem / (bw * 1e12) * 1e9
    return {
        "total_ns": max(comp_ns, mem_ns),
        "compute_ns": round(comp_ns, 2),
        "memory_ns": round(mem_ns, 2),
        "bound": "memory" if mem_ns > comp_ns else "compute",
        "weight_elems": weight_elems,
        "total_macs": total_mac,
        "oi": round(flops / total_mem, 2),
    }


def roofline_v4_attention(peak_perf, bw, cfg, bs, seq):
    """V4 core MQA (+ CSA indexer) 的 per-NPU roofline。

    core：S_eff 条 KV entry（每条 c 维，读一次，被 Hq 个 head 共享，用于 score+value）。
      - CSA：S_eff_full = top_k + n_win（与 L 无关）
      - HCA：S_eff_full = ceil(L/m') + n_win
      tp_s 把 entry 维均分到各 NPU -> per-NPU S_eff = ceil(S_eff_full / tp_s)
    indexer（仅 CSA）：扫 ceil(L/m) 条 block，n_I_h 头 × c_I 维点积，FP4；
      heads 按 tp_h、blocks 按 tp_s 切 -> per-NPU = full / n_npus。
    """
    variant = cfg["variant"]
    Hq = cfg["num_attention_heads"]
    c = cfg["c"]
    tp_s = cfg.get("tp_s", 1)
    tp_h = cfg.get("tp_h", 1)
    n_npus = tp_h * tp_s
    kv_bpe = cfg["kv_bytes_per_entry"]

    if variant == "csa":
        s_full = cfg["top_k"] + cfg["n_win"]
    else:
        s_full = math.ceil(seq / cfg["m_prime"]) + cfg["n_win"]
    s_eff = math.ceil(s_full / tp_s) if tp_s > 1 else s_full

    # ── core MQA：score(c) + value(c)，KV 读一次 ──
    core_mac = 2 * bs * Hq * s_eff * c            # score + value
    core_flops = 2 * core_mac
    kv_bytes = bs * s_eff * kv_bpe                # 一份 KV，被全部 head 复用
    core_comp_ns = core_flops / (peak_perf * 1e12) * 1e9
    core_mem_ns = kv_bytes / (bw * 1e12) * 1e9
    core_ns = max(core_comp_ns, core_mem_ns)
    core_ai = core_flops / kv_bytes if kv_bytes else 0

    # ── CSA indexer ──
    idx_ns = idx_comp_ns = idx_mem_ns = 0.0
    idx_flops = idx_blocks = 0
    if variant == "csa":
        blocks_full = math.ceil(seq / cfg["m"])
        blocks = math.ceil(blocks_full / n_npus)   # heads(tp_h)×blocks(tp_s) 均分
        idx_mac = bs * cfg["n_I_h"] * blocks * cfg["c_I"]
        idx_flops = 2 * idx_mac
        idx_bytes = bs * blocks * cfg["c_I"] * _BPE_FP4
        idx_comp_ns = idx_flops / (peak_perf * 1e12) * 1e9
        idx_mem_ns = idx_bytes / (bw * 1e12) * 1e9
        idx_ns = max(idx_comp_ns, idx_mem_ns)
        idx_blocks = blocks

    total_ns = core_ns + idx_ns
    return {
        "total_ns": round(total_ns, 2),
        "variant": variant,
        "s_eff_per_npu": s_eff,
        "kv_bytes_per_entry": kv_bpe,
        "core_ns": round(core_ns, 2),
        "core_ai_flop_per_byte": round(core_ai, 1),
        "core_bound": "memory" if core_mem_ns > core_comp_ns else "compute",
        "indexer_ns": round(idx_ns, 2),
        "indexer_blocks_per_npu": idx_blocks,
        "sub_ops": {
            "core_mqa": {"compute_ns": round(core_comp_ns, 2),
                         "memory_ns": round(core_mem_ns, 2),
                         "macs": core_mac},
            "indexer": {"compute_ns": round(idx_comp_ns, 2),
                        "memory_ns": round(idx_mem_ns, 2),
                        "flops": idx_flops},
        },
    }


def roofline_v4_output(peak_perf, bw, cfg, bs):
    """V4 grouped output 投影（absorbed），按 N_NPUS 切 value 维 c，与 roofline_mla_output 同构。"""
    d = cfg["d_model"]
    n_h = cfg["n_h_total"]              # 全部 heads；W_O 是 per-head 权重
    c = cfg["c"]
    tp_s = cfg.get("tp_s", 1)
    tp_h = cfg.get("tp_h", 1)
    n_npus = tp_h * tp_s
    c_shard = max(1, c // n_npus)

    # per-NPU 权重 = n_h × d × (c/N) = 总量/N（16 芯片下限）；勿用 Hq（会得到总量/64）。
    w_elems = n_h * d * c_shard
    mac = bs * n_h * d * c_shard
    act_elems = bs * (n_h * c_shard + d)
    total_mem = (w_elems + act_elems) * _BPE_FP8

    flops = 2 * mac
    comp_ns = flops / (peak_perf * 1e12) * 1e9
    mem_ns = total_mem / (bw * 1e12) * 1e9
    return {
        "total_ns": max(comp_ns, mem_ns),
        "compute_ns": round(comp_ns, 2),
        "memory_ns": round(mem_ns, 2),
        "bound": "memory" if mem_ns > comp_ns else "compute",
        "weight_elems": w_elems,
        "total_macs": mac,
        "oi": round(flops / total_mem, 2) if total_mem else 0,
    }


def calc_v4_strategy(variant, hw_cfg, bs, seq_list, link_bw=1.5,
                     hop_latency_ns=15, endpoint_delay_ns=10,
                     tp_h=4, tp_s=4, model="pro"):
    """V4 (CSA/HCA) "ours" 策略 roofline，返回结构对齐 calc_mla_strategy。

    comm 与 MLA 同构：QKV AllGather(tp_s) + score/lse AllReduce + o_comp AllReduce
    + ProjO tree-Reduce。CSA 额外一个 indexer top-k 选择的小 AllReduce（折进 score_ar，量级可忽略）。
    """
    cfg = make_v4_npu_config(variant, tp_h=tp_h, tp_s=tp_s, model=model)
    pp = hw_cfg["compute"]
    bw = hw_cfg["Bandwidth"]
    crossover = pp / bw

    Hq = cfg["num_attention_heads"]
    c = cfg["c"]
    d_model = cfg["d_model"]
    n_npus = tp_h * tp_s
    fixed_per_step = hop_latency_ns + endpoint_delay_ns
    heads_per_npu_weight = max(1, cfg["n_h_total"] // n_npus)

    # (0) QKV AllGather：每 NPU 算 heads_per_npu 个 head 的 Q，需 AllGather 重建全部 head
    qkv_ag_chunk = bs * heads_per_npu_weight * c
    qkv_ag_steps = n_npus - 1
    qkv_ag_ns = qkv_ag_steps * (fixed_per_step + qkv_ag_chunk / (link_bw * 1e3))
    # (0b) down-proj AllReduce：down/compress 投影按 N_NPUS 张量并行切分后, 重建完整 latent
    downproj_ar_msg = bs * (2 * c + cfg["d_c_prime"])
    downproj_ar_steps = 2 * int(math.log2(n_npus)) if n_npus > 1 else 0
    downproj_ar_ns = downproj_ar_steps * (fixed_per_step + downproj_ar_msg / (link_bw * 1e3))
    # (1) score/lse AllReduce within tp_s
    score_ar_msg = bs * Hq * 2 * 4
    score_ar_steps = 2 * int(math.log2(tp_s)) if tp_s > 1 else 0
    score_ar_ns = score_ar_steps * (fixed_per_step + score_ar_msg / (link_bw * 1e3))
    # (2) o_comp AllReduce within tp_s
    ocomp_ar_msg = bs * Hq * c
    ocomp_ar_steps = 2 * int(math.log2(tp_s)) if tp_s > 1 else 0
    ocomp_ar_ns = ocomp_ar_steps * (fixed_per_step + ocomp_ar_msg / (link_bw * 1e3))
    # (3) ProjO tree-Reduce across all NPUs
    proj_o_red_msg = bs * d_model
    proj_o_red_steps = int(math.log2(n_npus)) if n_npus > 1 else 0
    proj_o_red_ns = proj_o_red_steps * (fixed_per_step + proj_o_red_msg / (link_bw * 1e3))

    comm_total_ns = qkv_ag_ns + downproj_ar_ns + score_ar_ns + ocomp_ar_ns + proj_o_red_ns

    results = []
    for seq in seq_list:
        qkv = roofline_v4_qkv(pp, bw, cfg, bs)
        attn = roofline_v4_attention(pp, bw, cfg, bs, seq)
        out = roofline_v4_output(pp, bw, cfg, bs)
        gpu_ns = qkv["total_ns"] + attn["total_ns"] + out["total_ns"]
        wall_ns = gpu_ns + comm_total_ns
        results.append({
            "seq": seq, "batch": bs,
            "Proj_QKV": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in qkv.items()},
            "attention": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in attn.items()},
            "Proj_O": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()},
            "gpu_total_ns": round(gpu_ns, 2),
            "comm_total_ns": round(comm_total_ns, 2),
            "wall_total_ns": round(wall_ns, 2),
        })

    return {
        "strategy": f"v4_{variant}_ours",
        "config": cfg,
        "hardware": {"peak_perf_tflops": pp, "bandwidth_tb_s": bw,
                     "roofline_crossover_oi": round(crossover, 2), "link_bw_tb_s": link_bw},
        "comm": {"qkv_ag_ns": round(qkv_ag_ns, 2), "downproj_ar_ns": round(downproj_ar_ns, 2),
                 "score_ar_ns": round(score_ar_ns, 2), "ocomp_ar_ns": round(ocomp_ar_ns, 2),
                 "proj_o_reduce_ns": round(proj_o_red_ns, 2), "total_comm_ns": round(comm_total_ns, 2)},
        "data": results,
    }


if __name__ == "__main__":
    # 自检：打印 AMMA(ours) 下 CSA/HCA 单层 attention 随 CL 的 compute-bound 时延
    ours_hw = {"compute": 96.0, "Bandwidth": 2.5}
    seqs = [2048, 8192, 32768, 131072, 1048576]
    print(f"AMMA crossover OI = {96/2.5:.1f} FLOP/B")
    for variant in ("csa", "hca"):
        print(f"\n== V4-{variant.upper()} (ours, tp_h=4 tp_s=4) per-layer attn ns ==")
        r = calc_v4_strategy(variant, ours_hw, 1, seqs, tp_h=4, tp_s=4)
        for e in r["data"]:
            a = e["attention"]
            print(f"  CL={e['seq']:>8}  attn={a['total_ns']:>9.1f}ns "
                  f"(core={a['core_ns']:.1f} idx={a['indexer_ns']:.1f}, "
                  f"AI={a['core_ai_flop_per_byte']} {a['core_bound']}, "
                  f"S_eff/npu={a['s_eff_per_npu']})")
