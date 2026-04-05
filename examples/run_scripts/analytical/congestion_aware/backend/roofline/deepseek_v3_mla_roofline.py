#!/usr/bin/env python3
"""
DeepSeek-V3 MLA (Multi-head Latent Attention) Roofline Model

MLA 与 GQA 的核心区别:
  - KV 联合低秩压缩: d_c=512 (vs GQA 2*Hkv*dk=1024)
  - 矩阵吸收: W_UK 吸收到 W_Q, W_UV 吸收到 W_O → 权重更大但 KV cache 更小
  - 注意力在 d_c=512 压缩空间计算, 加上 d_r=64 RoPE 分量

数据来源: Deppseek-v3.md 附录 A

Per-NPU 模型 (16 NPU, tp_h=4, tp_s=4):
  - QKV 权重按 16 路均分
  - KV cache 按 tp_s=4 切分 (cache_seq = seq/4)
  - Proj_O 权重按 16 路均分
  - 通信: 与 RO_new 类似 (AllGather + Final Reduce)
  - Utilization: 固定 0.95 (用户指定)
"""
import bisect
import json
import math
import sys
from pathlib import Path

# ── DeepSeek-V3 MLA 参数 ──
MLA_CONFIG = {
    "name": "deepseek_v3_mla",
    "d_model": 7168,
    "n_h": 128,          # attention heads
    "d_h": 128,          # per-head dim (conceptual, not used in compressed space)
    "d_c": 512,          # KV compressed latent dim
    "d_c_prime": 1536,   # Q compressed dim
    "d_r": 64,           # RoPE dim
    "n_layers": 61,
}

# ── 架构参数 ──
N_NPUS = 16
TP_H = 4
TP_S = 4

# ── 通信参数 (与 RO_new 一致) ──
HOP_LATENCY = 15  # ns
ENDPOINT_DELAY = 10  # ns

# ── Utilization 加载 ──
_util_cache = {}

def _load_util(num_sa):
    """加载 deepseek3_util_{num_sa}.json, 缓存结果"""
    if num_sa not in _util_cache:
        base = Path(__file__).resolve().parent
        # 向上查找 congestion_aware 目录
        while base.name != "congestion_aware" and base != base.parent:
            base = base.parent
        path = base / f"deepseek3_util_{num_sa}.json"
        if not path.exists():
            import subprocess
            subprocess.run([sys.executable, str(base / "util.py"), "deepseek3", str(num_sa)],
                           capture_output=True, cwd=str(base))
        with open(path) as f:
            _util_cache[num_sa] = json.load(f)
    return _util_cache[num_sa]


def _lookup_util(util_map, key):
    """从 utilization map 中查找值 (支持 log2 线性插值)"""
    parsed = {}
    for k, v in util_map.items():
        if k.startswith("_"): continue
        parsed[int(k)] = v["utilization"] if isinstance(v, dict) else float(v)
    if key in parsed:
        return parsed[key]
    sorted_keys = sorted(parsed)
    if key <= sorted_keys[0]: return parsed[sorted_keys[0]]
    if key >= sorted_keys[-1]: return parsed[sorted_keys[-1]]
    idx = bisect.bisect_right(sorted_keys, key) - 1
    lo, hi = sorted_keys[idx], sorted_keys[idx + 1]
    t = (math.log2(key) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return parsed[lo] + t * (parsed[hi] - parsed[lo])


def mla_roofline(peak_perf, hbm_bw, link_bw, batch, seq, num_sa=None):
    """计算 MLA 单层 decode 的 per-NPU wall time (使用真实 SA utilization).

    Args:
        peak_perf: TFLOPS per NPU
        hbm_bw: TB/s per NPU (HBM bandwidth)
        link_bw: TB/s (D2D link bandwidth)
        batch: batch size
        seq: sequence length (total KV cache length)
        num_sa: SA 数量 (用于查找 utilization, 默认 = peak_perf)

    Returns:
        dict with per-module breakdown
    """
    if num_sa is None:
        num_sa = int(peak_perf)

    # 加载对应算力的 utilization
    util_data = _load_util(num_sa)
    qkv_util = _lookup_util(util_data["proj_qkv"], batch)
    # score/attention 用 effective_seq = cache_seq (seq/tp_s)
    cache_seq_for_lookup = seq // TP_S
    score_util = _lookup_util(util_data["score"], cache_seq_for_lookup)
    attn_v_util = _lookup_util(util_data["attention"], cache_seq_for_lookup)
    proj_o_util = _lookup_util(util_data["proj_o"], batch)

    cfg = MLA_CONFIG
    d_model = cfg["d_model"]
    n_h = cfg["n_h"]
    d_c = cfg["d_c"]
    d_c_prime = cfg["d_c_prime"]
    d_r = cfg["d_r"]

    # Per-NPU heads: n_h / N_NPUS
    heads_per_npu = n_h // N_NPUS  # 128/16 = 8
    cache_seq = seq // TP_S  # KV cache split by tp_s

    # ═══════════════════════════════════════════════
    #  Proj_QKV (Step 1 + Step 2 in MLA decode)
    # ═══════════════════════════════════════════════
    # Down projections (shared, but distributed):
    #   W_DKV: [d_c × d_model] = [512 × 7168] → per NPU: full (each NPU needs c_KV)
    #   W_KR:  [d_r × d_model] = [64 × 7168]  → per NPU: full
    #   W_DQ:  [d_c' × d_model] = [1536 × 7168] → per NPU: full
    # Absorbed Q (per head, distributed):
    #   W_Q_abs: [d_c × d_c'] per head → per NPU: heads_per_npu heads
    #   W_QR:    [d_r × d_c'] per head → per NPU: heads_per_npu heads

    # Weight elements per NPU (FP8 = 1 byte/elem)
    qkv_down_weights = (d_c + d_r + d_c_prime) * d_model  # shared down projections
    qkv_absorbed_weights = heads_per_npu * (d_c * d_c_prime + d_r * d_c_prime)  # per-NPU heads
    qkv_weight_elems = qkv_down_weights + qkv_absorbed_weights

    # Activation: input = bs * d_model, output = bs * (d_c + d_r + heads_per_npu*(d_c+d_r))
    qkv_act_elems = batch * d_model + batch * (d_c + d_r + heads_per_npu * (d_c + d_r))
    qkv_total_mem = qkv_weight_elems + qkv_act_elems

    # MACs per NPU
    qkv_down_macs = batch * (d_c + d_r + d_c_prime) * d_model  # down projections
    qkv_absorbed_macs = batch * heads_per_npu * (d_c + d_r) * d_c_prime  # absorbed Q
    qkv_total_macs = qkv_down_macs + qkv_absorbed_macs

    qkv_compute_ns = 2 * qkv_total_macs / (peak_perf * 1e12) * 1e9
    qkv_memory_ns = qkv_total_mem / (hbm_bw * 1e12) * 1e9
    qkv_adj_compute = qkv_compute_ns / qkv_util if qkv_util > 0 else qkv_compute_ns
    qkv_ns = max(qkv_adj_compute, qkv_memory_ns)

    # ═══════════════════════════════════════════════
    #  Attention (Step 3 + Step 5 in MLA decode)
    # ═══════════════════════════════════════════════
    # Per NPU: heads_per_npu heads, each over cache_seq positions
    # Score: q_abs[d_c] · c_KV[d_c] + q_R[d_r] · k_R[d_r] → d_c + d_r MACs per pos
    # Attn_V: α · c_KV → o_comp[d_c] → d_c MACs per pos
    # Total per position per head: d_c + d_r + d_c = 2*d_c + d_r = 1088

    attn_macs_per_pos = 2 * d_c + d_r  # score(d_c+d_r) + value(d_c)
    attn_total_macs = batch * heads_per_npu * cache_seq * attn_macs_per_pos

    # Memory: KV cache per NPU = cache_seq * (d_c + d_r) elements
    # (c_KV_cache[cache_seq × d_c] + k_R_cache[cache_seq × d_r])
    attn_kv_cache_elems = batch * cache_seq * (d_c + d_r)

    attn_compute_ns = 2 * attn_total_macs / (peak_perf * 1e12) * 1e9
    attn_memory_ns = attn_kv_cache_elems / (hbm_bw * 1e12) * 1e9
    # score 和 attn_v 分别调整, 取平均作为整体 attn util
    avg_attn_util = (score_util + attn_v_util) / 2 if (score_util + attn_v_util) > 0 else 1.0
    attn_adj_compute = attn_compute_ns / avg_attn_util
    attn_ns = max(attn_adj_compute, attn_memory_ns)

    # ═══════════════════════════════════════════════
    #  Proj_O (Step 6 in MLA decode)
    # ═══════════════════════════════════════════════
    # W_O_abs[i]: [d_model × d_c] per head
    # Per NPU: heads_per_npu heads

    proj_o_weight_elems = heads_per_npu * d_model * d_c
    proj_o_act_elems = batch * (heads_per_npu * d_c + d_model)
    proj_o_total_mem = proj_o_weight_elems + proj_o_act_elems

    proj_o_total_macs = batch * heads_per_npu * d_model * d_c

    proj_o_compute_ns = 2 * proj_o_total_macs / (peak_perf * 1e12) * 1e9
    proj_o_memory_ns = proj_o_total_mem / (hbm_bw * 1e12) * 1e9
    proj_o_adj_compute = proj_o_compute_ns / proj_o_util if proj_o_util > 0 else proj_o_compute_ns
    proj_o_ns = max(proj_o_adj_compute, proj_o_memory_ns)

    # ═══════════════════════════════════════════════
    #  Communication (RO_new style: AllGather + Final Reduce)
    # ═══════════════════════════════════════════════
    # QKV AllGather within tp_s group: reassemble d_c chunks
    g = TP_S  # GPUs per group
    fixed_per_step = HOP_LATENCY + ENDPOINT_DELAY  # 25ns

    # AllGather: (tp_s-1) steps, chunk = bs * heads_per_npu * d_c (per-head absorbed Q)
    ag_chunk = batch * (n_h // TP_H) * d_c // TP_S  # approximate
    ag_steps = TP_S - 1
    ag_bw_ns = ag_chunk / (link_bw * 1e3)
    ag_ns = ag_steps * (fixed_per_step + ag_bw_ns)

    # Final Reduce: log2(16) steps, msg = bs * d_model
    red_msg = batch * d_model
    red_steps = int(math.log2(N_NPUS))
    red_bw_ns = red_msg / (link_bw * 1e3)
    red_ns = red_steps * (fixed_per_step + red_bw_ns)

    comm_ns = ag_ns + red_ns

    gpu_ns = qkv_ns + attn_ns + proj_o_ns
    wall_ns = gpu_ns + comm_ns

    return {
        "seq": seq,
        "batch": batch,
        "peak_perf": peak_perf,
        "hbm_bw": hbm_bw,
        "link_bw": link_bw,
        "hybrid_Proj_QKV_ns": round(qkv_ns, 2),
        "hybrid_attn_ns": round(attn_ns, 2),
        "hybrid_Proj_O_ns": round(proj_o_ns, 2),
        "hybrid_gpu_ns": round(gpu_ns, 2),
        "comm_total_ns": round(comm_ns, 2),
        "hybrid_wall_ns": round(wall_ns, 2),
        "qkv_compute_ns": round(qkv_compute_ns, 2),
        "qkv_memory_ns": round(qkv_memory_ns, 2),
        "qkv_weight_elems": qkv_weight_elems,
        "attn_compute_ns": round(attn_compute_ns, 2),
        "attn_memory_ns": round(attn_memory_ns, 2),
        "attn_kv_cache_elems": attn_kv_cache_elems,
        "proj_o_compute_ns": round(proj_o_compute_ns, 2),
        "proj_o_memory_ns": round(proj_o_memory_ns, 2),
        "proj_o_weight_elems": proj_o_weight_elems,
        "qkv_util": round(qkv_util, 6),
        "score_util": round(score_util, 6),
        "attn_v_util": round(attn_v_util, 6),
        "proj_o_util": round(proj_o_util, 6),
        "cache_seq": cache_seq,
        "heads_per_npu": heads_per_npu,
    }


def generate_fig1_data(output_path):
    """生成 fig1 speedup 所需数据: seq sweep at bs=1,4,16,32"""
    seqs = [1024, 2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576]
    batches = [1, 4, 16, 32]
    peak_perf = 96  # 96T config
    hbm_bw = 2.5
    link_bw = 1.5

    results = {}
    for bs in batches:
        entries = []
        for seq in seqs:
            e = mla_roofline(peak_perf, hbm_bw, link_bw, bs, seq)
            entries.append(e)
        results[f"bs{bs}"] = entries

    with open(output_path, "w") as f:
        json.dump({
            "_description": "DeepSeek-V3 MLA roofline on 16-NPU (RO_new strategy)",
            "_model": MLA_CONFIG,
            "_architecture": f"{N_NPUS} NPUs, tp_h={TP_H}, tp_s={TP_S}",
            "_utilization": "SA model from util.py (per num_sa)",
            "_config": f"peak={peak_perf}T, HBM={hbm_bw}TB/s, D2D={link_bw}TB/s",
            "data": results,
        }, f, indent=2)
    print(f"Saved fig1 data: {output_path}")


def generate_heatmap_data(output_path):
    """生成 fig6 热力图数据: Compute × D2D BW × Batch"""
    sa_vals = [8, 16, 32, 64, 96, 128, 256]
    lbw_vals = [0.5, 1.0, 1.5, 2.0, 2.5]
    batch_vals = [1, 4, 16, 32]
    seq_vals = [8192, 65536, 131072]
    hbm_bw = 2.5

    results = []
    for bs in batch_vals:
        for seq in seq_vals:
            for pp in sa_vals:
                for lbw in lbw_vals:
                    e = mla_roofline(pp, hbm_bw, lbw, bs, seq)
                    results.append(e)

    with open(output_path, "w") as f:
        json.dump({
            "_description": "DeepSeek-V3 MLA heatmap sweep",
            "_model": MLA_CONFIG,
            "_utilization": "SA model from util.py (per num_sa)",
            "_parameters": {
                "compute_tflops": sa_vals,
                "link_bw_tbs": lbw_vals,
                "batch_sizes": batch_vals,
                "seq_lengths": seq_vals,
            },
            "data": results,
        }, f, indent=2)
    print(f"Saved heatmap data: {output_path}")


def main():
    base = Path(__file__).resolve().parent

    # 快速验证
    e = mla_roofline(96, 2.5, 1.5, 1, 65536)
    print("=== DeepSeek-V3 MLA @ 96T, bs=1, seq=64K ===")
    print(f"  QKV:   {e['hybrid_Proj_QKV_ns']:>8.0f} ns (comp={e['qkv_compute_ns']:.0f}, mem={e['qkv_memory_ns']:.0f}, weights={e['qkv_weight_elems']:,})")
    print(f"  Attn:  {e['hybrid_attn_ns']:>8.0f} ns (comp={e['attn_compute_ns']:.0f}, mem={e['attn_memory_ns']:.0f}, kv_cache={e['attn_kv_cache_elems']:,})")
    print(f"  ProjO: {e['hybrid_Proj_O_ns']:>8.0f} ns (comp={e['proj_o_compute_ns']:.0f}, mem={e['proj_o_memory_ns']:.0f}, weights={e['proj_o_weight_elems']:,})")
    print(f"  Comm:  {e['comm_total_ns']:>8.0f} ns")
    print(f"  Wall:  {e['hybrid_wall_ns']:>8.0f} ns")
    print()

    # 对比 Qwen3 GQA RO_new @ 96T
    try:
        with open(base / "reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_split4_bw1500_util96.json") as f:
            q3 = json.load(f)
        q3_e = next(x for x in q3["strategies"]["hmp_reo_new"]["data"] if x["seq"] == 65536)
        print(f"=== 对比 Qwen3 GQA RO_new @ 96T, bs=1, seq=64K ===")
        print(f"  QKV:   {q3_e['hybrid_Proj_QKV_ns']:>8.0f} ns")
        print(f"  Attn:  {q3_e['hybrid_attn_ns']:>8.0f} ns")
        print(f"  ProjO: {q3_e['hybrid_Proj_O_ns']:>8.0f} ns")
        print(f"  Comm:  {q3_e['comm_total_ns']:>8.0f} ns")
        print(f"  Wall:  {q3_e['hybrid_wall_ns']:>8.0f} ns")
        print()
        print(f"  MLA/GQA wall ratio: {e['hybrid_wall_ns'] / q3_e['hybrid_wall_ns']:.2f}x")
        print(f"  MLA QKV weights:  {e['qkv_weight_elems']:>12,} vs GQA: ~2,363,968")
        print(f"  MLA ProjO weights:{e['proj_o_weight_elems']:>12,} vs GQA: ~2,103,296")
        print(f"  MLA KV cache/NPU: {e['attn_kv_cache_elems']:>12,} vs GQA: ~838,860")
    except Exception as ex:
        print(f"(Qwen3 comparison skipped: {ex})")

    print()

    # 生成数据
    fig1_path = base / "paper_figures/fig1_e2e_latency/data"
    fig1_path.mkdir(parents=True, exist_ok=True)
    generate_fig1_data(fig1_path / "deepseek_v3_mla_fig1.json")

    fig6_path = base / "paper_figures/fig6_design_exploration/data"
    fig6_path.mkdir(parents=True, exist_ok=True)
    generate_heatmap_data(fig6_path / "deepseek_v3_mla_heatmap.json")


if __name__ == "__main__":
    main()
