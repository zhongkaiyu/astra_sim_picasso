#!/usr/bin/env python3
"""Absorbed MLA decode profiler — matches roofline model stages exactly.

Profiles the 4 stages used in deepseek_v3_mla_roofline.py / util_128.json:

    Stage       Absorbed MLA operation                  GEMM shape (batch=1)
    ─────────── ─────────────────────────────────────── ─────────────────────────
    Proj_QKV    W_DQ + W_DKV + W_UQ_absorbed            3 GEMMs (see below)
    Score       Q_abs · (c_kv || k_rope)^T              (n_heads, d_cr) × (d_cr, S)
    Attention   softmax(score) · c_kv                   (n_heads, S) × (S, d_c)
    Proj_O      absorbed O projection                   (1, n_heads*d_c) × (n_heads*d_c, d_model)

Proj_QKV sub-GEMMs:
    W_DQ:         (bs, 7168) @ (7168, 1536)       Q compression
    W_DKV:        (bs, 7168) @ (7168, 576)         KV compression (c_kv + k_rope)
    W_UQ_abs:     (bs, 1536) @ (1536, 73728)       Absorbed Q expansion (128 heads × 576)

All batch=1 decode GEMMs are memory-bound (AI ≈ 1 FLOP/byte).
BW utilization is the key metric and transfers across architectures.

Usage:
    python ncu_absorbed_mla_profile_dsv3.py --gpu 5 --seq 1024 4096 16384 65536 131072
"""
import argparse
import statistics
import torch

# ── DeepSeek-V3 MLA config (absorbed) ──────────────────────────
D_MODEL      = 7168
N_HEADS      = 128
D_C          = 512       # KV compressed latent dim
D_R          = 64        # RoPE dim
D_CR         = D_C + D_R # 576
D_C_PRIME    = 1536      # Q compressed dim

# Derived absorbed dimensions
PROJ_QKV_SUBS = {
    "W_DQ":      (D_MODEL, D_C_PRIME),           # 7168 × 1536
    "W_DKV":     (D_MODEL, D_CR),                 # 7168 × 576
    "W_UQ_abs":  (D_C_PRIME, N_HEADS * D_CR),     # 1536 × 73728
}
PROJ_O_SHAPE = (N_HEADS * D_C, D_MODEL)           # 65536 × 7168

# H100 specs
H100_PEAK_TFLOPS = 989.4   # FP16 Tensor Core dense
H100_HBM_BW_TBs  = 3.35    # TB/s


def _bench_gemm(M, K, N, dev, dtype, iters=200):
    """Benchmark (M, K) @ (K, N) → median latency in us."""
    A = torch.randn(M, K, device=dev, dtype=dtype)
    W = torch.randn(K, N, device=dev, dtype=dtype)
    # warmup
    for _ in range(10):
        _ = A @ W
    torch.cuda.synchronize(dev)

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = A @ W
        e.record()
        torch.cuda.synchronize(dev)
        times.append(s.elapsed_time(e) * 1000)  # ms → us
    return statistics.median(times)


def _bench_matmul(A_shape, B_shape, dev, dtype, iters=200):
    """Benchmark matmul(A, B) where A and B are pre-allocated."""
    A = torch.randn(*A_shape, device=dev, dtype=dtype)
    B = torch.randn(*B_shape, device=dev, dtype=dtype)
    for _ in range(10):
        _ = torch.matmul(A, B)
    torch.cuda.synchronize(dev)

    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        _ = torch.matmul(A, B)
        e.record()
        torch.cuda.synchronize(dev)
        times.append(s.elapsed_time(e) * 1000)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser(
        description="DSV3 Absorbed MLA decode profiler — 4 roofline stages")
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536, 131072])
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    B = 1  # batch size for decode

    peak_bw = H100_HBM_BW_TBs * 1e12   # bytes/s
    peak_flops = H100_PEAK_TFLOPS * 1e12  # FLOP/s

    print(f"\n{'='*110}")
    print(f"  DSV3 Absorbed MLA Decode Profiling — GPU {args.gpu} — "
          f"{args.iters} iters — batch={B}")
    print(f"  H100: {H100_PEAK_TFLOPS} TFLOPS FP16, "
          f"{H100_HBM_BW_TBs} TB/s HBM3")
    print(f"  Absorbed dims: d_model={D_MODEL}, n_heads={N_HEADS}, "
          f"d_c={D_C}, d_r={D_R}, d_c'={D_C_PRIME}")
    print(f"{'='*110}")

    # ═══════════════════════════════════════════════════════════════
    #  Stage 1: Proj_QKV (batch-dependent, seq-independent)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── Proj_QKV (3 sub-GEMMs, batch={B}) ──")
    print(f"{'Sub-GEMM':<12} | {'Shape':>20} | {'Weight':>10} | "
          f"{'FLOPs':>10} | {'Latency':>10} | {'Eff BW':>10} | "
          f"{'BW util':>8} | {'Eff TFLOP':>10} | {'Comp util':>9}")
    print("-" * 120)

    qkv_total_lat = 0
    qkv_total_flops = 0
    qkv_total_bytes = 0

    for name, (K, N) in PROJ_QKV_SUBS.items():
        flops = 2 * B * K * N
        wbytes = K * N * 2  # FP16
        lat_us = _bench_gemm(B, K, N, dev, dtype, args.iters)
        lat_s = lat_us * 1e-6
        eff_bw = wbytes / lat_s
        bw_util = eff_bw / peak_bw * 100
        eff_tf = flops / lat_s
        comp_util = eff_tf / peak_flops * 100

        qkv_total_lat += lat_us
        qkv_total_flops += flops
        qkv_total_bytes += wbytes

        print(f"{name:<12} | {f'({B},{K})×({K},{N})':>20} | "
              f"{wbytes/1e6:>8.1f}MB | {flops/1e6:>8.1f}M | "
              f"{lat_us:>8.1f}us | {eff_bw/1e9:>8.0f}GB/s | "
              f"{bw_util:>6.1f}% | {eff_tf/1e12:>8.3f} T | "
              f"{comp_util:>7.2f}%")

    qkv_eff_bw = qkv_total_bytes / (qkv_total_lat * 1e-6)
    qkv_bw_util = qkv_eff_bw / peak_bw * 100
    print(f"{'TOTAL':<12} | {'':>20} | "
          f"{qkv_total_bytes/1e6:>8.1f}MB | {qkv_total_flops/1e6:>8.1f}M | "
          f"{qkv_total_lat:>8.1f}us | {qkv_eff_bw/1e9:>8.0f}GB/s | "
          f"{qkv_bw_util:>6.1f}% |")

    # ═══════════════════════════════════════════════════════════════
    #  Stage 2 & 3: Score + Attention (seq-dependent)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n── Score: Q_abs @ KV_cache^T = ({N_HEADS},{D_CR}) × ({D_CR},S) ──")
    print(f"── Attention: softmax @ C_kv = ({N_HEADS},S) × (S,{D_C}) ──")
    print(f"{'Seq':>8} | {'score_lat':>10} | {'score_BW':>10} | "
          f"{'score_util':>10} | {'attn_lat':>10} | {'attn_BW':>10} | "
          f"{'attn_util':>10} | {'KV_cache':>10}")
    print("-" * 100)

    for S in args.seq:
        # Score: (N_HEADS, D_CR) @ (D_CR, S)
        score_flops = 2 * N_HEADS * D_CR * S
        # Memory: read KV cache (S × D_CR × 2B) + Q (tiny) + write output (tiny for large S)
        score_bytes = S * D_CR * 2 + N_HEADS * D_CR * 2 + N_HEADS * S * 2
        kv_mb = S * D_CR * 2 / 1e6

        try:
            score_lat = _bench_matmul(
                (N_HEADS, D_CR), (D_CR, S), dev, dtype, args.iters)
        except torch.cuda.OutOfMemoryError:
            print(f"{S:>8} | OOM")
            torch.cuda.empty_cache()
            continue

        score_eff_bw = score_bytes / (score_lat * 1e-6)
        score_bw_util = score_eff_bw / peak_bw * 100

        # Attention V: (N_HEADS, S) @ (S, D_C)
        attn_flops = 2 * N_HEADS * S * D_C
        # Memory: read attn_weights (N_HEADS×S×2B) + C_kv (S×D_C×2B) + write output
        attn_bytes = N_HEADS * S * 2 + S * D_C * 2 + N_HEADS * D_C * 2

        try:
            attn_lat = _bench_matmul(
                (N_HEADS, S), (S, D_C), dev, dtype, args.iters)
        except torch.cuda.OutOfMemoryError:
            print(f"{S:>8} | score OK, attn OOM")
            torch.cuda.empty_cache()
            continue

        attn_eff_bw = attn_bytes / (attn_lat * 1e-6)
        attn_bw_util = attn_eff_bw / peak_bw * 100

        print(f"{S:>8} | {score_lat:>8.1f}us | "
              f"{score_eff_bw/1e9:>8.0f}GB/s | {score_bw_util:>8.1f}% | "
              f"{attn_lat:>8.1f}us | "
              f"{attn_eff_bw/1e9:>8.0f}GB/s | {attn_bw_util:>8.1f}% | "
              f"{kv_mb:>8.1f}MB")

        torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    #  Stage 4: Proj_O (absorbed: n_heads*d_c → d_model)
    # ═══════════════════════════════════════════════════════════════
    K_o, N_o = PROJ_O_SHAPE
    o_flops = 2 * B * K_o * N_o
    o_bytes = K_o * N_o * 2
    o_lat = _bench_gemm(B, K_o, N_o, dev, dtype, args.iters)
    o_eff_bw = o_bytes / (o_lat * 1e-6)
    o_bw_util = o_eff_bw / peak_bw * 100
    o_comp_util = o_flops / (o_lat * 1e-6) / peak_flops * 100

    print(f"\n── Proj_O: ({B},{K_o}) × ({K_o},{N_o}) ──")
    print(f"  Weight: {o_bytes/1e6:.1f} MB | FLOPs: {o_flops/1e6:.1f} M")
    print(f"  Latency: {o_lat:.1f} us | Eff BW: {o_eff_bw/1e9:.0f} GB/s | "
          f"BW util: {o_bw_util:.1f}% | Comp util: {o_comp_util:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    #  Summary: BW utilization per stage (for Rubin transfer)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print(f"  Summary: H100 BW Utilization (transferable to Rubin)")
    print(f"  batch=1 decode → all stages memory-bound (AI ≈ 1)")
    print(f"{'='*80}")
    print(f"  Proj_QKV (aggregate):  {qkv_bw_util:>6.1f}%  "
          f"({qkv_total_lat:.1f}us, {qkv_total_bytes/1e6:.1f}MB)")
    print(f"  Proj_O:                {o_bw_util:>6.1f}%  "
          f"({o_lat:.1f}us, {o_bytes/1e6:.1f}MB)")
    print(f"  Score / Attention:     see seq-dependent table above")
    print(f"\n  To predict Rubin latency:")
    print(f"    t_stage = weight_bytes / (Rubin_HBM_BW × BW_util)")


if __name__ == "__main__":
    main()
