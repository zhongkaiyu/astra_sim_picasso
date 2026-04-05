#!/usr/bin/env python3
"""Attention-only decode profiler for DeepSeek-V3 MLA architecture.

Profiles one decode attention step (batch=1, seq_q=1) using Multi-head
Latent Attention (MLA) with expanded KV cache:

    Stage               Description
    ─────────────────── ──────────────────────────────────
    q_a_proj            Q compression  (d_model → q_lora_rank)
    q_a_norm            RMSNorm on compressed Q
    q_b_proj            Q expansion    (q_lora_rank → heads*(d_nope+d_rope))
    kv_a_proj           KV compression (d_model → kv_lora_rank + d_rope)
    kv_a_norm           RMSNorm on compressed KV
    kv_b_proj           KV expansion   (kv_lora_rank → heads*(d_nope+d_v))
    rope                Rotary position embedding (q_rope + k_rope)
    attn                FlashAttention-2  flash_attn_with_kvcache
    o_proj              Output GEMM    (heads*d_v → d_model, row-parallel)

Model config: DeepSeek-V3 MLA
    d_model=7168, n_heads=128, q_lora_rank=1536, kv_lora_rank=512
    qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128

Note: Uses expanded KV cache with head_dim=192 (nope+rope concatenated,
V padded to 192). This is an approximation; production MLA uses absorbed
weights with custom attention kernels operating in compressed space.

With --tp N, head dimensions are sharded (heads/tp) on a single GPU.

Usage:
    # e2e latency — TP=1 (full model, GPU 0):
    python ncu_attn_profile_dsv3.py --e2e 100 --seq 1024 4096 16384 65536 131072

    # e2e latency — TP=2 sharded dims (single GPU 1):
    python ncu_attn_profile_dsv3.py --tp 2 --e2e 100 --seq 1024 4096 16384 65536 131072

    # ncu profiling — TP=1 (GPU 0):
    sudo ncu --set full --target-processes all --launch-skip 8 --launch-count 200 \\
        -o H100_results/dsv3_attn_only \\
        python ncu_attn_profile_dsv3.py --seq 1024 4096 16384 65536

    # ncu profiling — TP=2 (GPU 1):
    sudo ncu --set full --target-processes all --launch-skip 8 --launch-count 200 \\
        -o H100_results/dsv3_attn_only_tp2 \\
        python ncu_attn_profile_dsv3.py --tp 2 --seq 1024 4096 16384 65536
"""
import argparse
import statistics
import torch
from flash_attn import flash_attn_with_kvcache

# ── Model config: DeepSeek-V3 MLA ──────────────────────────────
D_MODEL      = 7168
NUM_HEADS    = 128
D_HEAD_NOPE  = 128           # qk_nope_head_dim
D_HEAD_ROPE  = 64            # qk_rope_head_dim
D_HEAD_V     = 128           # v_head_dim
Q_LORA_RANK  = 1536          # q_lora_rank  (d_c')
KV_LORA_RANK = 512           # kv_lora_rank (d_c)
D_HEAD       = D_HEAD_NOPE + D_HEAD_ROPE  # 192 (combined for FlashAttn)
RMS_EPS      = 1e-6

# Derived
KV_A_OUT     = KV_LORA_RANK + D_HEAD_ROPE  # 576

# Default GPU: tp=1 → cuda:5, tp=2 → cuda:0+5
TP_DEFAULT_GPU = {1: 5, 2: 0}

# ── Absorbed MLA correction ─────────────────────────────────────
# Expanded (FlashAttn): K+V per token = 2 × NUM_HEADS × D_HEAD × 2B
# Absorbed (production): c_kv read 2× (score + value) + k_rope 1×
#   = (KV_LORA_RANK × 2 + D_HEAD_ROPE) × 2B per token
BYTES_EXPANDED_PER_TOK  = 2 * NUM_HEADS * D_HEAD * 2      # 98304
BYTES_ABSORBED_PER_TOK  = (2 * KV_LORA_RANK + D_HEAD_ROPE) * 2  # 2176
ATTN_CORRECTION_IDEAL   = BYTES_ABSORBED_PER_TOK / BYTES_EXPANDED_PER_TOK  # ~1/45
ATTN_CORRECTION_PRACTICAL = ATTN_CORRECTION_IDEAL * 3  # ×3 for SRAM tiling overhead


def rms_norm(x, weight, eps=RMS_EPS):
    """RMSNorm — maps to 1 fused CUDA kernel in production."""
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(orig)


def apply_rope(x, cos, sin):
    """RoPE on last dim — maps to 1 fused kernel in production.
    x: (..., D_HEAD_ROPE), cos/sin: (1, D_HEAD_ROPE//2)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _mla_forward(x, W_dq, rms_w_q, W_uq, W_dkv, rms_w_kv, W_ukv, W_o,
                 K_cache, V_cache, cache_seqlens, rope_cos, rope_sin,
                 B, num_heads_local, dev, torch_dtype):
    """Full MLA forward pass (for warmup / _full_layer)."""
    q_comp = rms_norm(x @ W_dq, rms_w_q)
    q_pe = q_comp @ W_uq

    nope_dim = num_heads_local * D_HEAD_NOPE
    q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
    q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)

    kv_a = x @ W_dkv
    c_kv = rms_norm(kv_a[..., :KV_LORA_RANK], rms_w_kv)
    k_rope_raw = kv_a[..., KV_LORA_RANK:]

    kv_b = c_kv @ W_ukv
    k_nope_dim = num_heads_local * D_HEAD_NOPE
    k_nope = kv_b[..., :k_nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
    v_raw = kv_b[..., k_nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)

    q_rope_out = apply_rope(q_rope_raw, rope_cos, rope_sin)
    k_rope_out = apply_rope(
        k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(-1, -1, num_heads_local, -1),
        rope_cos, rope_sin)

    q = torch.cat([q_nope, q_rope_out], dim=-1)
    k = torch.cat([k_nope, k_rope_out], dim=-1)
    v = torch.zeros(B, 1, num_heads_local, D_HEAD, device=dev, dtype=torch_dtype)
    v[..., :D_HEAD_V] = v_raw

    ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                 cache_seqlens=cache_seqlens.clone(), causal=True)
    o_proj_in = num_heads_local * D_HEAD_V
    return ao[..., :D_HEAD_V].contiguous().view(B, o_proj_in) @ W_o


def run_attn_e2e(S, dev, dtype_name, iters=100, tp=1):
    """Measure per-stage wall-clock latency for MLA attention block.

    With tp>1, uses TP-sharded dimensions (heads/tp) on a single GPU.
    """
    torch_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    B = 1

    # Per-rank dimensions
    num_heads_local = NUM_HEADS // tp
    nope_dim = num_heads_local * D_HEAD_NOPE
    q_proj_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_ROPE)
    kv_b_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_V)
    o_proj_in_local = num_heads_local * D_HEAD_V

    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)

    # ── Weights ──
    W_dq     = mk(D_MODEL, Q_LORA_RANK)
    rms_w_q  = torch.ones(Q_LORA_RANK, device=dev, dtype=torch_dtype)
    W_uq     = mk(Q_LORA_RANK, q_proj_out_local)
    W_dkv    = mk(D_MODEL, KV_A_OUT)
    rms_w_kv = torch.ones(KV_LORA_RANK, device=dev, dtype=torch_dtype)
    W_ukv    = mk(KV_LORA_RANK, kv_b_out_local)
    W_o      = mk(o_proj_in_local, D_MODEL)

    x = mk(B, D_MODEL)

    # KV cache (expanded: head_dim=D_HEAD=192, V padded)
    K_cache = mk(B, S + 1, num_heads_local, D_HEAD)
    V_cache = torch.zeros(B, S + 1, num_heads_local, D_HEAD,
                          device=dev, dtype=torch_dtype)
    V_cache[..., :D_HEAD_V] = torch.randn(
        B, S + 1, num_heads_local, D_HEAD_V, device=dev, dtype=torch_dtype)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    # RoPE cos/sin
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD_ROPE, 2,
                    device=dev, dtype=torch.float32) / D_HEAD_ROPE))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(torch_dtype).view(1, D_HEAD_ROPE // 2)
    rope_sin = emb.sin().to(torch_dtype).view(1, D_HEAD_ROPE // 2)

    STAGES = ["q_a_proj", "q_a_norm", "q_b_proj",
              "kv_a_proj", "kv_a_norm", "kv_b_proj",
              "rope", "attn", "o_proj"]

    ev = {s: (torch.cuda.Event(enable_timing=True),
              torch.cuda.Event(enable_timing=True)) for s in STAGES}
    ev["total"] = (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
    times = {s: [] for s in STAGES + ["total"]}

    # Warmup
    torch.cuda.synchronize(dev)
    for _ in range(5):
        _mla_forward(x, W_dq, rms_w_q, W_uq, W_dkv, rms_w_kv, W_ukv, W_o,
                     K_cache, V_cache, cache_seqlens, rope_cos, rope_sin,
                     B, num_heads_local, dev, torch_dtype)
    torch.cuda.synchronize(dev)

    # Timed iterations
    for _ in range(iters):
        ev["total"][0].record()

        # 1. q_a_proj
        ev["q_a_proj"][0].record()
        q_comp = x @ W_dq
        ev["q_a_proj"][1].record()

        # 2. q_a_norm
        ev["q_a_norm"][0].record()
        q_comp = rms_norm(q_comp, rms_w_q)
        ev["q_a_norm"][1].record()

        # 3. q_b_proj
        ev["q_b_proj"][0].record()
        q_pe = q_comp @ W_uq
        ev["q_b_proj"][1].record()

        # 4. kv_a_proj
        ev["kv_a_proj"][0].record()
        kv_a = x @ W_dkv
        ev["kv_a_proj"][1].record()

        # 5. kv_a_norm
        ev["kv_a_norm"][0].record()
        c_kv = kv_a[..., :KV_LORA_RANK]
        c_kv = rms_norm(c_kv, rms_w_kv)
        k_rope_raw = kv_a[..., KV_LORA_RANK:]
        ev["kv_a_norm"][1].record()

        # 6. kv_b_proj
        ev["kv_b_proj"][0].record()
        kv_b = c_kv @ W_ukv
        ev["kv_b_proj"][1].record()

        # 7. rope + reshape
        ev["rope"][0].record()
        q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)
        k_nope = kv_b[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        v_raw = kv_b[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)

        q_rope_out = apply_rope(q_rope_raw, rope_cos, rope_sin)
        k_rope_out = apply_rope(
            k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(
                -1, -1, num_heads_local, -1),
            rope_cos, rope_sin)

        q = torch.cat([q_nope, q_rope_out], dim=-1)
        k = torch.cat([k_nope, k_rope_out], dim=-1)
        v = torch.zeros(B, 1, num_heads_local, D_HEAD,
                        device=dev, dtype=torch_dtype)
        v[..., :D_HEAD_V] = v_raw
        ev["rope"][1].record()

        # 8. attn
        ev["attn"][0].record()
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(),
                                     causal=True)
        ev["attn"][1].record()

        # 9. o_proj
        ev["o_proj"][0].record()
        _ = ao[..., :D_HEAD_V].contiguous().view(B, o_proj_in_local) @ W_o
        ev["o_proj"][1].record()

        ev["total"][1].record()
        torch.cuda.synchronize(dev)

        for s in STAGES + ["total"]:
            times[s].append(ev[s][0].elapsed_time(ev[s][1]) * 1000)  # ms→us

    medians = {s: statistics.median(times[s]) for s in STAGES + ["total"]}
    return medians, STAGES


def run_attn_ncu(S, dev, dtype_name, tp=1):
    """Single profiled iteration with NVTX ranges (for ncu)."""
    torch_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    B = 1

    num_heads_local = NUM_HEADS // tp
    nope_dim = num_heads_local * D_HEAD_NOPE
    q_proj_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_ROPE)
    kv_b_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_V)
    o_proj_in_local = num_heads_local * D_HEAD_V

    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)

    W_dq     = mk(D_MODEL, Q_LORA_RANK)
    rms_w_q  = torch.ones(Q_LORA_RANK, device=dev, dtype=torch_dtype)
    W_uq     = mk(Q_LORA_RANK, q_proj_out_local)
    W_dkv    = mk(D_MODEL, KV_A_OUT)
    rms_w_kv = torch.ones(KV_LORA_RANK, device=dev, dtype=torch_dtype)
    W_ukv    = mk(KV_LORA_RANK, kv_b_out_local)
    W_o      = mk(o_proj_in_local, D_MODEL)

    x = mk(B, D_MODEL)

    K_cache = mk(B, S + 1, num_heads_local, D_HEAD)
    V_cache = torch.zeros(B, S + 1, num_heads_local, D_HEAD,
                          device=dev, dtype=torch_dtype)
    V_cache[..., :D_HEAD_V] = torch.randn(
        B, S + 1, num_heads_local, D_HEAD_V, device=dev, dtype=torch_dtype)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD_ROPE, 2,
                    device=dev, dtype=torch.float32) / D_HEAD_ROPE))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(torch_dtype).view(1, D_HEAD_ROPE // 2)
    rope_sin = emb.sin().to(torch_dtype).view(1, D_HEAD_ROPE // 2)
    torch.cuda.synchronize(dev)

    # Warmup
    for _ in range(2):
        _mla_forward(x, W_dq, rms_w_q, W_uq, W_dkv, rms_w_kv, W_ukv, W_o,
                     K_cache, V_cache, cache_seqlens, rope_cos, rope_sin,
                     B, num_heads_local, dev, torch_dtype)
    torch.cuda.synchronize(dev)

    # Profiled iteration
    tag = f"_S{S}_tp{tp}"

    torch.cuda.nvtx.range_push(f"q_a_proj{tag}")
    q_comp = x @ W_dq
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"q_a_norm{tag}")
    q_comp = rms_norm(q_comp, rms_w_q)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"q_b_proj{tag}")
    q_pe = q_comp @ W_uq
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"kv_a_proj{tag}")
    kv_a = x @ W_dkv
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"kv_a_norm{tag}")
    c_kv = kv_a[..., :KV_LORA_RANK]
    c_kv = rms_norm(c_kv, rms_w_kv)
    k_rope_raw = kv_a[..., KV_LORA_RANK:]
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"kv_b_proj{tag}")
    kv_b = c_kv @ W_ukv
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"rope{tag}")
    q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
    q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)
    k_nope = kv_b[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
    v_raw = kv_b[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)

    q_rope_out = apply_rope(q_rope_raw, rope_cos, rope_sin)
    k_rope_out = apply_rope(
        k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(
            -1, -1, num_heads_local, -1),
        rope_cos, rope_sin)

    q = torch.cat([q_nope, q_rope_out], dim=-1)
    k = torch.cat([k_nope, k_rope_out], dim=-1)
    v = torch.zeros(B, 1, num_heads_local, D_HEAD,
                    device=dev, dtype=torch_dtype)
    v[..., :D_HEAD_V] = v_raw
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"attn{tag}")
    ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                 cache_seqlens=cache_seqlens.clone(),
                                 causal=True)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"o_proj{tag}")
    _ = ao[..., :D_HEAD_V].contiguous().view(B, o_proj_in_local) @ W_o
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    print(f"[done] S={S}, tp={tp}, dtype={dtype_name}, gpu=cuda:{dev.index}")


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 MLA attention-only decode profiler "
                    "(q_proj → kv_proj → rope → attn → o_proj)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel degree (shards head dims, "
                             "still single-GPU execution)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id (default: tp=1→cuda:0, tp=2→cuda:1)")
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536, 131072, 262144])
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"])
    parser.add_argument("--e2e", type=int, default=0, metavar="N",
                        help="E2E latency: N iterations per seq")
    args = parser.parse_args()

    tp = args.tp
    assert NUM_HEADS % tp == 0, \
        f"tp={tp} must divide NUM_HEADS={NUM_HEADS}"

    gpu_id = args.gpu if args.gpu is not None else TP_DEFAULT_GPU.get(tp, 0)
    num_heads_local = NUM_HEADS // tp

    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")

    if args.e2e > 0:
        print(f"\n{'='*90}")
        print(f"  DSV3 MLA Attention E2E — GPU {gpu_id} — TP={tp} — "
              f"{args.e2e} iters")
        print(f"  Per-rank: heads={num_heads_local}, "
              f"d_nope={D_HEAD_NOPE}, d_rope={D_HEAD_ROPE}, d_v={D_HEAD_V}")
        print(f"  Q_lora={Q_LORA_RANK}, KV_lora={KV_LORA_RANK}")
        print(f"{'='*90}")

        stages_hdr = ["q_a_proj", "q_a_norm", "q_b_proj",
                      "kv_a_proj", "kv_a_norm", "kv_b_proj",
                      "rope", "attn", "o_proj"]

        print(f"{'Seq':>8} | ", end="")
        for s in stages_hdr:
            print(f"{s:>10} | ", end="")
        print(f"{'total':>10} | {'attn%':>6} | KV_cache")
        print("-" * (25 + 13 * len(stages_hdr) + 20))

        # Collect results for correction table
        all_results = []

        for S in args.seq:
            # KV cache memory (expanded MHA, head_dim=192, fp16)
            kv_mb = 2 * S * num_heads_local * D_HEAD * 2 / 1e6
            try:
                medians, STAGES = run_attn_e2e(
                    S, dev, args.dtype, iters=args.e2e, tp=tp)
            except torch.cuda.OutOfMemoryError:
                print(f"{S:>8} | OOM (KV cache = {kv_mb:.0f} MB)")
                torch.cuda.empty_cache()
                continue

            print(f"{S:>8} | ", end="")
            for s in STAGES:
                print(f"{medians[s]:>8.1f}us | ", end="")
            attn_pct = medians["attn"] / medians["total"] * 100
            print(f"{medians['total']:>8.1f}us | {attn_pct:>5.1f}% | "
                  f"{kv_mb:.0f}MB")

            all_results.append((S, medians, STAGES))
            torch.cuda.empty_cache()

        # ── Absorbed MLA correction table ─────────────────────────
        if all_results:
            non_attn_stages = [s for s in all_results[0][2] if s != "attn"]
            cf_i = ATTN_CORRECTION_IDEAL
            cf_p = ATTN_CORRECTION_PRACTICAL

            print(f"\n{'='*90}")
            print(f"  Absorbed MLA Correction (decode is memory-bound, "
                  f"t_attn ∝ KV cache bytes)")
            print(f"  Expanded: {BYTES_EXPANDED_PER_TOK} B/tok "
                  f"(128 heads × 192d × 2 × 2B)")
            print(f"  Absorbed: {BYTES_ABSORBED_PER_TOK} B/tok "
                  f"(c_kv×2 + k_rope, shared across heads)")
            print(f"  Ideal factor: ×{cf_i:.4f} (1/{1/cf_i:.0f}), "
                  f"Practical factor: ×{cf_p:.4f} (1/{1/cf_p:.0f}, "
                  f"~3× SRAM tiling overhead)")
            print(f"{'='*90}")
            print(f"{'Seq':>8} | {'attn_meas':>10} | "
                  f"{'attn_ideal':>10} | {'attn_prac':>10} | "
                  f"{'non_attn':>10} | "
                  f"{'total_meas':>10} | {'total_ideal':>11} | "
                  f"{'total_prac':>11}")
            print("-" * 105)

            for S, medians, STAGES in all_results:
                a_meas = medians["attn"]
                a_ideal = a_meas * cf_i
                a_prac = a_meas * cf_p
                non_attn = sum(medians[s] for s in non_attn_stages)
                t_meas = medians["total"]
                t_ideal = non_attn + a_ideal
                t_prac = non_attn + a_prac
                print(f"{S:>8} | {a_meas:>8.1f}us | "
                      f"{a_ideal:>8.1f}us | {a_prac:>8.1f}us | "
                      f"{non_attn:>8.1f}us | "
                      f"{t_meas:>8.1f}us | {t_ideal:>9.1f}us | "
                      f"{t_prac:>9.1f}us")
            print("-" * 105)
    else:
        # ncu mode
        print(f"\n{'='*60}")
        print(f"  DSV3 MLA Attention ncu mode — GPU {gpu_id} — TP={tp}")
        print(f"  Per-rank: heads={num_heads_local}, "
              f"d_nope={D_HEAD_NOPE}, d_rope={D_HEAD_ROPE}")
        print(f"{'='*60}")
        for S in args.seq:
            try:
                run_attn_ncu(S, dev, args.dtype, tp=tp)
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] S={S}")
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
