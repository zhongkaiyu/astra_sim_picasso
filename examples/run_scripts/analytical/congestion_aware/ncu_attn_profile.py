#!/usr/bin/env python3
"""Attention-only decode profiler — qkv_proj → rope → attn → o_proj.

Stages:
    qkv_proj   Fused QKV GEMM  (1 matmul, column-parallel)
    rope       Rotary position embedding
    attn       FlashAttention-2  flash_attn_with_kvcache
    o_proj     Output GEMM     (1 matmul, row-parallel)

With --tp N, dimensions are sharded as standard Megatron-style TP:
    Q heads  = 64/tp,  KV heads = 4/tp
    W_qkv: (D_MODEL, qkv_dim_local)  column-parallel
    W_o:   (q_dim_local, D_MODEL)     row-parallel
    Default GPU: tp=1 → cuda:0, tp=2 → cuda:1

Usage:
    # e2e latency — TP=1 (full model, GPU 0):
    python ncu_attn_profile.py --e2e 100 --seq 1024 4096 16384 65536 131072 262144 524288 1048576

    # e2e latency — TP=2 sharded dims (single GPU 1):
    python ncu_attn_profile.py --tp 2 --e2e 100 --seq 1024 4096 16384 65536 131072 262144 524288 1048576

    # ncu profiling — TP=1 (GPU 0):
    sudo ncu --set full --target-processes all --launch-skip 8 --launch-count 200 \
        -o H100_results/attn_only \
        python ncu_attn_profile.py --seq 1024 4096 16384 65536 131072 262144

    # ncu profiling — TP=2 (GPU 1):
    sudo ncu --set full --target-processes all --launch-skip 8 --launch-count 200 \
        -o H100_results/attn_only_tp2 \
        python ncu_attn_profile.py --tp 2 --seq 1024 4096 16384 65536 131072 262144

    # split Q / KV:
    python ncu_attn_profile.py --tp 2 --e2e 100 --split-qkv --seq 1024 65536 131072 524288
"""
import argparse
import statistics
import torch
from flash_attn import flash_attn_with_kvcache

# ── Model config: Qwen3-235B GQA ────────────────────────────────
D_MODEL      = 4096
NUM_Q_HEADS  = 64
NUM_KV_HEADS = 4
D_HEAD       = 128

# Default GPU: tp=1 → cuda:0, tp=2 → cuda:1
TP_DEFAULT_GPU = {1: 0, 2: 1}


def apply_rope(x, cos, sin):
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def run_attn_e2e(S, dev, dtype_name, iters=100, split_qkv=False, tp=1):
    """Measure per-stage wall-clock latency for attention block only.

    With tp>1, uses TP-sharded dimensions (heads/tp) on a single GPU.
    """
    torch_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    B = 1

    # Per-rank dimensions
    num_q_local  = NUM_Q_HEADS  // tp
    num_kv_local = NUM_KV_HEADS // tp
    q_dim_local  = num_q_local  * D_HEAD
    kv_dim_local = num_kv_local * D_HEAD
    qkv_dim_local = q_dim_local + 2 * kv_dim_local

    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)

    if split_qkv:
        W_q  = mk(D_MODEL, q_dim_local)
        W_kv = mk(D_MODEL, 2 * kv_dim_local)
    else:
        W_qkv = mk(D_MODEL, qkv_dim_local)
    W_o = mk(q_dim_local, D_MODEL)

    x = mk(B, D_MODEL)

    K_cache = mk(B, S + 1, num_kv_local, D_HEAD)
    V_cache = mk(B, S + 1, num_kv_local, D_HEAD)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD, 2,
                    device=dev, dtype=torch.float32) / D_HEAD))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(torch_dtype).view(1, D_HEAD // 2)
    rope_sin = emb.sin().to(torch_dtype).view(1, D_HEAD // 2)

    # Stage names
    if split_qkv:
        STAGES = ["q_proj", "kv_proj", "rope", "attn", "o_proj"]
    else:
        STAGES = ["qkv_proj", "rope", "attn", "o_proj"]

    ev = {s: (torch.cuda.Event(enable_timing=True),
              torch.cuda.Event(enable_timing=True)) for s in STAGES}
    ev["total"] = (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
    times = {s: [] for s in STAGES + ["total"]}

    # Warmup
    torch.cuda.synchronize(dev)
    for _ in range(5):
        if split_qkv:
            q = (x @ W_q).view(B, 1, num_q_local, D_HEAD)
            kv = x @ W_kv
            k, v = kv.split([kv_dim_local, kv_dim_local], -1)
            k = k.view(B, 1, num_kv_local, D_HEAD)
            v = v.view(B, 1, num_kv_local, D_HEAD)
        else:
            qkv = x @ W_qkv
            q, k, v = qkv.split([q_dim_local, kv_dim_local, kv_dim_local], -1)
            q = q.view(B, 1, num_q_local, D_HEAD)
            k = k.view(B, 1, num_kv_local, D_HEAD)
            v = v.view(B, 1, num_kv_local, D_HEAD)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        _ = ao.contiguous().view(B, q_dim_local) @ W_o
    torch.cuda.synchronize(dev)

    # Timed iterations
    for _ in range(iters):
        ev["total"][0].record()

        if split_qkv:
            ev["q_proj"][0].record()
            q_out = x @ W_q
            ev["q_proj"][1].record()

            ev["kv_proj"][0].record()
            kv_out = x @ W_kv
            ev["kv_proj"][1].record()

            ev["rope"][0].record()
            q = apply_rope(q_out.view(B, 1, num_q_local, D_HEAD), rope_cos, rope_sin)
            ktmp, v = kv_out.split([kv_dim_local, kv_dim_local], -1)
            k = apply_rope(ktmp.view(B, 1, num_kv_local, D_HEAD), rope_cos, rope_sin)
            v = v.view(B, 1, num_kv_local, D_HEAD)
            ev["rope"][1].record()
        else:
            ev["qkv_proj"][0].record()
            qkv = x @ W_qkv
            ev["qkv_proj"][1].record()

            ev["rope"][0].record()
            q, k, v = qkv.split([q_dim_local, kv_dim_local, kv_dim_local], -1)
            q = apply_rope(q.view(B, 1, num_q_local, D_HEAD), rope_cos, rope_sin)
            k = apply_rope(k.view(B, 1, num_kv_local, D_HEAD), rope_cos, rope_sin)
            v = v.view(B, 1, num_kv_local, D_HEAD)
            ev["rope"][1].record()

        ev["attn"][0].record()
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        ev["attn"][1].record()

        ev["o_proj"][0].record()
        _ = ao.contiguous().view(B, q_dim_local) @ W_o
        ev["o_proj"][1].record()

        ev["total"][1].record()
        torch.cuda.synchronize(dev)

        for s in STAGES + ["total"]:
            times[s].append(ev[s][0].elapsed_time(ev[s][1]) * 1000)  # ms→us

    medians = {s: statistics.median(times[s]) for s in STAGES + ["total"]}
    return medians, STAGES


def run_attn_ncu(S, dev, dtype_name, split_qkv=False, tp=1):
    """Single profiled iteration with NVTX ranges (for ncu)."""
    torch_dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
    B = 1

    # Per-rank dimensions
    num_q_local  = NUM_Q_HEADS  // tp
    num_kv_local = NUM_KV_HEADS // tp
    q_dim_local  = num_q_local  * D_HEAD
    kv_dim_local = num_kv_local * D_HEAD
    qkv_dim_local = q_dim_local + 2 * kv_dim_local

    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)

    if split_qkv:
        W_q  = mk(D_MODEL, q_dim_local)
        W_kv = mk(D_MODEL, 2 * kv_dim_local)
    else:
        W_qkv = mk(D_MODEL, qkv_dim_local)
    W_o = mk(q_dim_local, D_MODEL)
    x = mk(B, D_MODEL)

    K_cache = mk(B, S + 1, num_kv_local, D_HEAD)
    V_cache = mk(B, S + 1, num_kv_local, D_HEAD)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD, 2,
                    device=dev, dtype=torch.float32) / D_HEAD))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(torch_dtype).view(1, D_HEAD // 2)
    rope_sin = emb.sin().to(torch_dtype).view(1, D_HEAD // 2)
    torch.cuda.synchronize(dev)

    # Warmup
    for _ in range(2):
        if split_qkv:
            q = (x @ W_q).view(B, 1, num_q_local, D_HEAD)
            kv = x @ W_kv
            k, v = kv.split([kv_dim_local, kv_dim_local], -1)
            k = k.view(B, 1, num_kv_local, D_HEAD)
            v = v.view(B, 1, num_kv_local, D_HEAD)
        else:
            qkv = x @ W_qkv
            q, k, v = qkv.split([q_dim_local, kv_dim_local, kv_dim_local], -1)
            q = q.view(B, 1, num_q_local, D_HEAD)
            k = k.view(B, 1, num_kv_local, D_HEAD)
            v = v.view(B, 1, num_kv_local, D_HEAD)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        _ = ao.contiguous().view(B, q_dim_local) @ W_o
    torch.cuda.synchronize(dev)

    # Profiled iteration
    tag = f"_S{S}_tp{tp}"

    if split_qkv:
        torch.cuda.nvtx.range_push(f"q_proj{tag}")
        q_out = x @ W_q
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"kv_proj{tag}")
        kv_out = x @ W_kv
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"rope{tag}")
        q = apply_rope(q_out.view(B, 1, num_q_local, D_HEAD), rope_cos, rope_sin)
        ktmp, v = kv_out.split([kv_dim_local, kv_dim_local], -1)
        k = apply_rope(ktmp.view(B, 1, num_kv_local, D_HEAD), rope_cos, rope_sin)
        v = v.view(B, 1, num_kv_local, D_HEAD)
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()
    else:
        torch.cuda.nvtx.range_push(f"qkv_proj{tag}")
        qkv = x @ W_qkv
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push(f"rope{tag}")
        q, k, v = qkv.split([q_dim_local, kv_dim_local, kv_dim_local], -1)
        q = q.view(B, 1, num_q_local, D_HEAD)
        k = k.view(B, 1, num_kv_local, D_HEAD)
        v = v.view(B, 1, num_kv_local, D_HEAD)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"attn{tag}")
    ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                 cache_seqlens=cache_seqlens.clone(), causal=True)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push(f"o_proj{tag}")
    _ = ao.contiguous().view(B, q_dim_local) @ W_o
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    print(f"[done] S={S}, tp={tp}, dtype={dtype_name}, gpu=cuda:{dev.index}")


def main():
    parser = argparse.ArgumentParser(
        description="Attention-only decode profiler (qkv → rope → attn → o_proj)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel degree (shards head dims, "
                             "still single-GPU execution)")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU id (default: tp=1→cuda:0, tp=2→cuda:1)")
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 4096, 16384, 65536, 131072,
                                 262144, 524288, 1048576])
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"])
    parser.add_argument("--e2e", type=int, default=0, metavar="N",
                        help="E2E latency: N iterations per seq")
    parser.add_argument("--split-qkv", action="store_true",
                        help="Separate Q proj + KV proj instead of fused QKV")
    args = parser.parse_args()

    tp = args.tp
    assert NUM_Q_HEADS % tp == 0 and NUM_KV_HEADS % tp == 0, \
        f"tp={tp} must divide both NUM_Q_HEADS={NUM_Q_HEADS} and NUM_KV_HEADS={NUM_KV_HEADS}"

    gpu_id = args.gpu if args.gpu is not None else TP_DEFAULT_GPU.get(tp, 0)
    num_q_local  = NUM_Q_HEADS  // tp
    num_kv_local = NUM_KV_HEADS // tp

    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")

    mode = "split-qkv" if args.split_qkv else "fused-qkv"

    if args.e2e > 0:
        print(f"\n{'='*80}")
        print(f"  Attention E2E — GPU {gpu_id} — TP={tp} — {args.e2e} iters — {mode}")
        print(f"  Per-rank: Q_heads={num_q_local}, KV_heads={num_kv_local}, D_HEAD={D_HEAD}")
        print(f"{'='*80}")

        # Header
        if args.split_qkv:
            stages_hdr = ["q_proj", "kv_proj", "rope", "attn", "o_proj"]
        else:
            stages_hdr = ["qkv_proj", "rope", "attn", "o_proj"]

        print(f"{'Seq':>8} | ", end="")
        for s in stages_hdr:
            print(f"{s:>10} | ", end="")
        print(f"{'total':>10} | {'attn%':>6} | KV_cache")
        print("-" * (25 + 13 * len(stages_hdr) + 20))

        for S in args.seq:
            kv_mb = 2 * S * num_kv_local * D_HEAD * 2 / 1e6
            try:
                medians, STAGES = run_attn_e2e(
                    S, dev, args.dtype, iters=args.e2e,
                    split_qkv=args.split_qkv, tp=tp)
            except torch.cuda.OutOfMemoryError:
                print(f"{S:>8} | OOM (KV cache = {kv_mb:.0f} MB)")
                torch.cuda.empty_cache()
                continue

            print(f"{S:>8} | ", end="")
            for s in STAGES:
                print(f"{medians[s]:>8.1f}us | ", end="")
            attn_pct = medians["attn"] / medians["total"] * 100
            print(f"{medians['total']:>8.1f}us | {attn_pct:>5.1f}% | {kv_mb:.0f}MB")

            torch.cuda.empty_cache()
    else:
        # ncu mode
        print(f"\n{'='*60}")
        print(f"  Attention ncu mode — GPU {gpu_id} — TP={tp} — {mode}")
        print(f"  Per-rank: Q_heads={num_q_local}, KV_heads={num_kv_local}")
        print(f"{'='*60}")
        for S in args.seq:
            try:
                run_attn_ncu(S, dev, args.dtype,
                             split_qkv=args.split_qkv, tp=tp)
            except torch.cuda.OutOfMemoryError:
                print(f"[OOM] S={S}")
                torch.cuda.empty_cache()
                continue
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
