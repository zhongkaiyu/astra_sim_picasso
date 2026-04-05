#!/usr/bin/env python3
"""Production-style single-layer decode profiler for ncu.

Profiles one transformer decode step (batch=1, seq_q=1) matching real
inference-engine practice (vLLM / TensorRT-LLM / SGLang):

    Stage               Kernel type
    ─────────────────── ──────────────────────────────────
    rmsnorm_attn        fused RMSNorm
    qkv_proj            fused QKV GEMM  (1 matmul)
    rope                fused RoPE kernel
    attn                FlashAttention-2  flash_attn_with_kvcache
    o_proj              output GEMM     (1 matmul)
    allreduce_attn      NCCL all-reduce (TP mode only)
    residual_attn       element-wise add
    rmsnorm_mlp         fused RMSNorm
    gate_up_proj        fused gate+up GEMM (1 matmul)
    silu_mul            SiLU(gate) × up
    down_proj           down GEMM       (1 matmul)
    allreduce_mlp       NCCL all-reduce (TP mode only)
    residual_mlp        element-wise add

Modes:
    --tp 1       Single GPU on cuda:0 (default). 4 GEMMs + 1 FlashAttn.
    --tp 2       Tensor parallel across cuda:0 + cuda:3.
                 Weights sharded, 2× NCCL all-reduce per layer.
    --skip-nccl  Replace NCCL with local copy (required for ncu profiling).
    --e2e N      End-to-end latency mode: run N iterations per seq, measure
                 per-stage wall-clock time via CUDA events. Works with real
                 NCCL. Use with nsys for full timeline including comms.

Usage:
    # 1) ncu: per-kernel utilization (single GPU)
    sudo ncu --set full --target-processes all --launch-skip 24 --launch-count 200 \
        -o H100_results/single_gpu_decode \
        python ncu_decode_profile.py --seq 1024 4096 16384 65536 --dtype fp16

    # 2) ncu: per-kernel utilization (TP=2, skip NCCL)
    sudo ncu --set full --target-processes all --launch-skip 24 --launch-count 400 \
        -o H100_results/tp2_decode \
        python ncu_decode_profile.py --tp 2 --skip-nccl --seq 1024 4096 16384 65536

    # 3) e2e: wall-clock latency per stage (standalone, no ncu needed)
    python ncu_decode_profile.py --tp 1 --e2e 100 --seq 1024 4096 16384 65536
    python ncu_decode_profile.py --tp 2 --e2e 100 --seq 1024 4096 16384 65536

    # 4) nsys: full timeline with NCCL comms (use with --e2e)
    sudo nsys profile --trace=cuda,nvtx,nccl -o H100_results/tp2_nsys \
        python ncu_decode_profile.py --tp 2 --e2e 20 --seq 1024 4096 16384 65536
"""
import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_attn import flash_attn_with_kvcache

# ── Model config: Qwen3-235B GQA ────────────────────────────────
D_MODEL      = 4096
NUM_Q_HEADS  = 64
NUM_KV_HEADS = 4
D_HEAD       = 128
INTERMEDIATE = 14336             # per-expert MLP intermediate (adjust for model)
Q_DIM        = NUM_Q_HEADS  * D_HEAD   # 8192
KV_DIM       = NUM_KV_HEADS * D_HEAD   # 512
QKV_DIM      = Q_DIM + 2 * KV_DIM     # 9216  (fused Q+K+V)
GATE_UP_DIM  = 2 * INTERMEDIATE        # 28672 (fused gate+up)
RMS_EPS      = 1e-6

# TP GPU mapping: rank → cuda device id
TP_GPU_MAP = {0: 1, 1: 2}

DTYPE_CONFIGS = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8":  torch.float8_e4m3fn,
}


# ── Small fused-kernel stand-ins ─────────────────────────────────

def rms_norm(x, weight, eps=RMS_EPS):
    """RMSNorm — maps to 1 fused CUDA kernel in production."""
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(orig)


def apply_rope(x, cos, sin):
    """RoPE on last dim — maps to 1 fused kernel in production.
    x: (..., D_HEAD), cos/sin: (1, D_HEAD//2)."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


def _proj(x, w, sa=None, sb=None):
    """Linear projection; FP8 scaled_mm when scales provided."""
    if sa is not None:
        return torch._scaled_mm(
            x.to(torch.float8_e4m3fn), w.t(),
            scale_a=sa, scale_b=sb,
            out_dtype=torch.float16, use_fast_accum=True)
    return x @ w


# ── Decode step (supports both single-GPU and TP) ───────────────

def run_decode(S, dev, dtype_name, tp=1, rank=0, skip_nccl=False,
               split_qkv=False):
    """Run one decode layer.

    When tp > 1, weights are sharded and NCCL all-reduce is inserted
    after o_proj and down_proj (standard Megatron-style TP).

    skip_nccl: replace real all-reduce with a local copy (same memory
    traffic) so ncu can profile without deadlocking on NCCL replay.
    split_qkv: use separate Q proj and KV proj instead of fused QKV.
    """
    use_nccl = (tp > 1 and not skip_nccl)
    torch_dtype = DTYPE_CONFIGS[dtype_name]
    is_fp8 = dtype_name == "fp8"
    attn_dtype = torch.float16          # FlashAttn requires fp16/bf16
    B = 1

    # ── Per-rank dimensions (sharded for TP) ─────────────────────
    num_q_local  = NUM_Q_HEADS  // tp
    num_kv_local = NUM_KV_HEADS // tp
    q_dim_local  = num_q_local  * D_HEAD    # column-parallel
    kv_dim_local = num_kv_local * D_HEAD
    qkv_dim_local = q_dim_local + 2 * kv_dim_local
    inter_local   = INTERMEDIATE // tp
    gate_up_local = 2 * inter_local

    # ── Weights ───────────────────────────────────────────────────
    kv_proj_dim_local = 2 * kv_dim_local  # K+V fused
    if is_fp8:
        mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16) \
                             .to(torch.float8_e4m3fn)
        sa = torch.ones(1, device=dev, dtype=torch.float32)
        sb = torch.ones(1, device=dev, dtype=torch.float32)
        # FP8 layout: (out, in)
        if split_qkv:
            W_q       = mk(q_dim_local,      D_MODEL)
            W_kv      = mk(kv_proj_dim_local, D_MODEL)
        else:
            W_qkv     = mk(qkv_dim_local,    D_MODEL)
        W_gate_up = mk(gate_up_local,  D_MODEL)
        W_o       = mk(D_MODEL,        q_dim_local)
        W_down    = mk(D_MODEL,        inter_local)
    else:
        mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)
        sa = sb = None
        if split_qkv:
            W_q       = mk(D_MODEL,  q_dim_local)
            W_kv      = mk(D_MODEL,  kv_proj_dim_local)
        else:
            W_qkv     = mk(D_MODEL,       qkv_dim_local)
        W_gate_up = mk(D_MODEL,       gate_up_local)
        W_o       = mk(q_dim_local,   D_MODEL)
        W_down    = mk(inter_local,   D_MODEL)

    rms_w_attn = torch.ones(D_MODEL, device=dev, dtype=torch.float16)
    rms_w_mlp  = torch.ones(D_MODEL, device=dev, dtype=torch.float16)

    # ── Input + KV cache ──────────────────────────────────────────
    if is_fp8:
        x = torch.randn(B, D_MODEL, device=dev, dtype=torch.float16) \
                .to(torch.float8_e4m3fn)
        residual = x.half().clone()
    else:
        x = torch.randn(B, D_MODEL, device=dev, dtype=torch_dtype)
        residual = x.clone()

    # KV cache: per-rank heads
    K_cache = torch.randn(B, S + 1, num_kv_local, D_HEAD,
                          device=dev, dtype=attn_dtype)
    V_cache = torch.randn(B, S + 1, num_kv_local, D_HEAD,
                          device=dev, dtype=attn_dtype)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    # RoPE cos/sin for the new-token position
    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD, 2,
                    device=dev, dtype=torch.float32) / D_HEAD))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(attn_dtype).view(1, D_HEAD // 2)
    rope_sin = emb.sin().to(attn_dtype).view(1, D_HEAD // 2)

    torch.cuda.synchronize(dev)

    # ── Full-layer helper (for warmup) ────────────────────────────
    def _full_layer():
        nonlocal residual
        h = rms_norm(x.half() if is_fp8 else x, rms_w_attn)
        if is_fp8:
            h = h.to(torch.float8_e4m3fn)
        if split_qkv:
            q_out = _proj(h, W_q, sa, sb)
            kv_out = _proj(h, W_kv, sa, sb)
            q = q_out.half().view(B, 1, num_q_local, D_HEAD)
            k, v = kv_out.half().split([kv_dim_local, kv_dim_local], dim=-1)
            k = k.view(B, 1, num_kv_local, D_HEAD)
            v = v.view(B, 1, num_kv_local, D_HEAD)
        else:
            qkv = _proj(h, W_qkv, sa, sb)
            q, k, v = qkv.half().split([q_dim_local, kv_dim_local, kv_dim_local], dim=-1)
            q = q.view(B, 1, num_q_local, D_HEAD)
            k = k.view(B, 1, num_kv_local, D_HEAD)
            v = v.view(B, 1, num_kv_local, D_HEAD)
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)
        attn_out = flash_attn_with_kvcache(
            q, K_cache, V_cache, k=k, v=v,
            cache_seqlens=cache_seqlens.clone(), causal=True)
        af = attn_out.contiguous().view(B, q_dim_local)
        if is_fp8:
            af = af.to(torch.float8_e4m3fn)
        o = _proj(af, W_o, sa, sb)
        if use_nccl:
            dist.all_reduce(o)
        elif tp > 1:
            o = o.clone()  # simulate allreduce memory traffic
        residual = residual.half() + o.half()
        h2 = rms_norm(residual, rms_w_mlp)
        if is_fp8:
            h2 = h2.to(torch.float8_e4m3fn)
        gu = _proj(h2, W_gate_up, sa, sb)
        gate, up = gu.half().split([inter_local, inter_local], dim=-1)
        hidden = F.silu(gate.float()).to(attn_dtype) * up
        if is_fp8:
            hidden = hidden.to(torch.float8_e4m3fn)
        d = _proj(hidden, W_down, sa, sb)
        if use_nccl:
            dist.all_reduce(d)
        elif tp > 1:
            d = d.clone()
        residual = residual + d.half()

    # ── Warmup (skipped by ncu --launch-skip) ─────────────────────
    for _ in range(2):
        _full_layer()
    torch.cuda.synchronize(dev)

    # Reset residual
    residual = (x.half() if is_fp8 else x).clone()

    # ── Profiled iteration — each stage in its own NVTX range ─────
    tag = f"_S{S}" if tp == 1 else f"_S{S}_r{rank}"

    # 1. Pre-attention RMSNorm
    torch.cuda.nvtx.range_push(f"rmsnorm_attn{tag}")
    h = rms_norm(x.half() if is_fp8 else x, rms_w_attn)
    if is_fp8:
        h = h.to(torch.float8_e4m3fn)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 2. QKV projection
    if split_qkv:
        # 2a. Q projection
        torch.cuda.nvtx.range_push(f"q_proj{tag}")
        q_out = _proj(h, W_q, sa, sb)
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

        # 2b. KV projection
        torch.cuda.nvtx.range_push(f"kv_proj{tag}")
        kv_out = _proj(h, W_kv, sa, sb)
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()
    else:
        torch.cuda.nvtx.range_push(f"qkv_proj{tag}")
        qkv = _proj(h, W_qkv, sa, sb)
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

    # 3. RoPE
    torch.cuda.nvtx.range_push(f"rope{tag}")
    if split_qkv:
        q = q_out.half().view(B, 1, num_q_local, D_HEAD)
        k, v = kv_out.half().split([kv_dim_local, kv_dim_local], dim=-1)
        k = k.view(B, 1, num_kv_local, D_HEAD)
        v = v.view(B, 1, num_kv_local, D_HEAD)
    else:
        q, k, v = qkv.half().split([q_dim_local, kv_dim_local, kv_dim_local], dim=-1)
        q = q.view(B, 1, num_q_local,  D_HEAD)
        k = k.view(B, 1, num_kv_local, D_HEAD)
        v = v.view(B, 1, num_kv_local, D_HEAD)
    q = apply_rope(q, rope_cos, rope_sin)
    k = apply_rope(k, rope_cos, rope_sin)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 4. FlashAttention-2 decode
    torch.cuda.nvtx.range_push(f"attn{tag}")
    attn_out = flash_attn_with_kvcache(
        q, K_cache, V_cache,
        k=k, v=v,
        cache_seqlens=cache_seqlens.clone(),
        causal=True)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 5. Output projection (row-parallel)
    torch.cuda.nvtx.range_push(f"o_proj{tag}")
    attn_flat = attn_out.contiguous().view(B, q_dim_local)
    if is_fp8:
        attn_flat = attn_flat.to(torch.float8_e4m3fn)
    o_out = _proj(attn_flat, W_o, sa, sb)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 5b. All-reduce after o_proj (TP only)
    if tp > 1:
        torch.cuda.nvtx.range_push(f"allreduce_attn{tag}")
        if use_nccl:
            dist.all_reduce(o_out)
        else:
            o_out = o_out.clone()  # simulate allreduce memory traffic
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

    # 6. Residual add (attention)
    torch.cuda.nvtx.range_push(f"residual_attn{tag}")
    residual = residual.half() + o_out.half()
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 7. Pre-MLP RMSNorm
    torch.cuda.nvtx.range_push(f"rmsnorm_mlp{tag}")
    h2 = rms_norm(residual, rms_w_mlp)
    if is_fp8:
        h2 = h2.to(torch.float8_e4m3fn)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 8. Fused gate+up projection (single GEMM, column-parallel)
    torch.cuda.nvtx.range_push(f"gate_up_proj{tag}")
    gu = _proj(h2, W_gate_up, sa, sb)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 9. SiLU × gate
    torch.cuda.nvtx.range_push(f"silu_mul{tag}")
    gate, up = gu.half().split([inter_local, inter_local], dim=-1)
    hidden = F.silu(gate.float()).to(attn_dtype) * up
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 10. MLP down projection (row-parallel)
    torch.cuda.nvtx.range_push(f"down_proj{tag}")
    if is_fp8:
        hidden = hidden.to(torch.float8_e4m3fn)
    down = _proj(hidden, W_down, sa, sb)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 10b. All-reduce after down_proj (TP only)
    if tp > 1:
        torch.cuda.nvtx.range_push(f"allreduce_mlp{tag}")
        if use_nccl:
            dist.all_reduce(down)
        else:
            down = down.clone()
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

    # 11. Residual add (MLP)
    torch.cuda.nvtx.range_push(f"residual_mlp{tag}")
    residual = residual + down.half()
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    print(f"[done] S={S}, dtype={dtype_name}, tp={tp}, rank={rank}, "
          f"gpu=cuda:{dev.index}")


# ── E2E latency measurement ─────────────────────────────────────

def run_decode_e2e(S, dev, dtype_name, tp=1, rank=0, iters=100,
                   split_qkv=False):
    """Measure per-stage wall-clock latency using CUDA events.

    Returns dict of {stage_name: median_us}.
    Runs real NCCL when tp>1 (no skip).
    """
    torch_dtype = DTYPE_CONFIGS[dtype_name]
    is_fp8 = dtype_name == "fp8"
    attn_dtype = torch.float16
    B = 1
    use_nccl = (tp > 1)

    num_q_local  = NUM_Q_HEADS  // tp
    num_kv_local = NUM_KV_HEADS // tp
    q_dim_local  = num_q_local  * D_HEAD
    kv_dim_local = num_kv_local * D_HEAD
    qkv_dim_local = q_dim_local + 2 * kv_dim_local
    kv_proj_dim_local = 2 * kv_dim_local
    inter_local   = INTERMEDIATE // tp
    gate_up_local = 2 * inter_local

    # ── Weights ───────────────────────────────────────────────────
    if is_fp8:
        mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16) \
                             .to(torch.float8_e4m3fn)
        sa = torch.ones(1, device=dev, dtype=torch.float32)
        sb = torch.ones(1, device=dev, dtype=torch.float32)
        if split_qkv:
            W_q       = mk(q_dim_local,      D_MODEL)
            W_kv      = mk(kv_proj_dim_local, D_MODEL)
        else:
            W_qkv     = mk(qkv_dim_local, D_MODEL)
        W_gate_up = mk(gate_up_local, D_MODEL)
        W_o       = mk(D_MODEL,       q_dim_local)
        W_down    = mk(D_MODEL,       inter_local)
    else:
        mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)
        sa = sb = None
        if split_qkv:
            W_q       = mk(D_MODEL, q_dim_local)
            W_kv      = mk(D_MODEL, kv_proj_dim_local)
        else:
            W_qkv     = mk(D_MODEL,      qkv_dim_local)
        W_gate_up = mk(D_MODEL,      gate_up_local)
        W_o       = mk(q_dim_local,  D_MODEL)
        W_down    = mk(inter_local,  D_MODEL)

    rms_w_attn = torch.ones(D_MODEL, device=dev, dtype=torch.float16)
    rms_w_mlp  = torch.ones(D_MODEL, device=dev, dtype=torch.float16)

    if is_fp8:
        x = torch.randn(B, D_MODEL, device=dev, dtype=torch.float16) \
                .to(torch.float8_e4m3fn)
    else:
        x = torch.randn(B, D_MODEL, device=dev, dtype=torch_dtype)

    K_cache = torch.randn(B, S + 1, num_kv_local, D_HEAD,
                          device=dev, dtype=attn_dtype)
    V_cache = torch.randn(B, S + 1, num_kv_local, D_HEAD,
                          device=dev, dtype=attn_dtype)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD, 2,
                    device=dev, dtype=torch.float32) / D_HEAD))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(attn_dtype).view(1, D_HEAD // 2)
    rope_sin = emb.sin().to(attn_dtype).view(1, D_HEAD // 2)

    torch.cuda.synchronize(dev)

    # ── Stage definitions ─────────────────────────────────────────
    if split_qkv:
        STAGES = ["rmsnorm_attn", "q_proj", "kv_proj", "rope", "attn", "o_proj"]
    else:
        STAGES = ["rmsnorm_attn", "qkv_proj", "rope", "attn", "o_proj"]
    if tp > 1:
        STAGES.append("allreduce_attn")
    STAGES += ["residual_attn", "rmsnorm_mlp", "gate_up_proj", "silu_mul",
               "down_proj"]
    if tp > 1:
        STAGES.append("allreduce_mlp")
    STAGES.append("residual_mlp")

    # Pre-create CUDA events
    events = {s: (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True)) for s in STAGES}
    events["layer"] = (torch.cuda.Event(enable_timing=True),
                       torch.cuda.Event(enable_timing=True))

    all_times = {s: [] for s in STAGES}
    all_times["layer"] = []

    # ── Warmup ────────────────────────────────────────────────────
    for _ in range(5):
        residual = (x.half() if is_fp8 else x).clone()
        h = rms_norm(x.half() if is_fp8 else x, rms_w_attn)
        if is_fp8: h = h.to(torch.float8_e4m3fn)
        if split_qkv:
            q_out = _proj(h, W_q, sa, sb)
            kv_out = _proj(h, W_kv, sa, sb)
            q = apply_rope(q_out.half().view(B,1,num_q_local,D_HEAD), rope_cos, rope_sin)
            ktmp, v = kv_out.half().split([kv_dim_local, kv_dim_local], -1)
            k = apply_rope(ktmp.view(B,1,num_kv_local,D_HEAD), rope_cos, rope_sin)
            v = v.view(B,1,num_kv_local,D_HEAD)
        else:
            qkv = _proj(h, W_qkv, sa, sb)
            q, k, v = qkv.half().split([q_dim_local, kv_dim_local, kv_dim_local], -1)
            q = apply_rope(q.view(B,1,num_q_local,D_HEAD), rope_cos, rope_sin)
            k = apply_rope(k.view(B,1,num_kv_local,D_HEAD), rope_cos, rope_sin)
            v = v.view(B,1,num_kv_local,D_HEAD)
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        af = ao.contiguous().view(B, q_dim_local)
        if is_fp8: af = af.to(torch.float8_e4m3fn)
        o = _proj(af, W_o, sa, sb)
        if use_nccl: dist.all_reduce(o)
        residual = residual.half() + o.half()
        h2 = rms_norm(residual, rms_w_mlp)
        if is_fp8: h2 = h2.to(torch.float8_e4m3fn)
        gu = _proj(h2, W_gate_up, sa, sb)
        gt, up = gu.half().split([inter_local, inter_local], -1)
        hid = F.silu(gt.float()).to(attn_dtype) * up
        if is_fp8: hid = hid.to(torch.float8_e4m3fn)
        d = _proj(hid, W_down, sa, sb)
        if use_nccl: dist.all_reduce(d)
        residual = residual + d.half()
    torch.cuda.synchronize(dev)

    # ── Timed iterations ──────────────────────────────────────────
    for _ in range(iters):
        residual = (x.half() if is_fp8 else x).clone()

        events["layer"][0].record()

        # rmsnorm_attn
        events["rmsnorm_attn"][0].record()
        h = rms_norm(x.half() if is_fp8 else x, rms_w_attn)
        if is_fp8: h = h.to(torch.float8_e4m3fn)
        events["rmsnorm_attn"][1].record()

        # qkv / split q+kv proj
        if split_qkv:
            events["q_proj"][0].record()
            q_out = _proj(h, W_q, sa, sb)
            events["q_proj"][1].record()

            events["kv_proj"][0].record()
            kv_out = _proj(h, W_kv, sa, sb)
            events["kv_proj"][1].record()
        else:
            events["qkv_proj"][0].record()
            qkv = _proj(h, W_qkv, sa, sb)
            events["qkv_proj"][1].record()

        # rope
        events["rope"][0].record()
        if split_qkv:
            q = apply_rope(q_out.half().view(B,1,num_q_local,D_HEAD), rope_cos, rope_sin)
            ktmp, v = kv_out.half().split([kv_dim_local, kv_dim_local], -1)
            k = apply_rope(ktmp.view(B,1,num_kv_local,D_HEAD), rope_cos, rope_sin)
            v = v.view(B,1,num_kv_local,D_HEAD)
        else:
            q, k, v = qkv.half().split([q_dim_local, kv_dim_local, kv_dim_local], -1)
            q = apply_rope(q.view(B,1,num_q_local,D_HEAD), rope_cos, rope_sin)
            k = apply_rope(k.view(B,1,num_kv_local,D_HEAD), rope_cos, rope_sin)
            v = v.view(B,1,num_kv_local,D_HEAD)
        events["rope"][1].record()

        # attn
        events["attn"][0].record()
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        events["attn"][1].record()

        # o_proj
        events["o_proj"][0].record()
        af = ao.contiguous().view(B, q_dim_local)
        if is_fp8: af = af.to(torch.float8_e4m3fn)
        o = _proj(af, W_o, sa, sb)
        events["o_proj"][1].record()

        # allreduce_attn
        if tp > 1:
            events["allreduce_attn"][0].record()
            if use_nccl: dist.all_reduce(o)
            events["allreduce_attn"][1].record()

        # residual_attn
        events["residual_attn"][0].record()
        residual = residual.half() + o.half()
        events["residual_attn"][1].record()

        # rmsnorm_mlp
        events["rmsnorm_mlp"][0].record()
        h2 = rms_norm(residual, rms_w_mlp)
        if is_fp8: h2 = h2.to(torch.float8_e4m3fn)
        events["rmsnorm_mlp"][1].record()

        # gate_up_proj
        events["gate_up_proj"][0].record()
        gu = _proj(h2, W_gate_up, sa, sb)
        events["gate_up_proj"][1].record()

        # silu_mul
        events["silu_mul"][0].record()
        gt, up = gu.half().split([inter_local, inter_local], -1)
        hid = F.silu(gt.float()).to(attn_dtype) * up
        events["silu_mul"][1].record()

        # down_proj
        events["down_proj"][0].record()
        if is_fp8: hid = hid.to(torch.float8_e4m3fn)
        d = _proj(hid, W_down, sa, sb)
        events["down_proj"][1].record()

        # allreduce_mlp
        if tp > 1:
            events["allreduce_mlp"][0].record()
            if use_nccl: dist.all_reduce(d)
            events["allreduce_mlp"][1].record()

        # residual_mlp
        events["residual_mlp"][0].record()
        residual = residual + d.half()
        events["residual_mlp"][1].record()

        events["layer"][1].record()
        torch.cuda.synchronize(dev)

        # Collect times
        for s in STAGES:
            all_times[s].append(events[s][0].elapsed_time(events[s][1]) * 1000)  # ms→us
        all_times["layer"].append(
            events["layer"][0].elapsed_time(events["layer"][1]) * 1000)

    # ── Compute medians ───────────────────────────────────────────
    import statistics
    medians = {}
    for s in STAGES + ["layer"]:
        medians[s] = statistics.median(all_times[s])

    return medians, STAGES


def print_e2e_table(seq_results, tp, rank):
    """Print formatted e2e latency table."""
    if not seq_results:
        return

    _, STAGES = list(seq_results.values())[0]

    print()
    print("=" * 100)
    print(f"  E2E Latency (median of N iters) | TP={tp} rank={rank}")
    print("=" * 100)
    print(f"{'Seq':>8} | {'Stage':<18} | {'Median(us)':>11} | {'% of layer':>10}")
    print("-" * 100)

    for S, (medians, _) in sorted(seq_results.items()):
        layer_us = medians["layer"]
        for s in STAGES:
            pct = medians[s] / layer_us * 100 if layer_us > 0 else 0
            print(f"{S:>8} | {s:<18} | {medians[s]:>11.1f} | {pct:>9.1f}%")
        print(f"{S:>8} | {'LAYER TOTAL':<18} | {layer_us:>11.1f} |")
        print("-" * 100)


# ── TP worker (launched via mp.spawn) ────────────────────────────

def _tp_worker(rank, world_size, seq_list, dtype_name,
               skip_nccl=False, e2e_iters=0, split_qkv=False):
    """Worker for tensor-parallel mode."""
    gpu_id = TP_GPU_MAP[rank]
    dev = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(dev)

    need_nccl = (not skip_nccl) or (e2e_iters > 0)
    if need_nccl:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.set_float32_matmul_precision("high")

    if rank == 0:
        mode = "e2e" if e2e_iters > 0 else ("skip-nccl" if skip_nccl else "ncu")
        print(f"\n{'='*60}")
        print(f"  TP={world_size}  rank 0 → cuda:{TP_GPU_MAP[0]}, "
              f"rank 1 → cuda:{TP_GPU_MAP[1]}  ({mode})")
        print(f"{'='*60}")

    if e2e_iters > 0:
        # E2E latency mode
        seq_results = {}
        for S in seq_list:
            medians, stages = run_decode_e2e(
                S, dev, dtype_name, tp=world_size, rank=rank,
                iters=e2e_iters, split_qkv=split_qkv)
            seq_results[S] = (medians, stages)
            torch.cuda.empty_cache()
        if rank == 0:
            print_e2e_table(seq_results, tp=world_size, rank=rank)
    else:
        # ncu profiling mode
        for S in seq_list:
            run_decode(S, dev, dtype_name, tp=world_size, rank=rank,
                       skip_nccl=skip_nccl, split_qkv=split_qkv)
            torch.cuda.empty_cache()

    if need_nccl:
        dist.destroy_process_group()


# ── Entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Production-style decode profiler for ncu (FlashAttention-2)")
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384, 32768,
                                 65536, 131072, 262144, 524288, 1048576])
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp8"])
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2],
                        help="Tensor parallel degree (1=single GPU cuda:0, "
                             "2=TP across cuda:0 + cuda:3)")
    parser.add_argument("--skip-nccl", action="store_true",
                        help="Replace NCCL all-reduce with local copy. "
                             "Required for ncu profiling (ncu cannot replay "
                             "collective kernels). Each rank runs independently.")
    parser.add_argument("--e2e", type=int, default=0, metavar="N",
                        help="E2E latency mode: run N iterations per seq, "
                             "measure per-stage wall-clock time via CUDA events. "
                             "Works with real NCCL. Pair with nsys for timeline.")
    parser.add_argument("--split-qkv", action="store_true",
                        help="Use separate Q proj + KV proj instead of fused QKV. "
                             "Q proj: (D_MODEL → Q_DIM), KV proj: (D_MODEL → 2*KV_DIM)")
    args = parser.parse_args()

    if args.tp == 1:
        # ── Single GPU mode (cuda:0) ─────────────────────────────
        dev = torch.device(f"cuda:{TP_GPU_MAP[0]}")
        torch.cuda.init()
        torch.set_float32_matmul_precision("high")

        sq = args.split_qkv
        mode_str = "split-qkv" if sq else "fused-qkv"
        if args.e2e > 0:
            print(f"\n{'='*60}")
            print(f"  E2E — Single GPU cuda:{dev.index} — {args.e2e} iters — {mode_str}")
            print(f"{'='*60}")
            seq_results = {}
            for S in args.seq:
                medians, stages = run_decode_e2e(
                    S, dev, args.dtype, tp=1, rank=0, iters=args.e2e,
                    split_qkv=sq)
                seq_results[S] = (medians, stages)
                torch.cuda.empty_cache()
            print_e2e_table(seq_results, tp=1, rank=0)
        else:
            print(f"\n{'='*60}")
            print(f"  Single GPU — cuda:{dev.index} (ncu mode) — {mode_str}")
            print(f"{'='*60}")
            for S in args.seq:
                run_decode(S, dev, args.dtype, tp=1, rank=0, split_qkv=sq)
                torch.cuda.empty_cache()

    elif args.tp == 2:
        # ── Tensor Parallel mode (cuda:0 + cuda:3) ───────────────
        sq = args.split_qkv
        if args.e2e > 0:
            mp.spawn(_tp_worker,
                     args=(2, args.seq, args.dtype, False, args.e2e, sq),
                     nprocs=2, join=True)
        elif args.skip_nccl:
            _tp_worker(0, 2, args.seq, args.dtype,
                       skip_nccl=True, split_qkv=sq)
        else:
            mp.spawn(_tp_worker,
                     args=(2, args.seq, args.dtype, False, 0, sq),
                     nprocs=2, join=True)


if __name__ == "__main__":
    main()
