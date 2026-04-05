#!/usr/bin/env python3
"""Production-style single-layer decode profiler for DeepSeek-V3 (MLA + MoE).

Profiles one transformer decode step (batch=1, seq_q=1) matching real
inference-engine practice:

    Stage               Kernel type
    ─────────────────── ──────────────────────────────────
    rmsnorm_attn        fused RMSNorm (pre-attention)
    q_a_proj            Q compression    (d_model → q_lora_rank)
    q_a_norm            RMSNorm on compressed Q
    q_b_proj            Q expansion      (q_lora_rank → heads*(d_nope+d_rope))
    kv_a_proj           KV compression   (d_model → kv_lora_rank + d_rope)
    kv_a_norm           RMSNorm on compressed KV
    kv_b_proj           KV expansion     (kv_lora_rank → heads*(d_nope+d_v))
    rope                Rotary position embedding
    attn                FlashAttention-2  flash_attn_with_kvcache
    o_proj              Output GEMM      (heads*d_v → d_model, row-parallel)
    allreduce_attn      NCCL all-reduce (TP mode only)
    residual_attn       element-wise add
    rmsnorm_mlp         fused RMSNorm (pre-MLP)
    router              Expert routing   (d_model → n_experts, top-k)
    expert_gate_up      Routed experts gate+up (batched GEMM, top-8)
    expert_silu_mul     SiLU(gate) × up  (batched)
    expert_down         Routed experts down    (batched GEMM)
    expert_combine      Weighted sum of expert outputs
    shared_gate_up      Shared expert gate+up  (1 expert)
    shared_silu_mul     SiLU(gate) × up
    shared_down         Shared expert down
    allreduce_mlp       NCCL all-reduce (TP mode only)
    residual_mlp        element-wise add

Model config: DeepSeek-V3
    MLA: d_model=7168, 128 heads, q_lora=1536, kv_lora=512,
         d_nope=128, d_rope=64, d_v=128
    MoE: 256 routed experts (top-8) + 1 shared expert,
         per-expert intermediate=2048

Note: KV cache uses expanded form (head_dim=192, V padded) for
FlashAttention-2 compatibility.

Modes:
    --tp 1       Single GPU on cuda:1 (default). MLA + MoE.
    --tp 2       Tensor parallel across cuda:1 + cuda:2.
                 Heads sharded, experts column-parallel, 2× NCCL all-reduce.
    --skip-nccl  Replace NCCL with local copy (for ncu profiling).
    --e2e N      End-to-end latency mode: N iterations per seq.

Usage:
    # 1) ncu: per-kernel utilization (single GPU)
    sudo ncu --set full --target-processes all --launch-skip 24 --launch-count 300 \
        -o H100_results/dsv3_single_gpu_decode \
        python ncu_decode_profile_dsv3.py --seq 1024 4096 16384 65536 --dtype fp16

    # 2) ncu: per-kernel utilization (TP=2, skip NCCL)
    sudo ncu --set full --target-processes all --launch-skip 24 --launch-count 600 \
        -o H100_results/dsv3_tp2_decode \
        python ncu_decode_profile_dsv3.py --tp 2 --skip-nccl --seq 1024 4096 16384 65536

    # 3) e2e: wall-clock latency per stage
    python ncu_decode_profile_dsv3.py --tp 1 --e2e 100 --seq 1024 4096 16384 65536
    python ncu_decode_profile_dsv3.py --tp 2 --e2e 100 --seq 1024 4096 16384 65536

    # 4) nsys: full timeline with NCCL comms
    sudo nsys profile --trace=cuda,nvtx,nccl -o H100_results/dsv3_tp2_nsys \
        python ncu_decode_profile_dsv3.py --tp 2 --e2e 20 --seq 1024 4096 16384 65536
"""
import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from flash_attn import flash_attn_with_kvcache

# ── Model config: DeepSeek-V3 MLA ──────────────────────────────
D_MODEL       = 7168
NUM_HEADS     = 128
D_HEAD_NOPE   = 128          # qk_nope_head_dim
D_HEAD_ROPE   = 64           # qk_rope_head_dim
D_HEAD_V      = 128          # v_head_dim
Q_LORA_RANK   = 1536         # q compressed dim (d_c')
KV_LORA_RANK  = 512          # kv compressed latent dim (d_c)
D_HEAD        = D_HEAD_NOPE + D_HEAD_ROPE  # 192 (combined for FlashAttn)
RMS_EPS       = 1e-6

# MoE config
N_ROUTED_EXPERTS  = 256
N_SHARED_EXPERTS  = 1
TOP_K_EXPERTS     = 8
MOE_INTERMEDIATE  = 2048     # per-expert intermediate size

# Derived
KV_A_OUT = KV_LORA_RANK + D_HEAD_ROPE  # 576

# TP GPU mapping: rank → cuda device id
TP_GPU_MAP = {0: 5, 1: 0}

# ── Absorbed MLA correction ─────────────────────────────────────
# Expanded (FlashAttn): K+V per token = 2 × NUM_HEADS × D_HEAD × 2B
# Absorbed (production): c_kv read 2× (score+value) + k_rope 1×
BYTES_EXPANDED_PER_TOK  = 2 * NUM_HEADS * (D_HEAD_NOPE + D_HEAD_ROPE) * 2  # 98304
BYTES_ABSORBED_PER_TOK  = (2 * KV_LORA_RANK + D_HEAD_ROPE) * 2             # 2176
ATTN_CORRECTION_IDEAL   = BYTES_ABSORBED_PER_TOK / BYTES_EXPANDED_PER_TOK   # ~1/45
ATTN_CORRECTION_PRACTICAL = ATTN_CORRECTION_IDEAL * 3  # ×3 for SRAM tiling overhead

DTYPE_CONFIGS = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# ── Small fused-kernel stand-ins ─────────────────────────────────

def rms_norm(x, weight, eps=RMS_EPS):
    """RMSNorm — maps to 1 fused CUDA kernel in production."""
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(orig)


def apply_rope(x, cos, sin):
    """RoPE on last dim — maps to 1 fused kernel in production."""
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin,
                      x2 * cos + x1 * sin], dim=-1)


# ── Decode step (supports both single-GPU and TP) ───────────────

def run_decode(S, dev, dtype_name, tp=1, rank=0, skip_nccl=False):
    """Run one decode layer with MLA attention + MoE MLP.

    When tp > 1, attention heads are sharded and expert intermediate is
    column-parallel. NCCL all-reduce after o_proj and expert_down.
    """
    use_nccl = (tp > 1 and not skip_nccl)
    torch_dtype = DTYPE_CONFIGS[dtype_name]
    attn_dtype = torch.float16
    B = 1

    # ── Per-rank dimensions ──────────────────────────────────────
    num_heads_local = NUM_HEADS // tp
    nope_dim = num_heads_local * D_HEAD_NOPE
    q_proj_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_ROPE)
    kv_b_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_V)
    o_proj_in_local = num_heads_local * D_HEAD_V
    inter_local = MOE_INTERMEDIATE // tp
    gate_up_local = 2 * inter_local

    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)

    # ── Weights: MLA attention ───────────────────────────────────
    W_dq     = mk(D_MODEL, Q_LORA_RANK)
    rms_w_qa = torch.ones(Q_LORA_RANK, device=dev, dtype=attn_dtype)
    W_uq     = mk(Q_LORA_RANK, q_proj_out_local)
    W_dkv    = mk(D_MODEL, KV_A_OUT)
    rms_w_kva = torch.ones(KV_LORA_RANK, device=dev, dtype=attn_dtype)
    W_ukv    = mk(KV_LORA_RANK, kv_b_out_local)
    W_o      = mk(o_proj_in_local, D_MODEL)

    rms_w_attn = torch.ones(D_MODEL, device=dev, dtype=attn_dtype)
    rms_w_mlp  = torch.ones(D_MODEL, device=dev, dtype=attn_dtype)

    # ── Weights: MoE MLP ─────────────────────────────────────────
    W_router = mk(D_MODEL, N_ROUTED_EXPERTS)
    # Pre-allocate for TOP_K selected experts (batched GEMM simulation)
    W_expert_gate_up = mk(TOP_K_EXPERTS, D_MODEL, gate_up_local)
    W_expert_down    = mk(TOP_K_EXPERTS, inter_local, D_MODEL)
    # Shared expert
    W_shared_gate_up = mk(D_MODEL, gate_up_local)
    W_shared_down    = mk(inter_local, D_MODEL)

    # ── Input + KV cache ─────────────────────────────────────────
    x = mk(B, D_MODEL)
    residual = x.clone()

    K_cache = mk(B, S + 1, num_heads_local, D_HEAD)
    V_cache = torch.zeros(B, S + 1, num_heads_local, D_HEAD,
                          device=dev, dtype=torch_dtype)
    V_cache[..., :D_HEAD_V] = torch.randn(
        B, S + 1, num_heads_local, D_HEAD_V, device=dev, dtype=torch_dtype)
    cache_seqlens = torch.tensor([S], device=dev, dtype=torch.int32)

    freqs = 1.0 / (10000.0 ** (torch.arange(0, D_HEAD_ROPE, 2,
                    device=dev, dtype=torch.float32) / D_HEAD_ROPE))
    emb = torch.tensor([S], device=dev, dtype=torch.float32) * freqs
    rope_cos = emb.cos().to(attn_dtype).view(1, D_HEAD_ROPE // 2)
    rope_sin = emb.sin().to(attn_dtype).view(1, D_HEAD_ROPE // 2)

    torch.cuda.synchronize(dev)

    # ── Full-layer helper (for warmup) ────────────────────────────
    def _full_layer():
        nonlocal residual
        # Attention
        h = rms_norm(x, rms_w_attn)
        q_comp = rms_norm(h @ W_dq, rms_w_qa)
        q_pe = q_comp @ W_uq
        q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)

        kv_a = h @ W_dkv
        c_kv = rms_norm(kv_a[..., :KV_LORA_RANK], rms_w_kva)
        k_rope_raw = kv_a[..., KV_LORA_RANK:]
        kv_b = c_kv @ W_ukv
        k_nope = kv_b[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        v_raw = kv_b[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)

        q = torch.cat([q_nope, apply_rope(q_rope_raw, rope_cos, rope_sin)], dim=-1)
        k = torch.cat([k_nope, apply_rope(
            k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(-1, -1, num_heads_local, -1),
            rope_cos, rope_sin)], dim=-1)
        v = torch.zeros(B, 1, num_heads_local, D_HEAD, device=dev, dtype=torch_dtype)
        v[..., :D_HEAD_V] = v_raw

        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=v,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        o = ao[..., :D_HEAD_V].contiguous().view(B, o_proj_in_local) @ W_o
        if use_nccl:
            dist.all_reduce(o)
        elif tp > 1:
            o = o.clone()
        residual = residual + o

        # MoE MLP
        h2 = rms_norm(residual, rms_w_mlp)
        logits = h2 @ W_router
        routing_weights, _ = torch.topk(torch.softmax(logits, -1), TOP_K_EXPERTS)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        x_exp = h2.unsqueeze(0).expand(TOP_K_EXPERTS, -1, -1)
        gu_exp = torch.bmm(x_exp, W_expert_gate_up)
        gate_exp, up_exp = gu_exp.split(inter_local, dim=-1)
        hidden_exp = F.silu(gate_exp.float()).to(attn_dtype) * up_exp
        down_exp = torch.bmm(hidden_exp, W_expert_down)
        expert_out = (down_exp * routing_weights.t().unsqueeze(-1)).sum(0)

        gu_shared = h2 @ W_shared_gate_up
        gate_s, up_s = gu_shared.split(inter_local, dim=-1)
        hidden_s = F.silu(gate_s.float()).to(attn_dtype) * up_s
        shared_out = hidden_s @ W_shared_down

        mlp_out = expert_out + shared_out
        if use_nccl:
            dist.all_reduce(mlp_out)
        elif tp > 1:
            mlp_out = mlp_out.clone()
        residual = residual + mlp_out

    # ── Warmup ────────────────────────────────────────────────────
    for _ in range(2):
        _full_layer()
    torch.cuda.synchronize(dev)

    # Reset residual
    residual = x.clone()

    # ── Profiled iteration — each stage in its own NVTX range ────
    tag = f"_S{S}" if tp == 1 else f"_S{S}_r{rank}"

    # 1. Pre-attention RMSNorm
    torch.cuda.nvtx.range_push(f"rmsnorm_attn{tag}")
    h = rms_norm(x, rms_w_attn)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 2. Q compression
    torch.cuda.nvtx.range_push(f"q_a_proj{tag}")
    q_comp = h @ W_dq
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 3. Q norm
    torch.cuda.nvtx.range_push(f"q_a_norm{tag}")
    q_comp = rms_norm(q_comp, rms_w_qa)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 4. Q expansion
    torch.cuda.nvtx.range_push(f"q_b_proj{tag}")
    q_pe = q_comp @ W_uq
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 5. KV compression
    torch.cuda.nvtx.range_push(f"kv_a_proj{tag}")
    kv_a = h @ W_dkv
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 6. KV norm
    torch.cuda.nvtx.range_push(f"kv_a_norm{tag}")
    c_kv = rms_norm(kv_a[..., :KV_LORA_RANK], rms_w_kva)
    k_rope_raw = kv_a[..., KV_LORA_RANK:]
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 7. KV expansion
    torch.cuda.nvtx.range_push(f"kv_b_proj{tag}")
    kv_b = c_kv @ W_ukv
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 8. RoPE + reshape + assemble
    torch.cuda.nvtx.range_push(f"rope{tag}")
    q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
    q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)
    k_nope = kv_b[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
    v_raw = kv_b[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)

    q = torch.cat([q_nope, apply_rope(q_rope_raw, rope_cos, rope_sin)], dim=-1)
    k = torch.cat([k_nope, apply_rope(
        k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(-1, -1, num_heads_local, -1),
        rope_cos, rope_sin)], dim=-1)
    v = torch.zeros(B, 1, num_heads_local, D_HEAD, device=dev, dtype=torch_dtype)
    v[..., :D_HEAD_V] = v_raw
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 9. FlashAttention-2 decode
    torch.cuda.nvtx.range_push(f"attn{tag}")
    attn_out = flash_attn_with_kvcache(
        q, K_cache, V_cache, k=k, v=v,
        cache_seqlens=cache_seqlens.clone(), causal=True)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 10. Output projection (row-parallel)
    torch.cuda.nvtx.range_push(f"o_proj{tag}")
    o_out = attn_out[..., :D_HEAD_V].contiguous().view(B, o_proj_in_local) @ W_o
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 10b. All-reduce after o_proj (TP only)
    if tp > 1:
        torch.cuda.nvtx.range_push(f"allreduce_attn{tag}")
        if use_nccl:
            dist.all_reduce(o_out)
        else:
            o_out = o_out.clone()
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

    # 11. Residual add (attention)
    torch.cuda.nvtx.range_push(f"residual_attn{tag}")
    residual = residual + o_out
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 12. Pre-MLP RMSNorm
    torch.cuda.nvtx.range_push(f"rmsnorm_mlp{tag}")
    h2 = rms_norm(residual, rms_w_mlp)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 13. Router (expert selection)
    torch.cuda.nvtx.range_push(f"router{tag}")
    logits = h2 @ W_router
    routing_weights, _ = torch.topk(
        torch.softmax(logits, -1), TOP_K_EXPERTS)
    routing_weights = routing_weights / routing_weights.sum(
        dim=-1, keepdim=True)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 14. Routed experts — gate+up (batched GEMM, top-8)
    torch.cuda.nvtx.range_push(f"expert_gate_up{tag}")
    x_exp = h2.unsqueeze(0).expand(TOP_K_EXPERTS, -1, -1)
    gu_exp = torch.bmm(x_exp, W_expert_gate_up)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 15. Routed experts — SiLU × gate
    torch.cuda.nvtx.range_push(f"expert_silu_mul{tag}")
    gate_exp, up_exp = gu_exp.split(inter_local, dim=-1)
    hidden_exp = F.silu(gate_exp.float()).to(attn_dtype) * up_exp
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 16. Routed experts — down (batched GEMM)
    torch.cuda.nvtx.range_push(f"expert_down{tag}")
    down_exp = torch.bmm(hidden_exp, W_expert_down)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 17. Expert combine (weighted sum)
    torch.cuda.nvtx.range_push(f"expert_combine{tag}")
    expert_out = (down_exp * routing_weights.t().unsqueeze(-1)).sum(0)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 18. Shared expert — gate+up
    torch.cuda.nvtx.range_push(f"shared_gate_up{tag}")
    gu_shared = h2 @ W_shared_gate_up
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 19. Shared expert — SiLU × gate
    torch.cuda.nvtx.range_push(f"shared_silu_mul{tag}")
    gate_s, up_s = gu_shared.split(inter_local, dim=-1)
    hidden_s = F.silu(gate_s.float()).to(attn_dtype) * up_s
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 20. Shared expert — down
    torch.cuda.nvtx.range_push(f"shared_down{tag}")
    shared_out = hidden_s @ W_shared_down
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    # 21. Combine routed + shared
    mlp_out = expert_out + shared_out

    # 21b. All-reduce after MoE (TP only)
    if tp > 1:
        torch.cuda.nvtx.range_push(f"allreduce_mlp{tag}")
        if use_nccl:
            dist.all_reduce(mlp_out)
        else:
            mlp_out = mlp_out.clone()
        torch.cuda.synchronize(dev)
        torch.cuda.nvtx.range_pop()

    # 22. Residual add (MLP)
    torch.cuda.nvtx.range_push(f"residual_mlp{tag}")
    residual = residual + mlp_out
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    print(f"[done] S={S}, dtype={dtype_name}, tp={tp}, rank={rank}, "
          f"gpu=cuda:{dev.index}")


# ── E2E latency measurement ─────────────────────────────────────

def run_decode_e2e(S, dev, dtype_name, tp=1, rank=0, iters=100):
    """Measure per-stage wall-clock latency using CUDA events.

    Returns dict of {stage_name: median_us}.
    Runs real NCCL when tp>1 (no skip).
    """
    torch_dtype = DTYPE_CONFIGS[dtype_name]
    attn_dtype = torch.float16
    B = 1
    use_nccl = (tp > 1)

    # ── Per-rank dimensions ──────────────────────────────────────
    num_heads_local = NUM_HEADS // tp
    nope_dim = num_heads_local * D_HEAD_NOPE
    q_proj_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_ROPE)
    kv_b_out_local = num_heads_local * (D_HEAD_NOPE + D_HEAD_V)
    o_proj_in_local = num_heads_local * D_HEAD_V
    inter_local = MOE_INTERMEDIATE // tp
    gate_up_local = 2 * inter_local

    mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)

    # ── Weights: MLA attention ───────────────────────────────────
    W_dq     = mk(D_MODEL, Q_LORA_RANK)
    rms_w_qa = torch.ones(Q_LORA_RANK, device=dev, dtype=attn_dtype)
    W_uq     = mk(Q_LORA_RANK, q_proj_out_local)
    W_dkv    = mk(D_MODEL, KV_A_OUT)
    rms_w_kva = torch.ones(KV_LORA_RANK, device=dev, dtype=attn_dtype)
    W_ukv    = mk(KV_LORA_RANK, kv_b_out_local)
    W_o      = mk(o_proj_in_local, D_MODEL)

    rms_w_attn = torch.ones(D_MODEL, device=dev, dtype=attn_dtype)
    rms_w_mlp  = torch.ones(D_MODEL, device=dev, dtype=attn_dtype)

    # ── Weights: MoE MLP ─────────────────────────────────────────
    W_router = mk(D_MODEL, N_ROUTED_EXPERTS)
    W_expert_gate_up = mk(TOP_K_EXPERTS, D_MODEL, gate_up_local)
    W_expert_down    = mk(TOP_K_EXPERTS, inter_local, D_MODEL)
    W_shared_gate_up = mk(D_MODEL, gate_up_local)
    W_shared_down    = mk(inter_local, D_MODEL)

    # ── Input + KV cache ─────────────────────────────────────────
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
    rope_cos = emb.cos().to(attn_dtype).view(1, D_HEAD_ROPE // 2)
    rope_sin = emb.sin().to(attn_dtype).view(1, D_HEAD_ROPE // 2)

    torch.cuda.synchronize(dev)

    # ── Stage definitions ─────────────────────────────────────────
    STAGES = ["rmsnorm_attn", "q_a_proj", "q_a_norm", "q_b_proj",
              "kv_a_proj", "kv_a_norm", "kv_b_proj",
              "rope", "attn", "o_proj"]
    if tp > 1:
        STAGES.append("allreduce_attn")
    STAGES += ["residual_attn", "rmsnorm_mlp", "router",
               "expert_gate_up", "expert_silu_mul", "expert_down",
               "expert_combine",
               "shared_gate_up", "shared_silu_mul", "shared_down"]
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
        residual = x.clone()
        h = rms_norm(x, rms_w_attn)
        q_comp = rms_norm(h @ W_dq, rms_w_qa)
        q_pe = q_comp @ W_uq
        q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)
        kv_a = h @ W_dkv
        c_kv = rms_norm(kv_a[..., :KV_LORA_RANK], rms_w_kva)
        k_rope_raw = kv_a[..., KV_LORA_RANK:]
        kv_b = c_kv @ W_ukv
        k_nope = kv_b[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        v_raw = kv_b[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)
        q = torch.cat([q_nope, apply_rope(q_rope_raw, rope_cos, rope_sin)], dim=-1)
        k = torch.cat([k_nope, apply_rope(
            k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(-1, -1, num_heads_local, -1),
            rope_cos, rope_sin)], dim=-1)
        vv = torch.zeros(B, 1, num_heads_local, D_HEAD, device=dev, dtype=torch_dtype)
        vv[..., :D_HEAD_V] = v_raw
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=vv,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        o = ao[..., :D_HEAD_V].contiguous().view(B, o_proj_in_local) @ W_o
        if use_nccl:
            dist.all_reduce(o)
        residual = residual + o
        h2 = rms_norm(residual, rms_w_mlp)
        logits = h2 @ W_router
        rw, _ = torch.topk(torch.softmax(logits, -1), TOP_K_EXPERTS)
        rw = rw / rw.sum(dim=-1, keepdim=True)
        x_exp = h2.unsqueeze(0).expand(TOP_K_EXPERTS, -1, -1)
        gu_exp = torch.bmm(x_exp, W_expert_gate_up)
        gt_exp, up_exp = gu_exp.split(inter_local, dim=-1)
        hid_exp = F.silu(gt_exp.float()).to(attn_dtype) * up_exp
        dn_exp = torch.bmm(hid_exp, W_expert_down)
        exp_out = (dn_exp * rw.t().unsqueeze(-1)).sum(0)
        gu_s = h2 @ W_shared_gate_up
        gt_s, up_s = gu_s.split(inter_local, dim=-1)
        hid_s = F.silu(gt_s.float()).to(attn_dtype) * up_s
        sh_out = hid_s @ W_shared_down
        mlp_out = exp_out + sh_out
        if use_nccl:
            dist.all_reduce(mlp_out)
        residual = residual + mlp_out
    torch.cuda.synchronize(dev)

    # ── Timed iterations ──────────────────────────────────────────
    for _ in range(iters):
        residual = x.clone()

        events["layer"][0].record()

        # rmsnorm_attn
        events["rmsnorm_attn"][0].record()
        h = rms_norm(x, rms_w_attn)
        events["rmsnorm_attn"][1].record()

        # q_a_proj
        events["q_a_proj"][0].record()
        q_comp = h @ W_dq
        events["q_a_proj"][1].record()

        # q_a_norm
        events["q_a_norm"][0].record()
        q_comp = rms_norm(q_comp, rms_w_qa)
        events["q_a_norm"][1].record()

        # q_b_proj
        events["q_b_proj"][0].record()
        q_pe = q_comp @ W_uq
        events["q_b_proj"][1].record()

        # kv_a_proj
        events["kv_a_proj"][0].record()
        kv_a = h @ W_dkv
        events["kv_a_proj"][1].record()

        # kv_a_norm
        events["kv_a_norm"][0].record()
        c_kv = rms_norm(kv_a[..., :KV_LORA_RANK], rms_w_kva)
        k_rope_raw = kv_a[..., KV_LORA_RANK:]
        events["kv_a_norm"][1].record()

        # kv_b_proj
        events["kv_b_proj"][0].record()
        kv_b = c_kv @ W_ukv
        events["kv_b_proj"][1].record()

        # rope
        events["rope"][0].record()
        q_nope = q_pe[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        q_rope_raw = q_pe[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_ROPE)
        k_nope = kv_b[..., :nope_dim].view(B, 1, num_heads_local, D_HEAD_NOPE)
        v_raw = kv_b[..., nope_dim:].view(B, 1, num_heads_local, D_HEAD_V)
        q = torch.cat([q_nope, apply_rope(q_rope_raw, rope_cos, rope_sin)], dim=-1)
        k = torch.cat([k_nope, apply_rope(
            k_rope_raw.view(B, 1, 1, D_HEAD_ROPE).expand(-1, -1, num_heads_local, -1),
            rope_cos, rope_sin)], dim=-1)
        vv = torch.zeros(B, 1, num_heads_local, D_HEAD, device=dev, dtype=torch_dtype)
        vv[..., :D_HEAD_V] = v_raw
        events["rope"][1].record()

        # attn
        events["attn"][0].record()
        ao = flash_attn_with_kvcache(q, K_cache, V_cache, k=k, v=vv,
                                     cache_seqlens=cache_seqlens.clone(), causal=True)
        events["attn"][1].record()

        # o_proj
        events["o_proj"][0].record()
        o = ao[..., :D_HEAD_V].contiguous().view(B, o_proj_in_local) @ W_o
        events["o_proj"][1].record()

        # allreduce_attn
        if tp > 1:
            events["allreduce_attn"][0].record()
            if use_nccl:
                dist.all_reduce(o)
            events["allreduce_attn"][1].record()

        # residual_attn
        events["residual_attn"][0].record()
        residual = residual + o
        events["residual_attn"][1].record()

        # rmsnorm_mlp
        events["rmsnorm_mlp"][0].record()
        h2 = rms_norm(residual, rms_w_mlp)
        events["rmsnorm_mlp"][1].record()

        # router
        events["router"][0].record()
        logits = h2 @ W_router
        rw, _ = torch.topk(torch.softmax(logits, -1), TOP_K_EXPERTS)
        rw = rw / rw.sum(dim=-1, keepdim=True)
        events["router"][1].record()

        # expert_gate_up
        events["expert_gate_up"][0].record()
        x_exp = h2.unsqueeze(0).expand(TOP_K_EXPERTS, -1, -1)
        gu_exp = torch.bmm(x_exp, W_expert_gate_up)
        events["expert_gate_up"][1].record()

        # expert_silu_mul
        events["expert_silu_mul"][0].record()
        gt_exp, up_exp = gu_exp.split(inter_local, dim=-1)
        hid_exp = F.silu(gt_exp.float()).to(attn_dtype) * up_exp
        events["expert_silu_mul"][1].record()

        # expert_down
        events["expert_down"][0].record()
        dn_exp = torch.bmm(hid_exp, W_expert_down)
        events["expert_down"][1].record()

        # expert_combine
        events["expert_combine"][0].record()
        exp_out = (dn_exp * rw.t().unsqueeze(-1)).sum(0)
        events["expert_combine"][1].record()

        # shared_gate_up
        events["shared_gate_up"][0].record()
        gu_s = h2 @ W_shared_gate_up
        events["shared_gate_up"][1].record()

        # shared_silu_mul
        events["shared_silu_mul"][0].record()
        gt_s, up_s = gu_s.split(inter_local, dim=-1)
        hid_s = F.silu(gt_s.float()).to(attn_dtype) * up_s
        events["shared_silu_mul"][1].record()

        # shared_down
        events["shared_down"][0].record()
        sh_out = hid_s @ W_shared_down
        events["shared_down"][1].record()

        # combine routed + shared
        mlp_out = exp_out + sh_out

        # allreduce_mlp
        if tp > 1:
            events["allreduce_mlp"][0].record()
            if use_nccl:
                dist.all_reduce(mlp_out)
            events["allreduce_mlp"][1].record()

        # residual_mlp
        events["residual_mlp"][0].record()
        residual = residual + mlp_out
        events["residual_mlp"][1].record()

        events["layer"][1].record()
        torch.cuda.synchronize(dev)

        for s in STAGES:
            all_times[s].append(
                events[s][0].elapsed_time(events[s][1]) * 1000)  # ms→us
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
    print(f"  DSV3 E2E Latency (median of N iters) | TP={tp} rank={rank}")
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

    # ── Absorbed MLA correction summary ───────────────────────────
    cf_i = ATTN_CORRECTION_IDEAL
    cf_p = ATTN_CORRECTION_PRACTICAL
    print()
    print("=" * 110)
    print(f"  Absorbed MLA Correction | "
          f"Expanded={BYTES_EXPANDED_PER_TOK} B/tok, "
          f"Absorbed={BYTES_ABSORBED_PER_TOK} B/tok | "
          f"Ideal: 1/{1/cf_i:.0f}, Practical: 1/{1/cf_p:.0f}")
    print("=" * 110)
    print(f"{'Seq':>8} | {'attn_meas':>10} | {'attn_ideal':>10} | "
          f"{'attn_prac':>10} | {'non_attn':>10} | "
          f"{'layer_meas':>10} | {'layer_ideal':>11} | "
          f"{'layer_prac':>11}")
    print("-" * 110)

    for S, (medians, stg) in sorted(seq_results.items()):
        a_meas = medians["attn"]
        a_ideal = a_meas * cf_i
        a_prac = a_meas * cf_p
        non_attn = sum(medians[s] for s in stg if s != "attn")
        l_meas = medians["layer"]
        l_ideal = non_attn + a_ideal
        l_prac = non_attn + a_prac
        print(f"{S:>8} | {a_meas:>8.1f}us | {a_ideal:>8.1f}us | "
              f"{a_prac:>8.1f}us | {non_attn:>8.1f}us | "
              f"{l_meas:>8.1f}us | {l_ideal:>9.1f}us | "
              f"{l_prac:>9.1f}us")
    print("-" * 110)


# ── TP worker (launched via mp.spawn) ────────────────────────────

def _tp_worker(rank, world_size, seq_list, dtype_name,
               skip_nccl=False, e2e_iters=0):
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
        print(f"  DSV3 TP={world_size}  rank 0 → cuda:{TP_GPU_MAP[0]}, "
              f"rank 1 → cuda:{TP_GPU_MAP[1]}  ({mode})")
        print(f"{'='*60}")

    if e2e_iters > 0:
        seq_results = {}
        for S in seq_list:
            try:
                medians, stages = run_decode_e2e(
                    S, dev, dtype_name, tp=world_size, rank=rank,
                    iters=e2e_iters)
                seq_results[S] = (medians, stages)
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"[OOM] S={S}")
            torch.cuda.empty_cache()
        if rank == 0:
            print_e2e_table(seq_results, tp=world_size, rank=rank)
    else:
        for S in seq_list:
            try:
                run_decode(S, dev, dtype_name, tp=world_size, rank=rank,
                           skip_nccl=skip_nccl)
            except torch.cuda.OutOfMemoryError:
                if rank == 0:
                    print(f"[OOM] S={S}")
            torch.cuda.empty_cache()

    if need_nccl:
        dist.destroy_process_group()


# ── Entry point ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek-V3 decode profiler (MLA + MoE, FlashAttention-2)")
    parser.add_argument("--seq", type=int, nargs="+",
                        default=[1024, 2048, 4096, 8192, 16384, 32768,
                                 65536, 131072, 262144])
    parser.add_argument("--dtype", type=str, default="fp16",
                        choices=["fp16", "bf16"])
    parser.add_argument("--tp", type=int, default=1, choices=[1, 2],
                        help="Tensor parallel degree (1=single GPU cuda:1, "
                             "2=TP across cuda:1 + cuda:2)")
    parser.add_argument("--skip-nccl", action="store_true",
                        help="Replace NCCL all-reduce with local copy. "
                             "Required for ncu profiling.")
    parser.add_argument("--e2e", type=int, default=0, metavar="N",
                        help="E2E latency mode: run N iterations per seq, "
                             "measure per-stage wall-clock time via CUDA events.")
    args = parser.parse_args()

    if args.tp == 1:
        # ── Single GPU mode ──────────────────────────────────────
        dev = torch.device(f"cuda:{TP_GPU_MAP[0]}")
        torch.cuda.init()
        torch.set_float32_matmul_precision("high")

        if args.e2e > 0:
            print(f"\n{'='*60}")
            print(f"  DSV3 E2E — Single GPU cuda:{dev.index} — {args.e2e} iters")
            print(f"  MLA + MoE (top-{TOP_K_EXPERTS} of {N_ROUTED_EXPERTS} "
                  f"+ {N_SHARED_EXPERTS} shared)")
            print(f"{'='*60}")
            seq_results = {}
            for S in args.seq:
                try:
                    medians, stages = run_decode_e2e(
                        S, dev, args.dtype, tp=1, rank=0, iters=args.e2e)
                    seq_results[S] = (medians, stages)
                except torch.cuda.OutOfMemoryError:
                    print(f"[OOM] S={S}")
                torch.cuda.empty_cache()
            print_e2e_table(seq_results, tp=1, rank=0)
        else:
            print(f"\n{'='*60}")
            print(f"  DSV3 Single GPU — cuda:{dev.index} (ncu mode)")
            print(f"  MLA + MoE (top-{TOP_K_EXPERTS} of {N_ROUTED_EXPERTS} "
                  f"+ {N_SHARED_EXPERTS} shared)")
            print(f"{'='*60}")
            for S in args.seq:
                try:
                    run_decode(S, dev, args.dtype, tp=1, rank=0)
                except torch.cuda.OutOfMemoryError:
                    print(f"[OOM] S={S}")
                torch.cuda.empty_cache()

    elif args.tp == 2:
        # ── Tensor Parallel mode ─────────────────────────────────
        if args.e2e > 0:
            mp.spawn(_tp_worker,
                     args=(2, args.seq, args.dtype, False, args.e2e),
                     nprocs=2, join=True)
        elif args.skip_nccl:
            _tp_worker(0, 2, args.seq, args.dtype, skip_nccl=True)
        else:
            mp.spawn(_tp_worker,
                     args=(2, args.seq, args.dtype, False, 0),
                     nprocs=2, join=True)


if __name__ == "__main__":
    main()
