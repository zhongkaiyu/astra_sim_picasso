#!/usr/bin/env python3
"""Production-style single-layer **MoE FFN** decode profiler for ncu.

Companion to ncu_decode_profile.py (which profiles a *dense* MLP). This one
profiles the **MoE FFN** of a decode step (batch=1, seq_q=1) exactly the way an
inference engine runs it, so we can measure the real-card compute/memory
utilization of the *small expert GEMVs* — these are a different shape class from
the big dense gate_up/down GEMMs and tend to realize a lower fraction of peak.

Why a separate script:
    The exp1 FFN baseline (backend/lpu_ffn_decode) models DeepSeek-V3 / Qwen3-MoE
    with expert OI=1 (§6: every token re-reads its top_k experts' weights). The
    per-expert GEMV is (1, d_model) @ (d_model, 2*d_inter) — for DeepSeek
    d_inter=2048, far smaller than the dense inter=14336. We need *its* HBM-BW%
    and TensorCore% to put a defensible utilization into the Rubin FFN model
    instead of reusing the dense-MLP proxy.

MoE decode stages (per token, top_k experts):
    Stage               Kernel type                         shape (per expert)
    ─────────────────── ─────────────────────────────────── ─────────────────────
    rmsnorm_mlp         fused RMSNorm                        (1, d_model)
    gating              router GEMV                          (1,d) @ (d, n_experts)
    expert_gate_up      top_k × fused gate+up GEMV           (1,d) @ (d, 2*d_inter/tp)
    expert_silu         top_k × SiLU(gate)*up                (1, d_inter/tp)
    expert_down         top_k × down GEMV                    (1,d_inter/tp) @ (d_inter/tp, d)
    shared_gate_up      shared expert gate+up GEMV           (1,d) @ (d, 2*d_inter*ns/tp)
    shared_silu         shared SiLU*mul
    shared_down         shared down GEMV
    combine             weighted scatter-add of expert outs  (top_k+ns) × (1, d)
    allreduce_mlp       NCCL all-reduce (TP mode only)       (1, d)

Each token is routed to top_k distinct experts; to defeat L2 reuse (so the GEMV
is genuinely weight-memory-bound, OI≈1) every expert gets its own weight tensors.

Modes (same as ncu_decode_profile.py):
    --e2e N      End-to-end per-stage wall-clock via CUDA events (no sudo, real NCCL).
    (ncu mode)   Default: one NVTX-tagged profiled iteration for `sudo ncu`.
    --skip-nccl  Replace NCCL with local copy (required for ncu replay).

Usage:
    # 1) ncu: per-kernel utilization (single GPU, DeepSeek-V3 MoE, fp8)
    sudo /usr/local/cuda-13.0/bin/ncu --set full --target-processes all \
        --launch-skip 8 --launch-count 600 \
        -o H100_results/moe_decode_dsv3 \
        /home/haotian/miniconda3/envs/bench/bin/python ncu_moe_decode_profile.py \
            --model deepseek3 --dtype fp8 --seq 4096

    # 2) e2e per-stage latency (standalone, no ncu)
    python ncu_moe_decode_profile.py --model deepseek3 --e2e 100 --dtype fp8
    python ncu_moe_decode_profile.py --model qwen3-235b --e2e 100 --tp 2 --skip-nccl

Then read utilization with analyze_moe_ncu.py (companion) or the per-kernel
HBM-BW%/TensorCore% straight from the ncu UI / `ncu --import`.
"""
import argparse
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

# ── MoE model library (FFN-relevant dims; mirror lpu_config.MODEL_LIBRARY) ──
#   d_model, n_experts, top_k, d_inter (per-expert intermediate), n_shared
MOE_CONFIGS = {
    "deepseek3":    dict(d_model=7168, n_experts=256, top_k=8, d_inter=2048, n_shared=1),
    "qwen3-235b":   dict(d_model=4096, n_experts=128, top_k=8, d_inter=1536, n_shared=0),
    "gpt-oss-120b": dict(d_model=2880, n_experts=128, top_k=4, d_inter=2880, n_shared=0),
}

RMS_EPS = 1e-6
DTYPE_CONFIGS = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8":  torch.float8_e4m3fn,
}

# TP GPU mapping: rank → cuda device id
TP_GPU_MAP = {0: 0, 1: 1}


# ── Small fused-kernel stand-ins ─────────────────────────────────

def rms_norm(x, weight, eps=RMS_EPS):
    """RMSNorm — 1 fused CUDA kernel in production."""
    orig = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x * weight).to(orig)


def _proj(x, w, sa=None, sb=None):
    """Linear projection; FP8 scaled_mm when scales provided.

    Weight layout: w is (out, in) for FP8 (passed transposed), (in, out) otherwise.
    """
    if sa is not None:
        return torch._scaled_mm(
            x.to(torch.float8_e4m3fn), w.t(),
            scale_a=sa, scale_b=sb,
            out_dtype=torch.float16, use_fast_accum=True)
    return x @ w


def _make_expert_weights(cfg, tp, dev, dtype_name):
    """Allocate distinct weights for gating, top_k routed experts, shared experts.

    Returns a dict of tensors. Each routed expert gets its OWN gate_up/down so
    that running all top_k forces top_k separate weight reads (OI≈1 / no reuse).
    """
    d = cfg["d_model"]
    di = cfg["d_inter"] // tp                 # column-parallel shard of intermediate
    gate_up = 2 * di
    ns = cfg["n_shared"]
    is_fp8 = dtype_name == "fp8"
    torch_dtype = DTYPE_CONFIGS[dtype_name]

    if is_fp8:
        mk = lambda *s: torch.randn(*s, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
        # FP8 layout: (out, in)
        W_gate = mk(cfg["n_experts"], d)                       # router
        experts = [(mk(gate_up, d), mk(d, di)) for _ in range(cfg["top_k"])]
        shared = None
        if ns > 0:
            shared = (mk(2 * di * ns, d), mk(d, di * ns))
    else:
        mk = lambda *s: torch.randn(*s, device=dev, dtype=torch_dtype)
        # (in, out) layout
        W_gate = mk(d, cfg["n_experts"])
        experts = [(mk(d, gate_up), mk(di, d)) for _ in range(cfg["top_k"])]
        shared = None
        if ns > 0:
            shared = (mk(d, 2 * di * ns), mk(di * ns, d))
    return {"gate": W_gate, "experts": experts, "shared": shared, "di": di}


# ── One MoE FFN decode step (gating → top_k experts → shared → combine) ──

def _moe_ffn(h2, W, cfg, tp, sa, sb, is_fp8, attn_dtype, nvtx=False, tag="",
             use_nccl=False, events=None):
    """Run the routed+shared MoE FFN on a single normalized token h2 (B=1).

    If `events` is provided (e2e mode), each stage is timed; if `nvtx`, each
    stage is wrapped in an NVTX range for ncu. Returns the layer output.
    """
    di = W["di"]
    n_shared = cfg["n_shared"]
    dev = h2.device

    def _rec(name, fn):
        # helper: run fn() under nvtx/event instrumentation, return its result
        if events is not None:
            events[name][0].record()
            r = fn()
            events[name][1].record()
            return r
        if nvtx:
            torch.cuda.nvtx.range_push(f"{name}{tag}")
            r = fn()
            torch.cuda.synchronize(dev)
            torch.cuda.nvtx.range_pop()
            return r
        return fn()

    h2f = h2.to(torch.float8_e4m3fn) if is_fp8 else h2

    # 1) gating / router GEMV
    _rec("gating", lambda: _proj(h2f, W["gate"], sa, sb))

    # 2) routed experts: top_k × (gate_up GEMV, silu*mul, down GEMV)
    expert_outs = []

    def _expert_gate_up():
        outs = []
        for (W_gu, _W_dn) in W["experts"]:
            outs.append(_proj(h2f, W_gu, sa, sb))
        return outs

    gus = _rec("expert_gate_up", _expert_gate_up)

    def _expert_silu():
        hids = []
        for gu in gus:
            gate, up = gu.half().split([di, di], dim=-1)
            hids.append(F.silu(gate.float()).to(attn_dtype) * up)
        return hids

    hids = _rec("expert_silu", _expert_silu)

    def _expert_down():
        outs = []
        for hid, (_W_gu, W_dn) in zip(hids, W["experts"]):
            hi = hid.to(torch.float8_e4m3fn) if is_fp8 else hid
            outs.append(_proj(hi, W_dn, sa, sb))
        return outs

    expert_outs = _rec("expert_down", _expert_down)

    # 3) shared expert (dense small FFN, every token)
    if n_shared > 0:
        W_sgu, W_sdn = W["shared"]
        sgu = _rec("shared_gate_up", lambda: _proj(h2f, W_sgu, sa, sb))

        def _shared_silu():
            g, u = sgu.half().split([di * n_shared, di * n_shared], dim=-1)
            return F.silu(g.float()).to(attn_dtype) * u
        shid = _rec("shared_silu", _shared_silu)

        def _shared_down():
            hi = shid.to(torch.float8_e4m3fn) if is_fp8 else shid
            return _proj(hi, W_sdn, sa, sb)
        expert_outs.append(_rec("shared_down", _shared_down))

    # 4) combine: weighted sum of expert outputs (router weights ≈ uniform here)
    def _combine():
        acc = expert_outs[0].half()
        for o in expert_outs[1:]:
            acc = acc + o.half()
        return acc
    out = _rec("combine", _combine)

    # 5) TP all-reduce
    if tp > 1:
        def _ar():
            if use_nccl:
                dist.all_reduce(out)
                return out
            return out.clone()
        out = _rec("allreduce_mlp", _ar)

    return out


# ── ncu profiling mode (single NVTX-tagged iteration) ───────────

def run_decode(cfg, dev, dtype_name, tp=1, rank=0, skip_nccl=False):
    use_nccl = (tp > 1 and not skip_nccl)
    torch_dtype = DTYPE_CONFIGS[dtype_name]
    is_fp8 = dtype_name == "fp8"
    attn_dtype = torch.float16
    d = cfg["d_model"]
    sa = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None
    sb = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None

    W = _make_expert_weights(cfg, tp, dev, dtype_name)
    rms_w = torch.ones(d, device=dev, dtype=torch.float16)
    x = torch.randn(1, d, device=dev, dtype=torch.float16)
    if is_fp8:
        x = x.to(torch.float8_e4m3fn)

    torch.cuda.synchronize(dev)

    # warmup (ncu --launch-skip drops these)
    for _ in range(2):
        h2 = rms_norm(x.half() if is_fp8 else x, rms_w)
        _moe_ffn(h2, W, cfg, tp, sa, sb, is_fp8, attn_dtype,
                 use_nccl=use_nccl)
    torch.cuda.synchronize(dev)

    tag = "" if tp == 1 else f"_r{rank}"
    torch.cuda.nvtx.range_push(f"rmsnorm_mlp{tag}")
    h2 = rms_norm(x.half() if is_fp8 else x, rms_w)
    torch.cuda.synchronize(dev)
    torch.cuda.nvtx.range_pop()

    _moe_ffn(h2, W, cfg, tp, sa, sb, is_fp8, attn_dtype,
             nvtx=True, tag=tag, use_nccl=use_nccl)
    print(f"[done] model dims d={d} top_k={cfg['top_k']} di={cfg['d_inter']} "
          f"dtype={dtype_name} tp={tp} rank={rank} gpu=cuda:{dev.index}")


# ── e2e latency mode ─────────────────────────────────────────────

def run_decode_e2e(cfg, dev, dtype_name, tp=1, rank=0, iters=100):
    is_fp8 = dtype_name == "fp8"
    attn_dtype = torch.float16
    d = cfg["d_model"]
    use_nccl = (tp > 1)
    sa = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None
    sb = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None

    W = _make_expert_weights(cfg, tp, dev, dtype_name)
    rms_w = torch.ones(d, device=dev, dtype=torch.float16)
    x = torch.randn(1, d, device=dev, dtype=torch.float16)
    if is_fp8:
        x = x.to(torch.float8_e4m3fn)

    STAGES = ["rmsnorm_mlp", "gating", "expert_gate_up", "expert_silu",
              "expert_down"]
    if cfg["n_shared"] > 0:
        STAGES += ["shared_gate_up", "shared_silu", "shared_down"]
    STAGES += ["combine"]
    if tp > 1:
        STAGES.append("allreduce_mlp")

    events = {s: (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True)) for s in STAGES}
    events["layer"] = (torch.cuda.Event(enable_timing=True),
                       torch.cuda.Event(enable_timing=True))
    all_times = {s: [] for s in STAGES + ["layer"]}

    torch.cuda.synchronize(dev)
    for _ in range(5):
        h2 = rms_norm(x.half() if is_fp8 else x, rms_w)
        _moe_ffn(h2, W, cfg, tp, sa, sb, is_fp8, attn_dtype, use_nccl=use_nccl)
    torch.cuda.synchronize(dev)

    for _ in range(iters):
        events["layer"][0].record()
        events["rmsnorm_mlp"][0].record()
        h2 = rms_norm(x.half() if is_fp8 else x, rms_w)
        events["rmsnorm_mlp"][1].record()
        _moe_ffn(h2, W, cfg, tp, sa, sb, is_fp8, attn_dtype,
                 use_nccl=use_nccl, events=events)
        events["layer"][1].record()
        torch.cuda.synchronize(dev)
        for s in STAGES:
            all_times[s].append(events[s][0].elapsed_time(events[s][1]) * 1000)
        all_times["layer"].append(
            events["layer"][0].elapsed_time(events["layer"][1]) * 1000)

    import statistics
    medians = {s: statistics.median(all_times[s]) for s in STAGES + ["layer"]}
    return medians, STAGES


def print_e2e_table(cfg_name, medians, STAGES, tp, rank):
    print()
    print("=" * 70)
    print(f"  MoE FFN E2E Latency (median) | model={cfg_name} TP={tp} rank={rank}")
    print("=" * 70)
    print(f"{'Stage':<18} | {'Median(us)':>11} | {'% of FFN':>9}")
    print("-" * 70)
    layer_us = medians["layer"]
    for s in STAGES:
        pct = medians[s] / layer_us * 100 if layer_us > 0 else 0
        print(f"{s:<18} | {medians[s]:>11.1f} | {pct:>8.1f}%")
    print(f"{'FFN TOTAL':<18} | {layer_us:>11.1f} |")
    print("-" * 70)


# ── TP worker ────────────────────────────────────────────────────

def _tp_worker(rank, world_size, cfg, cfg_name, dtype_name,
               skip_nccl=False, e2e_iters=0):
    dev = torch.device(f"cuda:{TP_GPU_MAP[rank]}")
    torch.cuda.set_device(dev)
    need_nccl = (not skip_nccl) or (e2e_iters > 0)
    if need_nccl:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.set_float32_matmul_precision("high")
    if e2e_iters > 0:
        medians, stages = run_decode_e2e(cfg, dev, dtype_name,
                                         tp=world_size, rank=rank, iters=e2e_iters)
        if rank == 0:
            print_e2e_table(cfg_name, medians, stages, world_size, rank)
    else:
        run_decode(cfg, dev, dtype_name, tp=world_size, rank=rank,
                   skip_nccl=skip_nccl)
    if need_nccl:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description="MoE FFN decode profiler for ncu")
    p.add_argument("--model", default="deepseek3", choices=list(MOE_CONFIGS))
    p.add_argument("--dtype", default="fp8", choices=["fp16", "bf16", "fp8"])
    p.add_argument("--tp", type=int, default=1, choices=[1, 2])
    p.add_argument("--seq", type=int, default=4096,
                   help="KV length placeholder (FFN is seq-independent; kept for "
                        "symmetry with ncu_decode_profile.py call sites)")
    p.add_argument("--skip-nccl", action="store_true")
    p.add_argument("--e2e", type=int, default=0, metavar="N")
    args = p.parse_args()
    cfg = MOE_CONFIGS[args.model]

    if args.tp == 1:
        dev = torch.device(f"cuda:{TP_GPU_MAP[0]}")
        torch.cuda.init()
        torch.set_float32_matmul_precision("high")
        print(f"\n{'='*60}\n  MoE FFN — {args.model} — single GPU cuda:{dev.index} "
              f"— dtype={args.dtype}\n{'='*60}")
        if args.e2e > 0:
            medians, stages = run_decode_e2e(cfg, dev, args.dtype,
                                             tp=1, rank=0, iters=args.e2e)
            print_e2e_table(args.model, medians, stages, 1, 0)
        else:
            run_decode(cfg, dev, args.dtype, tp=1, rank=0)
    else:
        if args.e2e > 0:
            mp.spawn(_tp_worker,
                     args=(2, cfg, args.model, args.dtype, False, args.e2e),
                     nprocs=2, join=True)
        elif args.skip_nccl:
            _tp_worker(0, 2, cfg, args.model, args.dtype, skip_nccl=True)
        else:
            mp.spawn(_tp_worker,
                     args=(2, cfg, args.model, args.dtype, False, 0),
                     nprocs=2, join=True)


if __name__ == "__main__":
    main()
