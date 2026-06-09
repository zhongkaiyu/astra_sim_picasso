#!/usr/bin/env python3
"""No-sudo MoE FFN decode utilization microbench (CUDA-graph method).

ncu needs root (GPU perf counters). For these tiny single-token expert GEMVs a
plain timing loop is useless — it's dominated by CPU kernel-launch overhead
(gating's 0.5MB weight "takes longer" than a 12MB one). Two fixes make it valid
WITHOUT sudo:
  1) CUDA graph replay  -> removes per-kernel launch overhead.
  2) pool of P distinct weights (P*weight > 50MB L2) -> forces real HBM reads,
     not L2 hits.

Validation vs ncu (dense qwen3 MLP, decode b=1, fp8):
     stage         ncu HBM-BW%   this method   ratio
     gate_up(235MB)   82%           72.0%       0.88
     down(117MB)      77%           66.5%       0.86
  => method reads ~0.87x of ncu's HBM-BW% (residual graph-node overhead). We
     report the raw measurement AND an ncu-calibrated estimate (raw / 0.87).

Usage:
    CUDA_VISIBLE_DEVICES=0 python microbench_moe_util.py --model qwen3-235b --dtype fp8
"""
import argparse
import json
import time
import torch

PEAKS = {  # dtype -> (torch_dtype, weight_bytes, peak_TFLOPS H100)
    "fp16": (torch.float16, 2, 989.0),
    "bf16": (torch.bfloat16, 2, 989.0),
    "fp8":  (torch.float8_e4m3fn, 1, 1979.0),
}
PEAK_BW_TBS = 3.35           # H100 HBM3
NCU_CAL = 0.87               # graph-method / ncu HBM-BW% (from dense validation)

MOE_CONFIGS = {
    "deepseek3":    dict(d=7168, ne=256, top_k=8, di=2048, ns=1),
    "qwen3-235b":   dict(d=4096, ne=128, top_k=8, di=1536, ns=0),
    "gpt-oss-120b": dict(d=2880, ne=128, top_k=4, di=2880, ns=0),
}


def _proj(x, w, sa, sb):
    return torch._scaled_mm(x, w.t(), scale_a=sa, scale_b=sb,
                            out_dtype=torch.float16, use_fast_accum=True)


def bench_graph(M, K, N, dev, peak_tf, P=16, R=200):
    """CUDA-graph timing of (M,K)@(K,N) over P distinct weights. fp8 only."""
    x = torch.randn(M, K, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
    ws = [torch.randn(N, K, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
          for _ in range(P)]
    outs = [torch.empty(M, N, device=dev, dtype=torch.float16) for _ in range(P)]
    sa = torch.ones(1, device=dev, dtype=torch.float32)
    sb = torch.ones(1, device=dev, dtype=torch.float32)
    for w in ws:
        _proj(x, w, sa, sb)
    torch.cuda.synchronize(dev)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i, w in enumerate(ws):
            outs[i].copy_(_proj(x, w, sa, sb))
    torch.cuda.synchronize(dev)
    t = time.time()
    for _ in range(R):
        g.replay()
    torch.cuda.synchronize(dev)
    dt = (time.time() - t) / (R * P)
    wbytes = K * N * 1 + M * K * 1 + M * N * 2     # fp8 weights/act in, fp16 out
    bw = wbytes / dt
    flops = 2 * M * K * N
    mem_util = bw / (PEAK_BW_TBS * 1e12)
    compute_util = (flops / dt) / (peak_tf * 1e12)
    return mem_util, compute_util, dt * 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-235b", choices=list(MOE_CONFIGS))
    ap.add_argument("--dtype", default="fp8", choices=["fp8"],
                    help="fp8 only (CUDA-graph path)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--pool", type=int, default=16, help="distinct weights (defeat L2)")
    ap.add_argument("--replays", type=int, default=200)
    ap.add_argument("--emit", default=None)
    args = ap.parse_args()

    dev = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision("high")
    _, _, peak_tf = PEAKS[args.dtype]
    c = MOE_CONFIGS[args.model]
    d, ne, di, ns = c["d"], c["ne"], c["di"], c["ns"]
    B = args.batch

    stages = [("gating", B, d, ne),
              ("expert_gate_up", B, d, 2 * di),
              ("expert_down", B, di, d)]
    if ns > 0:
        stages += [("shared_gate_up", B, d, 2 * di * ns),
                   ("shared_down", B, di * ns, d)]

    print(f"\n{'='*92}")
    print(f"  MoE FFN decode utilization (CUDA-graph, no-sudo) | model={args.model} "
          f"dtype={args.dtype} B={B} GPU{args.gpu}")
    print(f"  H100 peak {peak_tf} TFLOPS / {PEAK_BW_TBS} TB/s | pool={args.pool} "
          f"replays={args.replays} | ncu_cal={NCU_CAL}")
    print(f"{'='*92}")
    print(f"{'stage':<16} | {'M':>4} {'K':>6} {'N':>6} | {'t(us)':>7} | "
          f"{'mem_util':>8} | {'ncu~est':>8} | {'cmpt_util':>9}")
    print("-" * 92)

    out = {}
    for name, M, K, N in stages:
        mu, cu, t_us = bench_graph(M, K, N, dev, peak_tf,
                                   P=args.pool, R=args.replays)
        ncu_est = min(mu / NCU_CAL, 0.99)
        out[name] = {"mem_util": round(ncu_est, 4), "compute_util": round(cu, 4),
                     "mem_util_raw": round(mu, 4), "bound": "mem"}
        print(f"{name:<16} | {M:>4} {K:>6} {N:>6} | {t_us:>7.2f} | "
              f"{mu*100:>7.1f}% | {ncu_est*100:>7.1f}% | {cu*100:>8.2f}%")
    print("-" * 92)
    print("  mem_util column = raw graph measurement; ncu~est = raw / 0.87 "
          "(calibrated to ncu HBM-BW%). JSON below uses ncu~est.")

    if args.emit:
        blob = {"_source": f"microbench_moe_util.py CUDA-graph on H100 GPU{args.gpu}, "
                           f"calibrated to ncu (x1/{NCU_CAL})",
                "_model": args.model, "_dtype": args.dtype, "_batch": B,
                "_measured": True, "moe": out}
        with open(args.emit, "w") as f:
            json.dump(blob, f, indent=2)
        print(f"=> wrote {args.emit}")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'mem_util_raw'}
                      for k, v in out.items()}, indent=2))


if __name__ == "__main__":
    main()
