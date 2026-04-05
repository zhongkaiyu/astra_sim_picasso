import time
import argparse
import torch
import csv
import os
import torch.nn.functional as F
from einops import rearrange, einsum

# {name: (torch_dtype, bytes_per_elem, H100_SXM_peak_TFLOPS)}
DTYPE_CONFIGS = {
    "fp16": (torch.float16,       2, 989.0),
    "bf16": (torch.bfloat16,      2, 989.0),
    "fp8":  (torch.float8_e4m3fn, 1, 1979.0),
}
DEFAULT_PEAK_BW_TBS = 3.35


MODELS = {
    "qwen3-235B": {
        "d_model": 4096, "num_attention_heads": 64,
        "num_kv_heads": 4, "d_head": 128, "num_layers": 94,
    },
    "deepseek": {
        "d_model": 7168, "num_attention_heads": 128,
        "num_kv_heads": 8, "d_head": 128, "num_layers": 61,
    },
    "llama4": {
        "d_model": 5120, "num_attention_heads": 40,
        "num_kv_heads": 8, "d_head": 128, "num_layers": 48,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="GQA Attention microbenchmark (single/multi-GPU, FP16/BF16/FP8)")
    parser.add_argument("--model", type=str, default="qwen3-235B",
                        choices=list(MODELS.keys()),
                        help="Model to benchmark (default: qwen3-235B)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU ids, e.g. '0' or '0,1'")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp8"],
                        help="Data type (default: fp16). All stages use the selected dtype.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="Peak TFLOPS per GPU (auto-set from dtype if omitted)")
    parser.add_argument("--peak-bw", type=float, default=DEFAULT_PEAK_BW_TBS,
                        help="Peak HBM bandwidth in TB/s per GPU (default: 3.35)")
    return parser.parse_args()


def _proj(x, w, sa=None, sb=None):
    """Linear projection: FP8 _scaled_mm or standard matmul.

    FP8:     x (M,K) fp8, w (N,K) fp8.  Returns (M,N) fp16.
    FP16/BF16: x (M,K), w (K,N).         Returns x @ w.
    """
    if sa is not None:
        x_fp8 = x if x.dtype == torch.float8_e4m3fn else x.to(torch.float8_e4m3fn)
        return torch._scaled_mm(x_fp8, w.t(), scale_a=sa, scale_b=sb,
                                out_dtype=torch.float16, use_fast_accum=True)
    return x @ w


def bench_attention(batch_sizes, seq_lengths, csv_writer,
                    model_config=None, dtype_name="fp16", devices=None,
                    peak_tflops=989.0, peak_bw_tbs=DEFAULT_PEAK_BW_TBS,
                    warmup=10, iters=50, label=""):
    if devices is None:
        devices = [torch.device("cuda:0")]
    num_gpus = len(devices)
    torch_dtype, dtype_bytes, _ = DTYPE_CONFIGS[dtype_name]
    is_fp8 = (dtype_name == "fp8")

    agg_peak_tflops = peak_tflops * num_gpus
    agg_peak_bw = peak_bw_tbs * num_gpus

    d_model = model_config["d_model"]
    num_q_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config["num_kv_heads"]
    d_head = model_config["d_head"]
    q_dim = num_q_heads * d_head       # 8192
    kv_dim = num_kv_heads * d_head     # 512
    num_head_groups = num_q_heads // num_kv_heads  # 16

    STAGES = ["proj_qkv", "attn", "proj_o"]

    print(f"\n{'='*90}")
    print(f"  {label} Full Attention (GQA)")
    print(f"  d_model={d_model}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, d_head={d_head}")
    print(f"  Wq:({d_model},{q_dim}) Wk:({d_model},{kv_dim}) Wv:({d_model},{kv_dim}) Wo:({q_dim},{d_model})")
    print(f"  GQA groups={num_head_groups}, dtype={dtype_name}, devices={devices}")
    print(f"  peak={peak_tflops} TFLOPS/GPU ({dtype_name}), BW={peak_bw_tbs} TB/s/GPU")
    print(f"{'='*90}")

    streams = [torch.cuda.Stream(device=d) for d in devices]

    for B in batch_sizes:
        for S in seq_lengths:
            if B < num_gpus:
                print(f"B={B:3d}, S={S:5d} | skipped (B < num_gpus)")
                continue

            tokens = B * S
            chunk = B // num_gpus
            remainder = B % num_gpus
            splits = [chunk + (1 if i < remainder else 0) for i in range(num_gpus)]

            # Estimate scores memory: B * num_q_heads * S * S * 2 bytes (fp16)
            scores_bytes = B * num_q_heads * S * S * 2
            # Skip if scores alone would exceed 40GB (leave room for other tensors)
            if scores_bytes > 40e9:
                print(f"\nB={B:3d}, S={S:5d} | skipped (scores={scores_bytes/1e9:.1f}GB, would OOM)")
                continue

            # Allocate per-device tensors
            try:
                per_dev = []
                for idx, dev in enumerate(devices):
                    bi = splits[idx]
                    toks_i = bi * S
                    sa = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None
                    sb = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None

                    if is_fp8:
                        x  = torch.randn(toks_i, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wq = torch.randn(q_dim,  d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wk = torch.randn(kv_dim, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wv = torch.randn(kv_dim, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wo = torch.randn(d_model, q_dim,  device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    else:
                        x  = torch.randn(toks_i, d_model, device=dev, dtype=torch_dtype)
                        Wq = torch.randn(d_model, q_dim,   device=dev, dtype=torch_dtype)
                        Wk = torch.randn(d_model, kv_dim,  device=dev, dtype=torch_dtype)
                        Wv = torch.randn(d_model, kv_dim,  device=dev, dtype=torch_dtype)
                        Wo = torch.randn(q_dim,   d_model,  device=dev, dtype=torch_dtype)

                    per_dev.append({
                        "x": x, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                        "sa": sa, "sb": sb, "B": bi, "tokens": toks_i,
                    })
            except torch.OutOfMemoryError:
                print(f"\nB={B:3d}, S={S:5d} | skipped (OOM during allocation)")
                for dev in devices:
                    torch.cuda.empty_cache()
                continue

            # ---- warmup (with OOM guard) ----
            try:
                for _ in range(warmup):
                    for i, dev in enumerate(devices):
                        with torch.cuda.stream(streams[i]):
                            p = per_dev[i]
                            q = _proj(p["x"], p["Wq"], p["sa"], p["sb"])
                            k = _proj(p["x"], p["Wk"], p["sa"], p["sb"])
                            v = _proj(p["x"], p["Wv"], p["sa"], p["sb"])
                            q = rearrange(q.view(p["B"], S, num_q_heads, d_head), "b n h d -> b h n d")
                            k = rearrange(k.view(p["B"], S, num_kv_heads, d_head), "b s h d -> b h s d")
                            v = rearrange(v.view(p["B"], S, num_kv_heads, d_head), "b s h d -> b h s d")
                            q = rearrange(q, "b (h g) n d -> b g h n d", g=num_head_groups)
                            scores = einsum(q, k, "b g h n d, b h s d -> b g h n s")
                            scores = F.softmax(scores, dim=-1)
                            attn_out = einsum(scores, v, "b g h n s, b h s d -> b g h n d")
                            attn_out = rearrange(attn_out, "b g h n d -> b n (h g) d")
                            attn_out = attn_out.contiguous().view(p["tokens"], q_dim)
                            _ = _proj(attn_out, p["Wo"], p["sa"], p["sb"])
                for dev in devices:
                    torch.cuda.synchronize(dev)
            except torch.OutOfMemoryError:
                print(f"\nB={B:3d}, S={S:5d} | skipped (OOM during warmup)")
                for dev in devices:
                    torch.cuda.empty_cache()
                continue

            # ---- Pre-create CUDA events: [stage][device][iter] ----
            ev_s = {st: [[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                         for _ in range(num_gpus)] for st in STAGES}
            ev_e = {st: [[torch.cuda.Event(enable_timing=True) for _ in range(iters)]
                         for _ in range(num_gpus)] for st in STAGES}

            # ---- Timed iterations ----
            for it in range(iters):
                for i, dev in enumerate(devices):
                    with torch.cuda.stream(streams[i]):
                        p = per_dev[i]

                        # === proj_qkv ===
                        ev_s["proj_qkv"][i][it].record()
                        q = _proj(p["x"], p["Wq"], p["sa"], p["sb"])
                        k = _proj(p["x"], p["Wk"], p["sa"], p["sb"])
                        v = _proj(p["x"], p["Wv"], p["sa"], p["sb"])
                        ev_e["proj_qkv"][i][it].record()

                        # === attn (always fp16: reshape + QK^T + softmax + AV) ===
                        ev_s["attn"][i][it].record()
                        q = rearrange(q.view(p["B"], S, num_q_heads, d_head), "b n h d -> b h n d")
                        k = rearrange(k.view(p["B"], S, num_kv_heads, d_head), "b s h d -> b h s d")
                        v = rearrange(v.view(p["B"], S, num_kv_heads, d_head), "b s h d -> b h s d")
                        q = rearrange(q, "b (h g) n d -> b g h n d", g=num_head_groups)
                        scores = einsum(q, k, "b g h n d, b h s d -> b g h n s")
                        scores = F.softmax(scores, dim=-1)
                        attn_out = einsum(scores, v, "b g h n s, b h s d -> b g h n d")
                        attn_out = rearrange(attn_out, "b g h n d -> b n (h g) d")
                        attn_out = attn_out.contiguous().view(p["tokens"], q_dim)
                        ev_e["attn"][i][it].record()

                        # === proj_o (includes fp16->fp8 cast if FP8 mode) ===
                        ev_s["proj_o"][i][it].record()
                        _ = _proj(attn_out, p["Wo"], p["sa"], p["sb"])
                        ev_e["proj_o"][i][it].record()

            for dev in devices:
                torch.cuda.synchronize(dev)

            # ---- Per-stage timing (max across GPUs per iter, then average) ----
            stage_ms = {}
            for st in STAGES:
                per_iter = []
                for it in range(iters):
                    t_max = max(ev_s[st][i][it].elapsed_time(ev_e[st][i][it])
                                for i in range(num_gpus))
                    per_iter.append(t_max)
                stage_ms[st] = sum(per_iter) / iters

            total_ms = sum(stage_ms.values())

            # ---- FLOPS per stage (total, precision-independent) ----
            flops = {}
            flops["proj_qkv"] = tokens * d_model * (q_dim + 2 * kv_dim) * 2
            flops["attn"] = (B * num_q_heads * S * d_head * S * 2
                             + B * num_q_heads * S * S * 5
                             + B * num_q_heads * S * S * d_head * 2)
            flops["proj_o"] = tokens * q_dim * d_model * 2

            # ---- Bytes per stage ----
            bytes_moved = {}
            if is_fp8:
                # proj_qkv: read x(fp8,1B)*3, read W(fp8,1B), write qkv(fp16,2B)
                bytes_moved["proj_qkv"] = (
                    (3 * tokens * d_model + d_model * q_dim + 2 * d_model * kv_dim) * 1
                    + (tokens * q_dim + 2 * tokens * kv_dim) * 2
                )
                # proj_o: read attn_out(fp16->fp8 cast, 2+1B) + read Wo(fp8,1B) + write out(fp16,2B)
                bytes_moved["proj_o"] = (
                    tokens * q_dim * 3      # fp16 read + fp8 write (cast)
                    + q_dim * d_model * 1   # read Wo (fp8)
                    + tokens * d_model * 2  # write output (fp16)
                )
            else:
                bytes_moved["proj_qkv"] = (
                    3 * tokens * d_model
                    + d_model * q_dim + 2 * d_model * kv_dim
                    + tokens * q_dim + 2 * tokens * kv_dim
                ) * dtype_bytes
                bytes_moved["proj_o"] = (
                    tokens * q_dim + q_dim * d_model + tokens * d_model
                ) * dtype_bytes

            # attn: uses dtype_bytes (fp8=1B, fp16=2B) for all tensors
            bytes_moved["attn"] = (
                B * num_q_heads * S * d_head          # read q
                + B * num_kv_heads * S * d_head       # read k
                + B * num_q_heads * S * S             # write scores
                + 2 * B * num_q_heads * S * S         # softmax read+write
                + B * num_q_heads * S * S             # read scores for AV
                + B * num_kv_heads * S * d_head       # read v
                + B * num_q_heads * S * d_head        # write attn_out
            ) * dtype_bytes

            # ---- Print ----
            print(f"\nB={B:3d}, S={S:5d} | total={total_ms:9.3f} ms")
            print(f"  {'stage':<10} | {'time_ms':>9} | {'%':>6} | {'TFLOPS':>8} | {'comp%':>6} | {'bytes_GB':>9} | {'BW_TB/s':>8} | {'bw%':>6}")
            print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}")

            for st in STAGES:
                t_ms = stage_ms[st]
                pct = t_ms / total_ms * 100 if total_ms > 0 else 0
                tf = flops[st] / (t_ms / 1000) / 1e12 if t_ms > 0 else 0
                cu = tf / agg_peak_tflops * 100
                b_gb = bytes_moved[st] / 1e9
                bw = bytes_moved[st] / (t_ms / 1000) / 1e12 if t_ms > 0 else 0
                bu = bw / agg_peak_bw * 100

                print(f"  {st:<10} | {t_ms:>9.3f} | {pct:>5.1f}% | {tf:>8.2f} | {cu:>5.1f}% | {b_gb:>9.4f} | {bw:>8.3f} | {bu:>5.1f}%")

                csv_writer.writerow({
                    "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
                    "batch": B, "seq_len": S, "stage": st,
                    "time_ms": f"{t_ms:.4f}", "time_pct": f"{pct:.2f}",
                    "TFLOPS": f"{tf:.3f}", "compute_util_pct": f"{cu:.2f}",
                    "bytes_GB": f"{b_gb:.6f}", "BW_TBs": f"{bw:.4f}",
                    "bw_util_pct": f"{bu:.2f}",
                })

            # Total row
            total_flops = sum(flops.values())
            total_bytes = sum(bytes_moved.values())
            total_tf = total_flops / (total_ms / 1000) / 1e12 if total_ms > 0 else 0
            total_cu = total_tf / agg_peak_tflops * 100
            total_b_gb = total_bytes / 1e9
            total_bw = total_bytes / (total_ms / 1000) / 1e12 if total_ms > 0 else 0
            total_bu = total_bw / agg_peak_bw * 100

            print(f"  {'TOTAL':<10} | {total_ms:>9.3f} | {100.0:>5.1f}% | {total_tf:>8.2f} | {total_cu:>5.1f}% | {total_b_gb:>9.4f} | {total_bw:>8.3f} | {total_bu:>5.1f}%")

            csv_writer.writerow({
                "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
                "batch": B, "seq_len": S, "stage": "TOTAL",
                "time_ms": f"{total_ms:.4f}", "time_pct": "100.00",
                "TFLOPS": f"{total_tf:.3f}", "compute_util_pct": f"{total_cu:.2f}",
                "bytes_GB": f"{total_b_gb:.6f}", "BW_TBs": f"{total_bw:.4f}",
                "bw_util_pct": f"{total_bu:.2f}",
            })


def bench_attention_flash(batch_sizes, seq_lengths, csv_writer,
                          model_config, dtype_name="fp16", devices=None,
                          peak_tflops=989.0, peak_bw_tbs=3.35,
                          warmup=5, iters=20, label=""):
    """FlashAttention (SDPA) benchmark. O(S) memory — supports up to 1M+ seq."""
    if devices is None:
        devices = [torch.device("cuda:0")]
    num_gpus = len(devices)
    torch_dtype, dtype_bytes, _ = DTYPE_CONFIGS[dtype_name]
    is_fp8 = (dtype_name == "fp8")

    agg_peak_tflops = peak_tflops * num_gpus
    agg_peak_bw = peak_bw_tbs * num_gpus

    d_model = model_config["d_model"]
    num_q_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config["num_kv_heads"]
    d_head = model_config["d_head"]
    q_dim = num_q_heads * d_head
    kv_dim = num_kv_heads * d_head

    STAGES = ["proj_qkv", "attn", "proj_o"]

    print(f"\n{'='*90}")
    print(f"  {label} FlashAttention (SDPA + GQA)")
    print(f"  d_model={d_model}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, d_head={d_head}")
    print(f"  dtype={dtype_name}, devices={devices}")
    print(f"  peak={peak_tflops} TFLOPS/GPU ({dtype_name}), BW={peak_bw_tbs} TB/s/GPU")
    print(f"{'='*90}")

    streams = [torch.cuda.Stream(device=d) for d in devices]

    for B in batch_sizes:
        for S in seq_lengths:
            if B < num_gpus:
                continue

            # Adaptive iterations for large S
            if S > 256 * 1024:
                cur_warmup, cur_iters = 1, 3
            elif S > 64 * 1024:
                cur_warmup, cur_iters = 2, 5
            elif S > 16 * 1024:
                cur_warmup, cur_iters = 3, 10
            else:
                cur_warmup, cur_iters = warmup, iters

            tokens = B * S
            chunk = B // num_gpus
            remainder = B % num_gpus
            splits = [chunk + (1 if i < remainder else 0) for i in range(num_gpus)]

            # Allocate
            try:
                per_dev = []
                for idx, dev in enumerate(devices):
                    bi = splits[idx]
                    toks_i = bi * S
                    sa = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None
                    sb = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None
                    if is_fp8:
                        x  = torch.randn(toks_i, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wq = torch.randn(q_dim,  d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wk = torch.randn(kv_dim, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wv = torch.randn(kv_dim, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                        Wo = torch.randn(d_model, q_dim,  device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    else:
                        x  = torch.randn(toks_i, d_model, device=dev, dtype=torch_dtype)
                        Wq = torch.randn(d_model, q_dim,   device=dev, dtype=torch_dtype)
                        Wk = torch.randn(d_model, kv_dim,  device=dev, dtype=torch_dtype)
                        Wv = torch.randn(d_model, kv_dim,  device=dev, dtype=torch_dtype)
                        Wo = torch.randn(q_dim,   d_model,  device=dev, dtype=torch_dtype)
                    per_dev.append({
                        "x": x, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                        "sa": sa, "sb": sb, "B": bi, "tokens": toks_i,
                    })
            except torch.OutOfMemoryError:
                print(f"\nB={B:3d}, S={S:7d} | skipped (OOM during allocation)")
                for dev in devices:
                    torch.cuda.empty_cache()
                continue

            # Warmup
            try:
                for _ in range(cur_warmup):
                    for i, dev in enumerate(devices):
                        with torch.cuda.stream(streams[i]):
                            p = per_dev[i]
                            q = _proj(p["x"], p["Wq"], p["sa"], p["sb"])
                            k = _proj(p["x"], p["Wk"], p["sa"], p["sb"])
                            v = _proj(p["x"], p["Wv"], p["sa"], p["sb"])
                            q_4d = q.view(p["B"], S, num_q_heads, d_head).transpose(1, 2)
                            k_4d = k.view(p["B"], S, num_kv_heads, d_head).transpose(1, 2)
                            v_4d = v.view(p["B"], S, num_kv_heads, d_head).transpose(1, 2)
                            attn_out = F.scaled_dot_product_attention(
                                q_4d, k_4d, v_4d, enable_gqa=True)
                            attn_out = attn_out.transpose(1, 2).contiguous().view(p["tokens"], q_dim)
                            _ = _proj(attn_out, p["Wo"], p["sa"], p["sb"])
                for dev in devices:
                    torch.cuda.synchronize(dev)
            except torch.OutOfMemoryError:
                print(f"\nB={B:3d}, S={S:7d} | skipped (OOM during warmup)")
                for dev in devices:
                    torch.cuda.empty_cache()
                continue

            # CUDA events
            ev_s = {st: [[torch.cuda.Event(enable_timing=True) for _ in range(cur_iters)]
                         for _ in range(num_gpus)] for st in STAGES}
            ev_e = {st: [[torch.cuda.Event(enable_timing=True) for _ in range(cur_iters)]
                         for _ in range(num_gpus)] for st in STAGES}

            # Timed iterations
            for it in range(cur_iters):
                for i, dev in enumerate(devices):
                    with torch.cuda.stream(streams[i]):
                        p = per_dev[i]

                        ev_s["proj_qkv"][i][it].record()
                        q = _proj(p["x"], p["Wq"], p["sa"], p["sb"])
                        k = _proj(p["x"], p["Wk"], p["sa"], p["sb"])
                        v = _proj(p["x"], p["Wv"], p["sa"], p["sb"])
                        ev_e["proj_qkv"][i][it].record()

                        ev_s["attn"][i][it].record()
                        q_4d = q.view(p["B"], S, num_q_heads, d_head).transpose(1, 2)
                        k_4d = k.view(p["B"], S, num_kv_heads, d_head).transpose(1, 2)
                        v_4d = v.view(p["B"], S, num_kv_heads, d_head).transpose(1, 2)
                        attn_out = F.scaled_dot_product_attention(
                            q_4d, k_4d, v_4d, enable_gqa=True)
                        attn_out = attn_out.transpose(1, 2).contiguous().view(p["tokens"], q_dim)
                        ev_e["attn"][i][it].record()

                        ev_s["proj_o"][i][it].record()
                        _ = _proj(attn_out, p["Wo"], p["sa"], p["sb"])
                        ev_e["proj_o"][i][it].record()

            for dev in devices:
                torch.cuda.synchronize(dev)

            # Compute timing
            stage_ms = {}
            for st in STAGES:
                per_iter = []
                for it in range(cur_iters):
                    t_max = max(ev_s[st][i][it].elapsed_time(ev_e[st][i][it])
                                for i in range(num_gpus))
                    per_iter.append(t_max)
                stage_ms[st] = sum(per_iter) / cur_iters
            total_ms = sum(stage_ms.values())

            # FLOPS (same math ops regardless of kernel)
            flops = {}
            flops["proj_qkv"] = tokens * d_model * (q_dim + 2 * kv_dim) * 2
            flops["attn"] = (B * num_q_heads * S * d_head * S * 2
                             + B * num_q_heads * S * S * 5
                             + B * num_q_heads * S * S * d_head * 2)
            flops["proj_o"] = tokens * q_dim * d_model * 2

            # Bytes — FlashAttention: O(S) HBM I/O for attn, no S^2 scores
            bytes_moved = {}
            if is_fp8:
                bytes_moved["proj_qkv"] = (
                    (3 * tokens * d_model + d_model * q_dim + 2 * d_model * kv_dim) * 1
                    + (tokens * q_dim + 2 * tokens * kv_dim) * 2)
                bytes_moved["proj_o"] = (
                    tokens * q_dim * 3 + q_dim * d_model * 1 + tokens * d_model * 2)
            else:
                bytes_moved["proj_qkv"] = (
                    3 * tokens * d_model + d_model * q_dim + 2 * d_model * kv_dim
                    + tokens * q_dim + 2 * tokens * kv_dim) * dtype_bytes
                bytes_moved["proj_o"] = (
                    tokens * q_dim + q_dim * d_model + tokens * d_model) * dtype_bytes
            # FlashAttention: read q,k,v once + write output (minimum HBM I/O)
            bytes_moved["attn"] = (
                B * num_q_heads * S * d_head       # read q
                + B * num_kv_heads * S * d_head    # read k
                + B * num_kv_heads * S * d_head    # read v
                + B * num_q_heads * S * d_head     # write output
            ) * 2  # fp16 (SDPA always runs in fp16)

            # Print
            print(f"\nB={B:3d}, S={S:7d} (iters={cur_iters}) | total={total_ms:11.3f} ms")
            print(f"  {'stage':<10} | {'time_ms':>11} | {'%':>6} | {'TFLOPS':>8} | {'comp%':>6} | {'bytes_GB':>9} | {'BW_TB/s':>8} | {'bw%':>6}")
            print(f"  {'-'*10}-+-{'-'*11}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}")

            for st in STAGES:
                t_ms = stage_ms[st]
                pct = t_ms / total_ms * 100 if total_ms > 0 else 0
                tf = flops[st] / (t_ms / 1000) / 1e12 if t_ms > 0 else 0
                cu = tf / agg_peak_tflops * 100
                b_gb = bytes_moved[st] / 1e9
                bw = bytes_moved[st] / (t_ms / 1000) / 1e12 if t_ms > 0 else 0
                bu = bw / agg_peak_bw * 100

                print(f"  {st:<10} | {t_ms:>11.3f} | {pct:>5.1f}% | {tf:>8.2f} | {cu:>5.1f}% | {b_gb:>9.4f} | {bw:>8.3f} | {bu:>5.1f}%")
                csv_writer.writerow({
                    "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
                    "batch": B, "seq_len": S, "stage": st,
                    "time_ms": f"{t_ms:.4f}", "time_pct": f"{pct:.2f}",
                    "TFLOPS": f"{tf:.3f}", "compute_util_pct": f"{cu:.2f}",
                    "bytes_GB": f"{b_gb:.6f}", "BW_TBs": f"{bw:.4f}",
                    "bw_util_pct": f"{bu:.2f}",
                })

            total_flops = sum(flops.values())
            total_bytes = sum(bytes_moved.values())
            total_tf = total_flops / (total_ms / 1000) / 1e12 if total_ms > 0 else 0
            total_cu = total_tf / agg_peak_tflops * 100
            total_b_gb = total_bytes / 1e9
            total_bw = total_bytes / (total_ms / 1000) / 1e12 if total_ms > 0 else 0
            total_bu = total_bw / agg_peak_bw * 100

            print(f"  {'TOTAL':<10} | {total_ms:>11.3f} | {100.0:>5.1f}% | {total_tf:>8.2f} | {total_cu:>5.1f}% | {total_b_gb:>9.4f} | {total_bw:>8.3f} | {total_bu:>5.1f}%")
            csv_writer.writerow({
                "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
                "batch": B, "seq_len": S, "stage": "TOTAL",
                "time_ms": f"{total_ms:.4f}", "time_pct": "100.00",
                "TFLOPS": f"{total_tf:.3f}", "compute_util_pct": f"{total_cu:.2f}",
                "bytes_GB": f"{total_b_gb:.6f}", "BW_TBs": f"{total_bw:.4f}",
                "bw_util_pct": f"{total_bu:.2f}",
            })

            # Free memory for next config
            del per_dev
            torch.cuda.empty_cache()


def bench_decode(seq_lengths, csv_writer,
                 model_config, dtype_name="fp16", devices=None,
                 peak_tflops=989.0, peak_bw_tbs=3.35,
                 warmup=20, iters=100, label=""):
    """Decode benchmark: 1 new token with pre-filled KV cache of length S.

    Multi-GPU uses Tensor Parallelism: each GPU holds 1/N of the attention
    heads and weight matrices.  All GPUs process the same single token in
    parallel.  An all-reduce for proj_o output is estimated but not measured.

    Models autoregressive token generation (memory-bound regime).
    Stages: proj_qkv (weight load), attn (KV cache read), proj_o (weight load).
    """
    if devices is None:
        devices = [torch.device("cuda:0")]
    num_gpus = len(devices)
    torch_dtype, dtype_bytes, _ = DTYPE_CONFIGS[dtype_name]
    is_fp8 = (dtype_name == "fp8")

    # Per-GPU peak (roofline is per-GPU; timing is max across GPUs)
    per_gpu_peak_tflops = peak_tflops
    per_gpu_peak_bw = peak_bw_tbs
    # Aggregate for throughput metrics
    agg_peak_tflops = peak_tflops * num_gpus
    agg_peak_bw = peak_bw_tbs * num_gpus

    d_model = model_config["d_model"]
    num_q_heads = model_config["num_attention_heads"]
    num_kv_heads = model_config["num_kv_heads"]
    d_head = model_config["d_head"]
    q_dim = num_q_heads * d_head
    kv_dim = num_kv_heads * d_head

    # TP split: each GPU gets a slice of heads
    assert num_q_heads % num_gpus == 0, f"num_q_heads={num_q_heads} not divisible by {num_gpus} GPUs"
    assert num_kv_heads % num_gpus == 0, f"num_kv_heads={num_kv_heads} not divisible by {num_gpus} GPUs"
    q_heads_per_gpu = num_q_heads // num_gpus
    kv_heads_per_gpu = num_kv_heads // num_gpus
    q_dim_per_gpu = q_heads_per_gpu * d_head
    kv_dim_per_gpu = kv_heads_per_gpu * d_head

    # Roofline crossover: arithmetic intensity where compute = memory time
    oi_crossover = per_gpu_peak_tflops * 1e12 / (per_gpu_peak_bw * 1e12)  # FLOP/byte

    STAGES = ["proj_qkv", "attn", "proj_o"]

    print(f"\n{'='*110}")
    print(f"  {label} Decode (1 new token, KV cache = S)")
    print(f"  d_model={d_model}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, d_head={d_head}")
    tp_str = f"TP={num_gpus}" if num_gpus > 1 else "no TP"
    print(f"  dtype={dtype_name}, devices={devices}, {tp_str}")
    if num_gpus > 1:
        print(f"  TP split: {q_heads_per_gpu} q_heads + {kv_heads_per_gpu} kv_heads per GPU")
    print(f"  peak={peak_tflops} TFLOPS/GPU ({dtype_name}), BW={peak_bw_tbs} TB/s/GPU")
    print(f"  roofline crossover OI = {oi_crossover:.1f} FLOP/byte")
    print(f"{'='*110}")

    streams = [torch.cuda.Stream(device=d) for d in devices]

    for S in seq_lengths:
        # Adaptive iterations: larger KV caches need fewer iters
        if S > 256 * 1024:
            cur_warmup, cur_iters = 5, 20
        elif S > 64 * 1024:
            cur_warmup, cur_iters = 10, 50
        else:
            cur_warmup, cur_iters = warmup, iters

        B = 1

        # Allocate TP-sharded tensors per GPU
        try:
            per_dev = []
            for idx, dev in enumerate(devices):
                sa = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None
                sb = torch.ones(1, device=dev, dtype=torch.float32) if is_fp8 else None

                # Input: single token embedding (replicated on each GPU)
                if is_fp8:
                    x = torch.randn(1, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    # TP-sharded weights: each GPU holds 1/N of heads
                    Wq = torch.randn(q_dim_per_gpu,  d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    Wk = torch.randn(kv_dim_per_gpu, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    Wv = torch.randn(kv_dim_per_gpu, d_model, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    Wo = torch.randn(d_model, q_dim_per_gpu,  device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                else:
                    x = torch.randn(1, d_model, device=dev, dtype=torch_dtype)
                    Wq = torch.randn(d_model, q_dim_per_gpu,   device=dev, dtype=torch_dtype)
                    Wk = torch.randn(d_model, kv_dim_per_gpu,  device=dev, dtype=torch_dtype)
                    Wv = torch.randn(d_model, kv_dim_per_gpu,  device=dev, dtype=torch_dtype)
                    Wo = torch.randn(q_dim_per_gpu,   d_model,  device=dev, dtype=torch_dtype)

                # TP-sharded KV cache: each GPU holds its kv_heads slice
                # FP8 mode: store KV cache in FP8 (consistent with roofline model)
                if is_fp8:
                    K_cache = torch.randn(B, kv_heads_per_gpu, S, d_head, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    V_cache = torch.randn(B, kv_heads_per_gpu, S, d_head, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                else:
                    K_cache = torch.randn(B, kv_heads_per_gpu, S, d_head, device=dev, dtype=torch.float16)
                    V_cache = torch.randn(B, kv_heads_per_gpu, S, d_head, device=dev, dtype=torch.float16)

                per_dev.append({
                    "x": x, "Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo,
                    "K_cache": K_cache, "V_cache": V_cache,
                    "sa": sa, "sb": sb,
                })
        except torch.OutOfMemoryError:
            print(f"\nS={S:>7} | skipped (OOM during allocation)")
            for dev in devices:
                torch.cuda.empty_cache()
            continue

        # Warmup
        try:
            for _ in range(cur_warmup):
                for i, dev in enumerate(devices):
                    with torch.cuda.stream(streams[i]):
                        p = per_dev[i]
                        q = _proj(p["x"], p["Wq"], p["sa"], p["sb"])
                        k_new = _proj(p["x"], p["Wk"], p["sa"], p["sb"])
                        v_new = _proj(p["x"], p["Wv"], p["sa"], p["sb"])
                        q_4d = q.float().view(B, q_heads_per_gpu, 1, d_head)
                        k_new_4d = k_new.float().view(B, kv_heads_per_gpu, 1, d_head)
                        v_new_4d = v_new.float().view(B, kv_heads_per_gpu, 1, d_head)
                        k_full = torch.cat([p["K_cache"].half(), k_new_4d.half()], dim=2)
                        v_full = torch.cat([p["V_cache"].half(), v_new_4d.half()], dim=2)
                        attn_out = F.scaled_dot_product_attention(
                            q_4d.half(), k_full, v_full, enable_gqa=True)
                        attn_out = attn_out.contiguous().view(1, q_dim_per_gpu)
                        _ = _proj(attn_out, p["Wo"], p["sa"], p["sb"])
            for dev in devices:
                torch.cuda.synchronize(dev)
        except torch.OutOfMemoryError:
            print(f"\nS={S:>7} | skipped (OOM during warmup)")
            for dev in devices:
                torch.cuda.empty_cache()
            continue

        # CUDA events
        ev_s = {st: [[torch.cuda.Event(enable_timing=True) for _ in range(cur_iters)]
                      for _ in range(num_gpus)] for st in STAGES}
        ev_e = {st: [[torch.cuda.Event(enable_timing=True) for _ in range(cur_iters)]
                      for _ in range(num_gpus)] for st in STAGES}

        # Timed iterations
        for it in range(cur_iters):
            for i, dev in enumerate(devices):
                with torch.cuda.stream(streams[i]):
                    p = per_dev[i]

                    ev_s["proj_qkv"][i][it].record()
                    q = _proj(p["x"], p["Wq"], p["sa"], p["sb"])
                    k_new = _proj(p["x"], p["Wk"], p["sa"], p["sb"])
                    v_new = _proj(p["x"], p["Wv"], p["sa"], p["sb"])
                    ev_e["proj_qkv"][i][it].record()

                    ev_s["attn"][i][it].record()
                    q_4d = q.float().view(B, q_heads_per_gpu, 1, d_head)
                    k_new_4d = k_new.float().view(B, kv_heads_per_gpu, 1, d_head)
                    v_new_4d = v_new.float().view(B, kv_heads_per_gpu, 1, d_head)
                    k_full = torch.cat([p["K_cache"].half(), k_new_4d.half()], dim=2)
                    v_full = torch.cat([p["V_cache"].half(), v_new_4d.half()], dim=2)
                    attn_out = F.scaled_dot_product_attention(
                        q_4d.half(), k_full, v_full, enable_gqa=True)
                    attn_out = attn_out.contiguous().view(1, q_dim_per_gpu)
                    ev_e["attn"][i][it].record()

                    ev_s["proj_o"][i][it].record()
                    _ = _proj(attn_out, p["Wo"], p["sa"], p["sb"])
                    ev_e["proj_o"][i][it].record()

        for dev in devices:
            torch.cuda.synchronize(dev)

        # Compute timing (max across GPUs per iter = TP latency)
        stage_ms = {}
        for st in STAGES:
            per_iter = []
            for it in range(cur_iters):
                t_max = max(ev_s[st][i][it].elapsed_time(ev_e[st][i][it])
                            for i in range(num_gpus))
                per_iter.append(t_max)
            stage_ms[st] = sum(per_iter) / cur_iters
        total_ms = sum(stage_ms.values())

        # === FLOPS per stage (full model, not per-GPU) ===
        flops = {}
        flops["proj_qkv"] = 1 * d_model * (q_dim + 2 * kv_dim) * 2
        S1 = S + 1
        flops["attn"] = (B * num_q_heads * 1 * d_head * S1 * 2
                         + B * num_q_heads * 1 * S1 * 5
                         + B * num_q_heads * 1 * S1 * d_head * 2)
        flops["proj_o"] = 1 * q_dim * d_model * 2

        # === Bytes moved per stage (per-GPU for roofline, total for throughput) ===
        # Per-GPU bytes (what each GPU actually reads/writes)
        bytes_per_gpu = {}
        if is_fp8:
            bytes_per_gpu["proj_qkv"] = (
                (3 * 1 * d_model + d_model * q_dim_per_gpu + 2 * d_model * kv_dim_per_gpu) * 1
                + (1 * q_dim_per_gpu + 2 * 1 * kv_dim_per_gpu) * 2)
            bytes_per_gpu["proj_o"] = (
                1 * q_dim_per_gpu * 3 + q_dim_per_gpu * d_model * 1 + 1 * d_model * 2)
        else:
            bytes_per_gpu["proj_qkv"] = (
                3 * 1 * d_model + d_model * q_dim_per_gpu + 2 * d_model * kv_dim_per_gpu
                + 1 * q_dim_per_gpu + 2 * 1 * kv_dim_per_gpu) * dtype_bytes
            bytes_per_gpu["proj_o"] = (
                1 * q_dim_per_gpu + q_dim_per_gpu * d_model + 1 * d_model) * dtype_bytes
        # KV cache stored in dtype_bytes (FP8=1B, FP16=2B); q/output always fp16
        kv_cache_bytes = dtype_bytes  # 1 for FP8, 2 for FP16
        bytes_per_gpu["attn"] = (
            B * q_heads_per_gpu * 1 * d_head * 2              # read q (fp16)
            + B * kv_heads_per_gpu * S1 * d_head * kv_cache_bytes  # read K cache
            + B * kv_heads_per_gpu * S1 * d_head * kv_cache_bytes  # read V cache
            + B * q_heads_per_gpu * 1 * d_head * 2              # write output (fp16)
        )

        # Total bytes (all GPUs combined)
        bytes_total = {st: bytes_per_gpu[st] * num_gpus for st in STAGES}

        # === FLOPS per-GPU (TP-sharded) ===
        flops_per_gpu = {st: flops[st] // num_gpus for st in STAGES}

        # === Arithmetic Intensity & Roofline ===
        ai = {}  # FLOP/byte (per-GPU)
        roofline_bound = {}
        for st in STAGES:
            ai[st] = flops_per_gpu[st] / bytes_per_gpu[st] if bytes_per_gpu[st] > 0 else 0
            roofline_bound[st] = "compute" if ai[st] >= oi_crossover else "memory"

        # Print header
        print(f"\nS={S:>7} (iters={cur_iters}) | total={total_ms:9.4f} ms   TP={num_gpus}")
        print(f"  {'stage':<10} | {'time_ms':>9} | {'%':>5} | {'TFLOPS':>7} {'comp%':>6} | {'BW_TB/s':>7} {'bw%':>5} | {'AI':>6} {'bound':>7} | {'ideal_ms':>8} {'overhead':>8}")
        print(f"  {'-'*10}-+-{'-'*9}-+-{'-'*5}-+-{'-'*7}-{'-'*6}-+-{'-'*7}-{'-'*5}-+-{'-'*6}-{'-'*7}-+-{'-'*8}-{'-'*8}")

        for st in STAGES:
            t_ms = stage_ms[st]
            pct = t_ms / total_ms * 100 if total_ms > 0 else 0
            # Per-GPU achieved TFLOPS and BW
            tf_per_gpu = flops_per_gpu[st] / (t_ms / 1000) / 1e12 if t_ms > 0 else 0
            cu = tf_per_gpu / per_gpu_peak_tflops * 100
            bw_per_gpu = bytes_per_gpu[st] / (t_ms / 1000) / 1e12 if t_ms > 0 else 0
            bu = bw_per_gpu / per_gpu_peak_bw * 100
            # Ideal time from roofline
            compute_ms = flops_per_gpu[st] / (per_gpu_peak_tflops * 1e12) * 1000
            memory_ms = bytes_per_gpu[st] / (per_gpu_peak_bw * 1e12) * 1000
            ideal_ms = max(compute_ms, memory_ms)
            overhead = t_ms / ideal_ms if ideal_ms > 0 else 0
            bound_str = roofline_bound[st]

            print(f"  {st:<10} | {t_ms:>9.4f} | {pct:>4.1f}% | {tf_per_gpu:>7.3f} {cu:>5.1f}% | {bw_per_gpu:>7.4f} {bu:>4.1f}% | {ai[st]:>6.1f} {bound_str:>7} | {ideal_ms:>8.4f} {overhead:>7.1f}x")

            csv_writer.writerow({
                "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
                "batch": B, "seq_len": S, "stage": st,
                "time_ms": f"{t_ms:.4f}", "time_pct": f"{pct:.2f}",
                "TFLOPS": f"{tf_per_gpu:.3f}", "compute_util_pct": f"{cu:.2f}",
                "bytes_GB": f"{bytes_per_gpu[st] / 1e9:.6f}",
                "BW_TBs": f"{bw_per_gpu:.4f}", "bw_util_pct": f"{bu:.2f}",
                "arith_intensity": f"{ai[st]:.2f}",
                "roofline_bound": bound_str,
                "ideal_ms": f"{ideal_ms:.4f}",
                "overhead_x": f"{overhead:.2f}",
            })

        # Total row
        total_flops_val = sum(flops.values())
        total_bytes_per_gpu = sum(bytes_per_gpu.values())
        total_flops_per_gpu = sum(flops_per_gpu.values())
        total_tf_per_gpu = total_flops_per_gpu / (total_ms / 1000) / 1e12 if total_ms > 0 else 0
        total_cu = total_tf_per_gpu / per_gpu_peak_tflops * 100
        total_bw_per_gpu = total_bytes_per_gpu / (total_ms / 1000) / 1e12 if total_ms > 0 else 0
        total_bu = total_bw_per_gpu / per_gpu_peak_bw * 100
        total_ai = total_flops_per_gpu / total_bytes_per_gpu if total_bytes_per_gpu > 0 else 0
        total_bound = "compute" if total_ai >= oi_crossover else "memory"
        total_ideal = sum(max(flops_per_gpu[st] / (per_gpu_peak_tflops * 1e12),
                              bytes_per_gpu[st] / (per_gpu_peak_bw * 1e12))
                         for st in STAGES) * 1000
        total_overhead = total_ms / total_ideal if total_ideal > 0 else 0

        print(f"  {'TOTAL':<10} | {total_ms:>9.4f} | 100.% | {total_tf_per_gpu:>7.3f} {total_cu:>5.1f}% | {total_bw_per_gpu:>7.4f} {total_bu:>4.1f}% | {total_ai:>6.1f} {total_bound:>7} | {total_ideal:>8.4f} {total_overhead:>7.1f}x")

        csv_writer.writerow({
            "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
            "batch": B, "seq_len": S, "stage": "TOTAL",
            "time_ms": f"{total_ms:.4f}", "time_pct": "100.00",
            "TFLOPS": f"{total_tf_per_gpu:.3f}", "compute_util_pct": f"{total_cu:.2f}",
            "bytes_GB": f"{total_bytes_per_gpu / 1e9:.6f}",
            "BW_TBs": f"{total_bw_per_gpu:.4f}", "bw_util_pct": f"{total_bu:.2f}",
            "arith_intensity": f"{total_ai:.2f}",
            "roofline_bound": total_bound,
            "ideal_ms": f"{total_ideal:.4f}",
            "overhead_x": f"{total_overhead:.2f}",
        })

        del per_dev
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    devices = [torch.device(f"cuda:{g}") for g in args.gpus.split(",")]
    num_gpus = len(devices)
    dtype_name = args.dtype

    _, _, default_peak = DTYPE_CONFIGS[dtype_name]
    peak_tflops = args.peak_tflops if args.peak_tflops is not None else default_peak

    if dtype_name == "fp8":
        assert hasattr(torch, '_scaled_mm'), "FP8 requires PyTorch 2.2+ with _scaled_mm"

    torch.cuda.init()
    torch.set_float32_matmul_precision("high")

    model_config = MODELS[args.model]

    print(f"Using {num_gpus} GPU(s): {devices}")
    print(f"model={args.model}, dtype={dtype_name}, peak_tflops={peak_tflops}/GPU, peak_bw={args.peak_bw} TB/s/GPU")

    # ---- Sweep 1: batch x seq grid ----
    batch_sizes = [1, 2, 4, 8, 16, 32]
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"{args.model}_attention_{dtype_name}_{num_gpus}gpu.csv")

    fieldnames = ["model", "dtype", "num_gpus", "batch", "seq_len", "stage",
                  "time_ms", "time_pct", "TFLOPS", "compute_util_pct",
                  "bytes_GB", "BW_TBs", "bw_util_pct"]

    file_exists = os.path.exists(results_path)
    with open(results_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        bench_attention(batch_sizes, seq_lengths, writer,
                        model_config=model_config, dtype_name=dtype_name,
                        devices=devices, peak_tflops=peak_tflops, peak_bw_tbs=args.peak_bw,
                        warmup=args.warmup, iters=args.iters,
                        label=f"{args.model}-GQA")

    print(f"\n✅ Results saved to: {results_path}")

    # ---- Sweep 2: B=1 FlashAttention long-sequence scaling (1K to 1M) ----
    flash_seq_lengths = [
        1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576,
    ]

    flash_results_path = os.path.join(results_dir,
        f"{args.model}_attention_flash_{dtype_name}_{num_gpus}gpu.csv")

    with open(flash_results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        print(f"\n{'#'*90}")
        print(f"  FlashAttention sweep: B=1, S from 1K to 1M")
        print(f"  SDPA kernel, O(S) memory, enables seq >> 16K")
        print(f"{'#'*90}")

        bench_attention_flash([1], flash_seq_lengths, writer,
                              model_config=model_config, dtype_name=dtype_name,
                              devices=devices, peak_tflops=peak_tflops,
                              peak_bw_tbs=args.peak_bw,
                              warmup=args.warmup, iters=args.iters,
                              label=f"{args.model}-GQA-flash")

    print(f"\n✅ FlashAttention results saved to: {flash_results_path}")

    # ---- Sweep 3: Decode benchmark (1 token, KV cache = S) ----
    decode_seq_lengths = [
        1024, 2048, 4096, 8192, 16384, 32768,
        65536, 131072, 262144, 524288, 1048576,
    ]

    decode_results_path = os.path.join(results_dir,
        f"{args.model}_attention_decode_{dtype_name}_{num_gpus}gpu.csv")

    decode_fieldnames = fieldnames + [
        "arith_intensity", "roofline_bound", "ideal_ms", "overhead_x"]

    with open(decode_results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=decode_fieldnames)
        writer.writeheader()

        print(f"\n{'#'*90}")
        print(f"  Decode sweep: 1 new token, KV cache from 1K to 1M")
        print(f"  Memory-bound regime: weight load + KV cache read")
        print(f"{'#'*90}")

        bench_decode(decode_seq_lengths, writer,
                     model_config=model_config, dtype_name=dtype_name,
                     devices=devices, peak_tflops=peak_tflops,
                     peak_bw_tbs=args.peak_bw,
                     warmup=args.warmup, iters=args.iters,
                     label=f"{args.model}-GQA-decode")

    print(f"\n✅ Decode results saved to: {decode_results_path}")
