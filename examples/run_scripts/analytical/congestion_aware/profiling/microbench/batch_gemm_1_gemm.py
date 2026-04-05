import time
import argparse
import torch
import csv
import os

# {name: (torch_dtype, bytes_per_elem, H100_SXM_peak_TFLOPS)}
DTYPE_CONFIGS = {
    "fp16": (torch.float16,       2, 989.0),
    "bf16": (torch.bfloat16,      2, 989.0),
    "fp8":  (torch.float8_e4m3fn, 1, 1979.0),
}
DEFAULT_PEAK_BW_TBS = 3.35  # H100 SXM HBM3


MODELS = {
    "qwen3-235B":  {"gemms": [(4096, 1536)],  "label": "Qwen3-235B"},
    "deepseek":    {"gemms": [(7168, 2048)],  "label": "DeepSeek"},
    "llama4":      {"gemms": [(5120, 8192)],  "label": "Llama4"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="GEMM microbenchmark (single/multi-GPU, FP16/BF16/FP8)")
    parser.add_argument("--model", type=str, default="qwen3-235B",
                        choices=list(MODELS.keys()),
                        help="Model to benchmark (default: qwen3-235B)")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU ids, e.g. '0' or '0,1'")
    parser.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp8"],
                        help="Data type for GEMM (default: fp16)")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--peak-tflops", type=float, default=None,
                        help="Peak TFLOPS per GPU (auto-set from dtype if omitted)")
    parser.add_argument("--peak-bw", type=float, default=DEFAULT_PEAK_BW_TBS,
                        help="Peak HBM bandwidth in TB/s per GPU (default: 3.35)")
    parser.add_argument("--mode", type=str, default="prefill",
                        choices=["prefill", "decode"],
                        help="Batch sweep mode: 'prefill' (B=1..8K), 'decode' (B=1..1M seq-aligned)")
    return parser.parse_args()


def _proj(x, w, sa=None, sb=None):
    """Linear projection: FP8 _scaled_mm or standard matmul.

    For FP8:  x is (M, K) fp8, w is (N, K) fp8 row-major.
              Returns (M, N) fp16 via _scaled_mm(x, w.t()).
    For FP16/BF16: x is (M, K), w is (K, N). Returns x @ w.
    """
    if sa is not None:
        x_fp8 = x if x.dtype == torch.float8_e4m3fn else x.to(torch.float8_e4m3fn)
        return torch._scaled_mm(x_fp8, w.t(), scale_a=sa, scale_b=sb,
                                out_dtype=torch.float16, use_fast_accum=True)
    return x @ w


def bench_one(shape_in, shape_out, batches, csv_writer,
              dtype_name, devices, peak_tflops, peak_bw_tbs,
              warmup=10, iters=50, label="", mode="prefill"):
    in_dim = shape_in
    out_dim = shape_out
    num_gpus = len(devices)
    torch_dtype, dtype_bytes, _ = DTYPE_CONFIGS[dtype_name]
    is_fp8 = (dtype_name == "fp8")

    agg_peak_tflops = peak_tflops * num_gpus
    agg_peak_bw = peak_bw_tbs * num_gpus
    oi_crossover = peak_tflops / peak_bw_tbs  # FLOP/byte per GPU

    print(f"\n{'='*120}")
    print(f"  {label} Gemm: (B, {in_dim}) @ ({in_dim}, {out_dim}) -> (B, {out_dim})")
    print(f"  dtype={dtype_name}, devices={devices}, peak={peak_tflops} TFLOPS/GPU, BW={peak_bw_tbs} TB/s/GPU")
    print(f"  roofline crossover OI = {oi_crossover:.1f} FLOP/byte")
    print(f"{'='*120}")
    print(f"{'B':>6} | {'time_ms':>9} | {'TFLOPS':>8} {'comp%':>7} | {'BW_TB/s':>8} {'bw%':>7} | {'AI':>7} {'bound':>7} | {'ideal_ms':>8} {'overhead':>8}")
    print(f"{'-'*6}-+-{'-'*9}-+-{'-'*8}-{'-'*7}-+-{'-'*8}-{'-'*7}-+-{'-'*7}-{'-'*7}-+-{'-'*8}-{'-'*8}")

    streams = [torch.cuda.Stream(device=d) for d in devices]

    for B in batches:
        if B < num_gpus:
            print(f"{B:>6} | skipped (B < num_gpus={num_gpus})")
            continue

        # Adaptive iterations for large batch sizes
        if B > 256 * 1024:
            cur_warmup, cur_iters = 2, 5
        elif B > 64 * 1024:
            cur_warmup, cur_iters = 3, 10
        elif B > 16 * 1024:
            cur_warmup, cur_iters = 5, 20
        else:
            cur_warmup, cur_iters = warmup, iters

        chunk = B // num_gpus
        remainder = B % num_gpus
        splits = [chunk + (1 if i < remainder else 0) for i in range(num_gpus)]

        # Allocate per-device tensors
        try:
            xs, ws, sas, sbs = [], [], [], []
            for i, dev in enumerate(devices):
                bi = splits[i]
                if is_fp8:
                    x = torch.randn(bi, in_dim, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    w = torch.randn(out_dim, in_dim, device=dev, dtype=torch.float16).to(torch.float8_e4m3fn)
                    sa = torch.ones(1, device=dev, dtype=torch.float32)
                    sb = torch.ones(1, device=dev, dtype=torch.float32)
                else:
                    x = torch.randn(bi, in_dim, device=dev, dtype=torch_dtype)
                    w = torch.randn(in_dim, out_dim, device=dev, dtype=torch_dtype)
                    sa, sb = None, None
                xs.append(x); ws.append(w); sas.append(sa); sbs.append(sb)
        except torch.OutOfMemoryError:
            print(f"{B:>6} | skipped (OOM during allocation)")
            for dev in devices:
                torch.cuda.empty_cache()
            continue

        # warmup
        try:
            for _ in range(cur_warmup):
                for i, dev in enumerate(devices):
                    with torch.cuda.stream(streams[i]):
                        _ = _proj(xs[i], ws[i], sas[i], sbs[i])
            for dev in devices:
                torch.cuda.synchronize(dev)
        except torch.OutOfMemoryError:
            print(f"{B:>6} | skipped (OOM during warmup)")
            del xs, ws, sas, sbs
            for dev in devices:
                torch.cuda.empty_cache()
            continue

        # timed iterations
        start = time.time()
        for _ in range(cur_iters):
            for i, dev in enumerate(devices):
                with torch.cuda.stream(streams[i]):
                    _ = _proj(xs[i], ws[i], sas[i], sbs[i])
        for dev in devices:
            torch.cuda.synchronize(dev)
        end = time.time()

        avg_ms = (end - start) * 1000.0 / cur_iters

        del xs, ws, sas, sbs
        torch.cuda.empty_cache()

        # FLOPS (same regardless of precision)
        flops = B * in_dim * out_dim * 2
        tflops = flops / (avg_ms / 1000) / 1e12
        compute_util = tflops / agg_peak_tflops * 100

        # Bytes: for FP8, input/weight are 1B, output is 2B (fp16)
        if is_fp8:
            bytes_moved = (B * in_dim + in_dim * out_dim) * 1 + B * out_dim * 2
        else:
            bytes_moved = (B * in_dim + in_dim * out_dim + B * out_dim) * dtype_bytes
        bytes_gb = bytes_moved / 1e9
        bw_tbs = bytes_moved / (avg_ms / 1000) / 1e12
        bw_util = bw_tbs / agg_peak_bw * 100

        # Roofline analysis
        arith_intensity = flops / bytes_moved if bytes_moved > 0 else 0
        roofline_bound = "compute" if arith_intensity >= oi_crossover else "memory"
        compute_ms = flops / (agg_peak_tflops * 1e12) * 1000
        memory_ms = bytes_moved / (agg_peak_bw * 1e12) * 1000
        ideal_ms = max(compute_ms, memory_ms)
        overhead = avg_ms / ideal_ms if ideal_ms > 0 else 0

        print(f"{B:>6} | {avg_ms:>9.3f} | {tflops:>8.2f} {compute_util:>6.1f}% | {bw_tbs:>8.3f} {bw_util:>6.1f}% | {arith_intensity:>7.1f} {roofline_bound:>7} | {ideal_ms:>8.4f} {overhead:>7.1f}x")

        csv_writer.writerow({
            "model": label, "dtype": dtype_name, "num_gpus": num_gpus,
            "batch": B, "in_dim": in_dim, "out_dim": out_dim,
            "avg_ms": f"{avg_ms:.4f}", "TFLOPS": f"{tflops:.3f}",
            "compute_util_pct": f"{compute_util:.2f}",
            "bytes_GB": f"{bytes_gb:.6f}", "BW_TBs": f"{bw_tbs:.4f}",
            "bw_util_pct": f"{bw_util:.2f}",
            "arith_intensity": f"{arith_intensity:.2f}",
            "roofline_bound": roofline_bound,
            "ideal_ms": f"{ideal_ms:.4f}",
            "overhead_x": f"{overhead:.2f}",
        })


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

    model_cfg = MODELS[args.model]

    print(f"Using {num_gpus} GPU(s): {devices}")
    print(f"model={args.model}, dtype={dtype_name}, peak_tflops={peak_tflops}/GPU, peak_bw={args.peak_bw} TB/s/GPU")

    if args.mode == "decode":
        # Decode-aligned: B maps to prefill seq length (1K to 1M)
        batches = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
            2048, 4096, 8192, 16384, 32768, 65536,
            131072, 262144, 524288, 1048576,
        ]
    else:
        batches = [
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
            2048, 3072, 4096, 5120, 6144, 7168, 8192,
        ]

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    mode_suffix = f"_{args.mode}" if args.mode != "prefill" else ""
    results_path = os.path.join(results_dir, f"{args.model}_gemm{mode_suffix}_{dtype_name}_{num_gpus}gpu.csv")

    with open(results_path, mode="w", newline="") as f:
        fieldnames = ["model", "dtype", "num_gpus", "batch", "in_dim", "out_dim",
                      "avg_ms", "TFLOPS", "compute_util_pct",
                      "bytes_GB", "BW_TBs", "bw_util_pct",
                      "arith_intensity", "roofline_bound", "ideal_ms", "overhead_x"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for in_dim, out_dim in model_cfg["gemms"]:
            label = f"{model_cfg['label']} ({in_dim}x{out_dim})"
            bench_one(in_dim, out_dim, batches, writer, dtype_name=dtype_name,
                      devices=devices, peak_tflops=peak_tflops, peak_bw_tbs=args.peak_bw,
                      warmup=args.warmup, iters=args.iters, label=label)

    print(f"\n✅ Results saved to: {results_path}")
