#!/usr/bin/env python3
"""Command-line query for compute_util / memory(bw)_util / power of one kernel.

Examples
--------
  # GQA (qwen3 / llama4) on AMMA
  python3 query_util.py --model qwen3  --arch amma  --op qkv  --bs 8
  python3 query_util.py --model llama4 --arch amma  --op attn --seq 16384
  python3 query_util.py --model qwen3  --arch amma  --op o    --bs 8 --strategy tp16

  # MLA (deepseek3) on AMMA
  python3 query_util.py --model deepseek3 --arch amma --op qkv  --bs 8
  python3 query_util.py --model deepseek3 --arch amma --op attn --seq 16384

  # Any model on Rubin / B200 (compute_util == 0, bw_util from measured profile)
  python3 query_util.py --model qwen3 --arch rubin --op qkv --bs 8

Tweak hardware peaks with --tflops / --bw / --num-sa; TP with --tp.
"""
import argparse

import utilization_lib as ul

CONFIGS = {
    "qwen3": ul.QWEN3_DEFAULTS,
    "llama4": ul.LLAMA4_DEFAULTS,
    "deepseek3": ul.DEEPSEEK_V3_DEFAULTS,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS))
    ap.add_argument("--arch", default="amma", choices=["amma", "rubin", "b200"])
    ap.add_argument("--op", required=True, choices=["qkv", "o", "attn"])
    ap.add_argument("--bs", type=int, default=1, help="batch size (qkv / o)")
    ap.add_argument("--seq", type=int, default=4096, help="sequence length (attn)")
    ap.add_argument("--tp", type=int, default=16)
    ap.add_argument("--strategy", default="HMP_reo",
                    help="GQA only: hmp | HMP_reo | hmp_reo_new | tp16 | rubin")
    ap.add_argument("--tflops", type=float, default=50.0, help="peak TFLOPS / NPU")
    ap.add_argument("--bw", type=float, default=1.5, help="HBM TB/s / NPU")
    ap.add_argument("--num-sa", type=int, default=64, help="systolic arrays (amma)")
    args = ap.parse_args()

    cfg = CONFIGS[args.model]
    hw = {
        "arch": args.arch,
        "peak_per_npu_tflops": args.tflops,
        "bw_per_npu_tbs": args.bw,
        "num_sa_for_util": args.num_sa,
    }
    is_mla = args.model == "deepseek3"

    if args.op == "qkv":
        if is_mla:
            cu = ul.compute_utilization_qkv(cfg, args.bs, hw, args.tp)
            mu = ul.memory_utilization_qkv(cfg, args.bs, hw, args.tp)
            pw = ul.power_qkv(mu, cu, hw)
        else:
            cu = ul.compute_utilization_qkv_gqa(cfg, args.bs, hw, args.tp, args.strategy)
            mu = ul.memory_utilization_qkv_gqa(cfg, args.bs, hw, args.tp, args.strategy)
            pw = ul.power_qkv_gqa(mu, cu, hw)
        x = f"bs={args.bs}"
    elif args.op == "o":
        if is_mla:
            cu = ul.compute_utilization_o(cfg, args.bs, hw, args.tp)
            mu = ul.memory_utilization_o(cfg, args.bs, hw, args.tp)
            pw = ul.power_o(mu, cu, hw)
        else:
            cu = ul.compute_utilization_o_gqa(cfg, args.bs, hw, args.tp, args.strategy)
            mu = ul.memory_utilization_o_gqa(cfg, args.bs, hw, args.tp, args.strategy)
            pw = ul.power_o_gqa(mu, cu, hw)
        x = f"bs={args.bs}"
    else:  # attn
        if is_mla:
            cu = ul.compute_utilization_attention(cfg, args.seq, hw, args.tp)
            mu = ul.memory_utilization_attention(cfg, args.seq, hw, args.tp)
            pw = ul.power_attention(mu, cu, hw)
        else:
            cu = ul.compute_utilization_attention_gqa(cfg, args.seq, hw, args.tp, args.strategy)
            mu = ul.memory_utilization_attention_gqa(cfg, args.seq, hw, args.tp, args.strategy)
            pw = ul.power_attention_gqa(mu, cu, hw)
        x = f"seq={args.seq}"

    tag = "MLA" if is_mla else f"GQA/{args.strategy}"
    print(f"[{args.model} | {args.arch} | {tag}] {args.op} {x} TP={args.tp}")
    print(f"  compute_util = {cu:.4f}")
    print(f"  bw_util      = {mu:.4f}")
    print(f"  power        = {pw:.1f} W")


if __name__ == "__main__":
    main()
