#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run.py — LPU FFN decode 仿真的命令行入口。

职责：
  1) 组装 LPUSpec / FFNModelSpec / RunSpec（默认值 + CLI 覆盖 + 可选 JSON 覆盖）；
  2) 调 ffn_decode.simulate_decode 跑仿真；
  3) 把【参数 meta + 结果】打成一份结构化 JSON 落盘，并在终端打印摘要。

与现有 backend 解耦：本目录自成一体，只 import 同目录的 lpu_config / ffn_decode。

用法示例：
  python run.py --model deepseek3 --tp 16 --batch 1 -o data/deepseek3_tp16_b1.json
  python run.py --model gpt-oss-120b --tp 8 --batch 4 --expert-mode batched
  python run.py --model qwen3-235b --w-bytes 1 --sram-bw 18.4
  # 对 batch 做 sweep：
  for b in 1 2 4 8 16 32; do python run.py --model deepseek3 --batch $b \
      -o data/deepseek3_b$b.json; done
"""

from __future__ import annotations
import argparse
import json
import os

from lpu_config import RunSpec, get_model, get_lpu
from ffn_decode import simulate_decode


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="LPU (SHIP) FFN decode latency simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- 模型 ---
    ap.add_argument("--model", default="deepseek3",
                    help="预置模型名（deepseek3 / gpt-oss-120b / qwen3-235b / llama3-70b）")
    ap.add_argument("--layers", type=int, default=None, help="覆盖模型层数")

    # --- 并行 / batch ---
    ap.add_argument("--tp", type=int, default=16, help="tensor-parallel 切分数")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel 级数（仅报告聚合）")
    ap.add_argument("--batch", type=int, default=1, help="decode batch size")
    ap.add_argument("--diameter", type=int, default=3, help="TP partition 网络直径（跳数）")
    ap.add_argument("--expert-mode", choices=["per_token", "batched"],
                    default="per_token", help="MoE expert 执行方式（per_token → OI=1）")

    # --- LPU 代次（基线规格），之后再叠加下面的逐字段覆盖 ---
    ap.add_argument("--lpu", choices=["lpuv1", "lpx"], default="lpuv1",
                    help="LPU 代次：lpuv1=论文 SHIP 代次，lpx=Groq 3 LPX(下一代)")

    # --- LPU 硬件覆盖（不填则用所选代次的默认值） ---
    ap.add_argument("--sram-bw", type=float, default=None, help="单颗 SRAM 带宽 TB/s")
    ap.add_argument("--compute", type=float, default=None, help="单颗算力 TFLOP/s")
    ap.add_argument("--c2c-bw", type=float, default=None, help="单颗 C2C 带宽 GB/s")
    ap.add_argument("--hop-ns", type=float, default=None, help="每跳延迟 ns")
    ap.add_argument("--w-bytes", type=int, default=None, help="权重字节数（FP8=1, FP16=2）")

    # --- 输出 ---
    ap.add_argument("--output", "-o", default=None, help="结果 JSON 路径（不填只打印）")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    # 1) 模型规格（可覆盖层数）
    model = get_model(args.model)
    if args.layers is not None:
        model.num_layers = args.layers

    # 2) LPU 规格：先取所选代次，再按 CLI 逐字段覆盖（None 表示不改）
    lpu = get_lpu(args.lpu)
    for cli_val, field in [
        (args.sram_bw, "sram_bw_TBs"),
        (args.compute, "compute_TFLOPs"),
        (args.c2c_bw, "c2c_bw_GBs"),
        (args.hop_ns, "hop_latency_ns"),
        (args.w_bytes, "w_bytes"),
    ]:
        if cli_val is not None:
            setattr(lpu, field, cli_val)

    # 3) 运行配置
    run = RunSpec(TP=args.tp, PP=args.pp, batch=args.batch,
                  tp_diameter_hops=args.diameter, expert_mode=args.expert_mode)

    # 4) 跑仿真
    result = simulate_decode(model, lpu, run)

    # 5) 拼装结构化输出：meta（全部入参）+ result（逐层 + 整模）
    out = {
        "meta": {
            "lpu_gen": args.lpu,
            "lpu": lpu.to_dict(),
            "model": model.to_dict(),
            "run": run.to_dict(),
        },
        **result,
    }

    # 6) 终端摘要
    dec = result["decode"]
    bd = dec["bound_breakdown_per_layer_ns"]
    print(f"== LPU FFN decode | lpu={args.lpu} model={model.name} TP={run.TP} "
          f"batch={run.batch} expert_mode={run.expert_mode} ==")
    print(f"  单层 FFN 延迟      : {dec['per_layer_ffn_ns']:>12,.1f} ns")
    print(f"  整模 FFN TPOT      : {dec['ffn_tpot_ns']:>12,.1f} ns "
          f"({dec['ffn_tpot_us']:.3f} us, {model.num_layers} 层)")
    print(f"  单层 bound 拆解(ns): compute={bd['compute']:,.1f} "
          f"mem={bd['mem']:,.1f} comm={bd['comm']:,.1f}")
    print("  逐 stage:")
    for s in result["per_layer"]["stages"]:
        if s["name"] == "AllReduce":
            print(f"    {s['name']:<14s} {s['t_ns']:>10,.1f} ns "
                  f"[comm, eff_bw={s['eff_bw_GBs']:.1f} GB/s]")
        else:
            print(f"    {s['name']:<14s} {s['t_ns']:>10,.1f} ns [{s['bound']}]")

    # 7) 落盘
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n=> 已写入 {args.output}")


if __name__ == "__main__":
    main()
