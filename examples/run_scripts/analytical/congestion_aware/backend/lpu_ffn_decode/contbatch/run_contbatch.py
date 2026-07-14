#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_contbatch.py — continuous-batch system 层的 CLI / 扫描入口。

把 exp1 的单步 TPOT（StepLatencyTable）+ 合成/trace workload + 迭代级调度器（sim）
串起来：固定 B_max=32，扫到达率 λ，比较 GPU+GPU / GPU+LPU / AMMA+LPU 三种硬件在
【同一套调度策略 + 同一份 request 流】下的平均 TPOT。

用法:
  # 默认：deepseek3, prompt 64K, B_max 32, 扫 λ
  python run_contbatch.py -o data/contbatch_sweep.json
  # 单点
  python run_contbatch.py --lambda-list 50 --b-max 32
  # 用 trace 文件
  python run_contbatch.py --trace mytrace.json
"""
from __future__ import annotations

import argparse
import json
import os

from workload import (WorkloadSpec, synth_workload, reseed_arrivals,
                      load_trace, dump_trace, workload_stats)
from step_latency import StepLatencyTable, SETTINGS, SETTING_LABEL
from sim import simulate


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Continuous-batch decode serving: avg TPOT for GPU+GPU / GPU+LPU / AMMA+LPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="deepseek3")
    ap.add_argument("--b-max", type=int, default=32, help="最大并发 batch")
    ap.add_argument("--b-max-gpu", type=int, default=None,
                    help="GPU 设定(gpu_gpu/gpu_lpu)的 B_max；默认同 --b-max")
    ap.add_argument("--b-max-amma", type=int, default=None,
                    help="AMMA 设定(amma_lpu/amma_lpu_ideal)的 B_max；默认同 --b-max")
    ap.add_argument("--sched", choices=["greedy", "throughput"], default="greedy",
                    help="调度：greedy 贪婪填满 B_max；throughput 吞吐感知（只批到撑住 λ 的最小 batch）")
    ap.add_argument("--link", default="nic_cx7", help="跨池链路（三 setting 统一）")
    ap.add_argument("--settings", default=",".join(SETTINGS))
    # workload（合成）
    ap.add_argument("--n-requests", type=int, default=2000)
    ap.add_argument("--prompt-mean", type=float, default=65536.0)
    ap.add_argument("--prompt-sigma", type=float, default=0.6)
    ap.add_argument("--output-mean", type=float, default=256.0)
    ap.add_argument("--output-sigma", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0, help="长度/到达的随机种子（复现）")
    # 扫描轴：到达率 λ（req/s）
    ap.add_argument("--lambda-list", default="0.5,1,1.5,2,3,5,8,12,20,40",
                    help="逗号分隔的 λ(req/s) 列表")
    # trace 输入（给了就忽略合成）
    ap.add_argument("--trace", default=None, help="trace JSON；给了则忽略合成 workload")
    ap.add_argument("--dump-trace", default=None, help="把合成 workload 落盘成 trace")
    ap.add_argument("--output", "-o", default=None)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    settings = [s.strip() for s in args.settings.split(",") if s.strip()]
    lambdas = [float(x) for x in args.lambda_list.split(",")]

    # 每设定的 B_max：GPU 与 AMMA 可不同（默认都 = --b-max）
    GPU_SETTINGS = {"gpu_gpu", "gpu_lpu"}
    bmax_gpu = args.b_max_gpu or args.b_max
    bmax_amma = args.b_max_amma or args.b_max
    bmax_by_setting = {s: (bmax_gpu if s in GPU_SETTINGS else bmax_amma) for s in settings}
    table_bmax = max(bmax_by_setting.values())

    print(f"== building step-latency table (model={args.model}, B_max≤{table_bmax} "
          f"[gpu={bmax_gpu}, amma={bmax_amma}], link={args.link}) ==")
    table = StepLatencyTable(model_key=args.model, link=args.link,
                             b_max=table_bmax, settings=settings)

    # workload：trace 优先，否则合成「长度模板」（固定 seed）
    if args.trace:
        base_reqs = load_trace(args.trace)
        print(f"== trace: {args.trace}  ({len(base_reqs)} reqs) ==")
        lambdas = [None]      # trace 自带到达时刻，不扫 λ
    else:
        spec = WorkloadSpec(n_requests=args.n_requests,
                            arrival_rate_rps=lambdas[0],
                            prompt_mean=args.prompt_mean, prompt_sigma_log=args.prompt_sigma,
                            output_mean=args.output_mean, output_sigma_log=args.output_sigma,
                            seed=args.seed)
        base_reqs = synth_workload(spec)
        st = workload_stats(base_reqs)
        print(f"== synth workload: n={st['n']} prompt_mean={st['prompt_mean']:.0f} "
              f"(p99 {st['prompt_p99']:.0f}) output_mean={st['output_mean']:.1f} "
              f"total_out_tok={st['total_output_tokens']} ==")
        if args.dump_trace:
            dump_trace(base_reqs, args.dump_trace, meta=spec.to_dict())
            print(f"   -> dumped trace: {args.dump_trace}")

    rows = []
    print("\n  avg TPOT (us)  [batch = 时间加权平均 batch]")
    hdr = f"  {'lambda':>7} " + " ".join(f"{SETTING_LABEL[s][:13]:>20}" for s in settings)
    print(hdr)
    for lam in lambdas:
        if lam is None:
            reqs = base_reqs
        else:
            reqs = reseed_arrivals(base_reqs, lam, seed=args.seed + 1)
        per_setting = {}
        for s in settings:
            per_setting[s] = simulate(s, reqs, table, b_max=bmax_by_setting[s],
                                      sched=args.sched, lam_rps=lam,
                                      out_mean=args.output_mean,
                                      seq_ref=args.prompt_mean)
        rows.append({"lambda_rps": lam, "metrics": per_setting})
        cells = " ".join(
            f"{per_setting[s]['avg_tpot_us']:>13.1f}[{per_setting[s]['mean_batch']:>4.1f}]"
            for s in settings)
        lam_s = f"{lam:>7.1f}" if lam is not None else "  trace"
        print(f"  {lam_s} {cells}")

    if args.output:
        out = {
            "meta": {
                "table": table.meta(),
                "workload": (spec.to_dict() if not args.trace else {"trace": args.trace}),
                "lambda_list": lambdas, "b_max": args.b_max,
                "b_max_by_setting": bmax_by_setting, "sched": args.sched,
                "settings": settings,
            },
            "sweep": rows,
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n=> wrote {args.output}")


if __name__ == "__main__":
    main()
