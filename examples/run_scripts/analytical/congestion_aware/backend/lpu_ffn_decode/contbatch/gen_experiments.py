#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_experiments.py — exp5 的三个扩展实验，统一输出到一个 JSON。

  A) accum  : request 积累的 breakdown —— 随 λ 变大，AMMA+LPU vs GPU+LPU 的
              在系统内 request 数 (waiting+active) 随时间如何堆积（过载者线性积压）。
  B) bmax   : B_max 扫描 —— 固定饱和负载，B_max∈{1..64} 时三 setting 的 avg TPOT /
              吞吐 / 平均 batch 如何变化（latency-throughput 权衡旋钮）。
  C) length : request 平均长度扫描 —— prompt 均值 {4K,16K,64K,1024K}，在 B_max=32 下
              低负载(batch≈1) 与 饱和(batch→32) 两种 regime 的 serving TPOT 随长度变化。

复用 step_latency 的 (B,seq) 表（一次建到 b_max=64、seq grid 扩到 ~3M），sim 的 timeline。
用法: python gen_experiments.py -o data/contbatch_experiments.json
"""
from __future__ import annotations

import argparse
import json
import os

from workload import WorkloadSpec, synth_workload, reseed_arrivals
from step_latency import StepLatencyTable, DEFAULT_SEQ_GRID, SETTING_LABEL
from sim import simulate

SETTINGS = ["gpu_gpu", "gpu_lpu", "amma_lpu", "amma_lpu_ideal"]
SEED = 0
N_REQ = 1500
EXT_SEQ_GRID = sorted(set(DEFAULT_SEQ_GRID) | {2097152, 3145728})   # 扩到 ~3M 支持 1024K
B_MAX_TABLE = 64


def exp_accum(table, base_reqs):
    """随时间的积累：amma_lpu vs gpu_lpu，几个 λ（含过载点）。"""
    lambdas = [2.0, 5.0, 12.0]
    settings = ["gpu_lpu", "amma_lpu"]
    out = {"lambdas": lambdas, "settings": settings, "series": []}
    for lam in lambdas:
        reqs = reseed_arrivals(base_reqs, lam, seed=SEED + 1)
        entry = {"lambda_rps": lam, "by_setting": {}}
        for s in settings:
            m = simulate(s, reqs, table, b_max=32, record_timeline=True)
            entry["by_setting"][s] = {
                "timeline": m["timeline"],
                "mean_in_system": m["mean_in_system"],
                "mean_backlog": m["mean_backlog"],
                "backlog_peak": m["backlog_peak"],
                "avg_tpot_us": m["avg_tpot_us"],
                "queue_mean_us": m["queue_mean_us"],
                "throughput_tok_s": m["throughput_tok_s"],
            }
        out["series"].append(entry)
    return out


def exp_bmax(table, base_reqs):
    """B_max 扫描，固定饱和负载（λ 很大 → 始终有积压 → batch 顶到 B_max）。"""
    bmaxes = [1, 2, 4, 8, 16, 32, 48, 64]
    lam = 1e6                      # 近似无限到达 → 饱和
    reqs = reseed_arrivals(base_reqs, lam, seed=SEED + 1)
    out = {"b_max_list": bmaxes, "settings": SETTINGS, "lambda_rps": lam, "rows": []}
    for bm in bmaxes:
        row = {"b_max": bm, "by_setting": {}}
        for s in SETTINGS:
            m = simulate(s, reqs, table, b_max=bm)
            row["by_setting"][s] = {
                "avg_tpot_us": m["avg_tpot_us"],
                "throughput_tok_s": m["throughput_tok_s"],
                "mean_batch": m["mean_batch"],
            }
        out["rows"].append(row)
    return out


def exp_length(table):
    """prompt 平均长度扫描，B_max=32，低负载 vs 饱和两种 regime。"""
    means = [4096, 16384, 65536, 1048576]
    regimes = {"light": 0.5, "saturated": 1e6}     # λ
    out = {"prompt_means": means, "settings": SETTINGS, "regimes": list(regimes),
           "rows": []}
    for pm in means:
        spec = WorkloadSpec(n_requests=N_REQ, arrival_rate_rps=1.0,
                            prompt_mean=float(pm), prompt_sigma_log=0.3,
                            prompt_lo=64, prompt_hi=3145728,
                            output_mean=256.0, output_sigma_log=0.7, seed=SEED)
        base = synth_workload(spec)
        row = {"prompt_mean": pm, "by_regime": {}}
        for rname, lam in regimes.items():
            reqs = reseed_arrivals(base, lam, seed=SEED + 1)
            row["by_regime"][rname] = {
                s: {
                    "avg_tpot_us": (mm := simulate(s, reqs, table, b_max=32))["avg_tpot_us"],
                    "mean_batch": mm["mean_batch"],
                } for s in SETTINGS
            }
        out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek3")
    ap.add_argument("--output", "-o", default="data/contbatch_experiments.json")
    ap.add_argument("--link", default="nic_cx7")
    args = ap.parse_args()
    MODEL = args.model

    print(f"== building (B,seq) table: model={MODEL}, b_max={B_MAX_TABLE}, "
          f"seq grid to {EXT_SEQ_GRID[-1]} ==")
    table = StepLatencyTable(model_key=MODEL, link=args.link, b_max=B_MAX_TABLE,
                             settings=SETTINGS, seq_grid=EXT_SEQ_GRID)
    base = synth_workload(WorkloadSpec(n_requests=N_REQ, prompt_mean=65536.0,
                                       prompt_sigma_log=0.6, seed=SEED))

    print("== A) accumulation timeline ==")
    accum = exp_accum(table, base)
    for e in accum["series"]:
        gl = e["by_setting"]["gpu_lpu"]; am = e["by_setting"]["amma_lpu"]
        print(f"  λ={e['lambda_rps']:>5.1f}  mean in-system: "
              f"GPU+LPU {gl['mean_in_system']:>7.1f} (peak backlog {gl['backlog_peak']})  | "
              f"AMMA+LPU {am['mean_in_system']:>7.1f} (peak {am['backlog_peak']})")

    print("== B) B_max sweep (saturated) ==")
    bmax = exp_bmax(table, base)
    print("  B_max " + "".join(f"{SETTING_LABEL[s][:12]:>14}" for s in SETTINGS))
    for row in bmax["rows"]:
        cells = "".join(f"{row['by_setting'][s]['avg_tpot_us']/1e3:>10.1f}ms" for s in SETTINGS)
        print(f"  {row['b_max']:>4}  {cells}")

    print("== C) length sweep (B_max=32) ==")
    length = exp_length(table)
    for row in length["rows"]:
        sat = row["by_regime"]["saturated"]; lit = row["by_regime"]["light"]
        print(f"  prompt~{row['prompt_mean']//1024}K  "
              f"[light]  AMMA {lit['amma_lpu']['avg_tpot_us']/1e3:6.1f}ms "
              f"GPU+LPU {lit['gpu_lpu']['avg_tpot_us']/1e3:6.1f}ms  | "
              f"[sat]  AMMA {sat['amma_lpu']['avg_tpot_us']/1e3:6.1f}ms "
              f"GPU+LPU {sat['gpu_lpu']['avg_tpot_us']/1e3:6.1f}ms")

    out = {
        "meta": {"model": MODEL, "link": args.link, "table_b_max": B_MAX_TABLE,
                 "seq_grid": EXT_SEQ_GRID, "settings": SETTINGS, "n_requests": N_REQ},
        "accum": accum, "bmax": bmax, "length": length,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"\n=> wrote {args.output}")


if __name__ == "__main__":
    main()
