#!/usr/bin/env python3
"""
Fig 6 数据生成: Compute Power × D2D BW × Batch Size 设计空间扫描

数据来源与计算流程:
  1. roofline_gqa_calc.py --peak-perf PP --bandwidth 2.5 --link-bw LBW --batch BS
     → 生成 roofline JSON, 包含:
       - compute_ns = 2 * MACs / (PP * 1e12) * 1e9  (反比于 peak_perf)
       - memory_ns  = total_mem / (2.5 * 1e12) * 1e9  (固定 HBM BW = 2.5 TB/s)
       - comm_ns    = steps * (hop_latency + chunk / (LBW * 1e12) * 1e9)  (反比于 link_bw)

  2. 从 roofline JSON 中直接读取 RO_new 策略的 wall_time:
       wall = QKV_total + Attn_total + ProjO_total + comm_total
     其中 QKV_total = max(compute_ns, memory_ns), 同理 Attn 和 ProjO

  注意: 本扫描使用 **原始 roofline 模型** (不含 utilization 调整),
        因为 utilization 是特定硬件的 profiling 结果, 不适用于设计空间探索。
        utilization 的影响在 Fig 1/3/5 中已单独分析。

参数范围:
  - Compute Power: 8, 16, 32, 64, 96, 128, 256 TFLOPS/NPU
  - D2D Link BW:   0.5, 1.0, 1.5, 2.0, 2.5 TB/s
  - Batch Size:     1, 4, 16, 32
  - HBM BW:         固定 2.5 TB/s/NPU
  - Seq Length:      固定 65536 (64K)
  - Model:           Qwen3-235B
  - Strategy:        hmp_reo_new (RO_new)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[4]  # congestion_aware/
OUT = Path(__file__).resolve().parents[1] / "data"

# ── 扫描参数 ──
COMPUTE_VALS = [8, 16, 32, 64, 96, 128, 256]      # TFLOPS/NPU
LINKBW_VALS  = [0.5, 1.0, 1.5, 2.0, 2.5]          # TB/s
BATCH_VALS   = [1, 4, 16, 32]                       # batch size
HBM_BW       = 2.5                                   # TB/s (固定)
SEQ          = 65536                                  # 序列长度 (固定)
MODEL        = "qwen3"
HOP_LATENCY  = 15                                    # ns
STRATEGY     = "hmp_reo_new"


def run_roofline(pp, lbw, bs):
    """调用 roofline_gqa_calc.py 生成 roofline JSON, 返回 RO_new wall time"""
    tmp_path = f"/tmp/sweep_pp{pp}_lbw{lbw}_bs{bs}.json"
    cmd = [
        sys.executable, str(BASE / "roofline_gqa_calc.py"),
        "--peak-perf", str(pp),
        "--bandwidth", str(HBM_BW),
        "--link-bw", str(lbw),
        "--hop-latency", str(HOP_LATENCY),
        "--batch", str(bs),
        "--model", MODEL,
        "--seq", str(SEQ),
        "-o", tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: pp={pp} lbw={lbw} bs={bs}: {result.stderr[:200]}", file=sys.stderr)
        return None

    with open(tmp_path) as f:
        data = json.load(f)

    # 提取 RO_new 策略数据
    strat_data = data.get("strategies", {}).get(STRATEGY, {}).get("data", [])
    entry = next((e for e in strat_data if e["seq"] == SEQ), None)
    if not entry:
        return None

    qkv = entry.get("Proj_QKV", {})
    attn = entry.get("attention", {})
    proj_o = entry.get("Proj_O", {})

    # 提取 wall time 各组件
    qkv_ns = qkv.get("total_ns", max(qkv.get("compute_ns", 0), qkv.get("memory_ns", 0)))
    attn_ns = attn.get("total_ns_fused", attn.get("total_ns", 0))
    proj_o_ns = proj_o.get("total_ns", max(proj_o.get("compute_ns", 0), proj_o.get("memory_ns", 0)))

    # 通信
    comm = data["strategies"][STRATEGY].get("comm", {})
    comm_ns = comm.get("total_comm_ns", 0)

    gpu_ns = qkv_ns + attn_ns + proj_o_ns
    wall_ns = gpu_ns + comm_ns

    return {
        "peak_perf": pp,
        "link_bw": lbw,
        "batch": bs,
        "seq": SEQ,
        "qkv_ns": round(qkv_ns, 2),
        "qkv_compute_ns": round(qkv.get("compute_ns", 0), 2),
        "qkv_memory_ns": round(qkv.get("memory_ns", 0), 2),
        "attn_ns": round(attn_ns, 2),
        "proj_o_ns": round(proj_o_ns, 2),
        "proj_o_compute_ns": round(proj_o.get("compute_ns", 0), 2),
        "proj_o_memory_ns": round(proj_o.get("memory_ns", 0), 2),
        "gpu_ns": round(gpu_ns, 2),
        "comm_ns": round(comm_ns, 2),
        "wall_ns": round(wall_ns, 2),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(COMPUTE_VALS) * len(LINKBW_VALS) * len(BATCH_VALS)
    count = 0

    for bs in BATCH_VALS:
        for pp in COMPUTE_VALS:
            for lbw in LINKBW_VALS:
                count += 1
                r = run_roofline(pp, lbw, bs)
                if r:
                    results.append(r)
                    if count % 20 == 0:
                        print(f"  [{count}/{total}] pp={pp}T lbw={lbw} bs={bs}: wall={r['wall_ns']:.0f}ns")

    # 保存完整结果
    out_path = OUT / "sweep_compute_bw_batch.json"
    with open(out_path, "w") as f:
        json.dump({
            "_description": "Design space sweep: Compute Power × D2D BW × Batch Size",
            "_model": f"{MODEL}, seq={SEQ}, HBM_BW={HBM_BW}TB/s, hop={HOP_LATENCY}ns",
            "_strategy": STRATEGY,
            "_note": "Raw roofline model (no utilization adjustment)",
            "_parameters": {
                "compute_tflops": COMPUTE_VALS,
                "link_bw_tbs": LINKBW_VALS,
                "batch_sizes": BATCH_VALS,
            },
            "data": results,
        }, f, indent=2)

    print(f"\nSaved {len(results)} entries to: {out_path}")


if __name__ == "__main__":
    main()
