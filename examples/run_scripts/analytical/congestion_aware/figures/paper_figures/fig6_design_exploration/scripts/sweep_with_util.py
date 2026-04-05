#!/usr/bin/env python3
"""
Fig 6 数据生成: 带 utilization 调整的设计空间扫描

与 sweep_compute_bw.py 的区别:
  - 使用 util.py + main.py 的 SA utilization 模型, 为每个算力生成精确利用率
  - 通过 merge_gqa_results.py 的 utilization 调整, 得到更真实的延时估算

数据流:
  1. util.py qwen3 NUM_SA → qwen3_util_{NUM_SA}.json (SA 利用率, 依赖 num_sa)
  2. roofline_gqa_calc.py --peak-perf PP --link-bw LBW --batch BS → roofline JSON
  3. merge_gqa_results.py --roofline-data ... --utilization qwen3_util_{NUM_SA}.json
     → hybrid JSON (利用率调整后的延时)
  4. 从 hybrid JSON 读取 RO_new wall time

参数:
  - num_sa (=TFLOPS): 8, 16, 32, 64, 96, 128, 256
  - D2D BW: 0.5, 1.0, 1.5, 2.0, 2.5 TB/s
  - Batch: 1, 4, 16, 32
  - HBM BW: 固定 2.5 TB/s, seq: 固定 65536
"""
import json
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[4]  # congestion_aware/
OUT = Path(__file__).resolve().parents[1] / "data"

# ── 扫描参数 ──
SA_VALS     = [8, 16, 32, 64, 96, 128, 256]
LINKBW_VALS = [0.5, 1.0, 1.5, 2.0, 2.5]
BATCH_VALS  = [1, 4, 16, 32]
HBM_BW      = 2.5
SEQ         = 65536
MODEL       = "qwen3"
HOP_LAT     = 15

# AstraSim 数据 (通信来源, bs=1 仿真)
AS_DATA = str(BASE / "reports/qwen3-235B/astrasim/gqa_seq_scaling_bs1_80t_split4_bw1500_data.json")
RUBIN_UTIL = str(BASE / "h100_rubin_utilization.json")


def ensure_util(num_sa):
    """确保 utilization JSON 存在, 不存在则调用 util.py 生成"""
    util_path = BASE / f"qwen3_util_{num_sa}.json"
    if not util_path.exists():
        subprocess.run(
            [sys.executable, str(BASE / "util.py"), MODEL, str(num_sa)],
            capture_output=True, cwd=str(BASE)
        )
    return str(util_path)


def run_pipeline(num_sa, link_bw, batch):
    """运行 roofline → merge pipeline, 返回 RO_new 各模块延时"""
    pp = num_sa  # num_sa 直接映射到 TFLOPS

    # Step 1: Roofline
    rf_path = f"/tmp/sweep_rf_pp{pp}_lbw{link_bw}_bs{batch}.json"
    subprocess.run([
        sys.executable, str(BASE / "roofline_gqa_calc.py"),
        "--peak-perf", str(pp), "--bandwidth", str(HBM_BW),
        "--link-bw", str(link_bw), "--hop-latency", str(HOP_LAT),
        "--batch", str(batch), "--model", MODEL, "--seq", str(SEQ),
        "-o", rf_path,
    ], capture_output=True)

    # Step 2: Merge with utilization
    util_path = ensure_util(num_sa)
    hybrid_path = f"/tmp/sweep_hybrid_pp{pp}_lbw{link_bw}_bs{batch}.json"
    subprocess.run([
        sys.executable, str(BASE / "merge_gqa_results.py"),
        "--astrasim-data", AS_DATA,
        "--roofline-data", rf_path,
        "--utilization", util_path,
        "--rubin-utilization", RUBIN_UTIL,
        "-o", hybrid_path,
    ], capture_output=True)

    # Step 3: 读取 RO_new 结果
    try:
        with open(hybrid_path) as f:
            hybrid = json.load(f)
    except Exception:
        return None

    strat = "hmp_reo_new"
    entry = next(
        (e for e in hybrid.get("strategies", {}).get(strat, {}).get("data", [])
         if e["seq"] == SEQ), None
    )
    if not entry:
        return None

    cb = entry.get("compute_breakdown", {})
    return {
        "num_sa": num_sa,
        "peak_perf": pp,
        "link_bw": link_bw,
        "batch": batch,
        "seq": SEQ,
        # 各模块延时
        "qkv_ns": entry["hybrid_Proj_QKV_ns"],
        "attn_ns": entry["hybrid_attn_ns"],
        "proj_o_ns": entry["hybrid_Proj_O_ns"],
        "gpu_ns": entry["hybrid_gpu_ns"],
        "comm_ns": entry["comm_total_ns"],
        "wall_ns": entry["hybrid_wall_ns"],
        # 利用率信息
        "qkv_util": cb.get("Proj_QKV_utilization", 0),
        "score_util": cb.get("score_utilization", 0),
        "attn_v_util": cb.get("attn_v_utilization", 0),
        "proj_o_util": cb.get("Proj_O_utilization", 0),
        "qkv_source": cb.get("Proj_QKV_source", ""),
        "attn_source": cb.get("attn_source", ""),
        "proj_o_source": cb.get("Proj_O_source", ""),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    total = len(SA_VALS) * len(LINKBW_VALS) * len(BATCH_VALS)
    count = 0

    for bs in BATCH_VALS:
        for sa in SA_VALS:
            for lbw in LINKBW_VALS:
                count += 1
                r = run_pipeline(sa, lbw, bs)
                if r:
                    results.append(r)
                if count % 20 == 0 or count == total:
                    print(f"  [{count}/{total}] sa={sa} lbw={lbw} bs={bs}: "
                          f"wall={r['wall_ns']:.0f}ns" if r else f"  [{count}/{total}] FAILED")

    out_path = OUT / "sweep_with_util.json"
    with open(out_path, "w") as f:
        json.dump({
            "_description": "Design space sweep WITH utilization adjustment",
            "_model": f"Qwen3-235B, seq={SEQ}, HBM_BW={HBM_BW}TB/s, hop={HOP_LAT}ns",
            "_strategy": "hmp_reo_new (RO_new)",
            "_utilization": "util.py SA model (per num_sa), merge_gqa_results.py adjusted",
            "_data_sources": {
                "utilization": "util.py + main.py → qwen3_util_{num_sa}.json",
                "roofline": "roofline_gqa_calc.py → compute_ns, memory_ns, comm_ns",
                "hybrid": "merge_gqa_results.py → max(compute_ns/util, memory_ns) + comm",
                "astrasim_comm": AS_DATA,
            },
            "_parameters": {
                "num_sa_tflops": SA_VALS,
                "link_bw_tbs": LINKBW_VALS,
                "batch_sizes": BATCH_VALS,
            },
            "data": results,
        }, f, indent=2)

    print(f"\nSaved {len(results)} entries to: {out_path}")


if __name__ == "__main__":
    main()
