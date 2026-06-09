#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_compose.py — 解耦 attention(AMMA) + FFN(LPU) 的完整 decode TPOT 组合器。

============================================================================
TPOT 计算（严格串行求和）
----------------------------------------------------------------------------
每个 decoder layer 内 4 项严格串行：
    t_layer(CL) = t_attn_AMMA(CL) + t_xfer(AMMA→LPU) + t_ffn_LPU + t_xfer(LPU→AMMA)
整模一个输出 token 要穿过所有层各一次：
    TPOT = num_layers × t_layer(CL)          (+ embed/lm_head，忽略)
注：TPOT 是「单用户延迟」，PP/流水线只提吞吐、不改 TPOT，故这里是求和不是取 max。
============================================================================

数据来源（保持解耦，roofline/ 零改动）：
  - t_attn ：只读复用 roofline/attention.py 的 single_layer_*_time + hmp 通信，
             硬件用 AMMA 的 per-NPU 配置 hardware_config.hbm4_npu_config，
             策略 "hmp"（即用户所说的 "ours"）。KV 格式由该模型内部决定（对齐 attention 模型）。
  - t_ffn  ：本模块 ffn_decode.simulate_layer，硬件用 LPU（默认 LPX）。
  - t_xfer ：本模块 interconnect.transfer_time。

用法:
  python decode_compose.py --model qwen3-235b --cl 4096 --batch 1
  python decode_compose.py --model qwen3-235b --cl-sweep 2048,8192,32768,131072 \
         -o data/qwen3_tpot_sweep.json
  python decode_compose.py --model deepseek3 --t-attn-ns 1800   # 直接给 attention 单层延时
"""

from __future__ import annotations
import argparse
import json
import os
import sys

from lpu_config import RunSpec, get_model, get_lpu
from ffn_decode import simulate_layer
from interconnect import get_link, transfer_time

# ---------------------------------------------------------------------------
# 只读复用 roofline/attention.py（不修改其源码）：把 roofline/ 加进 import 路径
# ---------------------------------------------------------------------------
_ROOFLINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roofline")
sys.path.insert(0, os.path.abspath(_ROOFLINE_DIR))
import attention as _attn               # noqa: E402  (roofline/attention.py, GQA)
import hardware_config as _hwc          # noqa: E402  (roofline/hardware_config.py)
import roofline_gqa_calc as _mla        # noqa: E402  (roofline/roofline_gqa_calc.py, MLA)
import roofline_v4_calc as _v4          # noqa: E402  (roofline/roofline_v4_calc.py, CSA/HCA)


# ---------------------------------------------------------------------------
# attention 来源：每个模型都走 backend 真实模型，不再手编维度。
#   - "gqa": qwen3-235B → roofline/attention.py 的 GQA 模型 (MODEL_CONFIGS["qwen3"])
#   - "mla": deepseek3  → roofline/roofline_gqa_calc.py 的 MLA 模型 (MLA_CONFIG)
# 没有 backend attention 模型的模型（如 gpt-oss）一律拒绝，避免编造。
# ---------------------------------------------------------------------------
ATTN_BACKEND = {
    "qwen3-235b": "gqa",
    "deepseek3":  "mla",
    "deepseek-v4-csa": "csa",   # DeepSeek-V4 Compressed Sparse Attention
    "deepseek-v4-hca": "hca",   # DeepSeek-V4 Heavily Compressed Attention
}


def attention_layer_ns(model_key: str, batch: int, seq: int,
                       tp_h: int, tp_hd: int) -> dict:
    """
    AMMA(ours) 单层 attention decode 延时，全部来自 backend roofline 模型。
    返回 {t_ns, parts:{qkv,attn,output,comm}, source}。CL 通过 seq 传入。
    """
    kind = ATTN_BACKEND.get(model_key)
    if kind is None:
        raise KeyError(f"模型 '{model_key}' 没有 backend attention 模型，"
                       f"仅支持 {list(ATTN_BACKEND)}（不编造维度）")
    # AMMA per-NPU 硬件。compute 用 fig1 验证过的 "Ours" 规格 96 TFLOPS（hbm4_npu.json
    # 里的 40 是更老的 onering16 配置 → 会让 GQA attention 错误地 compute-bound）。
    # 与 fig1_methodology §6 硬件表一致。
    hw = dict(_hwc.hbm4_npu_config)
    hw["compute"] = 96.0                            # fig1 "Ours (per NPU)" = 96 TFLOPS FP8

    if kind == "gqa":
        # GQA：roofline/attention.py 的 qwen3 配置 + hmp 策略
        mc = dict(_attn.MODEL_CONFIGS["qwen3"])
        cfg = _attn.make_strategy_config("hmp", tp_h=tp_h, tp_hd=tp_hd, model_config=mc)
        # decode 用「融合」attention kernel：t = max(KV_mem, einsum_compute)，不是串行相加。
        # 不设此项时 single_layer_attention_time 走 2×concat+2×einsum 串行路径，会把 GQA
        # attention 高估 ~3×（einsum 与 KV 读叠加），导致 attention 错误地成为瓶颈。
        cfg["fused_attention"] = True
        qkv = _attn.single_layer_qkv_time(hw, cfg, batch, seq) * 1e9
        attn = _attn.single_layer_attention_time(hw, cfg, batch, seq) * 1e9
        out = _attn.single_layer_output_time(hw, cfg, batch, seq) * 1e9
        comm = _attn.hmp_final_reduce_comm_time(hw, cfg, batch) * 1e9
        source = "roofline/attention.py:GQA(qwen3,hmp,fused,96T)"
    elif kind == "mla":
        # MLA：用 roofline/roofline_gqa_calc.py 的真实 MLA 模型 (DeepSeek-V3)
        res = _mla.calc_mla_strategy(hw, batch, [seq], link_bw=1.5,
                                     hop_latency_ns=15, endpoint_delay_ns=10,
                                     tp_h=tp_h, tp_s=tp_hd)
        e = res["data"][0]
        qkv = e["Proj_QKV"]["total_ns"]
        attn = e["attention"]["total_ns"]
        out = e["Proj_O"]["total_ns"]
        comm = e["comm_total_ns"]
        source = "roofline/roofline_gqa_calc.py:MLA(deepseek3,ours)"
    else:
        # CSA / HCA：DeepSeek-V4 混合注意力 (roofline/roofline_v4_calc.py)
        res = _v4.calc_v4_strategy(kind, hw, batch, [seq], link_bw=1.5,
                                   hop_latency_ns=15, endpoint_delay_ns=10,
                                   tp_h=tp_h, tp_s=tp_hd)
        e = res["data"][0]
        qkv = e["Proj_QKV"]["total_ns"]
        attn = e["attention"]["total_ns"]
        out = e["Proj_O"]["total_ns"]
        comm = e["comm_total_ns"]
        source = f"roofline/roofline_v4_calc.py:{kind.upper()}(deepseek-v4,ours)"

    return {
        "t_ns": qkv + attn + out + comm,
        "parts": {"qkv": qkv, "attn": attn, "output": out, "comm": comm},
        "source": source,
        "approx": None,
    }


def compose_layer(model, lpu, run, link, batch, seq, tp_h, tp_hd,
                  t_attn_override_ns=None) -> dict:
    """组装单层的 4 个串行项，返回逐项明细 + layer_total_ns。"""
    # 1) attention（AMMA）
    if t_attn_override_ns is not None:
        attn = {"t_ns": float(t_attn_override_ns), "parts": {},
                "approx": "user-override", "source": "user-override"}
    else:
        attn = attention_layer_ns(model.name, batch, seq, tp_h, tp_hd)

    # 2) FFN（LPU）—— 与 CL 无关
    ffn = simulate_layer(model, lpu, run)
    t_ffn = ffn["layer_total_ns"]

    # 3) 跨池传输：两次，payload = hidden state(B × d_model × act_bytes)
    xfer_bytes = batch * model.d_model * lpu.act_bytes
    xa2f = transfer_time(xfer_bytes, link, "a2f")
    xf2a = transfer_time(xfer_bytes, link, "f2a")

    t_layer = attn["t_ns"] + xa2f["t_ns"] + t_ffn + xf2a["t_ns"]
    return {
        "t_attn_ns": attn["t_ns"],
        "attn_parts": attn["parts"],
        "attn_approx": attn["approx"],
        "t_xfer_a2f_ns": xa2f["t_ns"],
        "t_ffn_ns": t_ffn,
        "ffn_bound_breakdown_ns": ffn["bound_breakdown_ns"],
        "t_xfer_f2a_ns": xf2a["t_ns"],
        "attn_source": attn.get("source"),
        "layer_total_ns": t_layer,
    }


def compose_decode(model, lpu, run, link, batch, seq, tp_h, tp_hd,
                   t_attn_override_ns=None) -> dict:
    """整模 TPOT = num_layers × 单层串行和。"""
    layer = compose_layer(model, lpu, run, link, batch, seq, tp_h, tp_hd,
                          t_attn_override_ns)
    n = model.num_layers
    tpot_ns = layer["layer_total_ns"] * n

    # 把各类时间按 attn / ffn / xfer 归并到整模口径
    attn_total = layer["t_attn_ns"] * n
    ffn_total = layer["t_ffn_ns"] * n
    xfer_total = (layer["t_xfer_a2f_ns"] + layer["t_xfer_f2a_ns"]) * n
    return {
        "per_layer": layer,
        "decode": {
            "context_length": seq,
            "num_layers": n,
            "tpot_ns": tpot_ns,
            "tpot_us": tpot_ns / 1e3,
            "throughput_tok_s": 1e9 / tpot_ns if tpot_ns > 0 else 0.0,
            "breakdown_ns": {"attn": attn_total, "ffn": ffn_total, "xfer": xfer_total},
            "breakdown_frac": {
                "attn": attn_total / tpot_ns if tpot_ns else 0,
                "ffn": ffn_total / tpot_ns if tpot_ns else 0,
                "xfer": xfer_total / tpot_ns if tpot_ns else 0,
            },
        },
    }


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Disaggregated decode TPOT compose (AMMA attention + LPU FFN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="qwen3-235b",
                    help="模型（需在 lpu_config.MODEL_LIBRARY 与 ATTN_MODEL_DIMS 中）")
    ap.add_argument("--cl", type=int, default=4096, help="上下文长度（单点）")
    ap.add_argument("--cl-sweep", default=None,
                    help="逗号分隔的 CL 列表，给出则做 sweep（覆盖 --cl）")
    ap.add_argument("--batch", type=int, default=1, help="decode batch")
    # FFN(LPU) 侧
    ap.add_argument("--lpu", choices=["lpuv1", "lpx"], default="lpx", help="LPU 代次")
    ap.add_argument("--tp-ffn", type=int, default=16, help="FFN 的 TP（LPU）")
    ap.add_argument("--expert-mode", choices=["per_token", "batched"], default="per_token")
    # attention(AMMA) 侧
    ap.add_argument("--tp-h", type=int, default=4, help="AMMA attention head-group 数")
    ap.add_argument("--tp-hd", type=int, default=4, help="AMMA attention head-dim 切分")
    ap.add_argument("--t-attn-ns", type=float, default=None,
                    help="直接指定 attention 单层延时(ns)，绕过 roofline 模型")
    # 跨池 link：先选 AMMA 论文预置，再可用 --link-bw/--link-lat 覆盖
    ap.add_argument("--link", choices=["ucie3_d2d", "nvlink_rubin", "nvlink_h100"],
                    default="ucie3_d2d", help="跨池链路预置（AMMA 论文参数）")
    ap.add_argument("--link-bw", type=float, default=None, help="覆盖跨池带宽 GB/s")
    ap.add_argument("--link-lat", type=float, default=None, help="覆盖跨池固定延迟 ns")
    ap.add_argument("--output", "-o", default=None, help="结果 JSON")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    model = get_model(args.model)
    lpu = get_lpu(args.lpu)
    run = RunSpec(TP=args.tp_ffn, batch=args.batch, expert_mode=args.expert_mode)
    link = get_link(args.link)
    if args.link_bw is not None:
        link.interpool_bw_GBs = args.link_bw
    if args.link_lat is not None:
        link.interpool_latency_ns = args.link_lat

    cls = ([int(x) for x in args.cl_sweep.split(",")] if args.cl_sweep else [args.cl])

    meta = {
        "model": model.to_dict(),
        "attn_hw": "AMMA(hbm4_npu, hmp/ours)",
        "ffn_hw": f"LPU({args.lpu})",
        "lpu": lpu.to_dict(),
        "run": run.to_dict(),
        "link": link.to_dict(),
        "tp_attn": {"tp_h": args.tp_h, "tp_hd": args.tp_hd},
    }

    results = []
    print(f"== Disaggregated decode TPOT | model={model.name} "
          f"attn=AMMA(hmp) ffn=LPU({args.lpu}) B={args.batch} ==")
    print(f"  {'CL':>8} {'TPOT(us)':>10} {'tok/s':>8} | "
          f"{'attn%':>6} {'ffn%':>6} {'xfer%':>6}")
    for cl in cls:
        r = compose_decode(model, lpu, run, link, args.batch, cl,
                           args.tp_h, args.tp_hd, args.t_attn_ns)
        results.append(r)
        d = r["decode"]
        f = d["breakdown_frac"]
        print(f"  {cl:>8} {d['tpot_us']:>10.2f} {d['throughput_tok_s']:>8.0f} | "
              f"{f['attn']*100:>5.1f}% {f['ffn']*100:>5.1f}% {f['xfer']*100:>5.1f}%")

    # 打印单点的逐层 4 项明细（用第一条结果）
    pl = results[0]["per_layer"]
    print("\n  单层 4 项串行明细 (CL=%d, ns):" % cls[0])
    print(f"    t_attn      = {pl['t_attn_ns']:>10,.1f}  (parts={ {k: round(v,1) for k,v in pl['attn_parts'].items()} })")
    print(f"    t_xfer_a2f  = {pl['t_xfer_a2f_ns']:>10,.1f}")
    print(f"    t_ffn       = {pl['t_ffn_ns']:>10,.1f}")
    print(f"    t_xfer_f2a  = {pl['t_xfer_f2a_ns']:>10,.1f}")
    print(f"    --------------------------------")
    print(f"    t_layer     = {pl['layer_total_ns']:>10,.1f}")
    if pl["attn_approx"]:
        print(f"    [注] attention 近似: {pl['attn_approx']}")

    if args.output:
        out = {"meta": meta, "sweep": results}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"\n=> 已写入 {args.output}")


if __name__ == "__main__":
    main()
