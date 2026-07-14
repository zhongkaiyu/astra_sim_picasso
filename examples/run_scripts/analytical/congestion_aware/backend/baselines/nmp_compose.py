#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmp_compose.py — Duplex / Helios / Stratum 三条 NMP decode baseline 的 TPOT 组合器 + CLI。

范式（见 plan/PLAN.md §0）：decode 时 attention 与 FFN 同址跑在一块近存设备上，无跨池 NIC：
    t_layer(CL) = t_attn_nmp(CL) + t_ffn_nmp        # NoC 已并入各自 comm
    TPOT        = num_layers × t_layer(CL)

对齐 fig1 + exp1（见 plan/PLAN.md §8）：
  - attention（fig1 路径，只换硬件）：
      * GQA(qwen3)   : roofline/attention.py 的 fused 单层模型 + h100_rubin_utilization derate
      * MLA(deepseek3): roofline/deepseek_v3_mla_roofline.py 的 absorbed roofline + h100_mla_profile，
                        按 device 数(TP)除 + NoC AllReduce  —— 与 fig1 同一路径（非 exp1 的 calc_mla_strategy）
  - FFN（exp1 路径）：lpu_ffn_decode/ffn_decode.py 的 simulate_layer，100% roofline（决策 A，乐观上界）
  - 组合/输出（exp1 schema）：t_layer = t_attn + t_ffn，breakdown 去 xfer、加 noc

用法:
  python nmp_compose.py --baseline duplex  --model qwen3-235b --cl 8192 --batch 1
  python nmp_compose.py --baseline stratum --model deepseek3  --cl-sweep 2048,8192,32768,131072,1048576 \
         -o out.json
"""
from __future__ import annotations
import argparse
import json
import os
import sys

# --- 复用 lpu_ffn_decode（FFN + NoC AllReduce + 模型库）-----------------------
_LPU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lpu_ffn_decode")
sys.path.insert(0, os.path.abspath(_LPU_DIR))
from lpu_config import RunSpec, get_model          # noqa: E402
import ffn_decode as _ffn                           # noqa: E402  simulate_layer + _allreduce_c2c

# --- 只读复用 roofline 的 attention 模型（GQA + fig1 MLA）---------------------
_ROOFLINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roofline")
sys.path.insert(0, os.path.abspath(_ROOFLINE_DIR))
import attention as _attn                            # noqa: E402  GQA
import deepseek_v3_mla_roofline as _mla              # noqa: E402  fig1 MLA（sourceless .pyc，已验证可 import）

# --- 本目录参数层 -----------------------------------------------------------
from nmp_config import (get_nmp, get_attn_hw, get_run_defaults,   # noqa: E402
                        NMP_LABELS, NMP_MEM_TECH)

ATTN_BACKEND = {"qwen3-235b": "gqa", "deepseek3": "mla"}

# GQA attention 利用率（H100 实测 → 应用为 roofline / util，同 fig1 / exp1 baseline_compose）
_RUBIN_UTIL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "h100_rubin_utilization.json")


def _load_gqa_util():
    with open(os.path.abspath(_RUBIN_UTIL_PATH)) as f:
        return json.load(f)


def _interp_seq(profile: dict, seq: int) -> float:
    """{seq_str: frac} 分段线性插值（attn_fused 用）。"""
    pts = sorted((int(k), float(v)) for k, v in profile.items())
    if seq <= pts[0][0]:
        return pts[0][1]
    if seq >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= seq <= x1:
            return y0 + (y1 - y0) * (seq - x0) / (x1 - x0)
    return pts[-1][1]


def _noc_allreduce_ns(model, batch, run, spec) -> float:
    """attention-output 的片内 NoC AllReduce（payload = B×d_model×act_bytes）。
    复用 ffn_decode._allreduce_c2c：diameter×hop_lat + 2(TP-1)/TP × payload/eff_bw。
    注意：run.TP 已是 tp_lat（node 内），故这是 node 内 NVLink-class collective。"""
    payload = batch * model.d_model * spec.act_bytes
    return _ffn._allreduce_c2c(payload, spec, run)["t_ns"]


def _ep_crossnode_ns(model, batch, defaults, spec) -> float:
    """跨 node 的 MoE expert-parallel all-to-all（>8 卡后才出现）。
    单 token decode 每层需 **2 次跨 node all-to-all-v**：dispatch（token→远端 expert node）+
    combine（结果→home node）。每次 all-to-all-v 一个 node 要与 (nn-1) 个对端交换，小包 latency-bound
    → 关键路径 ≈ (nn-1) × 单跳 NIC 延迟（保守上界：交换串行）；带宽项 (nn-1)/nn × payload/nic_bw。
    NIC 延迟用 exp1 同口径实测量级（Duplex/Helios 2µs IB/RDMA、Stratum 0.7µs cross-chip）。
    仅 MoE 且 num_nodes>1 时计。这是 NMP 在 >8 卡上的 **主要隐藏延时**（用户 #4 指正）。"""
    if not getattr(model, "is_moe", False) or defaults.num_nodes <= 1:
        return 0.0
    nn = defaults.num_nodes
    per_phase_payload = batch * model.d_model * spec.act_bytes      # 单次 all-to-all 的激活
    lat = (nn - 1) * defaults.nic_hop_ns                            # 单次 all-to-all-v 关键路径（ns）
    bw = ((nn - 1) / nn) * per_phase_payload / (defaults.nic_bw_GBs * 1e9) * 1e9
    return 2.0 * (lat + bw)                                         # ×2：dispatch + combine


def attention_layer_ns_nmp(model, batch, seq, hw, run, spec, defaults,
                           use_util=True) -> dict:
    """单层 attention decode 延时，跑在 NMP 上（fig1 roofline，硬件换 NMP）。"""
    kind = ATTN_BACKEND.get(model.name)
    if kind is None:
        raise KeyError(f"模型 '{model.name}' 无 backend attention 模型，仅 {list(ATTN_BACKEND)}")
    comm = _noc_allreduce_ns(model, batch, run, spec)
    u = _load_gqa_util() if use_util else None

    if kind == "gqa":
        # GQA：attention.py fused 模型，hw=NMP，head 切分=tp_h×tp_hd（聚合带宽 = 单设备×TP）
        mc = dict(_attn.MODEL_CONFIGS["qwen3"])
        cfg = _attn.make_strategy_config("hmp", tp_h=defaults.tp_h, tp_hd=defaults.tp_hd,
                                         model_config=mc)
        cfg["fused_attention"] = True
        qkv = _attn.single_layer_qkv_time(hw, cfg, batch, seq) * 1e9
        attn = _attn.single_layer_attention_time(hw, cfg, batch, seq) * 1e9
        out = _attn.single_layer_output_time(hw, cfg, batch, seq) * 1e9
        source = f"attention.py:GQA(qwen3,hmp,fused)@{NMP_LABELS.get(model.name,'')}"
        if u:
            qkv /= max(u["proj_qkv"].get(str(batch), u["proj_qkv"]["1"]), 1e-6)
            attn /= max(_interp_seq(u["attn_fused"], seq), 1e-6)
            out /= max(u["proj_o"].get(str(batch), u["proj_o"]["1"]), 1e-6)
            source += " /h100_rubin_util"
        approx = None
    else:
        # MLA：fig1 路径 deepseek_v3_mla_roofline.mla_single_gpu_baseline（absorbed + h100_mla_profile），
        #      单设备 roofline 已含 util；按 device 数(TP)除 → TP 切分（权重+KV 各分 TP 份）。
        r = _mla.mla_single_gpu_baseline(peak_perf_tflops=hw["compute"],
                                         hbm_bw_tbs=hw["Bandwidth"],
                                         batch=batch, seq=seq, bytes_per_elem=2)
        tp = max(run.TP, 1)
        qkv = r["hybrid_Proj_QKV_ns"] / tp
        attn = r["hybrid_attn_ns"] / tp
        out = r["hybrid_Proj_O_ns"] / tp
        source = "deepseek_v3_mla_roofline.py:MLA(absorbed,fig1)/TP@" + NMP_LABELS.get(model.name, "")
        approx = None

    return {"t_ns": qkv + attn + out + comm,
            "parts": {"qkv": qkv, "attn": attn, "output": out, "comm": comm},
            "source": source, "approx": approx}


def compose_layer(baseline, model, hw, run, spec, defaults, batch, seq,
                  use_attn_util=True) -> dict:
    """单层：attention(NMP) + FFN(NMP)，无跨池 xfer。"""
    attn = attention_layer_ns_nmp(model, batch, seq, hw, run, spec, defaults, use_attn_util)
    ffn = _ffn.simulate_layer(model, spec, run)           # exp1 路径，100% roofline（run.TP=tp_lat）
    t_ffn = ffn["layer_total_ns"]
    t_ep = _ep_crossnode_ns(model, batch, defaults, spec)  # >8 卡跨 node EP all-to-all（NIC）
    t_layer = attn["t_ns"] + t_ffn + t_ep
    return {
        "baseline": baseline,
        "t_attn_ns": attn["t_ns"],
        "attn_parts": attn["parts"],
        "attn_source": attn["source"],
        "t_ffn_ns": t_ffn,
        "t_ep_ns": t_ep,
        "ffn_bound_breakdown_ns": ffn["bound_breakdown_ns"],
        "layer_total_ns": t_layer,
    }


def compose_decode(baseline, model, hw, run, spec, defaults, batch, seq,
                   use_attn_util=True) -> dict:
    layer = compose_layer(baseline, model, hw, run, spec, defaults, batch, seq, use_attn_util)
    n = model.num_layers
    tpot_ns = layer["layer_total_ns"] * n
    attn_total = layer["t_attn_ns"] * n
    ffn_total = layer["t_ffn_ns"] * n
    ep_total = layer.get("t_ep_ns", 0.0) * n
    # noc 分量：attention comm（node 内）+ FFN comm（node 内）+ EP all-to-all（跨 node NIC）
    noc_total = (layer["attn_parts"]["comm"] + layer["ffn_bound_breakdown_ns"].get("comm", 0.0)
                 + layer.get("t_ep_ns", 0.0)) * n
    return {
        "per_layer": layer,
        "decode": {
            "baseline": baseline,
            "context_length": seq,
            "num_layers": n,
            "tpot_ns": tpot_ns,
            "tpot_us": tpot_ns / 1e3,
            "throughput_tok_s": 1e9 / tpot_ns if tpot_ns > 0 else 0.0,
            "breakdown_ns": {"attn": attn_total, "ffn": ffn_total, "noc": noc_total,
                             "ep_noc": ep_total},
            "breakdown_frac": {
                "attn": attn_total / tpot_ns if tpot_ns else 0,
                "ffn": ffn_total / tpot_ns if tpot_ns else 0,
                "noc": noc_total / tpot_ns if tpot_ns else 0,
            },
        },
    }


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Duplex/Helios/Stratum NMP decode TPOT baseline (attn+FFN co-located, no NIC)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--baseline", choices=["duplex", "helios", "stratum"], required=True)
    ap.add_argument("--model", default="deepseek3", help="deepseek3 (MLA) / qwen3-235b (GQA)")
    ap.add_argument("--cl", type=int, default=8192)
    ap.add_argument("--cl-sweep", default=None, help="逗号分隔 CL 列表（覆盖 --cl）")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--expert-mode", choices=["per_token", "batched"], default="per_token")
    ap.add_argument("--p-hot", type=float, default=None,
                    help="Stratum tiering 命中率（默认 0.9，仅 stratum 生效）")
    ap.add_argument("--no-attn-util", action="store_true", help="关闭 GQA attention util derate")
    ap.add_argument("--output", "-o", default=None)
    return ap


def main():
    args = build_argparser().parse_args()
    model = get_model(args.model)
    defaults = get_run_defaults(args.baseline, args.model)
    spec = (get_nmp(args.baseline, p_hot=args.p_hot) if (args.baseline == "stratum" and args.p_hot is not None)
            else get_nmp(args.baseline))
    hw = get_attn_hw(args.baseline)
    run = RunSpec(TP=defaults.tp_lat, batch=args.batch, expert_mode=args.expert_mode,
                  tp_diameter_hops=defaults.diameter_hops)   # roofline 用 tp_lat（node 封顶）
    cls = ([int(x) for x in args.cl_sweep.split(",")] if args.cl_sweep else [args.cl])

    meta = {
        "baseline": args.baseline,
        "label": NMP_LABELS[args.baseline],
        "mem_tech": NMP_MEM_TECH[args.baseline],
        "model": model.to_dict(),
        "nmp_spec": spec.to_dict(),
        "attn_hw": hw,
        "run": run.to_dict(),
        "run_defaults": {"tp": defaults.tp, "tp_h": defaults.tp_h, "tp_hd": defaults.tp_hd,
                         "diameter_hops": defaults.diameter_hops, "noc": defaults.noc},
        "ffn_util": "100%_roofline_optimistic",   # 决策 A：标记！
        "attn_util": "h100_rubin/h100_mla derate (inherited from exp1/fig1)",
        "dtype": "fp16",
        "align": "attn=fig1 roofline (GQA attention.py / MLA deepseek_v3_mla_roofline); "
                 "ffn=exp1 ffn_decode; compose=exp1 (no NIC xfer)",
    }

    print(f"== NMP baseline {NMP_LABELS[args.baseline]} | model={model.name} "
          f"B={args.batch} TP={defaults.tp} ({defaults.noc}) ==")
    print(f"  {'CL':>8} {'TPOT(us)':>10} {'tok/s':>8} | {'attn%':>6} {'ffn%':>6} {'noc%':>6}")
    rows = []
    for cl in cls:
        r = compose_decode(args.baseline, model, hw, run, spec, defaults,
                           args.batch, cl, use_attn_util=not args.no_attn_util)
        rows.append(r)
        d = r["decode"]; f = d["breakdown_frac"]
        print(f"  {cl:>8} {d['tpot_us']:>10.2f} {d['throughput_tok_s']:>8.0f} | "
              f"{f['attn']*100:>5.1f}% {f['ffn']*100:>5.1f}% {f['noc']*100:>5.1f}%")

    pl = rows[0]["per_layer"]
    print(f"\n  单层明细 (CL={cls[0]}, ns): t_attn={pl['t_attn_ns']:.1f} "
          f"(parts={ {k: round(v,1) for k,v in pl['attn_parts'].items()} })  "
          f"t_ffn={pl['t_ffn_ns']:.1f}  => t_layer={pl['layer_total_ns']:.1f}")

    if args.output:
        out = {"meta": meta, "sweep": rows}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"\n=> wrote {args.output}")


if __name__ == "__main__":
    main()
