#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
baseline_compose.py — the two **non-AMMA decode baselines**, one decode layer
(and the integrated TPOT), to compare against the AMMA+LPU "ours" (decode_compose.py).

    baseline        attention        FFN          cross-pool xfer?
    ─────────────── ──────────────── ──────────── ────────────────
    GPU+GPU         Rubin GPU        Rubin GPU    NO  (same device domain)
    GPU+LPU         Rubin GPU        LPU (LPX)    YES (GPU<->LPU, like exp1)
    [ours, exp1]    AMMA             LPU (LPX)    YES (AMMA<->LPU)

Per-layer latency (strict serial, single-user TPOT; PP does not change TPOT):
    GPU+GPU : t_layer = t_attn_rubin + t_ffn_rubin
    GPU+LPU : t_layer = t_attn_rubin + t_xfer + t_ffn_lpu + t_xfer
    TPOT    = num_layers × t_layer

What's new here vs decode_compose.py:
  - FFN can run on **Rubin GPU** (rubin_ffn_decode.py), derated by REAL-CARD GPU
    FFN utilization (data/gpu_ffn_utilization.json) — the requested baseline.
  - attention runs on **Rubin** instead of AMMA: same roofline attention models as
    decode_compose (roofline/attention.py GQA, roofline/roofline_gqa_calc.py MLA),
    but hw=rubin_single_layer_config and derated by h100_rubin_utilization.json
    (the repo's measured H100->Rubin attention efficiency). Override with --t-attn-ns.

Usage:
  python baseline_compose.py --model deepseek3 --cl 8192 --batch 1
  python baseline_compose.py --model qwen3-235b --cl-sweep 2048,8192,32768,131072
  python baseline_compose.py --model deepseek3 --t-attn-ns 1800   # plug your own attn
"""
from __future__ import annotations
import argparse
import json
import os
import sys

from lpu_config import RunSpec, get_model, get_lpu
import ffn_decode as lpu_ffn
import rubin_ffn_decode as rubin_ffn
from interconnect import get_link, transfer_time

# read-only reuse of roofline attention models (same as decode_compose.py)
_ROOFLINE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "roofline")
sys.path.insert(0, os.path.abspath(_ROOFLINE_DIR))
import attention as _attn               # noqa: E402  GQA
import hardware_config as _hwc          # noqa: E402
import roofline_gqa_calc as _mla        # noqa: E402  MLA
import roofline_v4_calc as _v4          # noqa: E402  CSA/HCA

ATTN_BACKEND = {"qwen3-235b": "gqa", "deepseek3": "mla",
                "deepseek-v4-csa": "csa", "deepseek-v4-hca": "hca"}

# Rubin attention utilization (H100-measured, applied as roofline / bw_util)
_RUBIN_UTIL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "h100_rubin_utilization.json")


def _load_rubin_attn_util():
    with open(os.path.abspath(_RUBIN_UTIL_PATH)) as f:
        return json.load(f)


def _interp_seq(profile: dict, seq: int) -> float:
    """Piecewise-linear interp of a {seq_str: frac} profile (e.g. attn_fused)."""
    pts = sorted((int(k), float(v)) for k, v in profile.items())
    if seq <= pts[0][0]:
        return pts[0][1]
    if seq >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= seq <= x1:
            return y0 + (y1 - y0) * (seq - x0) / (x1 - x0)
    return pts[-1][1]


def _attn_allreduce_ns(model, batch, tp_h, tp_hd):
    """Attention-output TP all-reduce on Rubin over NVLink (B x d_model activation).
    Replaces the AMMA-mesh comm model (per-hop latency x 16-cube), which is wrong on
    a GPU: a real GPU does one NVLink ring all-reduce over the attention TP group."""
    tp_attn = max(tp_h * tp_hd, 1)
    payload = batch * model.d_model * rubin_ffn.RUBIN.act_bytes
    run_attn = RunSpec(TP=tp_attn, batch=batch)
    return rubin_ffn._allreduce_nvlink(payload, rubin_ffn.RUBIN, run_attn)["t_ns"]


def attention_layer_ns_rubin(model_key: str, batch: int, seq: int,
                             tp_h: int, tp_hd: int, use_util: bool = True,
                             model=None) -> dict:
    """Rubin single-layer attention decode latency: compute/memory roofline on rubin
    hw (roofline/attention.py GQA, roofline/roofline_gqa_calc.py MLA), derated by
    h100_rubin_utilization, with a single NVLink all-reduce for the TP collective."""
    kind = ATTN_BACKEND.get(model_key)
    if kind is None:
        raise KeyError(f"model '{model_key}' has no backend attention model; "
                       f"only {list(ATTN_BACKEND)}")
    hw = _hwc.rubin_single_layer_config
    u = _load_rubin_attn_util() if use_util else None
    if model is None:
        model = get_model(model_key)
    comm = _attn_allreduce_ns(model, batch, tp_h, tp_hd)

    if kind == "gqa":
        mc = dict(_attn.MODEL_CONFIGS["qwen3"])
        cfg = _attn.make_strategy_config("hmp", tp_h=tp_h, tp_hd=tp_hd, model_config=mc)
        qkv = _attn.single_layer_qkv_time(hw, cfg, batch, seq) * 1e9
        attn = _attn.single_layer_attention_time(hw, cfg, batch, seq) * 1e9
        out = _attn.single_layer_output_time(hw, cfg, batch, seq) * 1e9
        source = "roofline/attention.py:GQA(qwen3) @rubin +nvlink_ar"
        if u:
            qkv /= max(u["proj_qkv"].get(str(batch), u["proj_qkv"]["1"]), 1e-6)
            attn /= max(_interp_seq(u["attn_fused"], seq), 1e-6)
            out /= max(u["proj_o"].get(str(batch), u["proj_o"]["1"]), 1e-6)
            source += " /h100_rubin_util"
    elif kind == "mla":
        res = _mla.calc_mla_strategy(hw, batch, [seq], link_bw=1.8,
                                     hop_latency_ns=900, endpoint_delay_ns=10,
                                     tp_h=tp_h, tp_s=tp_hd)
        e = res["data"][0]
        qkv = e["Proj_QKV"]["total_ns"]
        attn = e["attention"]["total_ns"]
        out = e["Proj_O"]["total_ns"]
        source = "roofline/roofline_gqa_calc.py:MLA(deepseek3) @rubin +nvlink_ar"
        if u:
            # GQA-derived util applied to MLA as approximation (KV-read efficiency)
            attn /= max(_interp_seq(u["attn_fused"], seq), 1e-6)
            qkv /= max(u["proj_qkv"]["1"], 1e-6)
            out /= max(u["proj_o"]["1"], 1e-6)
            source += " /h100_rubin_util(approx)"
    else:
        # CSA / HCA on Rubin (roofline/roofline_v4_calc.py)。dense-MLA util 作近似 derate。
        res = _v4.calc_v4_strategy(kind, hw, batch, [seq], link_bw=1.8,
                                   hop_latency_ns=900, endpoint_delay_ns=10,
                                   tp_h=tp_h, tp_s=tp_hd)
        e = res["data"][0]
        qkv = e["Proj_QKV"]["total_ns"]
        attn = e["attention"]["total_ns"]
        out = e["Proj_O"]["total_ns"]
        source = f"roofline/roofline_v4_calc.py:{kind.upper()}(deepseek-v4) @rubin +nvlink_ar"
        if u:
            attn /= max(_interp_seq(u["attn_fused"], seq), 1e-6)
            qkv /= max(u["proj_qkv"]["1"], 1e-6)
            out /= max(u["proj_o"]["1"], 1e-6)
            source += " /h100_rubin_util(approx)"

    return {"t_ns": qkv + attn + out + comm,
            "parts": {"qkv": qkv, "attn": attn, "output": out, "comm": comm},
            "source": source,
            "approx": "V4/MLA-as-GQA-util" if kind in ("mla", "csa", "hca") and use_util else None}


def compose_layer(baseline: str, model, run, link, batch, seq, tp_h, tp_hd,
                  lpu=None, gpu=None, util=None, t_attn_override_ns=None,
                  use_attn_util=True) -> dict:
    """One decoder layer for a given baseline ('gpu_gpu' or 'gpu_lpu')."""
    # 1) attention on Rubin (or user override)
    if t_attn_override_ns is not None:
        attn = {"t_ns": float(t_attn_override_ns), "parts": {},
                "approx": "user-override", "source": "user-override"}
    else:
        attn = attention_layer_ns_rubin(model.name, batch, seq, tp_h, tp_hd,
                                        use_util=use_attn_util, model=model)

    # 2) cross-pool transfer: BOTH baselines disaggregate attention-pool from
    #    FFN-pool over a NIC (attention on one Rubin pool, FFN on a SEPARATE pool
    #    — Rubin GPUs for gpu_gpu, LPUs for gpu_lpu). Same hidden-state payload,
    #    same link, so the only difference between the two baselines is the FFN hw.
    xfer_bytes = batch * model.d_model * gpu.act_bytes
    xa2f = transfer_time(xfer_bytes, link, "a2f")
    xf2a = transfer_time(xfer_bytes, link, "f2a")

    # 3) FFN on a separate Rubin GPU pool or on an LPU pool
    if baseline == "gpu_gpu":
        ffn = rubin_ffn.simulate_layer(model, gpu, run, util)
        ffn_hw = f"Rubin({gpu.name})"
    elif baseline == "gpu_lpu":
        ffn = lpu_ffn.simulate_layer(model, lpu, run)
        ffn_hw = "LPU(lpx)"
    else:
        raise ValueError(f"unknown baseline {baseline}")
    t_ffn = ffn["layer_total_ns"]

    t_layer = attn["t_ns"] + xa2f["t_ns"] + t_ffn + xf2a["t_ns"]
    return {
        "baseline": baseline,
        "t_attn_ns": attn["t_ns"],
        "attn_parts": attn["parts"],
        "attn_source": attn.get("source"),
        "attn_approx": attn.get("approx"),
        "t_xfer_a2f_ns": xa2f["t_ns"],
        "t_ffn_ns": t_ffn,
        "ffn_hw": ffn_hw,
        "ffn_bound_breakdown_ns": ffn["bound_breakdown_ns"],
        "t_xfer_f2a_ns": xf2a["t_ns"],
        "layer_total_ns": t_layer,
    }


def compose_decode(baseline, model, run, link, batch, seq, tp_h, tp_hd,
                   lpu=None, gpu=None, util=None, t_attn_override_ns=None,
                   use_attn_util=True) -> dict:
    layer = compose_layer(baseline, model, run, link, batch, seq, tp_h, tp_hd,
                          lpu, gpu, util, t_attn_override_ns, use_attn_util)
    n = model.num_layers
    tpot_ns = layer["layer_total_ns"] * n
    attn_total = layer["t_attn_ns"] * n
    ffn_total = layer["t_ffn_ns"] * n
    xfer_total = (layer["t_xfer_a2f_ns"] + layer["t_xfer_f2a_ns"]) * n
    return {
        "per_layer": layer,
        "decode": {
            "baseline": baseline,
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


def build_argparser():
    ap = argparse.ArgumentParser(
        description="GPU+GPU / GPU+LPU decode baselines (Rubin attn + Rubin/LPU FFN)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--model", default="deepseek3")
    ap.add_argument("--cl", type=int, default=8192)
    ap.add_argument("--cl-sweep", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--baselines", default="gpu_gpu,gpu_lpu",
                    help="comma list: gpu_gpu,gpu_lpu")
    # FFN parallelism
    ap.add_argument("--tp-ffn", type=int, default=16)
    ap.add_argument("--expert-mode", choices=["per_token", "batched"], default="per_token")
    ap.add_argument("--lpu", choices=["lpuv1", "lpx"], default="lpx")
    # attention parallelism (Rubin)
    ap.add_argument("--tp-h", type=int, default=4)
    ap.add_argument("--tp-hd", type=int, default=4)
    ap.add_argument("--t-attn-ns", type=float, default=None)
    ap.add_argument("--no-attn-util", action="store_true",
                    help="disable h100_rubin_utilization derating of attention")
    # GPU<->LPU link (only used by gpu_lpu). Discrete GPU+LPU over a NIC -> nic_cx7.
    ap.add_argument("--link", choices=["nic_cx7", "nic_cx8", "nvlink_rubin",
                                       "nvlink_h100", "ucie3_d2d"],
                    default="nic_cx7",
                    help="GPU<->LPU cross-pool link. GPU+LPU are discrete and "
                         "transfer over a NIC -> nic_cx7 (~2us latency) by default.")
    ap.add_argument("--link-bw", type=float, default=None, help="override link BW GB/s")
    ap.add_argument("--link-lat", type=float, default=None, help="override link latency ns")
    ap.add_argument("--output", "-o", default=None)
    return ap


def main():
    args = build_argparser().parse_args()
    model = get_model(args.model)
    lpu = get_lpu(args.lpu)
    gpu = rubin_ffn.RUBIN
    util = rubin_ffn.FFNUtil(model_name=model.name)
    run = RunSpec(TP=args.tp_ffn, batch=args.batch, expert_mode=args.expert_mode)
    link = get_link(args.link)
    if args.link_bw is not None:
        link.interpool_bw_GBs = args.link_bw
    if args.link_lat is not None:
        link.interpool_latency_ns = args.link_lat
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    cls = ([int(x) for x in args.cl_sweep.split(",")] if args.cl_sweep else [args.cl])

    meta = {
        "model": model.to_dict(),
        "gpu": gpu.to_dict(),
        "lpu": lpu.to_dict(),
        "run": run.to_dict(),
        "link_gpu_lpu": link.to_dict(),
        "attn_hw": "Rubin(rubin_single_layer) + h100_rubin_util"
                   + ("" if not args.no_attn_util else " [util OFF]"),
        "ffn_util_profile": util.path,
        "tp_attn": {"tp_h": args.tp_h, "tp_hd": args.tp_hd},
    }

    results = {}
    for b in baselines:
        print(f"\n== Baseline {b.upper()} | model={model.name} "
              f"B={args.batch} TP_ffn={args.tp_ffn} ==")
        print(f"  {'CL':>8} {'TPOT(us)':>10} {'tok/s':>8} | "
              f"{'attn%':>6} {'ffn%':>6} {'xfer%':>6}")
        rows = []
        for cl in cls:
            r = compose_decode(b, model, run, link, args.batch, cl,
                               args.tp_h, args.tp_hd, lpu=lpu, gpu=gpu, util=util,
                               t_attn_override_ns=args.t_attn_ns,
                               use_attn_util=not args.no_attn_util)
            rows.append(r)
            d = r["decode"]; f = d["breakdown_frac"]
            print(f"  {cl:>8} {d['tpot_us']:>10.2f} {d['throughput_tok_s']:>8.0f} | "
                  f"{f['attn']*100:>5.1f}% {f['ffn']*100:>5.1f}% {f['xfer']*100:>5.1f}%")
        results[b] = rows
        pl = rows[0]["per_layer"]
        print(f"  per-layer (CL={cls[0]}): t_attn={pl['t_attn_ns']:.1f}  "
              f"xfer_a2f={pl['t_xfer_a2f_ns']:.1f}  t_ffn={pl['t_ffn_ns']:.1f} "
              f"({pl['ffn_hw']})  xfer_f2a={pl['t_xfer_f2a_ns']:.1f}  "
              f"=> t_layer={pl['layer_total_ns']:.1f} ns")

    if args.output:
        out = {"meta": meta, "baselines": {b: results[b] for b in baselines}}
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as fp:
            json.dump(out, fp, indent=2)
        print(f"\n=> wrote {args.output}")


if __name__ == "__main__":
    main()
