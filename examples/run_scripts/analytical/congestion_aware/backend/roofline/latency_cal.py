#!/usr/bin/env python3
"""
Config-driven latency model for one Transformer layer (GQA) on HBM-NMP.

Supports config files:
- JSON:  *.json
- TOML:  *.toml  (Python 3.11+ uses tomllib)
- YAML:  *.yml / *.yaml (requires PyYAML: pip install pyyaml)

Strategies:
- hmp:      Duplicate Wo^g inside each Cube Group + local projection + final add-tree reduction.
- baseline: Standard TP/SP sharded Wo + explicit comm_ops (AR/AG/A2A from config).

Time model: roofline-style upper bound: time = max(compute_time, mem_time) + alpha terms.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python 3.11+
except Exception:
    tomllib = None


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def log2_ceil(x: int) -> int:
    if x <= 1:
        return 0
    return math.ceil(math.log2(x))


def bw_bytes_per_s(gbs: float) -> float:
    return gbs * 1e9


def fmt_bytes(x: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    v = float(x)
    i = 0
    while v >= 1024.0 and i < len(units) - 1:
        v /= 1024.0
        i += 1
    return f"{v:.3f} {units[i]}"


def s_to_ns(x: float) -> float:
    return x * 1e9


def fmt_ns(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:.4f} s"
    if ns >= 1e6:
        return f"{ns/1e6:.4f} ms"
    if ns >= 1e3:
        return f"{ns/1e3:.4f} us"
    return f"{ns:.4f} ns"


def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str) -> Dict[str, Any]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "rb") as f:
        data = f.read()

    if ext == ".json":
        return json.loads(data.decode("utf-8"))
    if ext in (".toml",):
        if tomllib is None:
            raise RuntimeError("tomllib not available. Use Python 3.11+ or provide JSON/YAML.")
        return tomllib.loads(data.decode("utf-8"))
    if ext in (".yml", ".yaml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML not installed. Run: pip install pyyaml") from e
        return yaml.safe_load(data.decode("utf-8")) or {}

    raise ValueError(f"Unsupported config extension: {ext} (use .json/.toml/.yml/.yaml)")


@dataclass
class Params:
    # Model
    B: int = 64
    D: int = 4096
    Hq: int = 64
    Hkv: int = 4
    G: int = 4
    Cg: int = 4
    dk: int = 128
    Lq: int = 1024
    Lk: int = 1024

    # Data types (bytes)
    act_bytes: int = 2
    w_bytes: int = 2
    scalar_bytes: int = 4

    # Hardware
    BW_hbm_gbs: float = 2000.0
    BW_sram_gbs: float = 2500.0
    BW_ip_gbs: float = 2000.0
    alpha_ns: float = 100.0
    compute_tflops: float = 30.0

    # Efficiencies
    eta_compute: float = 1.0
    eta_hbm: float = 1.0
    eta_sram: float = 1.0
    eta_ip: float = 1.0

    # Scheduling
    overlap_q_kv: bool = False

    @staticmethod
    def from_config(cfg: Dict[str, Any]) -> "Params":
        base = asdict(Params())
        valid_keys = set(base.keys())
        merged = deep_merge(base, cfg)

        # Allow structured config sections: model/dtype/hardware/efficiency/schedule
        flat: Dict[str, Any] = {}
        for section in ("model", "dtype", "hardware", "efficiency", "schedule"):
            if isinstance(merged.get(section), dict):
                flat.update(merged[section])

        # Also allow top-level flat keys
        for k, v in merged.items():
            if k in valid_keys:
                flat[k] = v

        # Map alternative key spellings (optional)
        aliases = {
            "BW_hbm": "BW_hbm_gbs",
            "BW_sram": "BW_sram_gbs",
            "BW_ip": "BW_ip_gbs",
            "alpha": "alpha_ns",
            "compute": "compute_tflops",
        }
        for k, v in list(flat.items()):
            if k in aliases and aliases[k] not in flat:
                flat[aliases[k]] = v

        # Construct
        kwargs = {k: flat[k] for k in base.keys() if k in flat}
        return Params(**kwargs)


def t_compute(flops: float, p: Params) -> float:
    eff = max(p.eta_compute, 1e-9)
    return flops / (eff * p.compute_tflops * 1e12)


def t_mem(bytes_: float, bw_gbs: float, eta: float) -> float:
    eff = max(eta, 1e-9)
    return bytes_ / (eff * bw_bytes_per_s(bw_gbs))


def roofline(flops: float, bytes_: float, bw_gbs: float, eta_bw: float, p: Params) -> float:
    return max(t_compute(flops, p), t_mem(bytes_, bw_gbs, eta_bw))


def stage_q_proj(p: Params, Dq_g: int) -> dict:
    flops = 2.0 * p.B * p.Lq * p.D * Dq_g
    bytes_in = p.B * p.Lq * p.D * p.act_bytes
    bytes_w = p.D * Dq_g * p.w_bytes
    bytes_out = p.B * p.Lq * Dq_g * p.act_bytes
    bytes_total = bytes_in + bytes_w + bytes_out
    t = roofline(flops, bytes_total, p.BW_hbm_gbs, p.eta_hbm, p)
    return {"name": "Q_proj", "flops": flops, "bytes": bytes_total, "time_ns": s_to_ns(t)}


def stage_kv_proj(p: Params, Dkv_g: int, Lk_c: int) -> dict:
    flops = 4.0 * p.B * Lk_c * p.D * Dkv_g
    bytes_in = p.B * Lk_c * p.D * p.act_bytes
    bytes_w = 2.0 * p.D * Dkv_g * p.w_bytes
    bytes_out = 2.0 * p.B * Lk_c * Dkv_g * p.act_bytes
    bytes_total = bytes_in + bytes_w + bytes_out
    t = roofline(flops, bytes_total, p.BW_hbm_gbs, p.eta_hbm, p)
    return {"name": "KV_proj", "flops": flops, "bytes": bytes_total, "time_ns": s_to_ns(t)}


def stage_attention_local(p: Params, Hq_g: int, Dq_g: int, Dkv_g: int, Lk_c: int) -> dict:
    flops = 4.0 * p.B * Hq_g * p.Lq * Lk_c * p.dk
    bytes_q = p.B * p.Lq * Dq_g * p.act_bytes
    bytes_kv = 2.0 * p.B * Lk_c * Dkv_g * p.act_bytes
    bytes_out = p.B * p.Lq * Dq_g * p.act_bytes
    bytes_total = bytes_q + bytes_kv + bytes_out
    t = roofline(flops, bytes_total, p.BW_hbm_gbs, p.eta_hbm, p)
    return {"name": "Attn_local", "flops": flops, "bytes": bytes_total, "time_ns": s_to_ns(t)}


def stage_softmax_stats_reduce(p: Params, Hq_g: int) -> dict:
    count = p.B * p.Lq * Hq_g
    bytes_total = float(count) * 2.0 * p.scalar_bytes
    alpha = (p.alpha_ns * 1e-9) / max(p.eta_ip, 1e-9)
    t = 2.0 * (alpha + t_mem(bytes_total, p.BW_ip_gbs, p.eta_ip))
    return {"name": "Softmax_stats_reduce", "flops": 0.0, "bytes": bytes_total, "time_ns": s_to_ns(t)}


def stage_wo_local(p: Params, Dq_g: int) -> dict:
    flops = 2.0 * p.B * p.Lq * Dq_g * p.D
    bytes_in = p.B * p.Lq * Dq_g * p.act_bytes
    bytes_w = Dq_g * p.D * p.w_bytes
    bytes_out = p.B * p.Lq * p.D * p.act_bytes
    bytes_total = bytes_in + bytes_w + bytes_out
    t = roofline(flops, bytes_total, p.BW_sram_gbs, p.eta_sram, p)
    return {"name": "Wo_local", "flops": flops, "bytes": bytes_total, "time_ns": s_to_ns(t)}


def stage_final_reduction(p: Params) -> dict:
    bytes_payload = p.B * p.Lq * p.D * p.act_bytes
    depth = log2_ceil(p.Cg) + log2_ceil(p.G)
    alpha = (p.alpha_ns * 1e-9) / max(p.eta_ip, 1e-9)
    t_step = alpha + t_mem(float(bytes_payload), p.BW_ip_gbs, p.eta_ip)
    t = float(depth) * t_step
    return {"name": "Final_addtree_reduce", "flops": 0.0, "bytes": float(bytes_payload), "time_ns": s_to_ns(t)}


# ---------------------------------------------------------------------------
# Baseline: generic communication stage from config comm_ops
# ---------------------------------------------------------------------------

def comm_time(payload_bytes: float, participants: int,
              bw_gbs: float, alpha_ns: float, eta_ip: float) -> float:
    """Tree-model communication time (seconds):  depth * (alpha + payload/bw)."""
    depth = log2_ceil(participants)
    alpha = (alpha_ns * 1e-9) / max(eta_ip, 1e-9)
    return depth * (alpha + t_mem(payload_bytes, bw_gbs, eta_ip))


def stage_comm_ops(p: Params, comm_ops: List[Dict[str, Any]]) -> List[dict]:
    """Build stage dicts for each explicit comm_op defined in strategy.comm_ops."""
    stages: List[dict] = []
    for op in comm_ops:
        payload = float(op["payload_bytes"])
        participants = int(op["participants"])
        t = comm_time(payload, participants, p.BW_ip_gbs, p.alpha_ns, p.eta_ip)
        stages.append({
            "name": f"COMM_{op['name']}",
            "flops": 0.0,
            "bytes": payload,
            "time_ns": s_to_ns(t),
        })
    return stages


def stage_wo_baseline(p: Params, Dq_g: int) -> dict:
    """Wo projection in baseline (sharded, row-parallel):
    each participant holds Wo shard of shape (Dq_g, D/participants).
    But the total FLOPs per device are the same as HMP Wo_local because
    the model-parallel split still does the same 2*B*Lq*Dq_g*D per group.
    The key difference is: baseline uses HBM (not SRAM) and needs
    follow-up allreduce, while HMP uses local SRAM + no allreduce.
    """
    flops = 2.0 * p.B * p.Lq * Dq_g * p.D
    bytes_in = p.B * p.Lq * Dq_g * p.act_bytes
    bytes_w = Dq_g * p.D * p.w_bytes
    bytes_out = p.B * p.Lq * p.D * p.act_bytes
    bytes_total = bytes_in + bytes_w + bytes_out
    # Baseline Wo reads from HBM (no local SRAM shortcut)
    t = roofline(flops, bytes_total, p.BW_hbm_gbs, p.eta_hbm, p)
    return {"name": "Wo_proj(sharded)", "flops": flops, "bytes": bytes_total, "time_ns": s_to_ns(t)}


# ---------------------------------------------------------------------------
# run_model: dispatch by strategy
# ---------------------------------------------------------------------------

def run_model(p: Params, strategy: Optional[Dict[str, Any]] = None) -> dict:
    if strategy is None:
        strategy = {}
    strategy_name = strategy.get("name", "hmp")

    if p.Hq % p.G != 0 or p.Hkv % p.G != 0:
        raise ValueError("Hq and Hkv must be divisible by G for this slicing.")
    Hq_g = p.Hq // p.G
    Hkv_g = p.Hkv // p.G

    Dq_g = Hq_g * p.dk
    Dkv_g = Hkv_g * p.dk
    Lk_c = ceil_div(p.Lk, p.Cg)

    derived = {
        "Hq_per_group": Hq_g,
        "Hkv_per_group": Hkv_g,
        "Dq_g": Dq_g,
        "Dkv_g": Dkv_g,
        "Lk_per_cube": Lk_c,
        "strategy": strategy_name,
        "wo_mode": strategy.get("wo_mode", "duplicate" if strategy_name == "hmp" else "sharded"),
    }

    if strategy_name == "baseline":
        # ----- Baseline: sharded Wo + explicit comm_ops from config -----
        stages = [
            stage_q_proj(p, Dq_g),
            stage_kv_proj(p, Dkv_g, Lk_c),
            stage_attention_local(p, Hq_g, Dq_g, Dkv_g, Lk_c),
        ]
        # Insert comm_ops at pipeline position (between attn and Wo, or after Wo)
        comm_ops = strategy.get("comm_ops", [])
        # Separate pre-Wo and post-Wo comm ops by name convention:
        #   ops whose name starts with "AR_" or "AG_" before Wo go before Wo
        #   the rest go after Wo
        pre_wo_ops = []
        post_wo_ops = []
        for op in comm_ops:
            # Simple heuristic: put all listed ops in order after attention
            # User controls order via comm_ops list position
            pass

        # Baseline Wo (sharded, reads from HBM)
        stages.append(stage_wo_baseline(p, Dq_g))
        # All comm ops appended after Wo in config-listed order
        stages.extend(stage_comm_ops(p, comm_ops))
    else:
        # ----- HMP: duplicate Wo + local SRAM + tree reduction -----
        stages = [
            stage_q_proj(p, Dq_g),
            stage_kv_proj(p, Dkv_g, Lk_c),
            stage_attention_local(p, Hq_g, Dq_g, Dkv_g, Lk_c),
            stage_softmax_stats_reduce(p, Hq_g),
            stage_wo_local(p, Dq_g),
            stage_final_reduction(p),
        ]

    if p.overlap_q_kv:
        t0 = max(stages[0]["time_ns"], stages[1]["time_ns"])
        total_ns = t0 + sum(s["time_ns"] for s in stages[2:])
    else:
        total_ns = sum(s["time_ns"] for s in stages)

    return {
        "config": asdict(p),
        "strategy": strategy,
        "derived": derived,
        "stages": stages,
        "total_time_ns": total_ns,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Config-driven per-layer latency model (HMP / Baseline)"
    )
    ap.add_argument("--config", required=True, help="Path to config file (.json/.toml/.yml/.yaml)")
    ap.add_argument("--output", "-o", default=None, help="Save results to JSON file")
    args = ap.parse_args()

    cfg = load_config(args.config)
    p = Params.from_config(cfg)
    strategy = cfg.get("strategy", {})
    out = run_model(p, strategy)

    total_ns = out["total_time_ns"]
    strategy_name = out["derived"].get("strategy", "hmp")

    print(f"== Strategy: {strategy_name} ==")
    print("== Derived ==")
    for k, v in out["derived"].items():
        print(f"  {k}: {v}")

    print(f"\n== Stages (per-cube critical path)  [unit: ns] ==")
    print(f"  {'Stage':<25s} {'time_ns':>14s} {'pct':>7s}  {'FLOPs':>12s}  {'Bytes':>12s}")
    print(f"  {'-'*25} {'-'*14} {'-'*7}  {'-'*12}  {'-'*12}")
    for s in out["stages"]:
        pct = (s["time_ns"] / total_ns * 100.0) if total_ns > 0 else 0.0
        print(
            f"  {s['name']:<25s} {s['time_ns']:>14,.1f} {pct:>6.2f}%"
            f"  {s['flops']:>12.3e}  {fmt_bytes(s['bytes']):>12s}"
        )

    print(f"\n== Total layer latency ==")
    print(f"  {total_ns:,.1f} ns  ({fmt_ns(total_ns)})")

    if args.output:
        out_path = args.output
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n=> Results saved to {out_path}")


if __name__ == "__main__":
    main()