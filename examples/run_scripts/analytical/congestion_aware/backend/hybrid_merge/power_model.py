#!/usr/bin/env python3
"""GQA single-layer power model: estimate Static / Compute / Memory / D2D power
from hybrid merged report data.

Power formulas (total system power):
    H100:  P = 17.7% × 700  + 829  × U_cmpt + 580  × U_mem
    Rubin: P = 17.7% × 2200 + 2631 × U_cmpt + 1800 × U_mem
    Ours:  P = 17.7% × 1440 + 394  × U_cmpt + 1120 × U_mem  [+ D2D]

Utilization:
    U_mem:  (hbm_bytes / time) / peak_hbm_bw   (per cube / per die)
    U_cmpt: (flops / time) / peak_flops         (per cube / per die)
    D2D:    energy-based = d2d_bits × pJ_per_bit / time

Three architectures:
    Ours:      16 HBM4 cubes (each with HBM + small compute die + D2D links)
    Rubin:     2 compute dies + 8 HBM4 cubes (single chip, no inter-chip D2D)
    H100_TP2:  2 × H100 GPUs with NVLink
"""
import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Power model coefficients:  P = static_w + cmpt_coeff × U_cmpt + mem_coeff × U_mem
# ---------------------------------------------------------------------------

POWER_COEFFS = {
    "ours": {
        "static_ratio": 0.177,
        "tdp_w": 1440,
        "cmpt_coeff": 394,       # W at full compute utilization
        "mem_coeff": 1120,        # W at full memory utilization
        "formula": "P = 17.7%×1440 + 394×U_cmpt + 1120×U_mem + D2D",
    },
    "rubin": {
        "static_ratio": 0.177,
        "tdp_w": 2200,
        "cmpt_coeff": 2631,
        "mem_coeff": 1800,
        "formula": "P = 17.7%×2200 + 2631×U_cmpt + 1800×U_mem",
    },
    "h100": {
        "static_ratio": 0.177,
        "tdp_w": 700,
        "cmpt_coeff": 829,
        "mem_coeff": 580,
        "formula": "P = 17.7%×700 + 829×U_cmpt + 580×U_mem",
    },
}

# Derived static power (for convenience)
for _k, _v in POWER_COEFFS.items():
    _v["static_w"] = _v["static_ratio"] * _v["tdp_w"]

# ---------------------------------------------------------------------------
# Hardware specs for utilization computation
# ---------------------------------------------------------------------------

OURS_DEFAULT = {
    "n_cubes": 16,
    "peak_flops_per_cube": 80,    # TFLOPS
    "hbm_bw_per_cube": 2.5,      # TB/s
    "d2d_pj_per_bit": 0.38,
}

NVLINK_PJ_PER_BIT = 1.3   # NVLink energy ~1.3 pJ/bit (inter-chip)

OURS_STRATEGIES = ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]
N_PHYSICAL_CUBES = 16


# ---------------------------------------------------------------------------
# Core power functions
# ---------------------------------------------------------------------------

def _time_weighted_util(module_utils, module_times):
    """Compute time-weighted average utilization.

    Parameters
    ----------
    module_utils : list of float  -- per-module utilization (0-1)
    module_times : list of float  -- per-module time (ns)

    Returns weighted average utilization.
    """
    total_t = sum(module_times)
    if total_t <= 0:
        return 0.0
    return sum(u * t for u, t in zip(module_utils, module_times)) / total_t


def calc_power_ours(entry, hw_cfg, power_coeffs=None):
    """Power estimate for one strategy running on Ours architecture (16 cubes).

    Utilization source:
        U_cmpt — from profiled utilization in compute_breakdown
                  (Proj_QKV_utilization, score_utilization, attn_v_utilization, Proj_O_utilization)
        U_mem  — computed from hbm_bytes / (gpu_time × peak_hbm_bw)
                  (only counts GPU-active time, not communication)
    """
    if power_coeffs is None:
        power_coeffs = POWER_COEFFS["ours"]

    agg = entry["aggregate_metrics"]
    cb = entry.get("compute_breakdown", {})

    n = hw_cfg["n_cubes"]
    time_ns = agg["total_time_ns"]
    time_s = time_ns * 1e-9

    # --- U_cmpt: time-weighted profiled compute utilization ---
    qkv_ns = entry.get("hybrid_Proj_QKV_ns", 0)
    attn_ns = entry.get("hybrid_attn_ns", 0)
    projo_ns = entry.get("hybrid_Proj_O_ns", 0)

    qkv_util = cb.get("Proj_QKV_utilization", 0)
    score_util = cb.get("score_utilization", 0)
    attn_v_util = cb.get("attn_v_utilization", 0)
    projo_util = cb.get("Proj_O_utilization", 0)

    # Attn consists of score + attn_v; split attn_ns proportionally
    score_plus_attnv = score_util + attn_v_util
    if score_plus_attnv > 0:
        score_ns = attn_ns * score_util / score_plus_attnv
        attn_v_ns = attn_ns * attn_v_util / score_plus_attnv
    else:
        score_ns = attn_ns / 2
        attn_v_ns = attn_ns / 2

    cmpt_util = _time_weighted_util(
        [qkv_util, score_util, attn_v_util, projo_util],
        [qkv_ns, score_ns, attn_v_ns, projo_ns],
    )

    # --- U_mem: HBM bandwidth utilization during GPU-active time ---
    hbm_bytes_per_cube = agg["hbm_read_bytes_per_cube"]
    gpu_ns = entry.get("hybrid_gpu_ns", qkv_ns + attn_ns + projo_ns)
    gpu_s = gpu_ns * 1e-9
    peak_hbm_bw = hw_cfg["hbm_bw_per_cube"] * 1e12  # bytes/s
    hbm_util = (hbm_bytes_per_cube / gpu_s) / peak_hbm_bw if gpu_s > 0 else 0

    cmpt_util = min(cmpt_util, 1.0)
    hbm_util = min(hbm_util, 1.0)

    # --- Power: P = static + cmpt_coeff × U_cmpt + mem_coeff × U_mem + D2D ---
    static_power = power_coeffs["static_w"]
    cmpt_power = power_coeffs["cmpt_coeff"] * cmpt_util
    mem_power = power_coeffs["mem_coeff"] * hbm_util

    d2d_bytes_total = agg["d2d_link_transfer_bytes_all_cubes"]
    d2d_bits = d2d_bytes_total * 8
    d2d_energy_j = d2d_bits * hw_cfg["d2d_pj_per_bit"] * 1e-12
    d2d_power = d2d_energy_j / time_s if time_s > 0 else 0

    total_power = static_power + cmpt_power + mem_power + d2d_power
    energy_nj = total_power * time_s * 1e9

    return {
        "arch": "ours",
        "n_cubes": n,
        "time_ns": time_ns,
        "hbm_util_per_cube": round(hbm_util, 6),
        "cmpt_util_per_cube": round(cmpt_util, 6),
        "per_module_cmpt_util": {
            "Proj_QKV": round(qkv_util, 6),
            "Score": round(score_util, 6),
            "Attn_V": round(attn_v_util, 6),
            "Proj_O": round(projo_util, 6),
        },
        "static_power_w": round(static_power, 4),
        "hbm_power_total_w": round(mem_power, 4),       # mem_coeff × U_mem
        "cmpt_power_total_w": round(cmpt_power, 4),     # cmpt_coeff × U_cmpt
        "d2d_energy_nj": round(d2d_energy_j * 1e9, 4),
        "d2d_power_w": round(d2d_power, 4),
        "total_power_w": round(total_power, 4),
        "energy_per_token_nj": round(energy_nj, 4),
        "tdp_total_w": power_coeffs["tdp_w"],
    }


def calc_power_rubin(entry, power_coeffs=None):
    """Power estimate for Rubin architecture.

    Utilization source:
        Both U_cmpt and U_mem come from H100 profiling (BW utilization),
        stored in entry as Proj_QKV_bw_util, attn_bw_util, Proj_O_bw_util.
        For Rubin decode (bs=1), workload is memory-bound, so BW utilization
        is the dominant factor; compute utilization is negligible.
    """
    if power_coeffs is None:
        power_coeffs = POWER_COEFFS["rubin"]

    agg = entry["aggregate_metrics"]
    time_ns = agg["total_time_ns"]
    time_s = time_ns * 1e-9

    # --- U_mem: time-weighted H100 BW utilization from profiling ---
    qkv_ns = entry.get("hybrid_Proj_QKV_ns", 0)
    attn_ns = entry.get("hybrid_attn_ns", 0)
    projo_ns = entry.get("hybrid_Proj_O_ns", 0)

    qkv_bw_util = entry.get("Proj_QKV_bw_util", 0)
    attn_bw_util = entry.get("attn_bw_util", 0)
    projo_bw_util = entry.get("Proj_O_bw_util", 0)

    hbm_util = _time_weighted_util(
        [qkv_bw_util, attn_bw_util, projo_bw_util],
        [qkv_ns, attn_ns, projo_ns],
    )

    # For Rubin decode (bs=1 GEMV), compute utilization is negligible
    cmpt_util = 0.0

    hbm_util = min(hbm_util, 1.0)

    # --- Power: P = static + cmpt_coeff × U_cmpt + mem_coeff × U_mem ---
    static_power = power_coeffs["static_w"]
    cmpt_power = power_coeffs["cmpt_coeff"] * cmpt_util
    mem_power = power_coeffs["mem_coeff"] * hbm_util

    total_power = static_power + cmpt_power + mem_power
    energy_nj = total_power * time_s * 1e9

    return {
        "arch": "rubin",
        "time_ns": time_ns,
        "hbm_util_per_cube": round(hbm_util, 6),
        "cmpt_util_per_die": round(cmpt_util, 6),
        "per_module_bw_util": {
            "Proj_QKV": round(qkv_bw_util, 6),
            "Attn": round(attn_bw_util, 6),
            "Proj_O": round(projo_bw_util, 6),
        },
        "static_power_w": round(static_power, 4),
        "hbm_power_total_w": round(mem_power, 4),       # mem_coeff × U_mem
        "cmpt_power_total_w": round(cmpt_power, 4),     # cmpt_coeff × U_cmpt
        "d2d_power_w": 0.0,
        "total_power_w": round(total_power, 4),
        "energy_per_token_nj": round(energy_nj, 4),
        "tdp_total_w": power_coeffs["tdp_w"],
    }


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _fix_tp16_per_cube(agg):
    """tp16 in merge has n_cubes=1, but per_cube IS per physical NPU.
    Return a copy with n_cubes set to N_PHYSICAL_CUBES and all_cubes
    recomputed."""
    if agg.get("n_cubes", 1) != 1:
        return agg
    fixed = dict(agg)
    fixed["n_cubes"] = N_PHYSICAL_CUBES
    fixed["hbm_read_bytes_all_cubes"] = fixed["hbm_read_bytes_per_cube"] * N_PHYSICAL_CUBES
    fixed["compute_flops_all_cubes"] = fixed["compute_flops_per_cube"] * N_PHYSICAL_CUBES
    return fixed


def load_hybrid(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GQA single-layer power model from hybrid merged data."
    )
    parser.add_argument("--data", required=True,
                        help="Path to hybrid merged JSON (e.g. gqa_hybrid_merged_80T.json)")
    parser.add_argument("--ours-peak-flops", type=float, default=OURS_DEFAULT["peak_flops_per_cube"],
                        help="Ours: peak TFLOPS per cube (default: %(default)s)")
    parser.add_argument("--ours-hbm-bw", type=float, default=OURS_DEFAULT["hbm_bw_per_cube"],
                        help="Ours: HBM BW per cube in TB/s (default: %(default)s)")
    parser.add_argument("--ours-n-cubes", type=int, default=OURS_DEFAULT["n_cubes"],
                        help="Ours: number of physical cubes (default: %(default)s)")
    parser.add_argument("--d2d-pj-per-bit", type=float, default=OURS_DEFAULT["d2d_pj_per_bit"],
                        help="D2D energy in pJ/bit (default: %(default)s)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output JSON path (default: auto)")
    args = parser.parse_args()

    ours_hw = {
        "n_cubes": args.ours_n_cubes,
        "peak_flops_per_cube": args.ours_peak_flops,
        "hbm_bw_per_cube": args.ours_hbm_bw,
        "d2d_pj_per_bit": args.d2d_pj_per_bit,
    }
    ours_pwr = POWER_COEFFS["ours"]
    rubin_pwr = POWER_COEFFS["rubin"]

    data = load_hybrid(args.data)
    strategies_data = data.get("strategies", {})

    all_strategies = OURS_STRATEGIES + ["rubin", "rubin_tp2", "h100", "h100_tp2"]
    result = {
        "_type": "power_estimation",
        "_description": "GQA single-layer power model: Static + Compute + Memory + D2D",
        "_units": {
            "power": "W (watts)",
            "energy": "nJ (nanojoules per token)",
            "time": "ns",
            "utilization": "fraction (0-1)",
        },
        "metadata": {
            "source_data": args.data,
            "power_formulas": {
                "ours": ours_pwr["formula"],
                "rubin": rubin_pwr["formula"],
                "h100": POWER_COEFFS["h100"]["formula"],
            },
            "decomposition": "P_total = P_static + P_cmpt_dynamic + P_mem_dynamic [+ P_d2d]",
            "d2d_power_formula": "P_d2d = d2d_bits × pJ_per_bit / time",
            "workflow": [
                "1. roofline_gqa_calc.py  → reports/roofline/*.json  (analytical compute + comm estimates)",
                "2. collect_gqa_data.py   → reports/astrasim/*.json  (AstraSim simulation data)",
                "3. merge_gqa_results.py  → reports/hybrid/*.json    (merged hybrid estimates with aggregate_metrics)",
                "4. power_model.py        → reports/power/*.json     (THIS FILE: power estimation from hybrid data)",
            ],
            "ours_hw": ours_hw,
            "ours_power_coeffs": ours_pwr,
            "rubin_power_coeffs": rubin_pwr,
        },
        "strategies": {},
    }

    # Collect seq list from first available strategy
    first_strat = next((s for s in all_strategies if s in strategies_data), None)
    if first_strat is None:
        print("ERROR: no strategies found in data", file=sys.stderr)
        sys.exit(1)
    seq_list = [e["seq"] for e in strategies_data[first_strat]["data"]]

    summary_rows = []

    for strat in all_strategies:
        if strat not in strategies_data:
            print(f"[WARN] Strategy '{strat}' not in data, skipping", file=sys.stderr)
            continue

        entries = strategies_data[strat]["data"]
        power_entries = []
        for entry in entries:
            if strat == "tp16":
                # Fix tp16 per-cube aggregation before passing to calc
                entry = dict(entry)
                entry["aggregate_metrics"] = _fix_tp16_per_cube(entry["aggregate_metrics"])

            if strat == "rubin":
                pw = calc_power_rubin(entry, rubin_pwr)
            elif strat == "h100":
                h100_pwr = POWER_COEFFS["h100"]
                pw = calc_power_rubin(entry, h100_pwr)
                pw["arch"] = "h100"
            elif strat in ("rubin_tp2", "h100_tp2"):
                # TP2 = 2× single-chip: use single-chip formula × 2 + NVLink D2D
                base_pwr = POWER_COEFFS["h100"] if strat == "h100_tp2" else rubin_pwr
                pw = calc_power_rubin(entry, base_pwr)
                n_gpus = entry.get("aggregate_metrics", {}).get("n_cubes", 2)
                for k in ("static_power_w", "hbm_power_total_w", "cmpt_power_total_w"):
                    pw[k] = round(pw[k] * n_gpus, 4)
                comm_bd = entry.get("comm_breakdown", {})
                nvlink_bytes = sum(op.get("msg_bytes", 0) * 2 for op in comm_bd.get("ops", []))
                nvlink_bits = nvlink_bytes * 8
                nvlink_energy_j = nvlink_bits * NVLINK_PJ_PER_BIT * 1e-12
                time_s = pw["time_ns"] * 1e-9
                pw["d2d_power_w"] = round(nvlink_energy_j / time_s, 4) if time_s > 0 else 0
                pw["d2d_energy_nj"] = round(nvlink_energy_j * 1e9, 4)
                pw["total_power_w"] = round(
                    pw["static_power_w"] + pw["hbm_power_total_w"]
                    + pw["cmpt_power_total_w"] + pw["d2d_power_w"], 4)
                pw["energy_per_token_nj"] = round(pw["total_power_w"] * time_s * 1e9, 4)
                pw["tdp_total_w"] = base_pwr["tdp_w"] * n_gpus
                pw["arch"] = strat
                pw["n_gpus"] = n_gpus
            else:
                pw = calc_power_ours(entry, ours_hw, ours_pwr)

            pw["seq"] = entry["seq"]
            agg = entry["aggregate_metrics"]
            pw["hybrid_wall_ns"] = entry.get("hybrid_wall_ns", agg["total_time_ns"])

            # Memory power breakdown by module (proportional to time)
            qkv_ns = entry.get("hybrid_Proj_QKV_ns", 0)
            attn_ns = entry.get("hybrid_attn_ns", 0)
            projo_ns = entry.get("hybrid_Proj_O_ns", 0)
            total_gpu_ns = qkv_ns + attn_ns + projo_ns
            if total_gpu_ns > 0:
                mem_total = pw["hbm_power_total_w"]
                pw["hbm_breakdown"] = {
                    "Proj_QKV_hbm_w": round(mem_total * qkv_ns / total_gpu_ns, 4),
                    "Attn_hbm_w": round(mem_total * attn_ns / total_gpu_ns, 4),
                    "Proj_O_hbm_w": round(mem_total * projo_ns / total_gpu_ns, 4),
                    "Proj_QKV_frac": round(qkv_ns / total_gpu_ns, 6),
                    "Attn_frac": round(attn_ns / total_gpu_ns, 6),
                    "Proj_O_frac": round(projo_ns / total_gpu_ns, 6),
                }

            power_entries.append(pw)

        result["strategies"][strat] = {"data": power_entries}

    # --- Summary table ---
    for seq in seq_list:
        row = {"seq": seq}
        for strat in all_strategies:
            if strat not in result["strategies"]:
                continue
            for pe in result["strategies"][strat]["data"]:
                if pe["seq"] == seq:
                    row[strat] = {
                        "total_power_w": pe["total_power_w"],
                        "static_power_w": pe["static_power_w"],
                        "hbm_power_w": pe["hbm_power_total_w"],
                        "cmpt_power_w": pe["cmpt_power_total_w"],
                        "d2d_power_w": pe.get("d2d_power_w", 0),
                        "energy_nj": pe["energy_per_token_nj"],
                        "time_ns": pe["time_ns"],
                    }
                    if "hbm_breakdown" in pe:
                        row[strat]["hbm_breakdown"] = pe["hbm_breakdown"]
                    break
        summary_rows.append(row)
    result["summary_table"] = summary_rows

    # --- Terminal table ---
    def seq_label(s):
        if s >= 1048576:
            return f"{s // 1048576}M"
        if s >= 1024:
            return f"{s // 1024}K"
        return str(s)

    header_strats = [s for s in all_strategies if s in result["strategies"]]
    col_w = 18

    print(f"\n{'='*90}")
    print(f"  GQA Power Model — {Path(args.data).name}")
    print(f"{'='*90}")

    # Power formula summary
    print(f"  Ours:  P = {ours_pwr['static_ratio']:.1%}×{ours_pwr['tdp_w']}W + "
          f"{ours_pwr['cmpt_coeff']}×U_cmpt + {ours_pwr['mem_coeff']}×U_mem + D2D")
    print(f"         Static = {ours_pwr['static_w']:.1f}W, TDP = {ours_pwr['tdp_w']}W")
    print(f"  Rubin: P = {rubin_pwr['static_ratio']:.1%}×{rubin_pwr['tdp_w']}W + "
          f"{rubin_pwr['cmpt_coeff']}×U_cmpt + {rubin_pwr['mem_coeff']}×U_mem")
    print(f"         Static = {rubin_pwr['static_w']:.1f}W, TDP = {rubin_pwr['tdp_w']}W")
    print(f"  Rubin_TP2 = 2× Rubin  →  Total TDP = {rubin_pwr['tdp_w'] * 2}W")
    print()

    # Total power table
    print("--- Total Power (W) ---")
    hdr = f"{'seq':>8}"
    for s in header_strats:
        hdr += f" {s:>{col_w}}"
    print(hdr)
    print("-" * (8 + col_w * len(header_strats) + len(header_strats)))
    for row in summary_rows:
        line = f"{seq_label(row['seq']):>8}"
        for s in header_strats:
            if s in row:
                line += f" {row[s]['total_power_w']:>{col_w}.2f}"
            else:
                line += f" {'—':>{col_w}}"
        print(line)

    # Power breakdown table
    print("\n--- Power Breakdown at seq=65536 ---")
    target_seq = 65536
    target_row = next((r for r in summary_rows if r["seq"] == target_seq), summary_rows[-1])
    print(f"{'Component':<20}", end="")
    for s in header_strats:
        print(f" {s:>{col_w}}", end="")
    print()
    print("-" * (20 + col_w * len(header_strats) + len(header_strats)))
    for comp, key in [("Static Power (W)", "static_power_w"),
                      ("Mem Power (W)", "hbm_power_w"),
                      ("Cmpt Power (W)", "cmpt_power_w"),
                      ("D2D Power (W)", "d2d_power_w"),
                      ("Total Power (W)", "total_power_w"),
                      ("Energy/token (nJ)", "energy_nj"),
                      ("Wall time (ns)", "time_ns")]:
        line = f"{comp:<20}"
        for s in header_strats:
            if s in target_row:
                v = target_row[s][key]
                if key == "energy_nj":
                    line += f" {v:>{col_w},.2f}"
                elif key == "time_ns":
                    line += f" {v:>{col_w},.1f}"
                else:
                    line += f" {v:>{col_w}.2f}"
            else:
                line += f" {'—':>{col_w}}"
        print(line)

    # Utilization table
    print("\n--- Utilization at seq=65536 ---")
    print(f"{'Metric':<20}", end="")
    for s in header_strats:
        print(f" {s:>{col_w}}", end="")
    print()
    print("-" * (20 + col_w * len(header_strats) + len(header_strats)))
    hbm_line = f"{'HBM util':<20}"
    cmpt_line = f"{'Cmpt util':<20}"
    for s in header_strats:
        sd = result["strategies"].get(s, {}).get("data", [])
        pe = next((p for p in sd if p["seq"] == target_seq), sd[-1] if sd else None)
        if pe:
            hbm_line += f" {pe.get('hbm_util_per_cube', 0):>{col_w}.4%}"
            cu = pe.get("cmpt_util_per_cube", pe.get("cmpt_util_per_die", 0))
            cmpt_line += f" {cu:>{col_w}.4%}"
        else:
            hbm_line += f" {'—':>{col_w}}"
            cmpt_line += f" {'—':>{col_w}}"
    print(hbm_line)
    print(cmpt_line)

    # Energy efficiency
    print("\n--- Energy Efficiency ---")
    print(f"{'seq':>8}", end="")
    for s in header_strats:
        print(f" {s + ' (nJ)':>{col_w}}", end="")
    print()
    print("-" * (8 + col_w * len(header_strats) + len(header_strats)))
    for row in summary_rows:
        line = f"{seq_label(row['seq']):>8}"
        for s in header_strats:
            if s in row:
                line += f" {row[s]['energy_nj']:>{col_w},.2f}"
            else:
                line += f" {'—':>{col_w}}"
        print(line)

    # Save JSON
    if args.output:
        out_path = Path(args.output)
    else:
        stem = Path(args.data).stem
        out_path = Path(args.data).resolve().parent.parent / "power" / f"{stem}_power.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
