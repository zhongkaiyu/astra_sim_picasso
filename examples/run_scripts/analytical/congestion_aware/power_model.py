#!/usr/bin/env python3
"""GQA single-layer power model: estimate HBM / Compute / D2D power
from hybrid merged report data.

Power formula (per component):
    Power = static_ratio * TDP  +  TDP * utilization

Utilization:
    HBM:  (hbm_bytes / time) / peak_hbm_bw
    Cmpt: (flops / time) / peak_flops
    D2D:  energy-based = d2d_bits * pJ_per_bit / time

Two architectures:
    Ours:  16 HBM4 cubes (each with HBM + small compute die + D2D links)
    Rubin: 2 compute dies + 8 HBM4 cubes (single chip, no inter-chip D2D)
"""
import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Architecture defaults
# ---------------------------------------------------------------------------

OURS_DEFAULT = {
    "n_cubes": 16,
    "hbm_tdp_per_cube": 75,       # W
    "cmpt_tdp_per_cube": 15,      # W
    "peak_flops_per_cube": 80,    # TFLOPS
    "hbm_bw_per_cube": 2.5,      # TB/s
    "d2d_pj_per_bit": 0.38,
    "static_ratio": 0.1,
}

RUBIN_DEFAULT = {
    "n_hbm_cubes": 8,
    "n_cmpt_dies": 2,
    "hbm_tdp_per_cube": 75,       # W
    "cmpt_tdp_per_die": 800,      # W
    "peak_flops_per_die": 8750,   # TFLOPS  (17500 / 2)
    "hbm_bw_per_cube": 2.75,     # TB/s    (22 / 8)
    "d2d_pj_per_bit": 0,
    "static_ratio": 0.1,
}

OURS_STRATEGIES = ["HMP_reo", "hmp_reo_new", "hmp", "tp16"]
N_PHYSICAL_CUBES = 16


# ---------------------------------------------------------------------------
# Core power functions
# ---------------------------------------------------------------------------

DYNAMIC_POWER_SCALE = 0.9  # scale factor for dynamic power (tdp * utilization)


def _component_power(tdp, utilization, static_ratio):
    return static_ratio * tdp + DYNAMIC_POWER_SCALE * tdp * utilization


def calc_power_ours(agg, cfg):
    """Power estimate for one strategy running on Ours architecture (16 cubes).

    Parameters
    ----------
    agg : dict   -- aggregate_metrics from hybrid JSON entry
    cfg : dict   -- architecture config (OURS_DEFAULT or overridden)

    Returns dict with full power breakdown.
    """
    n = cfg["n_cubes"]
    time_ns = agg["total_time_ns"]
    time_s = time_ns * 1e-9

    hbm_bytes_per_cube = agg["hbm_read_bytes_per_cube"]
    flops_per_cube = agg["compute_flops_per_cube"]
    d2d_bytes_total = agg["d2d_link_transfer_bytes_all_cubes"]

    peak_hbm_bw = cfg["hbm_bw_per_cube"] * 1e12          # bytes/s
    peak_flops = cfg["peak_flops_per_cube"] * 1e12        # FLOP/s

    hbm_util = (hbm_bytes_per_cube / time_s) / peak_hbm_bw if time_s > 0 else 0
    cmpt_util = (flops_per_cube / time_s) / peak_flops if time_s > 0 else 0

    hbm_util = min(hbm_util, 1.0)
    cmpt_util = min(cmpt_util, 1.0)

    sr = cfg["static_ratio"]
    hbm_power_per_cube = _component_power(cfg["hbm_tdp_per_cube"], hbm_util, sr)
    cmpt_power_per_cube = _component_power(cfg["cmpt_tdp_per_cube"], cmpt_util, sr)

    hbm_power_total = n * hbm_power_per_cube
    cmpt_power_total = n * cmpt_power_per_cube

    d2d_bits = d2d_bytes_total * 8
    d2d_energy_j = d2d_bits * cfg["d2d_pj_per_bit"] * 1e-12
    d2d_power = d2d_energy_j / time_s if time_s > 0 else 0

    total_power = hbm_power_total + cmpt_power_total + d2d_power
    energy_nj = total_power * time_s * 1e9

    tdp_total = n * (cfg["hbm_tdp_per_cube"] + cfg["cmpt_tdp_per_cube"])

    return {
        "arch": "ours",
        "n_cubes": n,
        "time_ns": time_ns,
        "hbm_util_per_cube": round(hbm_util, 6),
        "cmpt_util_per_cube": round(cmpt_util, 6),
        "hbm_power_per_cube_w": round(hbm_power_per_cube, 4),
        "cmpt_power_per_cube_w": round(cmpt_power_per_cube, 4),
        "hbm_power_total_w": round(hbm_power_total, 4),
        "cmpt_power_total_w": round(cmpt_power_total, 4),
        "d2d_energy_nj": round(d2d_energy_j * 1e9, 4),
        "d2d_power_w": round(d2d_power, 4),
        "total_power_w": round(total_power, 4),
        "energy_per_token_nj": round(energy_nj, 4),
        "tdp_total_w": tdp_total,
    }


def calc_power_rubin(agg, cfg):
    """Power estimate for Rubin architecture (2 compute die + 8 HBM cubes).

    For Rubin, agg comes from the 'rubin' strategy where n_cubes=1 and
    per_cube values represent the ENTIRE chip.
    """
    time_ns = agg["total_time_ns"]
    time_s = time_ns * 1e-9

    n_hbm = cfg["n_hbm_cubes"]
    n_cmpt = cfg["n_cmpt_dies"]

    hbm_bytes_total = agg["hbm_read_bytes_per_cube"]
    flops_total = agg["compute_flops_per_cube"]

    hbm_bytes_per_cube = hbm_bytes_total / n_hbm
    flops_per_die = flops_total / n_cmpt

    peak_hbm_bw = cfg["hbm_bw_per_cube"] * 1e12
    peak_flops = cfg["peak_flops_per_die"] * 1e12

    hbm_util = (hbm_bytes_per_cube / time_s) / peak_hbm_bw if time_s > 0 else 0
    cmpt_util = (flops_per_die / time_s) / peak_flops if time_s > 0 else 0

    hbm_util = min(hbm_util, 1.0)
    cmpt_util = min(cmpt_util, 1.0)

    sr = cfg["static_ratio"]
    hbm_power_per_cube = _component_power(cfg["hbm_tdp_per_cube"], hbm_util, sr)
    cmpt_power_per_die = _component_power(cfg["cmpt_tdp_per_die"], cmpt_util, sr)

    hbm_power_total = n_hbm * hbm_power_per_cube
    cmpt_power_total = n_cmpt * cmpt_power_per_die

    d2d_power = 0.0

    total_power = hbm_power_total + cmpt_power_total + d2d_power
    energy_nj = total_power * time_s * 1e9

    tdp_total = n_hbm * cfg["hbm_tdp_per_cube"] + n_cmpt * cfg["cmpt_tdp_per_die"]

    return {
        "arch": "rubin",
        "n_hbm_cubes": n_hbm,
        "n_cmpt_dies": n_cmpt,
        "time_ns": time_ns,
        "hbm_util_per_cube": round(hbm_util, 6),
        "cmpt_util_per_die": round(cmpt_util, 6),
        "hbm_power_per_cube_w": round(hbm_power_per_cube, 4),
        "cmpt_power_per_die_w": round(cmpt_power_per_die, 4),
        "hbm_power_total_w": round(hbm_power_total, 4),
        "cmpt_power_total_w": round(cmpt_power_total, 4),
        "d2d_power_w": 0.0,
        "total_power_w": round(total_power, 4),
        "energy_per_token_nj": round(energy_nj, 4),
        "tdp_total_w": tdp_total,
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
    parser.add_argument("--ours-hbm-tdp", type=float, default=OURS_DEFAULT["hbm_tdp_per_cube"],
                        help="Ours: HBM TDP per cube in W (default: %(default)s)")
    parser.add_argument("--ours-cmpt-tdp", type=float, default=OURS_DEFAULT["cmpt_tdp_per_cube"],
                        help="Ours: Compute TDP per cube in W (default: %(default)s)")
    parser.add_argument("--ours-hbm-bw", type=float, default=OURS_DEFAULT["hbm_bw_per_cube"],
                        help="Ours: HBM BW per cube in TB/s (default: %(default)s)")
    parser.add_argument("--ours-n-cubes", type=int, default=OURS_DEFAULT["n_cubes"],
                        help="Ours: number of physical cubes (default: %(default)s)")
    parser.add_argument("--rubin-peak-flops", type=float,
                        default=RUBIN_DEFAULT["peak_flops_per_die"] * RUBIN_DEFAULT["n_cmpt_dies"],
                        help="Rubin: total peak TFLOPS (default: %(default)s)")
    parser.add_argument("--rubin-hbm-bw", type=float,
                        default=RUBIN_DEFAULT["hbm_bw_per_cube"] * RUBIN_DEFAULT["n_hbm_cubes"],
                        help="Rubin: total HBM BW in TB/s (default: %(default)s)")
    parser.add_argument("--rubin-cmpt-tdp", type=float, default=RUBIN_DEFAULT["cmpt_tdp_per_die"],
                        help="Rubin: compute die TDP in W (default: %(default)s)")
    parser.add_argument("--rubin-hbm-tdp", type=float, default=RUBIN_DEFAULT["hbm_tdp_per_cube"],
                        help="Rubin: HBM cube TDP in W (default: %(default)s)")
    parser.add_argument("--rubin-n-hbm", type=int, default=RUBIN_DEFAULT["n_hbm_cubes"],
                        help="Rubin: number of HBM cubes (default: %(default)s)")
    parser.add_argument("--rubin-n-cmpt", type=int, default=RUBIN_DEFAULT["n_cmpt_dies"],
                        help="Rubin: number of compute dies (default: %(default)s)")
    parser.add_argument("--d2d-pj-per-bit", type=float, default=OURS_DEFAULT["d2d_pj_per_bit"],
                        help="D2D energy in pJ/bit (default: %(default)s)")
    parser.add_argument("--static-ratio", type=float, default=OURS_DEFAULT["static_ratio"],
                        help="Static power ratio (default: %(default)s)")
    parser.add_argument("-o", "--output", type=str, default="",
                        help="Output JSON path (default: auto)")
    args = parser.parse_args()

    ours_cfg = {
        "n_cubes": args.ours_n_cubes,
        "hbm_tdp_per_cube": args.ours_hbm_tdp,
        "cmpt_tdp_per_cube": args.ours_cmpt_tdp,
        "peak_flops_per_cube": args.ours_peak_flops,
        "hbm_bw_per_cube": args.ours_hbm_bw,
        "d2d_pj_per_bit": args.d2d_pj_per_bit,
        "static_ratio": args.static_ratio,
    }
    rubin_cfg = {
        "n_hbm_cubes": args.rubin_n_hbm,
        "n_cmpt_dies": args.rubin_n_cmpt,
        "hbm_tdp_per_cube": args.rubin_hbm_tdp,
        "cmpt_tdp_per_die": args.rubin_cmpt_tdp,
        "peak_flops_per_die": args.rubin_peak_flops / args.rubin_n_cmpt,
        "hbm_bw_per_cube": args.rubin_hbm_bw / args.rubin_n_hbm,
        "d2d_pj_per_bit": 0,
        "static_ratio": args.static_ratio,
    }

    data = load_hybrid(args.data)
    strategies_data = data.get("strategies", {})

    all_strategies = OURS_STRATEGIES + ["rubin"]
    result = {
        "_type": "power_estimation",
        "_description": "GQA single-layer power model: HBM + Compute + D2D breakdown",
        "_units": {
            "power": "W (watts)",
            "energy": "nJ (nanojoules per token)",
            "time": "ns",
            "utilization": "fraction (0-1)",
        },
        "metadata": {
            "source_data": args.data,
            "power_formula": "P_component = static_ratio * TDP + TDP * utilization",
            "total_power_formula": "P_total = P_hbm + P_cmpt + P_d2d",
            "d2d_power_formula": "P_d2d = d2d_bits * pJ_per_bit / time",
            "workflow": [
                "1. roofline_gqa_calc.py  → reports/roofline/*.json  (analytical compute + comm estimates)",
                "2. collect_gqa_data.py   → reports/astrasim/*.json  (AstraSim simulation data)",
                "3. merge_gqa_results.py  → reports/hybrid/*.json    (merged hybrid estimates with aggregate_metrics)",
                "4. power_model.py        → reports/power/*.json     (THIS FILE: power estimation from hybrid data)",
            ],
            "ours_config": ours_cfg,
            "rubin_config": rubin_cfg,
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
            agg = entry["aggregate_metrics"]

            if strat == "rubin":
                pw = calc_power_rubin(agg, rubin_cfg)
            else:
                if strat == "tp16":
                    agg = _fix_tp16_per_cube(agg)
                pw = calc_power_ours(agg, ours_cfg)

            pw["seq"] = entry["seq"]
            pw["hybrid_wall_ns"] = entry.get("hybrid_wall_ns", agg["total_time_ns"])

            cb = entry.get("compute_breakdown", {})
            qkv_ns = cb.get("Proj_QKV_roofline_mem_ns", 0) or entry.get("hybrid_Proj_QKV_ns", 0)
            attn_ns = cb.get("attn_roofline_fused_ns", 0) or entry.get("hybrid_attn_ns", 0)
            projo_ns = cb.get("Proj_O_roofline_mem_ns", 0) or entry.get("hybrid_Proj_O_ns", 0)
            total_gpu_ns = qkv_ns + attn_ns + projo_ns
            if total_gpu_ns > 0:
                hbm_total = pw["hbm_power_total_w"]
                pw["hbm_breakdown"] = {
                    "Proj_QKV_hbm_w": round(hbm_total * qkv_ns / total_gpu_ns, 4),
                    "Attn_hbm_w": round(hbm_total * attn_ns / total_gpu_ns, 4),
                    "Proj_O_hbm_w": round(hbm_total * projo_ns / total_gpu_ns, 4),
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

    # TDP summary
    ours_tdp = ours_cfg["n_cubes"] * (ours_cfg["hbm_tdp_per_cube"] + ours_cfg["cmpt_tdp_per_cube"])
    rubin_tdp = rubin_cfg["n_hbm_cubes"] * rubin_cfg["hbm_tdp_per_cube"] + \
                rubin_cfg["n_cmpt_dies"] * rubin_cfg["cmpt_tdp_per_die"]
    print(f"  Ours TDP:  {ours_tdp:.0f}W  ({ours_cfg['n_cubes']} cubes × "
          f"({ours_cfg['hbm_tdp_per_cube']}W HBM + {ours_cfg['cmpt_tdp_per_cube']}W Cmpt))")
    print(f"  Rubin TDP: {rubin_tdp:.0f}W  ({rubin_cfg['n_hbm_cubes']}×{rubin_cfg['hbm_tdp_per_cube']}W HBM"
          f" + {rubin_cfg['n_cmpt_dies']}×{rubin_cfg['cmpt_tdp_per_die']}W Cmpt)")
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
    for comp, key in [("HBM Power (W)", "hbm_power_w"),
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
    for pe_strat in header_strats:
        sd = result["strategies"].get(pe_strat, {}).get("data", [])
        pe = next((p for p in sd if p["seq"] == target_seq), sd[-1] if sd else None)
        if pe is None:
            continue
    hbm_line = f"{'HBM util':<20}"
    cmpt_line = f"{'Cmpt util':<20}"
    for s in header_strats:
        sd = result["strategies"].get(s, {}).get("data", [])
        pe = next((p for p in sd if p["seq"] == target_seq), sd[-1] if sd else None)
        if pe:
            hbm_line += f" {pe['hbm_util_per_cube']:>{col_w}.4%}"
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
