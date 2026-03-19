#!/usr/bin/env python3
"""Query power JSON and print formatted tables.

Usage:
    python3 query_power.py --power reports/power/gqa_hybrid_merged_80T_split4_bw1500_power.json \
                           --hybrid reports/hybrid/gqa_hybrid_merged_80T_split4_bw1500.json

    # Only power JSON (no data-volume columns)
    python3 query_power.py --power reports/power/gqa_hybrid_merged_80T_split4_bw1500_power.json

    # Filter seq range
    python3 query_power.py --power ... --hybrid ... --seq-min 4096 --seq-max 65536

    # Select strategies
    python3 query_power.py --power ... --strategies HMP_reo hmp_reo_new tp16
"""
import argparse
import json
import sys
from pathlib import Path

ALL_STRATS = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]
STRAT_LABELS = {
    "HMP_reo": "HMP_reo",
    "hmp_reo_new": "HMP_reo_new",
    "hmp": "HMP",
    "tp16": "TP16",
    "rubin": "Rubin",
}


def _seq_label(s):
    if s >= 1048576:
        return f"{s // 1048576}M"
    if s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def _fmt_bytes(b):
    if b >= 1e9:
        return f"{b / 1e9:.2f} GB"
    if b >= 1e6:
        return f"{b / 1e6:.2f} MB"
    if b >= 1e3:
        return f"{b / 1e3:.2f} KB"
    return f"{b:.0f} B"


def _fmt_flops(f):
    if f >= 1e12:
        return f"{f / 1e12:.2f} TFLOP"
    if f >= 1e9:
        return f"{f / 1e9:.2f} GFLOP"
    if f >= 1e6:
        return f"{f / 1e6:.2f} MFLOP"
    return f"{f:.0f} FLOP"


def _build_lookup(data, strats):
    """strategy -> seq -> entry"""
    lookup = {}
    for s in strats:
        if s not in data.get("strategies", {}):
            continue
        lookup[s] = {}
        for e in data["strategies"][s]["data"]:
            lookup[s][e["seq"]] = e
    return lookup


def _print_per_strategy(pw_lookup, hy_lookup, strats, seqs, has_hybrid):
    """Print detailed per-strategy tables."""
    for strat in strats:
        if strat not in pw_lookup:
            continue
        label = STRAT_LABELS.get(strat, strat)
        is_ours = strat != "rubin"
        arch = "16 cubes (Ours)" if is_ours else "8 HBM + 2 Cmpt dies (Rubin)"

        print(f"\n{'=' * 180}")
        print(f"  策略: {label}  ({arch})")
        print(f"{'=' * 180}")

        # Main table header
        cols = [
            ("Seq", 8), ("Wall(ns)", 10), ("Total(W)", 10), ("HBM(W)", 10),
            ("Cmpt(W)", 10), ("D2D(W)", 10), ("E/tok(nJ)", 14),
        ]
        if has_hybrid:
            cols += [
                ("HBM Bytes/cube", 16), ("FLOPS/cube", 16),
                ("D2D Bytes", 16),
            ]
        cols += [("HBM%", 6), ("Cmpt%", 6)]

        hdr = " | ".join(f"{name:>{w}}" for name, w in cols)
        print(hdr)
        print("-" * len(hdr))

        for seq in seqs:
            pe = pw_lookup[strat].get(seq)
            if pe is None:
                continue
            he = hy_lookup.get(strat, {}).get(seq)
            agg = he["aggregate_metrics"] if he else {}

            hbm_util = pe.get("hbm_util_per_cube", 0)
            cmpt_util = pe.get("cmpt_util_per_cube", pe.get("cmpt_util_per_die", 0))

            parts = [
                f"{_seq_label(seq):>8}",
                f"{pe['time_ns']:>10.1f}",
                f"{pe['total_power_w']:>10.2f}",
                f"{pe['hbm_power_total_w']:>10.2f}",
                f"{pe['cmpt_power_total_w']:>10.2f}",
                f"{pe.get('d2d_power_w', 0):>10.4f}",
                f"{pe['energy_per_token_nj']:>14.2f}",
            ]
            if has_hybrid:
                parts += [
                    f"{_fmt_bytes(agg.get('hbm_read_bytes_per_cube', 0)):>16}",
                    f"{_fmt_flops(agg.get('compute_flops_per_cube', 0)):>16}",
                    f"{_fmt_bytes(agg.get('d2d_link_transfer_bytes_all_cubes', 0)):>16}",
                ]
            parts += [f"{hbm_util:>5.1%}", f"{cmpt_util:>5.1%}"]
            print(" | ".join(parts))

        # HBM breakdown sub-table
        has_bd = any(
            "hbm_breakdown" in pw_lookup[strat].get(s, {})
            for s in seqs
        )
        if has_bd:
            print(f"\n  HBM Power Breakdown (by module):")
            bd_hdr = (
                f"{'Seq':>8} | {'HBM Total(W)':>12} | {'Proj_QKV(W)':>12} | "
                f"{'Attn(W)':>12} | {'Proj_O(W)':>12} | "
                f"{'QKV%':>7} | {'Attn%':>7} | {'ProjO%':>7}"
            )
            print(bd_hdr)
            print("-" * len(bd_hdr))
            for seq in seqs:
                pe = pw_lookup[strat].get(seq)
                if pe is None:
                    continue
                hb = pe.get("hbm_breakdown", {})
                if not hb:
                    print(
                        f"{_seq_label(seq):>8} | {pe['hbm_power_total_w']:>12.2f} | "
                        f"{'N/A':>12} | {'N/A':>12} | {'N/A':>12} | "
                        f"{'N/A':>7} | {'N/A':>7} | {'N/A':>7}"
                    )
                    continue
                print(
                    f"{_seq_label(seq):>8} | {pe['hbm_power_total_w']:>12.2f} | "
                    f"{hb['Proj_QKV_hbm_w']:>12.2f} | {hb['Attn_hbm_w']:>12.2f} | "
                    f"{hb['Proj_O_hbm_w']:>12.2f} | "
                    f"{hb['Proj_QKV_frac']:>6.1%} | {hb['Attn_frac']:>6.1%} | "
                    f"{hb['Proj_O_frac']:>6.1%}"
                )


def _print_cross_strategy(title, pw_lookup, hy_lookup, strats, seqs, extractor):
    """Print a cross-strategy comparison table."""
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    col_w = 16
    hdr = f"{'Seq':>8}"
    avail = [s for s in strats if s in pw_lookup]
    for s in avail:
        hdr += f" | {STRAT_LABELS.get(s, s):>{col_w}}"
    print(hdr)
    print("-" * len(hdr))

    for seq in seqs:
        line = f"{_seq_label(seq):>8}"
        for s in avail:
            val = extractor(s, seq, pw_lookup, hy_lookup)
            if val is None:
                line += f" | {'—':>{col_w}}"
            else:
                line += f" | {val:>{col_w}}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Query power JSON → formatted tables")
    parser.add_argument("--power", required=True, help="Path to power estimation JSON")
    parser.add_argument("--hybrid", default="", help="Path to hybrid merged JSON (adds data-volume columns)")
    parser.add_argument("--strategies", nargs="+", default=None,
                        help=f"Strategies to show (default: all). Choices: {ALL_STRATS}")
    parser.add_argument("--seq-min", type=int, default=0, help="Min seq length to show")
    parser.add_argument("--seq-max", type=int, default=10**9, help="Max seq length to show")
    parser.add_argument("--no-detail", action="store_true", help="Skip per-strategy detail tables")
    parser.add_argument("--no-cross", action="store_true", help="Skip cross-strategy comparison tables")
    args = parser.parse_args()

    with open(args.power) as f:
        pw_data = json.load(f)

    hy_data = None
    if args.hybrid:
        with open(args.hybrid) as f:
            hy_data = json.load(f)

    strats = args.strategies or ALL_STRATS
    strats = [s for s in strats if s in pw_data.get("strategies", {})]
    if not strats:
        print("ERROR: no matching strategies found", file=sys.stderr)
        sys.exit(1)

    first = strats[0]
    seqs = sorted(set(
        e["seq"] for e in pw_data["strategies"][first]["data"]
        if args.seq_min <= e["seq"] <= args.seq_max
    ))
    if not seqs:
        print("ERROR: no seq values in range", file=sys.stderr)
        sys.exit(1)

    pw_lookup = _build_lookup(pw_data, strats)
    hy_lookup = _build_lookup(hy_data, strats) if hy_data else {}
    has_hybrid = bool(hy_lookup)

    print("=" * 120)
    print(f"  GQA 单层功耗数据  ({Path(args.power).stem})")
    print(f"  Seq range: {_seq_label(seqs[0])} .. {_seq_label(seqs[-1])}  |  "
          f"Strategies: {', '.join(STRAT_LABELS.get(s, s) for s in strats)}")
    print("=" * 120)

    # Metadata
    meta = pw_data.get("metadata", {})
    ours_cfg = meta.get("ours_config", {})
    rubin_cfg = meta.get("rubin_config", {})
    if ours_cfg:
        ours_tdp = ours_cfg.get("n_cubes", 16) * (
            ours_cfg.get("hbm_tdp_per_cube", 75) + ours_cfg.get("cmpt_tdp_per_cube", 15)
        )
        print(f"  Ours TDP:  {ours_tdp:.0f}W  "
              f"({ours_cfg.get('n_cubes', 16)} cubes × "
              f"({ours_cfg.get('hbm_tdp_per_cube', 75)}W HBM + "
              f"{ours_cfg.get('cmpt_tdp_per_cube', 15)}W Cmpt))")
    if rubin_cfg:
        rubin_tdp = (rubin_cfg.get("n_hbm_cubes", 8) * rubin_cfg.get("hbm_tdp_per_cube", 75)
                     + rubin_cfg.get("n_cmpt_dies", 2) * rubin_cfg.get("cmpt_tdp_per_die", 800))
        print(f"  Rubin TDP: {rubin_tdp:.0f}W  "
              f"({rubin_cfg.get('n_hbm_cubes', 8)}×{rubin_cfg.get('hbm_tdp_per_cube', 75)}W HBM"
              f" + {rubin_cfg.get('n_cmpt_dies', 2)}×{rubin_cfg.get('cmpt_tdp_per_die', 800)}W Cmpt)")

    # --- Per-strategy detail ---
    if not args.no_detail:
        _print_per_strategy(pw_lookup, hy_lookup, strats, seqs, has_hybrid)

    # --- Cross-strategy comparisons ---
    if not args.no_cross:
        # Total power
        _print_cross_strategy(
            "跨策略对比: Total Power (W)", pw_lookup, hy_lookup, strats, seqs,
            lambda s, seq, pw, hy: (
                f"{pw[s][seq]['total_power_w']:.2f}" if seq in pw.get(s, {}) else None
            ),
        )

        # Energy per token
        _print_cross_strategy(
            "跨策略对比: Energy per Token (nJ)", pw_lookup, hy_lookup, strats, seqs,
            lambda s, seq, pw, hy: (
                f"{pw[s][seq]['energy_per_token_nj']:,.0f}" if seq in pw.get(s, {}) else None
            ),
        )

        if has_hybrid:
            # HBM read bytes per cube
            _print_cross_strategy(
                "跨策略对比: HBM Read Bytes per Cube", pw_lookup, hy_lookup, strats, seqs,
                lambda s, seq, pw, hy: (
                    _fmt_bytes(hy[s][seq]["aggregate_metrics"]["hbm_read_bytes_per_cube"])
                    if seq in hy.get(s, {}) else None
                ),
            )

            # D2D transfer bytes
            _print_cross_strategy(
                "跨策略对比: D2D Transfer Bytes (All Cubes)", pw_lookup, hy_lookup, strats, seqs,
                lambda s, seq, pw, hy: (
                    _fmt_bytes(hy[s][seq]["aggregate_metrics"]["d2d_link_transfer_bytes_all_cubes"])
                    if seq in hy.get(s, {}) else None
                ),
            )

            # Compute FLOPS per cube
            _print_cross_strategy(
                "跨策略对比: Compute FLOPS per Cube", pw_lookup, hy_lookup, strats, seqs,
                lambda s, seq, pw, hy: (
                    _fmt_flops(hy[s][seq]["aggregate_metrics"]["compute_flops_per_cube"])
                    if seq in hy.get(s, {}) else None
                ),
            )

    print()


if __name__ == "__main__":
    main()
