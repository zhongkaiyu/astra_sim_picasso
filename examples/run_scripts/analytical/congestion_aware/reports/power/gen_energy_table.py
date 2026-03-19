#!/usr/bin/env python3
"""Generate comprehensive energy table: static, dynamic, and dynamic breakdown.
Answers: Why does our (TDP 1440W) energy advantage vs Rubin (TDP 2200W) not look larger?
"""
import json
import sys
from pathlib import Path

POWER_JSON = Path(__file__).parent / "gqa_hybrid_merged_80T_split4_bw1500_power.json"

STRAT_LABELS = {
    "HMP_reo": "HMP_RO",
    "hmp_reo_new": "HMP_RO_new",
    "hmp": "HMP",
    "tp16": "TP16",
    "rubin": "Rubin",
}


def seq_label(s):
    if s >= 1048576:
        return f"{s // 1048576}M"
    if s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def main():
    with open(POWER_JSON) as f:
        data = json.load(f)

    strategies_data = data["strategies"]
    meta = data["metadata"]
    ours_cfg = meta.get("ours_config", {})
    rubin_cfg = meta.get("rubin_config", {})

    # Static power (W)
    ours_tdp = ours_cfg.get("n_cubes", 16) * (
        ours_cfg.get("hbm_tdp_per_cube", 75) + ours_cfg.get("cmpt_tdp_per_cube", 15)
    )
    rubin_tdp = (
        rubin_cfg.get("n_hbm_cubes", 8) * rubin_cfg.get("hbm_tdp_per_cube", 75)
        + rubin_cfg.get("n_cmpt_dies", 2) * rubin_cfg.get("cmpt_tdp_per_die", 800)
    )
    sr = ours_cfg.get("static_ratio", 0.1)
    p_static_ours = ours_tdp * sr
    p_static_rubin = rubin_tdp * sr

    # Per-component static (for dynamic breakdown)
    static_hbm_ours = 16 * 75 * sr
    static_cmpt_ours = 16 * 15 * sr
    static_hbm_rubin = 8 * 75 * sr
    static_cmpt_rubin = 2 * 800 * sr

    strats = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]
    seqs = sorted(e["seq"] for e in strategies_data["HMP_reo"]["data"])

    # Collect all rows
    rows = []
    for seq in seqs:
        for sk in strats:
            if sk not in strategies_data:
                continue
            entries = [e for e in strategies_data[sk]["data"] if e["seq"] == seq]
            if not entries:
                continue
            e = entries[0]
            time_ns = e["time_ns"]
            time_s = time_ns * 1e-9

            p_total = e["total_power_w"]
            p_static = p_static_ours if sk != "rubin" else p_static_rubin
            p_dynamic = p_total - p_static

            e_total = e["energy_per_token_nj"]
            e_static = p_static * time_ns  # W * ns = nJ (1W = 1e9 nJ/s, 1s = 1e9 ns)
            e_dynamic = p_dynamic * time_ns

            # Dynamic breakdown: HBM_dyn, Cmpt_dyn, D2D
            hbm_w = e.get("hbm_power_total_w", 0)
            cmpt_w = e.get("cmpt_power_total_w", 0)
            d2d_nj = e.get("d2d_energy_nj", 0)

            if sk == "rubin":
                shbm, scmpt = static_hbm_rubin, static_cmpt_rubin
            else:
                shbm, scmpt = static_hbm_ours, static_cmpt_ours

            hbm_dyn_w = max(0, hbm_w - shbm)
            cmpt_dyn_w = max(0, cmpt_w - scmpt)
            e_hbm_dyn = hbm_dyn_w * time_ns  # W * ns = nJ
            e_cmpt_dyn = cmpt_dyn_w * time_ns
            e_d2d = d2d_nj

            # HBM sub-breakdown (Proj_QKV, Attn, Proj_O)
            hb = e.get("hbm_breakdown", {})
            if hb:
                qkv_w = hb.get("Proj_QKV_hbm_w", 0)
                attn_w = hb.get("Attn_hbm_w", 0)
                projo_w = hb.get("Proj_O_hbm_w", 0)
                # Fraction of HBM power that is dynamic (approximate)
                hbm_dyn_frac = hbm_dyn_w / hbm_w if hbm_w > 0 else 0
                e_qkv_dyn = qkv_w * hbm_dyn_frac * time_ns
                e_attn_dyn = attn_w * hbm_dyn_frac * time_ns
                e_projo_dyn = projo_w * hbm_dyn_frac * time_ns
            else:
                e_qkv_dyn = e_attn_dyn = e_projo_dyn = 0

            rows.append({
                "seq": seq,
                "strat": sk,
                "e_total": e_total,
                "e_static": e_static,
                "e_dynamic": e_dynamic,
                "e_hbm_dyn": e_hbm_dyn,
                "e_cmpt_dyn": e_cmpt_dyn,
                "e_d2d": e_d2d,
                "e_qkv_dyn": e_qkv_dyn,
                "e_attn_dyn": e_attn_dyn,
                "e_projo_dyn": e_projo_dyn,
                "time_ns": time_ns,
            })

    # Build lookup: (seq, strat) -> row
    lookup = {(r["seq"], r["strat"]): r for r in rows}

    # ----- Print Table 1: E_static, E_dynamic, E_total by Seq x Strategy -----
    print("\n" + "=" * 140)
    print("  bw1500 能耗数据总表 | TDP: Ours=1440W, Rubin=2200W | P_static: Ours=144W, Rubin=220W")
    print("=" * 140)

    print("\n【表 1】 各 Seq 下五策略 动态能耗 / 静态能耗 / 总能耗 (nJ)")
    print("-" * 140)
    hdr = f"{'Seq':>8}"
    for sk in strats:
        if sk not in strategies_data:
            continue
        hdr += f" | {STRAT_LABELS.get(sk, sk):>12} E_dyn | {STRAT_LABELS.get(sk, sk):>12} E_sta | {STRAT_LABELS.get(sk, sk):>12} E_tot"
    print(hdr)
    print("-" * 140)

    for seq in seqs:
        line = f"{seq_label(seq):>8}"
        for sk in strats:
            if sk not in strategies_data:
                continue
            r = lookup.get((seq, sk))
            if r:
                line += f" | {r['e_dynamic']:>12,.0f} | {r['e_static']:>12,.0f} | {r['e_total']:>12,.0f}"
            else:
                line += " | " + " — " * 3
        print(line)

    # ----- Table 2: Dynamic energy breakdown (HBM/Cmpt/D2D) -----
    print("\n【表 2】 动态能耗 Breakdown: HBM | Cmpt | D2D (nJ)")
    print("-" * 140)
    hdr2 = f"{'Seq':>8}"
    for sk in strats:
        if sk not in strategies_data:
            continue
        lbl = STRAT_LABELS.get(sk, sk)
        hdr2 += f" | {lbl}_HBM | {lbl}_Cmpt | {lbl}_D2D"
    print(hdr2)
    print("-" * 140)

    for seq in seqs:
        line = f"{seq_label(seq):>8}"
        for sk in strats:
            if sk not in strategies_data:
                continue
            r = lookup.get((seq, sk))
            if r:
                line += f" | {r['e_hbm_dyn']:>8,.0f} | {r['e_cmpt_dyn']:>8,.0f} | {r['e_d2d']:>8,.0f}"
            else:
                line += " | " + " — " * 3
        print(line)

    # ----- Table 3: HBM dynamic sub-breakdown (Proj_QKV / Attn / Proj_O) -----
    print("\n【表 3】 动态能耗 HBM 子项: Proj_QKV | Attn | Proj_O (nJ)")
    print("-" * 120)
    hdr3 = f"{'Seq':>8}"
    for sk in strats:
        if sk not in strategies_data:
            continue
        lbl = STRAT_LABELS.get(sk, sk)
        hdr3 += f" | {lbl}_QKV | {lbl}_Attn | {lbl}_ProjO"
    print(hdr3)
    print("-" * 120)

    for seq in seqs:
        line = f"{seq_label(seq):>8}"
        for sk in strats:
            if sk not in strategies_data:
                continue
            r = lookup.get((seq, sk))
            if r:
                line += f" | {r['e_qkv_dyn']:>8,.0f} | {r['e_attn_dyn']:>8,.0f} | {r['e_projo_dyn']:>8,.0f}"
            else:
                line += " | " + " — " * 3
        print(line)

    # ----- Analysis: Why is our energy advantage vs Rubin not larger? -----
    print("\n" + "=" * 140)
    print("  【能耗优势分析】 为何 TDP 1440W vs 2200W 下能耗优势不够显著？")
    print("=" * 140)

    # Compare best ours (hmp) vs Rubin at key seq points
    key_seqs = [1024, 4096, 65536, 1048576]
    print("\n  最佳我方策略 (HMP) vs Rubin 能耗对比:")
    print(f"  {'Seq':>8} | {'HMP E_tot(nJ)':>14} | {'Rubin E_tot(nJ)':>14} | {'能耗比(HMP/Rubin)':>16} | {'理论TDP比':>12}")
    print("  " + "-" * 80)
    for seq in key_seqs:
        hmp = lookup.get((seq, "hmp"))
        rubin = lookup.get((seq, "rubin"))
        if hmp and rubin:
            ratio = hmp["e_total"] / rubin["e_total"]
            tdp_ratio = ours_tdp / rubin_tdp
            print(f"  {seq_label(seq):>8} | {hmp['e_total']:>14,.0f} | {rubin['e_total']:>14,.0f} | {ratio:>16.3f} (HMP/Rubin) | TDP {tdp_ratio:.2f}x")
        else:
            print(f"  {seq_label(seq):>8} | — | — | —")

    print("""
  【关键发现】

  1. 静态能耗占比差异:
     - Ours P_static=144W (10% TDP), Rubin P_static=220W (10% TDP)
     - 静态能耗 E_static = P_static × time，与 wall time 成正比
     - Rubin wall time 更短 (单核算力强) → 静态能耗绝对量不一定更大

  2. 动态功率与利用率:
     - 我方策略 (HMP/hmp_reo_new) 因 wall time 更长，P_total 更高 (1000–1400W)
     - Rubin 因 time 短，P_total 仅 ~765–815W，但 实际功耗/时间 乘积决定能耗
     - 能耗 = P × t；若 t 差异大，能耗不一定与 TDP 成比例

  3. 核心矛盾:
     - 我方: 16 cube 分布式，通信/调度开销 → wall time 长 → 总能耗 = 高功率 × 长时间
     - Rubin: 2 大算力核 + 8 HBM，集成度高 → wall time 短 → 总能耗 = 较低功率 × 短时间
     - TDP 优势 (1440 vs 2200) 体现在「峰值功耗上限」，但 实际运行 时我方平均功率接近甚至超过 Rubin
       因我方 time 更长、利用率（尤其是 HBM）更高

  4. 动态能耗 breakdown:
     - 动态能耗 ≈ HBM + Cmpt + D2D
     - 我方 HBM 利用率 85–99%，Rubin 71–73%
     - 我方 HBM 动态能耗 dominate；Rubin 的 Cmpt 占比更高
     - 长 seq 下 Attn (KV cache) 动态能耗在我方显著增长
""")


if __name__ == "__main__":
    main()
