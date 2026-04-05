#!/usr/bin/env python3
"""Analyze attn_scaling.ncu-rep — attention-only profiling (qkv→rope→attn→o_proj).

Produces per-stage, per-seq 3D utilization table:
  HBM BW%, SM Inst%, Tensor Core%, latency, effective bandwidth.

Usage:
    python analyze_attn_scaling.py                # TP=1 (default, reads attn_scaling_raw.csv)
    python analyze_attn_scaling.py --tp 1         # TP=1 (reads attn_scaling_tp1.csv)
    python analyze_attn_scaling.py --tp 2         # TP=2 (reads attn_scaling_tp2.csv)
"""
import argparse, csv, io, sys

# ── Model constants ──────────────────────────────────────────────
D_MODEL = 4096
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
D_HEAD = 128
HBM_PEAK_GBS  = 3350.0

SEQ_LENS = [1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]


def get_tp_dims(tp=1):
    num_q  = NUM_Q_HEADS  // tp
    num_kv = NUM_KV_HEADS // tp
    q_dim  = num_q  * D_HEAD
    kv_dim = num_kv * D_HEAD
    qkv_weight_gb = D_MODEL * (q_dim + 2 * kv_dim) * 2 / 1e9
    o_weight_gb   = q_dim * D_MODEL * 2 / 1e9
    return num_q, num_kv, qkv_weight_gb, o_weight_gb


def kv_cache_gb(S, tp=1):
    num_kv = NUM_KV_HEADS // tp
    return 2 * S * num_kv * D_HEAD * 2 / 1e9


def sf(v):
    try:
        v = v.replace(",", "")
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def classify(name):
    nl = name.lower()
    if "xmma_gemm" in nl or ("cutlass" in nl and "gemm" in nl):
        return "GEMM"
    if "flash_fwd_splitkv_k" in nl or ("flash_fwd_kernel" in nl and "combine" not in nl):
        return "FA_compute"
    if "flash_fwd_splitkv_c" in nl or "flash_fwd_combine" in nl:
        return "FA_combine"
    if "distribution_elementwise" in nl:
        return "RANDN"
    if "elementwise" in nl:
        return "ELEM"
    return "OTHER"


def parse(csv_path):
    """Parse long-format ncu CSV (one row per metric per kernel)."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)

        # Build per-kernel metric dicts
        kernels = {}  # id -> {kernel, metrics}
        for r in reader:
            if len(r) < 15:
                continue
            kid = int(r[0])
            if kid not in kernels:
                kernels[kid] = {"id": kid, "kernel": r[4], "metrics": {}}
            sec, mname, munit, mval = r[11], r[12], r[13], r[14]
            key = (sec, mname, munit)
            kernels[kid]["metrics"][key] = mval

    rows = []
    for kid in sorted(kernels):
        k = kernels[kid]
        m = k["metrics"]

        # Duration: prefer usecond
        dur_us = sf(m.get(("GPU Speed Of Light Throughput", "Duration", "usecond"), "0"))
        if dur_us == 0:
            dur_ms = sf(m.get(("GPU Speed Of Light Throughput", "Duration", "msecond"), "0"))
            dur_us = dur_ms * 1000

        # DRAM Throughput %
        dram_pct = sf(m.get(("GPU Speed Of Light Throughput", "DRAM Throughput", "%"), "0"))

        # Compute (SM) Throughput %
        sm_pct = sf(m.get(("GPU Speed Of Light Throughput", "Compute (SM) Throughput", "%"), "0"))

        # Effective memory throughput (GB/s)
        eff_bw = sf(m.get(("Memory Workload Analysis", "Memory Throughput", "Gbyte/second"), "0"))
        if eff_bw == 0:
            eff_bw_tb = sf(m.get(("Memory Workload Analysis", "Memory Throughput", "Tbyte/second"), "0"))
            eff_bw = eff_bw_tb * 1000

        # Estimate DRAM read from effective BW and duration
        dram_rd_gb = eff_bw * dur_us / 1e6 if dur_us > 0 else 0

        rows.append({
            "id":      kid,
            "kernel":  k["kernel"],
            "cls":     classify(k["kernel"]),
            "dram_rd": dram_rd_gb,
            "dram_rd_pct": dram_pct,
            "dram_wr": 0,
            "dur_us":  dur_us,
            "tc_pct":  0.0,  # not available in this CSV format
            "sm_pct":  sm_pct,
            "eff_bw":  eff_bw,
        })
    return rows


def group_by_seq(rows):
    """Split rows into per-seq groups using RANDN kernel clusters as boundaries.

    Pattern per seq: [warmup1 ~19 kernels] [warmup2 ~19 kernels] [profiled ~19 kernels]
    Between seqs: 4+ RANDN kernels (KV cache allocation for next seq)
    """
    # Find RANDN cluster start indices
    groups = []
    current = []
    in_randn = False

    for k in rows:
        if k["cls"] == "RANDN":
            if not in_randn and current:
                groups.append(current)
                current = []
            in_randn = True
        else:
            in_randn = False
            current.append(k)

    if current:
        groups.append(current)

    return groups


def extract_iterations(group):
    """Split a seq group into individual iterations.

    Each iteration: GEMM(qkv) → ELEM(rope) → FA_compute → FA_combine → GEMM(o_proj).
    GEMMs come in pairs; split between o_proj(even) and next qkv(odd).
    """
    # Find GEMM indices
    gemm_indices = [i for i, k in enumerate(group) if k["cls"] == "GEMM"]

    if len(gemm_indices) < 2:
        return [group] if group else []

    # Pair GEMMs: (qkv_idx, o_proj_idx) for each iteration
    iters = []
    for p in range(0, len(gemm_indices) - 1, 2):
        qkv_i = gemm_indices[p]
        o_i   = gemm_indices[p + 1]
        # Iteration spans from qkv GEMM to o_proj GEMM (inclusive)
        end = gemm_indices[p + 2] if p + 2 < len(gemm_indices) else len(group)
        iters.append(group[qkv_i:end])

    return iters


def analyze_iteration(kernels):
    """Extract per-stage metrics from one iteration's kernels."""
    stages = {}

    def make_stage(klist):
        """Build stage dict from a list of kernels."""
        main = klist[0]
        return {
            "dur_us": sum(k["dur_us"] for k in klist),
            "dram_rd_gb": sum(k["dram_rd"] for k in klist),
            "dram_wr_gb": sum(k.get("dram_wr", 0) for k in klist),
            "hbm_bw_pct": main["dram_rd_pct"],
            "sm_pct": main["sm_pct"],
            "tc_pct": main.get("tc_pct", 0),
            "eff_bw": main.get("eff_bw", 0),
        }

    # GEMMs: first = qkv_proj, second = o_proj
    gemms = [k for k in kernels if k["cls"] == "GEMM"]
    if len(gemms) >= 1:
        stages["qkv_proj"] = make_stage([gemms[0]])

    # RoPE: elementwise kernels between QKV and FA
    rope_kernels = [k for k in kernels if k["cls"] == "ELEM"]
    if rope_kernels:
        s = make_stage(rope_kernels)
        s["hbm_bw_pct"] = max(k["dram_rd_pct"] for k in rope_kernels)
        s["sm_pct"] = max(k["sm_pct"] for k in rope_kernels)
        stages["rope"] = s

    # FlashAttention
    fa_compute = [k for k in kernels if k["cls"] == "FA_compute"]
    fa_combine = [k for k in kernels if k["cls"] == "FA_combine"]
    fa_all = fa_compute + fa_combine
    if fa_all:
        stages["attn"] = make_stage(fa_all)
        if fa_compute:
            stages["  fa_splitkv"] = make_stage([fa_compute[0]])
        if fa_combine:
            stages["  fa_combine"] = make_stage([fa_combine[0]])

    if len(gemms) >= 2:
        stages["o_proj"] = make_stage([gemms[1]])

    total_dur = sum(k["dur_us"] for k in kernels if k["cls"] != "RANDN")
    return stages, total_dur


def get_eff_bw(stage):
    """Get effective bandwidth from stage dict (direct or computed)."""
    if stage.get("eff_bw", 0) > 0:
        return stage["eff_bw"]
    dur = stage.get("dur_us", 0)
    rd = stage.get("dram_rd_gb", 0)
    wr = stage.get("dram_wr_gb", 0)
    if dur <= 0:
        return 0.0
    return (rd + wr) / (dur / 1e6)  # GB/s


def bottleneck(hbm_pct, sm_pct, tc_pct):
    if hbm_pct > 30:
        return "MEM"
    if tc_pct > 5:
        return "COMPUTE"
    if sm_pct > 5:
        return "SM"
    return "LATENCY"


def main(tp_arg=None):
    tp = tp_arg if tp_arg is not None else 1
    if tp_arg is not None:
        csv_path = f"H100_results/attn_scaling_tp{tp_arg}.csv"
    else:
        csv_path = "H100_results/attn_scaling_raw.csv"

    num_q, num_kv, qkv_weight_gb, o_weight_gb = get_tp_dims(tp)

    rows = parse(csv_path)
    print(f"Loaded {len(rows)} kernels from {csv_path}")

    groups = group_by_seq(rows)
    print(f"Found {len(groups)} seq groups")

    STAGE_ORDER = ["qkv_proj", "rope", "attn", "  fa_splitkv", "  fa_combine", "o_proj"]

    # ═══ Detailed per-stage table ═══
    print()
    print("=" * 150)
    print(f"  Attention Scaling Analysis — Qwen3-235B GQA TP={tp} (D=4096, Q={num_q}, KV={num_kv}, D_HEAD=128)")
    print("  Stages: qkv_proj → rope → FlashAttn-2 → o_proj | B=1, seq_q=1 (decode)")
    print("=" * 150)

    hdr = (f"{'Seq':>8} | {'Stage':<14} | {'Time(us)':>9} | "
           f"{'HBM BW%':>8} | {'Eff BW':>9} | {'DRAM_rd':>10} | "
           f"{'SM Inst%':>8} | {'TC%':>8} | {'BN':<8}")
    print(hdr)
    print("-" * 150)

    all_results = []

    for gi, group in enumerate(groups):
        iters = extract_iterations(group)
        if not iters:
            continue

        # Use last iteration (profiled)
        profiled = iters[-1]
        stages, total_dur = analyze_iteration(profiled)

        seq = SEQ_LENS[gi] if gi < len(SEQ_LENS) else f"?{gi}"
        kv_gb = kv_cache_gb(seq, tp) if isinstance(seq, int) else 0

        all_results.append((seq, stages, total_dur))

        for sname in STAGE_ORDER:
            if sname not in stages:
                continue
            d = stages[sname]
            dur = d["dur_us"]
            bw = get_eff_bw(d)
            bn = bottleneck(d["hbm_bw_pct"], d["sm_pct"], d["tc_pct"])

            print(f"{seq:>8} | {sname:<14} | {dur:>9.1f} | "
                  f"{d['hbm_bw_pct']:>7.1f}% | {bw:>7.0f}GB/s | {d['dram_rd_gb']*1000:>8.1f}MB | "
                  f"{d['sm_pct']:>7.1f}% | {d['tc_pct']:>7.1f}% | {bn:<8}")

        # Stage sum + small ops
        heavy = sum(stages[s]["dur_us"] for s in ["qkv_proj","rope","attn","o_proj"] if s in stages)
        small = total_dur - heavy
        print(f"{seq:>8} | {'small_ops':<14} | {small:>9.1f} |")
        print(f"{seq:>8} | {'TOTAL':<14} | {total_dur:>9.1f} |"
              f"         |           | {'KV='+str(round(kv_gb*1000))+'MB':>10} |")
        print("-" * 150)

    # ═══ Summary: 3D utilization ═══
    print()
    print("=" * 120)
    print("  3-Dimensional Utilization Summary — Attention Scaling")
    print("=" * 120)
    print(f"{'Seq':>8} | {'qkv_proj':>10} {'':>4} | {'attn (FA2)':>10} {'':>4} | "
          f"{'o_proj':>10} {'':>4} | {'attn_dur':>8} | {'total':>8} | {'attn%':>6}")
    print(f"{'':>8} | {'HBM%':>5} {'SM%':>5} {'TC%':>5} | {'HBM%':>5} {'SM%':>5} {'TC%':>5} | "
          f"{'HBM%':>5} {'SM%':>5} {'TC%':>5} | {'(us)':>8} | {'(us)':>8} |")
    print("-" * 120)

    for seq, stages, total_dur in all_results:
        def g(s, f):
            return stages.get(s, {}).get(f, 0)

        attn_dur = g("attn", "dur_us")
        attn_pct = attn_dur / total_dur * 100 if total_dur > 0 else 0

        print(f"{seq:>8} | {g('qkv_proj','hbm_bw_pct'):>5.1f} {g('qkv_proj','sm_pct'):>5.1f} {g('qkv_proj','tc_pct'):>5.1f} | "
              f"{g('attn','hbm_bw_pct'):>5.1f} {g('attn','sm_pct'):>5.1f} {g('attn','tc_pct'):>5.1f} | "
              f"{g('o_proj','hbm_bw_pct'):>5.1f} {g('o_proj','sm_pct'):>5.1f} {g('o_proj','tc_pct'):>5.1f} | "
              f"{attn_dur:>8.1f} | {total_dur:>8.1f} | {attn_pct:>5.1f}%")
    print("-" * 120)

    # ═══ FlashAttention scaling detail ═══
    print()
    print("=" * 120)
    print("  FlashAttention-2 Scaling with Sequence Length")
    print(f"  KV cache: 2×S×{num_kv}heads×{D_HEAD}dim×fp16 | splitkv_kernel + combine_kernel")
    print("=" * 120)
    print(f"{'Seq':>8} | {'KV_cache':>8} | {'FA_dur':>8} | {'FA_rd':>10} | "
          f"{'Eff_BW':>9} | {'HBM%':>6} | {'SM%':>6} | {'TC%':>6} | "
          f"{'splitkv':>8} | {'combine':>8} | {'split%':>6}")
    print("-" * 120)

    for seq, stages, total_dur in all_results:
        attn = stages.get("attn", {})
        fa_s = stages.get("  fa_splitkv", {})
        fa_c = stages.get("  fa_combine", {})

        if not attn:
            continue

        kv_mb = kv_cache_gb(seq, tp) * 1000 if isinstance(seq, int) else 0
        dur = attn.get("dur_us", 0)
        rd = attn.get("dram_rd_gb", 0)
        bw = get_eff_bw(attn)
        split_dur = fa_s.get("dur_us", 0)
        comb_dur = fa_c.get("dur_us", 0)
        split_pct = split_dur / dur * 100 if dur > 0 else 0

        print(f"{seq:>8} | {kv_mb:>6.0f}MB | {dur:>6.1f}us | {rd*1000:>8.1f}MB | "
              f"{bw:>7.0f}GB/s | {attn.get('hbm_bw_pct',0):>5.1f}% | "
              f"{attn.get('sm_pct',0):>5.1f}% | {attn.get('tc_pct',0):>5.1f}% | "
              f"{split_dur:>6.1f}us | {comb_dur:>6.1f}us | {split_pct:>5.1f}%")
    print("-" * 120)

    # ═══ Stage time breakdown ═══
    print()
    print("=" * 100)
    print("  Latency Breakdown — Time per stage (us)")
    print("=" * 100)
    print(f"{'Seq':>8} | {'qkv_proj':>9} | {'rope':>8} | {'attn':>9} | "
          f"{'o_proj':>9} | {'small':>8} | {'total':>9} | {'attn%':>6}")
    print("-" * 100)

    for seq, stages, total_dur in all_results:
        def g(s):
            return stages.get(s, {}).get("dur_us", 0)

        heavy = g("qkv_proj") + g("rope") + g("attn") + g("o_proj")
        small = total_dur - heavy
        attn_pct = g("attn") / total_dur * 100 if total_dur > 0 else 0

        print(f"{seq:>8} | {g('qkv_proj'):>7.1f}us | {g('rope'):>6.1f}us | {g('attn'):>7.1f}us | "
              f"{g('o_proj'):>7.1f}us | {g('small'):>6.1f}us | {total_dur:>7.1f}us | {attn_pct:>5.1f}%")
    print("-" * 100)


def run_compare():
    """Compare TP=1 vs TP=2 side-by-side, per seq length."""
    csv1 = "H100_results/attn_scaling_tp1.csv"
    csv2 = "H100_results/attn_scaling_tp2.csv"

    def analyze_all(csv_path, tp):
        rows = parse(csv_path)
        groups = group_by_seq(rows)
        results = []
        for gi, group in enumerate(groups):
            iters = extract_iterations(group)
            if not iters:
                continue
            profiled = iters[-1]
            stages, total_dur = analyze_iteration(profiled)
            seq = SEQ_LENS[gi] if gi < len(SEQ_LENS) else 0
            results.append((seq, stages, total_dur))
        return results

    r1 = analyze_all(csv1, 1)
    r2 = analyze_all(csv2, 2)

    # Build lookup
    d1 = {seq: (st, td) for seq, st, td in r1}
    d2 = {seq: (st, td) for seq, st, td in r2}

    _, _, qkv1_gb, o1_gb = get_tp_dims(1)
    _, _, qkv2_gb, o2_gb = get_tp_dims(2)

    print()
    print("=" * 160)
    print("  TP=1 vs TP=2 — Per-Seq Comparison (Qwen3-235B GQA, single-GPU compute only)")
    print("  TP=1: Q=64, KV=4, W_qkv=76MB, W_o=67MB  |  TP=2: Q=32, KV=2, W_qkv=38MB, W_o=34MB")
    print("=" * 160)

    # ═══ Per-stage latency comparison ═══
    print()
    print("─" * 140)
    print(f"  Per-Stage Latency (us)")
    print("─" * 140)
    print(f"{'Seq':>8} | {'qkv_proj':^19} | {'rope':^19} | {'attn (FA2)':^23} | {'o_proj':^19} | {'total':^23}")
    print(f"{'':>8} | {'TP1':>6} {'TP2':>6} {'Δ':>5} | {'TP1':>6} {'TP2':>6} {'Δ':>5} | "
          f"{'TP1':>8} {'TP2':>8} {'Δ':>5} | {'TP1':>6} {'TP2':>6} {'Δ':>5} | "
          f"{'TP1':>8} {'TP2':>8} {'Δ':>5}")
    print("-" * 140)

    for seq in SEQ_LENS:
        if seq not in d1 or seq not in d2:
            continue
        s1, t1 = d1[seq]
        s2, t2 = d2[seq]

        def g(stages, name):
            return stages.get(name, {}).get("dur_us", 0)

        def fmt_delta(a, b):
            if a <= 0:
                return "  -  "
            ratio = b / a
            return f"{ratio:.2f}x"

        qkv1, qkv2 = g(s1, "qkv_proj"), g(s2, "qkv_proj")
        rope1, rope2 = g(s1, "rope"), g(s2, "rope")
        attn1, attn2 = g(s1, "attn"), g(s2, "attn")
        o1, o2 = g(s1, "o_proj"), g(s2, "o_proj")

        print(f"{seq:>8} | {qkv1:>5.1f} {qkv2:>5.1f} {fmt_delta(qkv2, qkv1):>5} | "
              f"{rope1:>5.1f} {rope2:>5.1f} {fmt_delta(rope2, rope1):>5} | "
              f"{attn1:>7.1f} {attn2:>7.1f} {fmt_delta(attn2, attn1):>5} | "
              f"{o1:>5.1f} {o2:>5.1f} {fmt_delta(o2, o1):>5} | "
              f"{t1:>7.1f} {t2:>7.1f} {fmt_delta(t2, t1):>5}")
    print("-" * 140)

    # ═══ Per-stage HBM BW% comparison ═══
    print()
    print("─" * 140)
    print(f"  HBM Bandwidth Utilization (%)")
    print("─" * 140)
    print(f"{'Seq':>8} | {'qkv_proj':^17} | {'attn (FA2)':^17} | {'o_proj':^17} | {'Eff BW qkv (GB/s)':^21} | {'Eff BW attn (GB/s)':^21} | {'Eff BW o (GB/s)':^21}")
    print(f"{'':>8} | {'TP1':>6} {'TP2':>6} {'Δ':>3} | {'TP1':>6} {'TP2':>6} {'Δ':>3} | "
          f"{'TP1':>6} {'TP2':>6} {'Δ':>3} | {'TP1':>8} {'TP2':>8} {'Δ':>3} | "
          f"{'TP1':>8} {'TP2':>8} {'Δ':>3} | {'TP1':>8} {'TP2':>8} {'Δ':>3}")
    print("-" * 140)

    for seq in SEQ_LENS:
        if seq not in d1 or seq not in d2:
            continue
        s1, _ = d1[seq]
        s2, _ = d2[seq]

        def hbm(stages, name):
            return stages.get(name, {}).get("hbm_bw_pct", 0)

        def ebw(stages, name):
            return get_eff_bw(stages.get(name, {}))

        def delta_pct(a, b):
            d = b - a
            return f"{d:+.0f}" if a > 0 else "  -"

        def delta_bw(a, b):
            d = b - a
            return f"{d:+.0f}" if a > 0 else "  -"

        print(f"{seq:>8} | {hbm(s1,'qkv_proj'):>5.1f} {hbm(s2,'qkv_proj'):>5.1f} {delta_pct(hbm(s1,'qkv_proj'),hbm(s2,'qkv_proj')):>3} | "
              f"{hbm(s1,'attn'):>5.1f} {hbm(s2,'attn'):>5.1f} {delta_pct(hbm(s1,'attn'),hbm(s2,'attn')):>3} | "
              f"{hbm(s1,'o_proj'):>5.1f} {hbm(s2,'o_proj'):>5.1f} {delta_pct(hbm(s1,'o_proj'),hbm(s2,'o_proj')):>3} | "
              f"{ebw(s1,'qkv_proj'):>7.0f} {ebw(s2,'qkv_proj'):>7.0f} {delta_bw(ebw(s1,'qkv_proj'),ebw(s2,'qkv_proj')):>5} | "
              f"{ebw(s1,'attn'):>7.0f} {ebw(s2,'attn'):>7.0f} {delta_bw(ebw(s1,'attn'),ebw(s2,'attn')):>5} | "
              f"{ebw(s1,'o_proj'):>7.0f} {ebw(s2,'o_proj'):>7.0f} {delta_bw(ebw(s1,'o_proj'),ebw(s2,'o_proj')):>5}")
    print("-" * 140)

    # ═══ Per-stage SM% comparison ═══
    print()
    print("─" * 140)
    print(f"  SM Compute Utilization (%)")
    print("─" * 140)
    print(f"{'Seq':>8} | {'qkv_proj':^17} | {'attn (FA2)':^17} | {'o_proj':^17}")
    print(f"{'':>8} | {'TP1':>6} {'TP2':>6} {'Δ':>3} | {'TP1':>6} {'TP2':>6} {'Δ':>3} | "
          f"{'TP1':>6} {'TP2':>6} {'Δ':>3}")
    print("-" * 140)

    for seq in SEQ_LENS:
        if seq not in d1 or seq not in d2:
            continue
        s1, _ = d1[seq]
        s2, _ = d2[seq]

        def sm(stages, name):
            return stages.get(name, {}).get("sm_pct", 0)

        def delta_pct(a, b):
            d = b - a
            return f"{d:+.0f}" if a > 0 else "  -"

        print(f"{seq:>8} | {sm(s1,'qkv_proj'):>5.1f} {sm(s2,'qkv_proj'):>5.1f} {delta_pct(sm(s1,'qkv_proj'),sm(s2,'qkv_proj')):>3} | "
              f"{sm(s1,'attn'):>5.1f} {sm(s2,'attn'):>5.1f} {delta_pct(sm(s1,'attn'),sm(s2,'attn')):>3} | "
              f"{sm(s1,'o_proj'):>5.1f} {sm(s2,'o_proj'):>5.1f} {delta_pct(sm(s1,'o_proj'),sm(s2,'o_proj')):>3}")
    print("-" * 140)

    # ═══ Attention speedup detail ═══
    print()
    print("─" * 120)
    print(f"  FlashAttention-2 Speedup (TP=1 vs TP=2)")
    print("─" * 120)
    print(f"{'Seq':>8} | {'KV TP1':>8} {'KV TP2':>8} | {'FA TP1':>8} {'FA TP2':>8} {'speedup':>7} | "
          f"{'BW TP1':>9} {'BW TP2':>9} | {'HBM% TP1':>8} {'HBM% TP2':>8}")
    print("-" * 120)

    for seq in SEQ_LENS:
        if seq not in d1 or seq not in d2:
            continue
        s1, _ = d1[seq]
        s2, _ = d2[seq]

        kv1_mb = kv_cache_gb(seq, 1) * 1000
        kv2_mb = kv_cache_gb(seq, 2) * 1000
        a1 = s1.get("attn", {}).get("dur_us", 0)
        a2 = s2.get("attn", {}).get("dur_us", 0)
        sp = a1 / a2 if a2 > 0 else 0
        bw1 = get_eff_bw(s1.get("attn", {}))
        bw2 = get_eff_bw(s2.get("attn", {}))
        hbm1 = s1.get("attn", {}).get("hbm_bw_pct", 0)
        hbm2 = s2.get("attn", {}).get("hbm_bw_pct", 0)

        print(f"{seq:>8} | {kv1_mb:>6.0f}MB {kv2_mb:>6.0f}MB | "
              f"{a1:>6.1f}us {a2:>6.1f}us {sp:>6.2f}x | "
              f"{bw1:>7.0f}GB/s {bw2:>7.0f}GB/s | "
              f"{hbm1:>7.1f}% {hbm2:>7.1f}%")
    print("-" * 120)

    # ═══ Total compute speedup ═══
    print()
    print("─" * 100)
    print(f"  Total Compute Speedup (TP=1 vs TP=2, no communication)")
    print("─" * 100)
    print(f"{'Seq':>8} | {'TP1 total':>10} {'TP2 total':>10} {'speedup':>8} | "
          f"{'TP1 attn%':>9} {'TP2 attn%':>9} | {'TP1 proj':>8} {'TP2 proj':>8}")
    print("-" * 100)

    for seq in SEQ_LENS:
        if seq not in d1 or seq not in d2:
            continue
        s1, t1 = d1[seq]
        s2, t2 = d2[seq]

        def g(stages, name):
            return stages.get(name, {}).get("dur_us", 0)

        sp = t1 / t2 if t2 > 0 else 0
        a1_pct = g(s1, "attn") / t1 * 100 if t1 > 0 else 0
        a2_pct = g(s2, "attn") / t2 * 100 if t2 > 0 else 0
        proj1 = g(s1, "qkv_proj") + g(s1, "o_proj")
        proj2 = g(s2, "qkv_proj") + g(s2, "o_proj")

        print(f"{seq:>8} | {t1:>8.1f}us {t2:>8.1f}us {sp:>7.2f}x | "
              f"{a1_pct:>8.1f}% {a2_pct:>8.1f}% | "
              f"{proj1:>6.1f}us {proj2:>6.1f}us")
    print("-" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze attention-only ncu profiling")
    parser.add_argument("--tp", type=int, default=None,
                        help="TP degree (reads attn_scaling_tp<N>.csv). "
                             "Default: reads attn_scaling_raw.csv")
    parser.add_argument("--compare", action="store_true",
                        help="Compare TP=1 vs TP=2 side-by-side")
    args = parser.parse_args()

    if args.compare:
        run_compare()
    else:
        main(args.tp)
