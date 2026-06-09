#!/usr/bin/env python3
"""Parse e2e_tp1.txt / e2e_tp2.txt (print_e2e_table output) and build a
TP=1 vs TP=2 comparison focused on Flash Decode (attn) latency.

Usage:
    python parse_e2e_tp_compare.py H100_results/e2e_tp1.txt H100_results/e2e_tp2.txt \
        -o H100_results/tp1_vs_tp2.csv
"""
import argparse
import csv
import re

ROW = re.compile(r"^\s*(\d+)\s*\|\s*([A-Za-z_ ]+?)\s*\|\s*([\d.]+)\s*\|")


def parse(path):
    """Return {seq: {stage: median_us}}."""
    out = {}
    with open(path) as f:
        for line in f:
            m = ROW.match(line)
            if not m:
                continue
            seq = int(m.group(1))
            stage = m.group(2).strip()
            val = float(m.group(3))
            out.setdefault(seq, {})[stage] = val
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tp1")
    ap.add_argument("tp2")
    ap.add_argument("-o", "--out", default="H100_results/tp1_vs_tp2.csv")
    args = ap.parse_args()

    d1, d2 = parse(args.tp1), parse(args.tp2)
    seqs = sorted(set(d1) | set(d2))

    cols = [
        "seq",
        "attn_tp1_us", "attn_tp2_us", "attn_speedup",
        "layer_tp1_us", "layer_tp2_us", "layer_speedup",
        "allreduce_tp2_us", "comm_frac_tp2",
    ]
    rows = []
    for s in seqs:
        a1 = d1.get(s, {}).get("attn", float("nan"))
        a2 = d2.get(s, {}).get("attn", float("nan"))
        l1 = d1.get(s, {}).get("LAYER TOTAL", float("nan"))
        l2 = d2.get(s, {}).get("LAYER TOTAL", float("nan"))
        ar = (d2.get(s, {}).get("allreduce_attn", 0.0)
              + d2.get(s, {}).get("allreduce_mlp", 0.0))
        rows.append({
            "seq": s,
            "attn_tp1_us": round(a1, 1), "attn_tp2_us": round(a2, 1),
            "attn_speedup": round(a1 / a2, 3) if a2 else "",
            "layer_tp1_us": round(l1, 1), "layer_tp2_us": round(l2, 1),
            "layer_speedup": round(l1 / l2, 3) if l2 else "",
            "allreduce_tp2_us": round(ar, 1),
            "comm_frac_tp2": round(ar / l2, 3) if l2 else "",
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    # pretty print
    print(f"Wrote {args.out}\n")
    hdr = f"{'seq':>8} | {'attn TP1':>9} {'attn TP2':>9} {'spdup':>6} | " \
          f"{'layer TP1':>10} {'layer TP2':>10} {'spdup':>6} | " \
          f"{'allred TP2':>10} {'comm%':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['seq']:>8} | {r['attn_tp1_us']:>9} {r['attn_tp2_us']:>9} "
              f"{r['attn_speedup']:>6} | {r['layer_tp1_us']:>10} "
              f"{r['layer_tp2_us']:>10} {r['layer_speedup']:>6} | "
              f"{r['allreduce_tp2_us']:>10} "
              f"{(r['comm_frac_tp2']*100 if r['comm_frac_tp2'] else 0):>5.1f}%")


if __name__ == "__main__":
    main()
