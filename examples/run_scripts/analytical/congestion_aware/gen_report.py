#!/usr/bin/env python3
"""Generate markdown report from hybrid merged JSON with utilization."""
import json
import sys

INPUT = "reports/qwen3-235B/hybrid/gqa_hybrid_merged_80T_split4_bw1500_util.json"
OUTPUT = "reports/qwen3-235B/hybrid/gqa_hybrid_merged_80T_split4_bw1500_util_report.md"

with open(INPUT) as f:
    d = json.load(f)

meta = d["metadata"]
strats = ["HMP_reo", "hmp_reo_new", "hmp", "tp16", "rubin"]
labels = {"HMP_reo": "HMP_RO", "hmp_reo_new": "RO_new", "hmp": "HMP", "tp16": "TP16", "rubin": "Rubin"}

data = {}
for s in strats:
    data[s] = {}
    for e in d.get("strategies", {}).get(s, {}).get("data", []):
        data[s][e["seq"]] = e

all_seqs = sorted(set().union(*(data[s].keys() for s in strats)))


def sl(s):
    if s >= 1048576:
        return f"{s // 1048576}M"
    if s >= 1024:
        return f"{s // 1024}K"
    return str(s)


def short_src(src, e=None):
    if "util_adj" in str(src):
        return "roofline+util"
    if "weight_load" in str(src):
        return "roofline_mem"
    if "kv_cache" in str(src):
        return "roofline_fused"
    if "astrasim" in str(src):
        return "astrasim"
    if e and e.get("utilization_applied"):
        return "roofline+H100_util"
    return "roofline"


L = []

L.append("# Qwen3-235B GQA Single-Layer Hybrid Estimation Report")
L.append("")
L.append(f'> **Roofline**: `{meta.get("roofline_file", "")}`')
L.append(f'> **AstraSim**: `{meta.get("astrasim_file", "")}`')
L.append(f'> **NPU utilization**: `{meta.get("utilization_file", "N/A")}` | '
         f'**Rubin utilization**: `{meta.get("rubin_utilization_file", "N/A")}`')
L.append("")

hw = meta.get("device_hw", {})
pp = float(hw.get("peak_perf_tflops", 0))
bw = float(hw.get("bandwidth_tb_s", 0))

L.append("## Hardware Configuration")
L.append("")
L.append("| Parameter | Ours (16 HBM4 NPUs) | Rubin |")
L.append("|-----------|---------------------|-------|")
L.append(f"| Peak Compute | {pp} TFLOPS/NPU ({pp*16:.0f} total) | 17,500 TFLOPS |")
L.append(f"| HBM BW | {bw} TB/s/NPU ({bw*16:.0f} total) | 22 TB/s |")
L.append(f'| D2D Link BW | {hw.get("link_bw_tb_s", "?")} TB/s | N/A (single chip) |')
L.append("| Topology | Mesh2D 4x4, block2x2 rank remap | Single GPU |")
L.append("")

L.append("## Utilization Model")
L.append("")
L.append("**Ours (NPU):** Compute utilization profiling. `adjusted_compute = compute_ns / util`, "
         "then `max(adjusted_compute, memory_ns)`. At bs=1: proj_qkv util=2.8%, proj_o util=4.7%.")
L.append("")
L.append("**Rubin:** H100 e2e HBM BW utilization from real GPU profiling. "
         "`adjusted_total = roofline_ns / bw_util`. "
         "proj_qkv BW util=68.7%, proj_o BW util=70.5%, attn BW util=4.2%-91.6% by seq.")
L.append("")

# === Wall Time Comparison ===
L.append("## Wall Time Comparison (ns)")
L.append("")
hdr = "| Seq |" + "|".join(f" {labels[s]} " for s in strats) + "|"
sep = "|-----|" + "|".join(["--------:"] * len(strats)) + "|"
L.append(hdr)
L.append(sep)
for seq in all_seqs:
    row = f"| {sl(seq)} |"
    for s in strats:
        e = data[s].get(seq)
        row += f" {e['hybrid_wall_ns']:,.0f} |" if e else " - |"
    L.append(row)
L.append("")

# === Detailed Breakdown ===
L.append("## Detailed Module Breakdown (ns)")
L.append("")
L.append("| Seq | Strategy | Proj_QKV | Attn | Proj_O | GPU | Comm | Wall | QKV Src | Attn Src | O Src |")
L.append("|-----|----------|--------:|-----:|------:|----:|-----:|-----:|---------|----------|-------|")
for seq in all_seqs:
    for s in strats:
        e = data[s].get(seq)
        if not e:
            continue
        cb = e.get("compute_breakdown", {})
        qsrc = short_src(cb.get("Proj_QKV_source", ""), e)
        asrc = short_src(cb.get("attn_source", ""), e)
        osrc = short_src(cb.get("Proj_O_source", ""), e)
        L.append(
            f"| {sl(seq)} | {labels[s]} "
            f"| {e['hybrid_Proj_QKV_ns']:,.0f} "
            f"| {e['hybrid_attn_ns']:,.0f} "
            f"| {e['hybrid_Proj_O_ns']:,.0f} "
            f"| {e['hybrid_gpu_ns']:,.0f} "
            f"| {e.get('comm_total_ns', 0):,.0f} "
            f"| {e['hybrid_wall_ns']:,.0f} "
            f"| {qsrc} | {asrc} | {osrc} |"
        )
L.append("")

# === Speedup vs Rubin ===
L.append("## Speedup vs Rubin")
L.append("")
ours = [s for s in strats if s != "rubin"]
hdr2 = "| Seq |" + "|".join(f" {labels[s]} " for s in ours) + "|"
sep2 = "|-----|" + "|".join(["------:"] * len(ours)) + "|"
L.append(hdr2)
L.append(sep2)
for seq in all_seqs:
    re = data["rubin"].get(seq)
    if not re:
        continue
    rw = re["hybrid_wall_ns"]
    row = f"| {sl(seq)} |"
    for s in ours:
        e = data[s].get(seq)
        if e and e["hybrid_wall_ns"] > 0:
            sp = rw / e["hybrid_wall_ns"]
            row += f" {sp:.2f}x |"
        else:
            row += " - |"
    L.append(row)
L.append("")

# === Key Findings ===
L.append("## Key Findings")
L.append("")

# Best strategy at each key seq
L.append("### Best Strategy per Seq (Ours only)")
L.append("")
for seq in [1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]:
    candidates = [(s, data[s][seq]["hybrid_wall_ns"]) for s in ours if seq in data[s]]
    if not candidates:
        continue
    best_s, best_w = min(candidates, key=lambda x: x[1])
    re = data["rubin"].get(seq)
    rw = re["hybrid_wall_ns"] if re else 0
    sp = rw / best_w if best_w > 0 and rw > 0 else 0
    L.append(f"- **seq={sl(seq)}**: Best = {labels[best_s]} ({best_w:,.0f} ns), "
             f"Rubin = {rw:,.0f} ns, **{sp:.2f}x speedup**")
L.append("")

# Crossover analysis
L.append("### Crossover Points (Ours vs Rubin)")
L.append("")
for s in ours:
    lost = None
    for seq in all_seqs:
        re = data["rubin"].get(seq)
        se = data[s].get(seq)
        if re and se and se["hybrid_wall_ns"] > re["hybrid_wall_ns"]:
            lost = seq
            break
    if lost:
        L.append(f"- **{labels[s]}** loses to Rubin starting at seq={sl(lost)}")
    else:
        L.append(f"- **{labels[s]}** beats Rubin at all seq lengths")
L.append("")

# Utilization impact
L.append("### Utilization Impact")
L.append("")
L.append("#### NPU (Ours)")
L.append("- **proj_qkv** (bs=1, util=2.8%): roofline_mem=946 ns -> util_adj=2,113 ns "
         "(compute-bound after adjustment, **+123%**)")
L.append("- **proj_o** (bs=1, util=4.7%): HMP_RO: roofline_mem=3,358 ns -> util_adj=4,448 ns "
         "(compute-bound, **+32%**)")
L.append("- **attn score/attn_v**: memory-bound at all seq for cache_seq (seq/4), "
         "util adjustment has limited impact")
L.append("")
L.append("#### Rubin (H100 BW Util)")
L.append("- **proj_qkv** (BW util=68.7%): 1,716 ns -> 2,498 ns (+46%)")
L.append("- **proj_o** (BW util=70.5%): 1,526 ns -> 2,164 ns (+42%)")
L.append("- **attn** (BW util varies):")
for seq in [1024, 16384, 65536, 262144, 1048576]:
    re = data["rubin"].get(seq)
    if not re:
        continue
    bu = re.get("attn_bw_util", 0)
    raw = re.get("attn_roofline_ns", 0)
    adj = re["hybrid_attn_ns"]
    pct = (adj / raw - 1) * 100 if raw > 0 else 0
    L.append(f"  - seq={sl(seq)}: BW util={bu*100:.1f}%, {raw:,.0f} ns -> {adj:,.0f} ns (+{pct:.0f}%)")
L.append("")

# Communication analysis
L.append("### Communication Overhead")
L.append("")
for seq in [65536, 1048576]:
    L.append(f"**seq={sl(seq)}:**")
    L.append("")
    L.append("| Strategy | GPU (ns) | Comm (ns) | Comm % | Wall (ns) |")
    L.append("|----------|--------:|----------:|-------:|----------:|")
    for s in strats:
        e = data[s].get(seq)
        if not e:
            continue
        gpu = e["hybrid_gpu_ns"]
        comm = e.get("comm_total_ns", 0)
        wall = e["hybrid_wall_ns"]
        cpct = comm / wall * 100 if wall > 0 else 0
        L.append(f"| {labels[s]} | {gpu:,.0f} | {comm:,.0f} | {cpct:.1f}% | {wall:,.0f} |")
    L.append("")

out = "\n".join(L) + "\n"
with open(OUTPUT, "w") as f:
    f.write(out)
print(f"Saved: {OUTPUT}")
print(f"Length: {len(L)} lines")
