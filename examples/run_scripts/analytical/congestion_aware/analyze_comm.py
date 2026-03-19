#!/usr/bin/env python3
"""
Analyze per-layer communication latency from ASTRA-sim logs.

Usage:
  # Single log
  python analyze_comm.py <log_file>

  # Compare multiple logs side-by-side
  python analyze_comm.py <log1> <log2> [<log3> ...]

  # Filter specific NPU (default: 0)
  python analyze_comm.py --npu 0 <log_file>

  # Show first N layers only (default: 2)
  python analyze_comm.py --layers 5 <log_file>

  # Show all layers
  python analyze_comm.py --layers 0 <log_file>
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CommEvent:
    npu: int
    node_id: int
    node_name: str
    comm_type: str
    size_mb: float
    size_bytes: int
    group_id: int
    latency_us: float = 0.0
    latency_ns: int = 0

    @property
    def layer_name(self) -> str:
        m = re.match(r"(transformer\.\d+)\.", self.node_name)
        return m.group(1) if m else "other"

    @property
    def op_name(self) -> str:
        name = self.node_name
        name = re.sub(r"^transformer\.\d+\.", "", name)
        name = re.sub(r"@\d+_X1COMM$", "", name)
        return name

    @property
    def module(self) -> str:
        """Classify into high-level module: weight_ag, attn_comm, output_comm, ffn_comm, other."""
        op = self.op_name
        if op == "mha.x" or op == "mha_res.x1":
            return "weight_ag"
        if "attn_kernel" in op or op == "mha.attn":
            return "attn_comm"
        if op in ("mha.o", "mha_res.y1"):
            return "output_comm"
        if "ffn" in op:
            return "ffn_comm"
        return "other_comm"


@dataclass
class CompEvent:
    npu: int
    node_id: int
    node_name: str
    op_type: str
    gflops: float
    size_mb: float
    runtime_us: float
    runtime_ns: int = 0

    @property
    def layer_name(self) -> str:
        m = re.match(r"(transformer\.\d+)\.", self.node_name)
        return m.group(1) if m else "other"

    @property
    def op_name(self) -> str:
        name = self.node_name
        name = re.sub(r"^transformer\.\d+\.", "", name)
        name = re.sub(r"@\d+_COMP$", "", name)
        return name

    @property
    def module(self) -> str:
        """Classify into high-level module: qkv_comp, attn_comp, output_comp, norm_res, ffn_comp, other."""
        op = self.op_name
        if op in ("mha.q", "mha.kv"):
            return "qkv_comp"
        if "attn_kernel" in op:
            return "attn_comp"
        if op in ("mha.o1", "mha.o"):
            return "output_comp"
        if op == "mha.kt" or op == "mha.vt":
            return "attn_comp"
        if any(k in op for k in ("input_norm", "post_attn_norm", "mha_res")):
            return "norm_res"
        if "ffn" in op:
            return "ffn_comp"
        return "other_comp"


_RE_COMM_START = re.compile(
    r"\[SIM_COMM\]\s+NPU=(\d+)\s+node_id=(\d+)\s+node_name=(\S+)\s+"
    r"type=(\S+)\s+size=([\d.]+)\s+MB\s+\((\d+)\s+bytes\)\s+comm_group_id=(\d+)"
)
_RE_COMM_FINISH = re.compile(
    r"\[SIM_COMM_FINISH\]\s+NPU=(\d+)\s+node_id=(\d+)\s+node_name=(\S+)\s+"
    r"type=(\S+)\s+size=([\d.]+)\s+MB\s+execution_time=([\d.]+)\s+us\s+"
    r"\((\d+)\s+ns\)\s+comm_group_id=(\d+)"
)
_RE_COMP = re.compile(
    r"\[SIM_COMP\]\s+NPU=(\d+)\s+node_id=(\d+)\s+node_name=(\S+)\s+"
    r"op_type=(\S+)\s+ops=([\d.]+)\s+GFLOPs\s+\([\d.e+]+\s+ops\)\s+"
    r"size=([\d.]+)\s+MB\s+\([\d.e+]+\s+bytes\)\s+runtime=([\d.]+)\s+us\s+\((\d+)\s+ns\)"
)
_RE_WALL_TIME = re.compile(
    r"sys\[\d+\],\s+Wall time:\s+(\d+)"
)
_RE_GPU_TIME = re.compile(
    r"sys\[\d+\],\s+GPU time:\s+(\d+)"
)
_RE_COMM_TIME = re.compile(
    r"sys\[\d+\],\s+Comm time:\s+(\d+)"
)


@dataclass
class LogSummary:
    path: str
    label: str
    events: list[CommEvent] = field(default_factory=list)
    comp_events: list[CompEvent] = field(default_factory=list)
    wall_time: int = 0
    gpu_time: int = 0
    comm_time: int = 0
    comm_group_map: dict = field(default_factory=dict)


def parse_comm_group_json(log_path: Path) -> dict[int, list[int]]:
    """Try to find the comm group JSON from the workload directory."""
    return {}


def parse_log(log_path: str, npu: int = 0) -> LogSummary:
    path = Path(log_path)
    label = path.stem
    summary = LogSummary(path=str(path), label=label)
    pending: dict[tuple[int, int], CommEvent] = {}

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _RE_COMP.search(line)
            if m:
                evt_npu = int(m.group(1))
                if evt_npu != npu:
                    continue
                summary.comp_events.append(CompEvent(
                    npu=evt_npu,
                    node_id=int(m.group(2)),
                    node_name=m.group(3),
                    op_type=m.group(4),
                    gflops=float(m.group(5)),
                    size_mb=float(m.group(6)),
                    runtime_us=float(m.group(7)),
                    runtime_ns=int(m.group(8)),
                ))
                continue

            m = _RE_COMM_START.search(line)
            if m:
                evt_npu = int(m.group(1))
                if evt_npu != npu:
                    continue
                node_id = int(m.group(2))
                evt = CommEvent(
                    npu=evt_npu,
                    node_id=node_id,
                    node_name=m.group(3),
                    comm_type=m.group(4),
                    size_mb=float(m.group(5)),
                    size_bytes=int(m.group(6)),
                    group_id=int(m.group(7)),
                )
                pending[(evt_npu, node_id)] = evt
                continue

            m = _RE_COMM_FINISH.search(line)
            if m:
                evt_npu = int(m.group(1))
                if evt_npu != npu:
                    continue
                node_id = int(m.group(2))
                key = (evt_npu, node_id)
                if key in pending:
                    evt = pending.pop(key)
                    evt.latency_us = float(m.group(6))
                    evt.latency_ns = int(m.group(7))
                    summary.events.append(evt)
                continue

            m = _RE_WALL_TIME.search(line)
            if m:
                summary.wall_time = int(m.group(1))
            m = _RE_GPU_TIME.search(line)
            if m:
                summary.gpu_time = int(m.group(1))
            m = _RE_COMM_TIME.search(line)
            if m:
                summary.comm_time = int(m.group(1))

    return summary


def group_by_layer(events: list[CommEvent]) -> dict[str, list[CommEvent]]:
    layers: dict[str, list[CommEvent]] = defaultdict(list)
    for evt in events:
        layers[evt.layer_name].append(evt)
    return dict(layers)


def print_layer_detail(layer: str, events: list[CommEvent]):
    total_us = sum(e.latency_us for e in events)
    print(f"\n  {layer}  (total comm = {total_us:.1f} us, {len(events)} ops)")
    print(f"  {'Op':<35} {'Type':<18} {'Size MB':>10} {'Group':>6} {'Latency':>12}")
    print(f"  {'-'*35} {'-'*18} {'-'*10} {'-'*6} {'-'*12}")
    for e in events:
        print(
            f"  {e.op_name:<35} {e.comm_type:<18} {e.size_mb:>10.4f} "
            f"{e.group_id:>6} {e.latency_us:>10.3f} us"
        )


def print_summary_table(summaries: list[LogSummary], max_layers: int):
    for s in summaries:
        layers = group_by_layer(s.events)
        layer_names = sorted(layers.keys(), key=lambda x: (
            int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 999, x
        ))

        print(f"\n{'='*90}")
        print(f"  {s.label}")
        print(f"  Wall={s.wall_time:,} cycles  GPU={s.gpu_time:,}  Comm={s.comm_time:,}")
        print(f"{'='*90}")

        show_layers = layer_names if max_layers == 0 else layer_names[:max_layers]
        for layer in show_layers:
            print_layer_detail(layer, layers[layer])

        if max_layers > 0 and len(layer_names) > max_layers:
            print(f"\n  ... ({len(layer_names) - max_layers} more layers omitted, use --layers 0 to show all)")

        per_layer_us = []
        for layer in layer_names:
            per_layer_us.append(sum(e.latency_us for e in layers[layer]))
        if per_layer_us:
            avg = sum(per_layer_us) / len(per_layer_us)
            total = sum(per_layer_us)
            print(f"\n  --- Per-layer comm summary ({len(layer_names)} layers) ---")
            print(f"  Avg per layer: {avg:.1f} us")
            print(f"  Total all layers: {total:.1f} us")

        by_type: dict[str, float] = defaultdict(float)
        by_type_count: dict[str, int] = defaultdict(int)
        for e in s.events:
            by_type[e.comm_type] += e.latency_us
            by_type_count[e.comm_type] += 1
        if by_type:
            print(f"\n  --- By comm type ---")
            print(f"  {'Type':<20} {'Count':>6} {'Total us':>12} {'Avg us':>10}")
            for t in sorted(by_type, key=lambda x: -by_type[x]):
                cnt = by_type_count[t]
                print(f"  {t:<20} {cnt:>6} {by_type[t]:>12.1f} {by_type[t]/cnt:>10.3f}")

        by_group: dict[int, float] = defaultdict(float)
        by_group_count: dict[int, int] = defaultdict(int)
        for e in s.events:
            by_group[e.group_id] += e.latency_us
            by_group_count[e.group_id] += 1
        if by_group:
            print(f"\n  --- By comm group ---")
            print(f"  {'Group':>6} {'Count':>6} {'Total us':>12} {'Avg us':>10}")
            for g in sorted(by_group, key=lambda x: -by_group[x]):
                cnt = by_group_count[g]
                print(f"  {g:>6} {cnt:>6} {by_group[g]:>12.1f} {by_group[g]/cnt:>10.3f}")


MODULE_LABELS = {
    "weight_ag": "Weight AllGather",
    "attn_comm": "Attn K/V/Reduce",
    "output_comm": "Output AR/RS/AG",
    "ffn_comm": "FFN Comm",
    "other_comm": "Other Comm",
    "qkv_comp": "QKV Projection",
    "attn_comp": "Attn Kernel",
    "output_comp": "Output Projection",
    "norm_res": "Norm + Residual",
    "ffn_comp": "FFN Compute",
    "other_comp": "Other Compute",
}

MODULE_ORDER = [
    "qkv_comp", "attn_comp", "output_comp", "norm_res", "ffn_comp", "other_comp",
    "weight_ag", "attn_comm", "output_comm", "ffn_comm", "other_comm",
]


def print_module_breakdown(summaries: list[LogSummary]):
    """Print per-module breakdown for layer 0 and aggregate across all layers."""
    print(f"\n{'='*100}")
    print(f"  MODULE BREAKDOWN (NPU 0)")
    print(f"{'='*100}")

    for s in summaries:
        layer0_comp = [e for e in s.comp_events if e.layer_name == "transformer.0"]
        layer0_comm = [e for e in s.events if e.layer_name == "transformer.0"]
        all_comp = [e for e in s.comp_events if e.layer_name != "other"]
        all_comm = [e for e in s.events if e.layer_name != "other"]
        n_layers = len(set(e.layer_name for e in s.comp_events if e.layer_name != "other"))

        print(f"\n  --- {s.label} ({n_layers} layers) ---")

        comp_by_mod: dict[str, float] = defaultdict(float)
        comm_by_mod: dict[str, float] = defaultdict(float)
        for e in layer0_comp:
            comp_by_mod[e.module] += e.runtime_us
        for e in layer0_comm:
            comm_by_mod[e.module] += e.latency_us

        total_comp = sum(comp_by_mod.values())
        total_comm = sum(comm_by_mod.values())
        total = total_comp + total_comm

        print(f"\n  [Layer 0 Breakdown]")
        print(f"  {'Module':<25} {'Time(us)':>10} {'% of Layer':>10} {'Category':>10}")
        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

        for mod in MODULE_ORDER:
            us = comp_by_mod.get(mod, 0) + comm_by_mod.get(mod, 0)
            if us > 0:
                pct = us / total * 100 if total else 0
                cat = "COMP" if mod.endswith("_comp") or mod == "norm_res" else "COMM"
                print(f"  {MODULE_LABELS.get(mod, mod):<25} {us:>10.3f} {pct:>9.1f}% {cat:>10}")

        print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        comp_pct = total_comp / total * 100 if total else 0
        comm_pct = total_comm / total * 100 if total else 0
        print(f"  {'Total Compute':<25} {total_comp:>10.3f} {comp_pct:>9.1f}%       COMP")
        print(f"  {'Total Communication':<25} {total_comm:>10.3f} {comm_pct:>9.1f}%       COMM")
        print(f"  {'Layer Total':<25} {total:>10.3f} {'100.0':>9}%")

        has_ffn_comp = any(e.module == "ffn_comp" for e in layer0_comp)
        has_ffn_comm = any(e.module == "ffn_comm" for e in layer0_comm)
        ffn_total = comp_by_mod.get("ffn_comp", 0) + comm_by_mod.get("ffn_comm", 0)
        mha_total = total - ffn_total - comp_by_mod.get("norm_res", 0) - comp_by_mod.get("other_comp", 0) - comm_by_mod.get("other_comm", 0)
        norm_total = comp_by_mod.get("norm_res", 0)

        print(f"\n  [High-Level Split]")
        print(f"  {'Component':<20} {'Time(us)':>10} {'% of Layer':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10}")
        print(f"  {'MHA (comp+comm)':<20} {mha_total:>10.3f} {mha_total/total*100 if total else 0:>9.1f}%")
        print(f"  {'FFN (comp+comm)':<20} {ffn_total:>10.3f} {ffn_total/total*100 if total else 0:>9.1f}%")
        print(f"  {'Norm/Residual':<20} {norm_total:>10.3f} {norm_total/total*100 if total else 0:>9.1f}%")
        if not has_ffn_comp and not has_ffn_comm:
            print(f"  ** NOTE: No FFN in this workload (decode_bypass model) **")

        agg_comp: dict[str, float] = defaultdict(float)
        agg_comm: dict[str, float] = defaultdict(float)
        for e in all_comp:
            agg_comp[e.module] += e.runtime_us
        for e in all_comm:
            agg_comm[e.module] += e.latency_us

        agg_total_comp = sum(agg_comp.values())
        agg_total_comm = sum(agg_comm.values())
        agg_total = agg_total_comp + agg_total_comm

        print(f"\n  [All {n_layers} Layers Aggregate]")
        print(f"  {'Module':<25} {'Time(us)':>12} {'% of Total':>10} {'Avg/Layer':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*12}")

        for mod in MODULE_ORDER:
            us = agg_comp.get(mod, 0) + agg_comm.get(mod, 0)
            if us > 0:
                pct = us / agg_total * 100 if agg_total else 0
                avg = us / n_layers if n_layers else 0
                print(f"  {MODULE_LABELS.get(mod, mod):<25} {us:>12.1f} {pct:>9.1f}% {avg:>10.3f} us")

        print(f"  {'-'*25} {'-'*12} {'-'*10} {'-'*12}")
        print(f"  {'Total Compute':<25} {agg_total_comp:>12.1f} {agg_total_comp/agg_total*100 if agg_total else 0:>9.1f}%")
        print(f"  {'Total Communication':<25} {agg_total_comm:>12.1f} {agg_total_comm/agg_total*100 if agg_total else 0:>9.1f}%")
        print(f"  {'Grand Total':<25} {agg_total:>12.1f} {'100.0':>9}%")

    if len(summaries) >= 2:
        print(f"\n{'='*100}")
        print(f"  MODULE COMPARISON (Layer 0)")
        print(f"{'='*100}")

        header = f"  {'Module':<25}"
        for s in summaries:
            short = s.label[:18]
            header += f" {short:>20}"
        print(header)
        print(f"  {'-'*25}" + f" {'-'*20}" * len(summaries))

        for mod in MODULE_ORDER:
            vals = []
            any_val = False
            for s in summaries:
                l0_comp = [e for e in s.comp_events if e.layer_name == "transformer.0"]
                l0_comm = [e for e in s.events if e.layer_name == "transformer.0"]
                us = sum(e.runtime_us for e in l0_comp if e.module == mod) + \
                     sum(e.latency_us for e in l0_comm if e.module == mod)
                vals.append(us)
                if us > 0:
                    any_val = True
            if any_val:
                row = f"  {MODULE_LABELS.get(mod, mod):<25}"
                for v in vals:
                    row += f" {f'{v:.1f} us':>20}" if v > 0 else f" {'---':>20}"
                print(row)

        print(f"  {'-'*25}" + f" {'-'*20}" * len(summaries))
        row_comp = f"  {'Compute Total':<25}"
        row_comm = f"  {'Comm Total':<25}"
        row_all = f"  {'Layer Total':<25}"
        for s in summaries:
            l0c = sum(e.runtime_us for e in s.comp_events if e.layer_name == "transformer.0")
            l0m = sum(e.latency_us for e in s.events if e.layer_name == "transformer.0")
            row_comp += f" {f'{l0c:.1f} us':>20}"
            row_comm += f" {f'{l0m:.1f} us':>20}"
            row_all += f" {f'{l0c+l0m:.1f} us':>20}"
        print(row_comp)
        print(row_comm)
        print(row_all)

        print(f"\n  {'Global Metric':<25}", end="")
        for s in summaries:
            print(f" {s.label[:18]:>20}", end="")
        print()
        print(f"  {'-'*25}" + f" {'-'*20}" * len(summaries))
        for metric, attr in [("Wall time (ns)", "wall_time"),
                             ("GPU time (ns)", "gpu_time"),
                             ("Comm time (ns)", "comm_time")]:
            print(f"  {metric:<25}", end="")
            for s in summaries:
                print(f" {getattr(s, attr):>20,}", end="")
            print()
        if summaries[0].wall_time:
            print(f"  {'Compute %':<25}", end="")
            for s in summaries:
                pct = s.gpu_time / s.wall_time * 100 if s.wall_time else 0
                print(f" {f'{pct:.1f}%':>20}", end="")
            print()
            print(f"  {'Comm %':<25}", end="")
            for s in summaries:
                pct = s.comm_time / s.wall_time * 100 if s.wall_time else 0
                print(f" {f'{pct:.1f}%':>20}", end="")
            print()


def print_comparison(summaries: list[LogSummary]):
    if len(summaries) < 2:
        return

    print(f"\n{'='*90}")
    print(f"  COMPARISON (layer 0, NPU 0)")
    print(f"{'='*90}")

    layer0_data = []
    for s in summaries:
        layers = group_by_layer(s.events)
        key = "transformer.0"
        evts = layers.get(key, [])
        total = sum(e.latency_us for e in evts)
        layer0_data.append((s.label, evts, total))

    max_label = max(len(d[0]) for d in layer0_data)
    all_ops = set()
    for _, evts, _ in layer0_data:
        for e in evts:
            all_ops.add(e.op_name)

    header = f"  {'Op':<35}"
    for label, _, _ in layer0_data:
        header += f" {label[:20]:>22}"
    print(header)
    print(f"  {'-'*35}" + f" {'-'*22}" * len(layer0_data))

    op_order = []
    for _, evts, _ in layer0_data:
        for e in evts:
            if e.op_name not in op_order:
                op_order.append(e.op_name)

    for op in op_order:
        row = f"  {op:<35}"
        for _, evts, _ in layer0_data:
            matching = [e for e in evts if e.op_name == op]
            if matching:
                parts = []
                for e in matching:
                    parts.append(f"{e.comm_type[:2]} g{e.group_id} {e.latency_us:.1f}us")
                row += f" {'; '.join(parts):>22}"
            else:
                row += f" {'---':>22}"
        print(row)

    row = f"  {'TOTAL per layer':<35}"
    for _, _, total in layer0_data:
        row += f" {f'{total:.1f} us':>22}"
    print(f"  {'-'*35}" + f" {'-'*22}" * len(layer0_data))
    print(row)

    print(f"\n  {'Metric':<35}", end="")
    for s in summaries:
        print(f" {s.label[:20]:>22}", end="")
    print()
    print(f"  {'-'*35}" + f" {'-'*22}" * len(summaries))
    for metric, attr in [("Wall time (cycles)", "wall_time"),
                         ("GPU time (cycles)", "gpu_time"),
                         ("Comm time (cycles)", "comm_time")]:
        print(f"  {metric:<35}", end="")
        for s in summaries:
            v = getattr(s, attr)
            print(f" {v:>22,}", end="")
        print()


def resolve_project_dir() -> str:
    """Walk up from this script to find the project root (contains 'build/' and 'examples/')."""
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "build").is_dir() and (d / "examples").is_dir():
            return str(d)
        d = d.parent
    return ""


def resolve_path(raw: str, project_dir: str) -> str:
    """Replace {PROJECT_DIR} placeholder and resolve the path."""
    if project_dir:
        raw = raw.replace("{PROJECT_DIR}", project_dir)
    return raw


def run_analysis(log_paths: list[str], npu: int = 0, max_layers: int = 2,
                  output: str | None = None, breakdown: bool = False) -> int:
    """Core analysis logic. If output is given, tee stdout to that file."""
    project_dir = resolve_project_dir()
    if project_dir:
        print(f"[INFO] PROJECT_DIR = {project_dir}", file=sys.stderr)

    summaries = []
    for log_path in log_paths:
        resolved = resolve_path(log_path, project_dir)
        p = Path(resolved)
        if not p.is_file():
            print(f"[WARN] File not found: {resolved}", file=sys.stderr)
            continue
        summaries.append(parse_log(resolved, npu=npu))

    if not summaries:
        print("No valid log files.", file=sys.stderr)
        return 1

    orig_stdout = sys.stdout
    out_file = None
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_file = open(out_path, "w", encoding="utf-8")
        sys.stdout = TeeWriter(orig_stdout, out_file)

    try:
        print_summary_table(summaries, max_layers=max_layers)
        if len(summaries) >= 2:
            print_comparison(summaries)
        if breakdown:
            print_module_breakdown(summaries)
    finally:
        sys.stdout = orig_stdout
        if out_file:
            out_file.close()
            print(f"[INFO] Report saved to: {output}", file=sys.stderr)

    return 0


class TeeWriter:
    """Write to two streams simultaneously."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze per-layer communication latency from ASTRA-sim logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("logs", nargs="+", help="Log file path(s). Supports {PROJECT_DIR} placeholder.")
    parser.add_argument("--npu", type=int, default=0, help="NPU id to analyze (default: 0)")
    parser.add_argument("--layers", type=int, default=2, help="Max layers to show detail (0=all)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save report to file (in addition to stdout)")
    parser.add_argument("--breakdown", "-b", action="store_true",
                        help="Show per-module breakdown (compute vs comm, MHA vs FFN)")
    args = parser.parse_args()

    return run_analysis(args.logs, npu=args.npu, max_layers=args.layers,
                        output=args.output, breakdown=args.breakdown)


if __name__ == "__main__":
    raise SystemExit(main())
