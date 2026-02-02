#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# ASTRA-sim ???? + ??????
# ============================================================================
# ???????? configs/ ????????? CSV + SQLite??????
# ============================================================================

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import Iterable, Optional

from cache_db import (
    append_csv,
    compute_key_hash,
    extract_key_fields,
    get_default_csv_path,
    get_default_db_path,
    get_example_dir,
    get_project_dir,
    is_comm_config_name,
    load_config,
    parse_config_name_meta,
    parse_log_metrics,
    query_result,
    save_result,
)


def iter_config_files(configs_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(configs_dir.glob(pattern))


def resolve_log_path(config: dict) -> Optional[Path]:
    output_dir = config.get("output_dir")
    log_file = config.get("log_file")
    if not output_dir or not log_file:
        return None
    return Path(output_dir) / "logs" / log_file


def run_simulation(run_script: Path, config_path: Path) -> int:
    result = subprocess.run(
        ["python3", str(run_script), str(config_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        text=True,
    )
    print(result.stdout)
    return result.returncode


def format_value(value: str, project_dir: Path, example_dir: Path) -> str:
    return value.format(PROJECT_DIR=str(project_dir), EXAMPLE_DIR=str(example_dir))


def load_trace_config(trace_path: Path, project_dir: Path, example_dir: Path) -> dict:
    with trace_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    resolved = {}
    for key, value in cfg.items():
        if isinstance(value, str):
            resolved[key] = format_value(value, project_dir, example_dir)
        else:
            resolved[key] = value
    return resolved


def build_trace_meta_index(trace_configs_dir: Path, project_dir: Path, example_dir: Path) -> dict:
    index: dict = {}
    for trace_path in iter_config_files(trace_configs_dir, "*.json"):
        trace_cfg = load_trace_config(trace_path, project_dir, example_dir)
        output_name = trace_cfg.get("output_name")
        output_dir = trace_cfg.get("output_dir")
        batch_size = trace_cfg.get("batch_size", trace_cfg.get("batch"))
        seq = trace_cfg.get("seq", trace_cfg.get("seq_length"))
        meta = {"batch_size": batch_size, "seq": seq}
        if output_name:
            index[output_name] = meta
        if output_dir and output_name:
            index[(str(output_dir), str(output_name))] = meta
    return index


def parse_parallel_factors(name: str) -> dict:
    factors = {"tp": None, "dp": None, "pp": None, "sp": None, "cp": None}
    for match in re.finditer(r"(tp|dp|pp|sp|cp)(\d+)", name.lower()):
        factors[match.group(1)] = int(match.group(2))
    return factors


def guess_model_type(workload_base: str) -> str:
    base = (workload_base or "").lower()
    if "decode_bypass_tp16" in base:
        return "qwen_decode_bypass_tp16"
    if "decode_bypass" in base:
        return "qwen_decode_bypass"
    if "decode_attention" in base:
        return "qwen_decode_attention"
    if "forward_only_bypass" in base:
        return "qwen_forward_only_bypass"
    if "forward_attention" in base:
        return "qwen_forward_attention"
    if "moe" in base:
        return "moe"
    return "dense"


def should_generate_trace(output_dir: Path, output_name: str, force: bool) -> bool:
    if force:
        return True
    trace_file = output_dir / f"{output_name}.0.et"
    return not trace_file.is_file()


def build_trace_cmd(main_script: Path, trace_cfg: dict) -> list[str]:
    if not trace_cfg.get("output_dir") or not trace_cfg.get("output_name"):
        raise SystemExit("Trace config must set output_dir and output_name")
    cmd = [
        "python3",
        str(main_script),
        "--output_dir",
        str(trace_cfg["output_dir"]),
        "--output_name",
        str(trace_cfg["output_name"]),
    ]
    flag_keys = [
        "dp",
        "tp",
        "pp",
        "sp",
        "ep",
        "dvocal",
        "dmodel",
        "dff",
        "batch",
        "micro_batch",
        "seq",
        "head",
        "kvhead",
        "num_stacks",
        "experts",
        "kexperts",
        "chakra_schema_version",
        "model_type",
        "weight_sharded",
        "activation_recompute",
        "tpsp",
        "mixed_precision",
        "print_gpu_vram",
    ]
    for key in flag_keys:
        if key in trace_cfg and trace_cfg[key] is not None:
            cmd.extend([f"--{key}", str(trace_cfg[key])])

    trace_args = trace_cfg.get("args")
    if trace_args:
        cmd.extend(shlex.split(str(trace_args)))
    return cmd


def run_trace_from_config(
    main_script: Path,
    trace_path: Path,
    trace_cfg: dict,
    trace_force: bool,
) -> int:
    output_dir = Path(trace_cfg["output_dir"])
    output_name = str(trace_cfg["output_name"])
    if not should_generate_trace(output_dir, output_name, trace_force):
        print(f"[SKIP] Trace exists: {output_dir}/{output_name}.0.et")
        return 0
    cmd = build_trace_cmd(main_script, trace_cfg)
    print(f"[TRACE] {trace_path.name} -> {output_dir}/{output_name}.%d.et")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def run_trace_generator(
    main_script: Path,
    config_path: Path,
    cfg: dict,
    trace_model_type: str | None,
    trace_args: str | None,
    trace_force: bool,
) -> int:
    workload_dir = cfg.get("workload_dir")
    workload_base = cfg.get("workload_base")
    if not workload_dir or not workload_base:
        print(f"[WARN] {config_path.name} missing workload_dir/workload_base, skip trace")
        return 0

    output_dir = Path(workload_dir)
    output_name = cfg.get("trace_output_name") or workload_base
    if not should_generate_trace(output_dir, output_name, trace_force):
        print(f"[SKIP] Trace exists: {output_dir}/{output_name}.0.et")
        return 0

    factors = parse_parallel_factors(config_path.stem)
    dp = factors["dp"] or 1
    tp = factors["tp"] or 1
    pp = factors["pp"] or 1
    model_type = trace_model_type or guess_model_type(workload_base)

    cmd = [
        "python3",
        str(main_script),
        "--output_dir",
        str(output_dir),
        "--output_name",
        str(output_name),
        "--dp",
        str(dp),
        "--tp",
        str(tp),
        "--pp",
        str(pp),
        "--model_type",
        model_type,
    ]
    if trace_args:
        cmd.extend(shlex.split(trace_args))

    print(f"[TRACE] {config_path.name} -> {output_dir}/{output_name}.%d.et")
    result = subprocess.run(cmd, check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch run all configs and save results to DB/CSV.")
    parser.add_argument("--configs-dir", type=str, default="", help="Configs directory")
    parser.add_argument("--pattern", type=str, default="*.json", help="Glob pattern in configs dir")
    parser.add_argument("--db", type=str, default="", help="SQLite db path")
    parser.add_argument("--csv", type=str, default="", help="CSV cache path")
    parser.add_argument("--force-run", action="store_true", help="Ignore cache and force run")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of configs (0 = all)")
    parser.add_argument("--gen-trace", action="store_true", help="Generate chakra trace via main.py")
    parser.add_argument(
        "--trace-configs-dir",
        type=str,
        default="",
        help="Separate trace configs directory (JSON)",
    )
    parser.add_argument("--trace-pattern", type=str, default="*.json", help="Glob pattern in trace configs dir")
    parser.add_argument("--trace-force", action="store_true", help="Force regenerate chakra trace")
    parser.add_argument(
        "--trace-model-type",
        type=str,
        default="",
        help="Override model_type passed to trace generator",
    )
    parser.add_argument(
        "--trace-args",
        type=str,
        default="",
        help="Extra args for trace generator (quoted string)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_script = script_dir / "run_from_config.py"
    project_dir = get_project_dir()
    trace_script = (project_dir / "symbolic_tensor_graph_picasso/main.py").resolve()
    configs_dir = Path(args.configs_dir) if args.configs_dir else (script_dir / "configs")
    trace_configs_dir = Path(args.trace_configs_dir) if args.trace_configs_dir else (script_dir / "trace_configs")
    cache_dir = script_dir / "cache_db_test"
    db_path = Path(args.db) if args.db else (cache_dir / "results.sqlite")
    csv_path = Path(args.csv) if args.csv else (cache_dir / "results_cache.csv")

    if not run_script.is_file():
        print(f"[ERROR] run_from_config.py not found: {run_script}")
        return 1
    if args.gen_trace and not trace_script.is_file():
        print(f"[ERROR] trace generator not found: {trace_script}")
        return 1
    if not configs_dir.is_dir():
        print(f"[ERROR] configs dir not found: {configs_dir}")
        return 1

    example_dir = get_example_dir(project_dir)

    trace_meta_index = {}
    if trace_configs_dir and trace_configs_dir.is_dir():
        trace_meta_index = build_trace_meta_index(trace_configs_dir, project_dir, example_dir)

    if args.gen_trace and trace_configs_dir:
        if not trace_configs_dir.is_dir():
            print(f"[ERROR] trace configs dir not found: {trace_configs_dir}")
            return 1
        trace_files = list(iter_config_files(trace_configs_dir, args.trace_pattern))
        if not trace_files:
            print(f"[WARN] No trace configs found: {trace_configs_dir}/{args.trace_pattern}")
            return 0
        print(f"[INFO] Total trace configs: {len(trace_files)}")
        for trace_path in trace_files:
            trace_cfg = load_trace_config(trace_path, project_dir, example_dir)
            trace_force = args.trace_force or bool(trace_cfg.get("force"))
            rc = run_trace_from_config(trace_script, trace_path, trace_cfg, trace_force)
            if rc != 0:
                print(f"[FAIL] Trace generation failed with code {rc}")
                return rc

    config_files = list(iter_config_files(configs_dir, args.pattern))
    if args.limit > 0:
        config_files = config_files[: args.limit]

    if not config_files:
        print(f"[WARN] No configs found: {configs_dir}/{args.pattern}")
        return 0

    print(f"[INFO] Total configs: {len(config_files)}")
    for config_path in config_files:
        cfg = load_config(config_path, project_dir, example_dir)
        is_comm = is_comm_config_name(config_path.stem)
        if trace_meta_index and not is_comm:
            workload_dir = cfg.get("workload_dir")
            workload_base = cfg.get("workload_base")
            meta = None
            if workload_dir and workload_base:
                meta = trace_meta_index.get((str(workload_dir), str(workload_base)))
            if not meta and workload_base:
                meta = trace_meta_index.get(str(workload_base))
            if meta:
                cfg["batch_size"] = meta.get("batch_size")
                cfg["seq"] = meta.get("seq")
        if not is_comm:
            name_meta = parse_config_name_meta(config_path.stem)
            if name_meta:
                if cfg.get("batch_size") is None:
                    cfg["batch_size"] = name_meta.get("batch_size")
                if cfg.get("seq") is None:
                    cfg["seq"] = name_meta.get("seq")
        key_fields = extract_key_fields(config_path, cfg)
        key_hash = compute_key_hash(key_fields)

        if args.gen_trace and not trace_configs_dir:
            config_trace_model = cfg.get("trace_model_type")
            config_trace_args = cfg.get("trace_args")
            config_trace_force = cfg.get("trace_force")
            trace_model_type = args.trace_model_type or config_trace_model or None
            trace_args = args.trace_args or config_trace_args or None
            trace_force = args.trace_force or bool(config_trace_force)
            rc = run_trace_generator(
                trace_script,
                config_path,
                cfg,
                trace_model_type,
                trace_args,
                trace_force,
            )
            if rc != 0:
                print(f"[FAIL] Trace generation failed with code {rc}")
                continue

        if not args.force_run:
            cached = query_result(db_path, key_hash)
            if cached:
                print(f"[HIT] {config_path.name} -> cached")
                continue

        print(f"[RUN] {config_path.name}")
        rc = run_simulation(run_script, config_path)
        if rc != 0:
            print(f"[FAIL] Simulation failed with code {rc}")
            save_result(
                db_path,
                key_hash,
                key_fields,
                None,
                {},
                status=f"FAILED: returncode={rc}",
            )
            append_csv(csv_path, key_hash, key_fields, {}, status=f"FAILED: returncode={rc}")
            continue

        log_path = resolve_log_path(cfg)
        metrics = parse_log_metrics(log_path) if log_path else {}
        status = "SUCCESS" if log_path and log_path.is_file() else "WARNING: log not found"
        save_result(db_path, key_hash, key_fields, log_path, metrics, status)
        append_csv(csv_path, key_hash, key_fields, metrics, status)
        print(f"[SAVE] {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

