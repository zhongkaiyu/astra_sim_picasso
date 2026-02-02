#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# ASTRA-sim ²éÑ¯»º´æ + Ö´ÐÐ½Å±¾
# ============================================================================

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Optional

from cache_db import (
    append_csv,
    compute_key_hash,
    extract_key_fields,
    get_default_csv_path,
    get_default_db_path,
    get_example_dir,
    get_project_dir,
    load_config,
    parse_log_metrics,
    query_result,
    save_result,
)


def resolve_config_path(config_path: str, configs_dir: Path) -> Path:
    if config_path.startswith("/") and Path(config_path).is_file():
        return Path(config_path)

    path = Path(config_path)
    if path.is_file():
        return path.resolve()

    candidate = configs_dir / config_path
    if candidate.is_file():
        return candidate

    basename_candidate = configs_dir / Path(config_path).name
    if basename_candidate.is_file():
        return basename_candidate

    return path


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


def get_log_path(config: dict) -> Optional[Path]:
    output_dir = config.get("output_dir")
    log_file = config.get("log_file")
    if not output_dir or not log_file:
        return None
    return Path(output_dir) / "logs" / log_file


def main() -> int:
    parser = argparse.ArgumentParser(description="Query cache, run if miss, and save results.")
    parser.add_argument("configs", nargs="+", help="Config json files")
    parser.add_argument("--db", type=str, default=str(get_default_db_path()), help="SQLite db path")
    parser.add_argument("--csv", type=str, default=str(get_default_csv_path()), help="CSV cache path")
    parser.add_argument("--force-run", action="store_true", help="Ignore cache and force run")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    run_script = script_dir / "run_from_config.py"
    configs_dir = script_dir / "configs"
    project_dir = get_project_dir()
    example_dir = get_example_dir(project_dir)
    db_path = Path(args.db)
    csv_path = Path(args.csv)

    if not run_script.is_file():
        print(f"[ERROR] run_from_config.py not found: {run_script}")
        return 1

    for config_arg in args.configs:
        config_path = resolve_config_path(config_arg, configs_dir)
        if not config_path.is_file():
            print(f"[MISS] Config not found: {config_arg}")
            continue

        cfg = load_config(config_path, project_dir, example_dir)
        key_fields = extract_key_fields(config_path, cfg)
        key_hash = compute_key_hash(key_fields)

        if not args.force_run:
            cached = query_result(db_path, key_hash)
            if cached:
                print(f"[HIT] {config_path.name} -> cached")
                print(
                    "  wall_time={wall_time}, gpu_time={gpu_time}, comm_time={comm_time}, "
                    "status={status}".format(
                        wall_time=cached.get("wall_time"),
                        gpu_time=cached.get("gpu_time"),
                        comm_time=cached.get("comm_time"),
                        status=cached.get("status"),
                    )
                )
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

        log_path = get_log_path(cfg)
        metrics = parse_log_metrics(log_path) if log_path else {}
        status = "SUCCESS" if log_path and log_path.is_file() else "WARNING: log not found"
        save_result(db_path, key_hash, key_fields, log_path, metrics, status)
        append_csv(csv_path, key_hash, key_fields, metrics, status)
        print(f"[SAVE] {status}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

