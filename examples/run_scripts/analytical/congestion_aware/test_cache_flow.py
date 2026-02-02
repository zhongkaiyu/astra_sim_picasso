#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# ASTRA-sim 缓存流程测试脚本
# ============================================================================
# 默认仅测试 DB/CSV 写入与查询（不运行仿真）
# 如需真实仿真：--mode run
# ============================================================================

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from cache_db import (
    append_csv,
    compute_key_hash,
    extract_key_fields,
    get_default_csv_path,
    get_default_db_path,
    get_example_dir,
    get_project_dir,
    load_config,
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


def test_db_flow(config_path: Path, db_path: Path, csv_path: Path) -> None:
    project_dir = get_project_dir()
    example_dir = get_example_dir(project_dir)
    cfg = load_config(config_path, project_dir, example_dir)
    key_fields = extract_key_fields(config_path, cfg)
    key_hash = compute_key_hash(key_fields)

    metrics = {
        "wall_time": 123456,
        "gpu_time": 80000,
        "comm_time": 40000,
        "overlap": 10000,
        "compute_util": 70.5,
        "memory_util": 55.2,
    }
    save_result(db_path, key_hash, key_fields, None, metrics, status="TEST_ONLY")
    append_csv(csv_path, key_hash, key_fields, metrics, status="TEST_ONLY")

    result = query_result(db_path, key_hash)
    if result:
        print(f"[HIT] {config_path.name} -> TEST_ONLY")
        print(
            "  wall_time={wall_time}, gpu_time={gpu_time}, comm_time={comm_time}, "
            "status={status}".format(
                wall_time=result.get("wall_time"),
                gpu_time=result.get("gpu_time"),
                comm_time=result.get("comm_time"),
                status=result.get("status"),
            )
        )
    else:
        print(f"[MISS] {config_path.name} -> test write failed")


def test_run_flow(config_path: Path, db_path: Path, csv_path: Path) -> None:
    script_dir = Path(__file__).resolve().parent
    run_script = script_dir / "run_with_cache.py"
    if not run_script.is_file():
        print(f"[ERROR] run_with_cache.py not found: {run_script}")
        return
    subprocess.run(
        ["python3", str(run_script), str(config_path), "--db", str(db_path), "--csv", str(csv_path)],
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Test cache pipeline (DB/CSV or real run).")
    parser.add_argument("configs", nargs="+", help="Config json files")
    parser.add_argument("--mode", choices=["db", "run"], default="db", help="Test mode")
    parser.add_argument("--db", type=str, default=str(get_default_db_path()), help="SQLite db path")
    parser.add_argument("--csv", type=str, default=str(get_default_csv_path()), help="CSV cache path")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    configs_dir = script_dir / "configs"
    db_path = Path(args.db)
    csv_path = Path(args.csv)

    for config_arg in args.configs:
        config_path = resolve_config_path(config_arg, configs_dir)
        if not config_path.is_file():
            print(f"[MISS] Config not found: {config_arg}")
            continue

        if args.mode == "db":
            test_db_flow(config_path, db_path, csv_path)
        else:
            test_run_flow(config_path, db_path, csv_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

