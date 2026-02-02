#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# ASTRA-sim 结果缓存查询脚本
# ============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cache_db import (
    compute_key_hash,
    extract_key_fields,
    get_default_db_path,
    get_example_dir,
    get_project_dir,
    load_config,
    query_result,
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Query cached results by config")
    parser.add_argument("configs", nargs="+", help="Config json files")
    parser.add_argument("--db", type=str, default=str(get_default_db_path()), help="SQLite db path")
    parser.add_argument("--print-json", action="store_true", help="Print full record as JSON")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    configs_dir = script_dir / "configs"
    project_dir = get_project_dir()
    example_dir = get_example_dir(project_dir)
    db_path = Path(args.db)

    for config_arg in args.configs:
        config_path = resolve_config_path(config_arg, configs_dir)
        if not config_path.is_file():
            print(f"[MISS] Config not found: {config_arg}")
            continue

        cfg = load_config(config_path, project_dir, example_dir)
        key_fields = extract_key_fields(config_path, cfg)
        key_hash = compute_key_hash(key_fields)
        result = query_result(db_path, key_hash)

        if not result:
            print(f"[MISS] {config_path.name} -> no cache")
            continue

        print(f"[HIT] {config_path.name}")
        if args.print_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(
                "  wall_time={wall_time}, gpu_time={gpu_time}, comm_time={comm_time}, "
                "status={status}".format(
                    wall_time=result.get("wall_time"),
                    gpu_time=result.get("gpu_time"),
                    comm_time=result.get("comm_time"),
                    status=result.get("status"),
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

