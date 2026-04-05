#!/usr/bin/env python3
# ============================================================================
# CSV -> SQLite 结果导入脚本
# ============================================================================
# 用途：
#   将 results_cache.csv 导入 results.sqlite
# ============================================================================

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

from cache_db import get_default_csv_path, get_default_db_path, init_db


def _to_int(value: Optional[str]) -> Optional[int]:
    if value is None or value == "":
        return None
    return int(value)


def _to_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)


def build_key_fields(row: Dict[str, str]) -> Dict[str, Any]:
    fields = {
        "config_name": row.get("config_name"),
        "config_path": row.get("config_path"),
        "model": row.get("model"),
        "operation": row.get("operation"),
        "batch_size": _to_int(row.get("batch_size")),
        "seq": _to_int(row.get("seq")),
        "tp": _to_int(row.get("tp")),
        "dp": _to_int(row.get("dp")),
        "pp": _to_int(row.get("pp")),
        "sp": _to_int(row.get("sp")),
        "cp": _to_int(row.get("cp")),
        "physical_topology": row.get("physical_topology"),
        "num_gpus": row.get("num_gpus"),
        "logical_topology": row.get("logical_topology"),
    }
    return {k: v for k, v in fields.items() if v is not None and v != ""}


def import_csv_to_sqlite(csv_path: Path, db_path: Path) -> int:
    if not csv_path.is_file():
        raise SystemExit(f"CSV 不存在: {csv_path}")

    init_db(db_path)
    inserted = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        with sqlite3.connect(db_path) as conn:
            for row in reader:
                key_hash = row.get("key_hash")
                if not key_hash:
                    continue
                key_fields = build_key_fields(row)
                record = (
                    key_hash,
                    row.get("config_name"),
                    row.get("config_path"),
                    json.dumps(key_fields, sort_keys=True, ensure_ascii=True),
                    None,
                    _to_int(row.get("wall_time")),
                    _to_int(row.get("gpu_time")),
                    _to_int(row.get("comm_time")),
                    _to_int(row.get("overlap")),
                    _to_float(row.get("compute_util")),
                    _to_float(row.get("memory_util")),
                    row.get("created_at"),
                    row.get("status"),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO results (
                        key_hash, config_name, config_path, key_fields, log_path,
                        wall_time, gpu_time, comm_time, overlap, compute_util, memory_util,
                        created_at, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    record,
                )
                inserted += 1

    return inserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Import results_cache.csv to results.sqlite")
    parser.add_argument("--csv", type=str, default=str(get_default_csv_path()), help="CSV 路径")
    parser.add_argument("--db", type=str, default=str(get_default_db_path()), help="SQLite 路径")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    db_path = Path(args.db)
    count = import_csv_to_sqlite(csv_path, db_path)
    print(f"[DONE] Imported {count} rows into {db_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




