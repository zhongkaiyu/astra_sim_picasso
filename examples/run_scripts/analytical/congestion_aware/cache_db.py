#!/usr/bin/env python3
# ============================================================================
# ASTRA-sim ?????????
# ============================================================================
# ???
#   - ?????????????
#   - ???????�? Key
#   - ??? SQLite + CSV
# ============================================================================

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def get_project_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    return (script_dir / "../../../..").resolve()


def get_example_dir(project_dir: Path) -> Path:
    return project_dir / "examples"


def get_cache_dir() -> Path:
    return Path(__file__).resolve().parent / "cache_db"


def get_default_db_path() -> Path:
    return get_cache_dir() / "results.sqlite"


def get_default_csv_path() -> Path:
    return get_cache_dir() / "results_cache.csv"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_value(value: Any, project_dir: Path, example_dir: Path) -> Any:
    if isinstance(value, str):
        return value.format(PROJECT_DIR=str(project_dir), EXAMPLE_DIR=str(example_dir))
    return value


def load_config(config_path: Path, project_dir: Path, example_dir: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    def get(key: str, default: Any = None) -> Any:
        val = cfg.get(key, default)
        if val is None:
            return None
        return format_value(val, project_dir, example_dir)

    trace_cfg = cfg.get("trace") if isinstance(cfg.get("trace"), dict) else {}
    trace_model_type = trace_cfg.get("model_type") if isinstance(trace_cfg, dict) else None
    trace_args = trace_cfg.get("args") if isinstance(trace_cfg, dict) else None
    trace_output_name = trace_cfg.get("output_name") if isinstance(trace_cfg, dict) else None
    trace_force = trace_cfg.get("force") if isinstance(trace_cfg, dict) else None

    return {
        "output_dir": get("output_dir"),
        "astra_sim": get("astra_sim"),
        "system": get("system"),
        "network": get("network"),
        "remote_memory": get("remote_memory"),
        "workload_dir": get("workload_dir"),
        "workload_base": get("workload_base"),
        "comm_group": get("comm_group"),
        "log_file": get("log_file"),
        "trace_model_type": get("trace_model_type", trace_model_type),
        "trace_args": get("trace_args", trace_args),
        "trace_output_name": get("trace_output_name", trace_output_name),
        "trace_force": get("trace_force", trace_force),
    }


def _parse_parallel_factors(name: str) -> Dict[str, Optional[int]]:
    factors = {"tp": None, "dp": None, "pp": None, "sp": None, "cp": None}
    for match in re.finditer(r"(tp|dp|pp|sp|cp)(\d+)", name.lower()):
        factors[match.group(1)] = int(match.group(2))
    return factors


def _parse_config_name_rules(config_name: str) -> dict:
    name = (config_name or "").lower()
    if not name:
        return {}
    pattern = re.compile(
        r"^(?P<par>(?:tp|dp|pp|sp|cp)\d+)_"
        r"(?P<device>[^_]+)_"
        r"(?P<physical>[^_]+)_"
        r"(?P<num_gpus>[^_]+)_"
        r"(?P<logical>[^_]+)_"
        r"(?P<rest>.+)$"
    )
    match = pattern.match(name)
    if match:
        physical = match.group("physical")
        num_gpus = match.group("num_gpus")
        logical = match.group("logical")
        rest_tokens = match.group("rest").split("_")
    else:
        tokens = name.split("_")
        if len(tokens) < 6:
            return {}
        physical = tokens[2]
        num_gpus = tokens[3]
        logical = tokens[4]
        rest_tokens = tokens[5:]
    batch_size = None
    seq = None
    filtered = []
    idx = 0
    while idx < len(rest_tokens):
        token = rest_tokens[idx]
        next_token = rest_tokens[idx + 1] if idx + 1 < len(rest_tokens) else ""
        if re.fullmatch(r"(?:b|batch|batch_size|batchsize|bs)\d+", token):
            batch_size = int(re.sub(r"\D+", "", token))
            idx += 1
            continue
        if re.fullmatch(r"(?:s|seq|seq_length|seqlen|sl)\d+", token):
            seq = int(re.sub(r"\D+", "", token))
            idx += 1
            continue
        if token in {"batch", "batch_size"} and re.fullmatch(r"size\d+", next_token):
            batch_size = int(re.sub(r"\D+", "", next_token))
            idx += 2
            continue
        if token in {"seq", "seq_length"} and re.fullmatch(r"length\d+", next_token):
            seq = int(re.sub(r"\D+", "", next_token))
            idx += 2
            continue
        if token:
            filtered.append(token)
        idx += 1

    model = filtered[0] if filtered else None
    operation = "_".join(filtered[1:]) if len(filtered) > 1 else None
    return {
        "physical_topology": physical,
        "num_gpus": num_gpus,
        "logical_topology": logical,
        "model": model,
        "operation": operation,
        "batch_size": batch_size,
        "seq": seq,
    }


def parse_config_name_meta(config_name: str) -> Dict[str, Optional[int]]:
    rules = _parse_config_name_rules(config_name)
    return {
        "batch_size": rules.get("batch_size"),
        "seq": rules.get("seq"),
    }


def _extract_model_type_from_config_name(config_name: str) -> Optional[str]:
    name = (config_name or "").lower()
    if not name:
        return None
    last_pos = -1
    last_token = ""
    for token in ("mesh2d", "mesh23", "ring", "alltoall", "direct"):
        pos = name.rfind(token)
        if pos > last_pos:
            last_pos = pos
            last_token = token
    if last_pos == -1:
        return None
    suffix = name[last_pos + len(last_token) :]
    if suffix.startswith("_"):
        suffix = suffix[1:]
    return suffix or None


def _guess_model(config_name: str) -> Optional[str]:
    model_type = _extract_model_type_from_config_name(config_name)
    if not model_type:
        return None
    return model_type.split("_", 1)[0]


def _guess_operation(config_name: str) -> Optional[str]:
    model_type = _extract_model_type_from_config_name(config_name)
    if not model_type or "_" not in model_type:
        return None
    return model_type.split("_", 1)[1]


def is_comm_config_name(config_name: str) -> bool:
    return (config_name or "").lower().startswith("comm_")


def extract_key_fields(config_path: Path, config: Dict[str, Any]) -> Dict[str, Any]:
    config_name = config_path.stem
    is_comm = is_comm_config_name(config_name)
    factors = {"tp": None, "dp": None, "pp": None, "sp": None, "cp": None}
    rules = {}
    model = None
    operation = None
    batch_size = None
    seq = None
    if not is_comm:
        factors = _parse_parallel_factors(config_name)
        rules = _parse_config_name_rules(config_name)
        model = rules.get("model") or _guess_model(config_name)
        operation = rules.get("operation") or _guess_operation(config_name)
        batch_size = config.get("batch_size") or config.get("batch") or rules.get("batch_size")
        seq = config.get("seq") or config.get("seq_length") or rules.get("seq")

    return {
        "config_name": config_name,
        "config_path": str(config_path.resolve()),
        "model": model,
        "operation": operation,
        "batch_size": batch_size,
        "seq": seq,
        "tp": factors["tp"],
        "dp": factors["dp"],
        "pp": factors["pp"],
        "cp": factors["cp"],
        "sp": factors["sp"],
        "physical_topology": rules.get("physical_topology"),
        "num_gpus": rules.get("num_gpus"),
        "logical_topology": rules.get("logical_topology"),
        "output_dir": config.get("output_dir"),
        "astra_sim": config.get("astra_sim"),
        "system": config.get("system"),
        "network": config.get("network"),
        "remote_memory": config.get("remote_memory"),
        "workload_dir": config.get("workload_dir"),
        "workload_base": config.get("workload_base"),
        "comm_group": config.get("comm_group"),
        "log_file": config.get("log_file"),
    }


def compute_key_hash(key_fields: Dict[str, Any]) -> str:
    key_json = json.dumps(key_fields, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(key_json.encode("utf-8")).hexdigest()


def init_db(db_path: Path) -> None:
    ensure_parent_dir(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                key_hash TEXT PRIMARY KEY,
                config_name TEXT,
                config_path TEXT,
                key_fields TEXT,
                log_path TEXT,
                wall_time INTEGER,
                gpu_time INTEGER,
                comm_time INTEGER,
                overlap INTEGER,
                compute_util REAL,
                memory_util REAL,
                created_at TEXT,
                status TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_config_name ON results(config_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_results_status ON results(status)")


def parse_log_metrics(log_path: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "wall_time": None,
        "gpu_time": None,
        "comm_time": None,
        "overlap": None,
        "compute_util": None,
        "memory_util": None,
    }
    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return metrics

    def _extract_int(pattern: str) -> Optional[int]:
        match = re.search(pattern, content)
        return int(match.group(1)) if match else None

    def _extract_float(pattern: str) -> Optional[float]:
        match = re.search(pattern, content)
        return float(match.group(1)) if match else None

    metrics["wall_time"] = _extract_int(r"sys\[0\].*Wall time: (\d+)")
    metrics["gpu_time"] = _extract_int(r"sys\[0\].*GPU time: (\d+)")
    metrics["comm_time"] = _extract_int(r"sys\[0\].*Comm time: (\d+)")
    metrics["overlap"] = _extract_int(r"sys\[0\].*Total compute-communication overlap: (\d+)")
    metrics["compute_util"] = _extract_float(r"sys\[0\].*Average compute utilization: ([\d.]+)")
    metrics["memory_util"] = _extract_float(r"sys\[0\].*Average memory utilization: ([\d.]+)")
    return metrics


def save_result(
    db_path: Path,
    key_hash: str,
    key_fields: Dict[str, Any],
    log_path: Optional[Path],
    metrics: Dict[str, Any],
    status: str,
) -> None:
    init_db(db_path)
    record = (
        key_hash,
        key_fields.get("config_name"),
        key_fields.get("config_path"),
        json.dumps(key_fields, sort_keys=True, ensure_ascii=True),
        str(log_path) if log_path else None,
        metrics.get("wall_time"),
        metrics.get("gpu_time"),
        metrics.get("comm_time"),
        metrics.get("overlap"),
        metrics.get("compute_util"),
        metrics.get("memory_util"),
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        status,
    )
    with sqlite3.connect(db_path) as conn:
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


def query_result(db_path: Path, key_hash: str) -> Optional[Dict[str, Any]]:
    if not db_path.is_file():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM results WHERE key_hash = ?", (key_hash,)).fetchone()
        if not row:
            return None
        result = dict(row)
        if result.get("key_fields"):
            try:
                result["key_fields"] = json.loads(result["key_fields"])
            except json.JSONDecodeError:
                pass
        return result


def append_csv(csv_path: Path, key_hash: str, key_fields: Dict[str, Any], metrics: Dict[str, Any], status: str) -> None:
    ensure_parent_dir(csv_path)
    header = [
        "key_hash",
        "config_name",
        "config_path",
        "model",
        "operation",
        "batch_size",
        "seq",
        "tp",
        "dp",
        "pp",
        "sp",
        "cp",
        "physical_topology",
        "num_gpus",
        "logical_topology",
        "wall_time",
        "gpu_time",
        "comm_time",
        "overlap",
        "compute_util",
        "memory_util",
        "status",
        "created_at",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "key_hash": key_hash,
                "config_name": key_fields.get("config_name"),
                "config_path": key_fields.get("config_path"),
                "model": key_fields.get("model"),
                "operation": key_fields.get("operation"),
                "batch_size": key_fields.get("batch_size"),
                "seq": key_fields.get("seq"),
                "tp": key_fields.get("tp"),
                "dp": key_fields.get("dp"),
                "pp": key_fields.get("pp"),
                "sp": key_fields.get("sp"),
                "cp": key_fields.get("cp"),
                "physical_topology": key_fields.get("physical_topology"),
                "num_gpus": key_fields.get("num_gpus"),
                "logical_topology": key_fields.get("logical_topology"),
                "wall_time": metrics.get("wall_time"),
                "gpu_time": metrics.get("gpu_time"),
                "comm_time": metrics.get("comm_time"),
                "overlap": metrics.get("overlap"),
                "compute_util": metrics.get("compute_util"),
                "memory_util": metrics.get("memory_util"),
                "status": status,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

