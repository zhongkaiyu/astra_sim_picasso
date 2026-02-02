#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
# ASTRA-sim 测试脚本 (从 JSON 配置读取并运行)
# ============================================================================

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def format_value(value, project_dir: Path, example_dir: Path):
    if isinstance(value, str):
        return value.format(PROJECT_DIR=str(project_dir), EXAMPLE_DIR=str(example_dir))
    return value


def load_config(config_path: Path, project_dir: Path, example_dir: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    def get(key, default=None):
        val = cfg.get(key, default)
        if val is None:
            return None
        return format_value(val, project_dir, example_dir)

    output_dir = get("output_dir", f"{project_dir}/output_qwen/test/dp16_h100_mesh2d_4x4_decode_bypass")
    astra_sim = get(
        "astra_sim",
        f"{project_dir}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
    )
    system = get("system", f"{example_dir}/system/native_collectives/DP16_H100.json")
    network = get("network", f"{example_dir}/network/analytical/Mesh2D_16gpus_4x4_H100.yml")
    remote_memory = get("remote_memory", f"{example_dir}/remote_memory/analytical/no_memory_expansion.json")
    workload_dir = get("workload_dir")
    workload_base = get("workload_base")
    comm_group = get("comm_group")
    log_file = get("log_file", "simulation_log_dp16_h100_mesh2d_4x4_decode_bypass.txt")

    if not workload_dir or not workload_base:
        raise SystemExit("Config must set workload_dir and workload_base")

    workload_base_path = f"{workload_dir}/{workload_base}"
    if comm_group is None:
        comm_group = f"{workload_base_path}.json"

    return {
        "output_dir": str(output_dir),
        "astra_sim": str(astra_sim),
        "system": str(system),
        "network": str(network),
        "remote_memory": str(remote_memory),
        "workload_dir": str(workload_dir),
        "workload_base": str(workload_base_path),
        "comm_group": str(comm_group),
        "log_file": str(log_file),
    }


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_dir = (script_dir / "../../../..").resolve()
    example_dir = project_dir / "examples"

    config_json = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / "configs" / "dp16_h100_mesh2d_4x4.json"
    if not config_json.is_file():
        print(f"ERROR: Config JSON not found: {config_json}")
        return 1

    config = load_config(config_json, project_dir, example_dir)

    output_dir = Path(config["output_dir"])
    output_dir.joinpath("logs").mkdir(parents=True, exist_ok=True)
    output_dir.joinpath("analysis").mkdir(parents=True, exist_ok=True)

    workload_base = config["workload_base"]
    workload_et = Path(f"{workload_base}.0.et")
    comm_group = Path(config["comm_group"])

    print("Checking workload files...")
    if not workload_et.is_file():
        print(f"WARNING: Workload file not found: {workload_et}")
        print("Please ensure workload files exist in:")
        print(f"  {config['workload_dir']}/")
        return 1
    print(f"? Workload files found: {workload_base}")

    if not comm_group.is_file():
        print(f"WARNING: Communication group file not found: {comm_group}")
        return 1
    print(f"? Comm group file found: {comm_group}")

    astra_sim = Path(config["astra_sim"])
    if not astra_sim.is_file():
        print(f"ERROR: ASTRA-sim binary not found: {astra_sim}")
        print("Please build ASTRA-sim first:")
        print("  cd build/astra_analytical && bash build.sh")
        return 1

    print("")
    print("==========================================================================")
    print("[ASTRA-sim] Run from config")
    print("==========================================================================")
    print(f"Workload: {workload_base}")
    print(f"Comm Group: {comm_group}")
    print(f"System: {config['system']}")
    print(f"Network: {config['network']}")
    print(f"Remote Memory: {config['remote_memory']}")
    print(f"Output: {output_dir}")
    print("==========================================================================")
    print("")

    print("Running simulation...")
    log_file = output_dir / "logs" / config["log_file"]
    with log_file.open("w", encoding="utf-8") as f:
        subprocess.run(
            [
                str(astra_sim),
                f"--workload-configuration={workload_base}",
                f"--system-configuration={config['system']}",
                f"--remote-memory-configuration={config['remote_memory']}",
                f"--network-configuration={config['network']}",
                f"--comm-group-configuration={comm_group}",
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print("")
    print("==========================================================================")
    print("[ASTRA-sim] Simulation Complete!")
    print("==========================================================================")
    print(f"Log file: {log_file}")
    try:
        with log_file.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if (
                    "Total execution time" in line
                    or "Total communication" in line
                    or "Total computation" in line
                ):
                    print(line.rstrip())
    except FileNotFoundError:
        pass
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())



