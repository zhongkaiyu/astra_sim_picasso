#!/bin/bash
# ============================================================================
# ASTRA-sim DP=16 测试脚本 (16 H100 GPUs, Mesh2D 4x4 Network)
# ============================================================================
# 说明：读取 JSON 配置文件，跑 DP=16 的 workload
# 用法：
#   bash run_DP16_H100_Mesh2D_4x4.sh [config.json]
# ============================================================================
set -e

# 获取脚本目录
SCRIPT_DIR=$(dirname "$(realpath "$0")")
# 项目根目录
PROJECT_DIR="${SCRIPT_DIR:?}/../../../.."
# 示例目录
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# 配置文件（默认同目录 JSON）
CONFIG_JSON="${1:-${SCRIPT_DIR:?}/run_DP16_H100_Mesh2D_4x4.json}"

if [ ! -f "${CONFIG_JSON}" ]; then
    echo "ERROR: Config JSON not found: ${CONFIG_JSON}"
    exit 1
fi

# 读取 JSON 配置
eval "$(
python3 - <<'PY'
import json
import os
import shlex
from pathlib import Path

config_path = os.environ["CONFIG_JSON"]
project_dir = os.environ["PROJECT_DIR"]
example_dir = os.environ["EXAMPLE_DIR"]

with open(config_path, "r") as f:
    cfg = json.load(f)

def fmt(value: str) -> str:
    return value.format(PROJECT_DIR=project_dir, EXAMPLE_DIR=example_dir)

def get(key, default=None):
    val = cfg.get(key, default)
    if val is None:
        return None
    if isinstance(val, str):
        return fmt(val)
    return val

output_dir = get("output_dir", f"{project_dir}/output_qwen/test/dp16_h100_mesh2d_4x4_decode_bypass")
astra_sim = get("astra_sim", f"{project_dir}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware")
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

def export(name, value):
    print(f'export {name}={shlex.quote(str(value))}')

export("OUTPUT_DIR", output_dir)
export("ASTRA_SIM", astra_sim)
export("SYSTEM", system)
export("NETWORK", network)
export("REMOTE_MEMORY", remote_memory)
export("WORKLOAD_DIR", workload_dir)
export("WORKLOAD_BASE", workload_base_path)
export("COMM_GROUP", comm_group)
export("LOG_FILE_NAME", log_file)
PY
)"

mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/analysis"

echo "Checking workload files..."
if [ ! -f "${WORKLOAD_BASE}.0.et" ]; then
    echo "WARNING: Workload file not found: ${WORKLOAD_BASE}.0.et"
    echo "Please ensure workload files exist in:"
    echo "  ${WORKLOAD_DIR}/"
    exit 1
else
    echo "✓ Workload files found: ${WORKLOAD_BASE}"
fi

if [ ! -f "${COMM_GROUP}" ]; then
    echo "WARNING: Communication group file not found: ${COMM_GROUP}"
    exit 1
else
    echo "✓ Comm group file found: ${COMM_GROUP}"
fi

if [ ! -f "$ASTRA_SIM" ]; then
    echo "ERROR: ASTRA-sim binary not found: $ASTRA_SIM"
    echo "Please build ASTRA-sim first:"
    echo "  cd build/astra_analytical && bash build.sh"
    exit 1
fi

echo ""
echo "=========================================================================="
echo "[ASTRA-sim] DP=16 (16 H100 GPUs) on Mesh2D 4x4"
echo "=========================================================================="
echo "Workload: ${WORKLOAD_BASE}"
echo "Comm Group: ${COMM_GROUP}"
echo "System: ${SYSTEM}"
echo "Network: ${NETWORK}"
echo "Remote Memory: ${REMOTE_MEMORY}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================================================="
echo ""

echo "Running simulation..."
LOG_FILE="${OUTPUT_DIR}/logs/${LOG_FILE_NAME}"
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD_BASE}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    > "${LOG_FILE}" 2>&1

echo ""
echo "=========================================================================="
echo "[ASTRA-sim] Simulation Complete!"
echo "=========================================================================="
echo "Log file: ${LOG_FILE}"
grep -E "(Total execution time|Total communication|Total computation)" "${LOG_FILE}" || true
echo ""

