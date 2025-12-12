#!/bin/bash
set -e

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../../../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="${EXAMPLE_DIR:?}/workload/microbenchmarks/all_gather/16npus_1MB/all_gather"
SYSTEM="${EXAMPLE_DIR:?}/system/native_collectives/Ring_4chunks.json"
NETWORK="${EXAMPLE_DIR:?}/network/analytical/Mesh2D_16npus.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"

# start
echo "[ASTRA-sim] Running Ring All-Gather on Mesh2D Physical Topology (4x4 grid, 16 NPUs)"
echo ""

# run ASTRA-sim
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    > simulation_log.txt 2>&1

# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."
echo "[ASTRA-sim] Log saved to: simulation_log.txt"

