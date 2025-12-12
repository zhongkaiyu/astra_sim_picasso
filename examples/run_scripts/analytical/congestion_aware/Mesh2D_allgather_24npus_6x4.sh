#!/bin/bash
set -e

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../../../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="/home/ohm/astra-sim/symbolic_tensor_graph/generated_attn_24npu/attention_tp24"
SYSTEM="${EXAMPLE_DIR:?}/system/native_collectives/Ring_4chunks.json"
NETWORK="${EXAMPLE_DIR:?}/network/analytical/Mesh2D_24npus_6x4.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"
COMM_GROUP="/home/ohm/astra-sim/symbolic_tensor_graph/generated_attn_24npu/attention_tp24.json"

# start
echo "[ASTRA-sim] Running Attention-Only Workload (TP=24) on Mesh2D Physical Topology (6x4 grid, 24 NPUs)"
echo ""
echo "  Mesh Layout:"
echo "       0 --- 1 --- 2 --- 3 --- 4 --- 5"
echo "       |     |     |     |     |     |"
echo "       6 --- 7 --- 8 --- 9 --- 10--- 11"
echo "       |     |     |     |     |     |"
echo "       12--- 13--- 14--- 15--- 16--- 17"
echo "       |     |     |     |     |     |"
echo "       18--- 19--- 20--- 21--- 22--- 23"
echo ""

# run ASTRA-sim
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    > simulation_log.txt 2>&1

# finalize
echo ""
echo "[ASTRA-sim] Finished the execution."
echo "[ASTRA-sim] Log saved to: simulation_log.txt"

