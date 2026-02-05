#!/bin/bash
set -e

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../../../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="${PROJECT_DIR:?}/symbolic_tensor_graph/generated/attn_tp16/attention_tp16"
SYSTEM="${EXAMPLE_DIR:?}/system/native_collectives/SparseMesh2D_16npus.json"
NETWORK="${EXAMPLE_DIR:?}/network/analytical/SparseMesh2D_16npus_snake.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"
COMM_GROUP="${PROJECT_DIR:?}/symbolic_tensor_graph/generated/attn_tp16/attention_tp16.json"

# start
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║           ASTRA-sim: Sparse Mesh2D with CUSTOM SNAKE PATTERN                  ║"
echo "╠═══════════════════════════════════════════════════════════════════════════════╣"
echo "║  Physical Topology:                                                           ║"
echo "║      0 --- 1 --- 2 --- 3 --- 4 --- 5                                          ║"
echo "║                  |     |     |     |                                          ║"
echo "║      x     x     9 --- 8 --- 7 --- 6  (reversed for ring optimization)        ║"
echo "║                  |     |     |     |                                          ║"
echo "║      x     x    12 ---11 ---10 ---13                                          ║"
echo "║                              |     |                                          ║"
echo "║      x     x     x     x    14 ---15                                          ║"
echo "║                                                                               ║"
echo "║  Ring: 0→1→2→3→4→5→6→7→8→9→10→11→12→13→14→15→0                                ║"
echo "║  Key benefit: 5→6 is now 1 hop (was 4 hops with auto-numbering!)              ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# check if workload exists
if [ ! -f "${WORKLOAD}.0.et" ]; then
    echo "[WARNING] Workload not found at ${WORKLOAD}"
    echo "[INFO] Generating workload with TP=16..."
    cd "${PROJECT_DIR:?}/symbolic_tensor_graph"
    python main.py \
        --output_dir generated/attn_tp16 \
        --output_name attention_tp16 \
        --model_type moe_attention \
        --tp 16 \
        --num_stacks 1 \
        --batch 1 \
        --seq 1024
    cd "${PROJECT_DIR:?}"
    echo "[INFO] Workload generated ✓"
    echo ""
fi

# check if binary exists
if [ ! -f "${ASTRA_SIM}" ]; then
    echo "[ERROR] ASTRA-sim binary not found. Please compile first:"
    echo "  cd ${PROJECT_DIR}"
    echo "  ./build/astra_analytical/build.sh"
    exit 1
fi

# Run simulation
echo "[RUN] Starting ASTRA-sim simulation..."
echo ""

"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD:?}" \
    --system-configuration="${SYSTEM:?}" \
    --network-configuration="${NETWORK:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    2>&1 | tee "${PROJECT_DIR:?}/simulation_log.txt"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                        SIMULATION COMPLETE                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo "[OUTPUT] Log saved to: ${PROJECT_DIR}/simulation_log.txt"
