#!/bin/bash
set -e

# find the absolute path to this script
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR="${SCRIPT_DIR:?}/../../../.."
EXAMPLE_DIR="${PROJECT_DIR:?}/examples"

# paths
ASTRA_SIM="${PROJECT_DIR:?}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware"
WORKLOAD="${PROJECT_DIR:?}/symbolic_tensor_graph_picasso/generated_attn_8npu/attention_tp8"
SYSTEM="${EXAMPLE_DIR:?}/system/native_collectives/Megatron_TP8.json"
NETWORK="${EXAMPLE_DIR:?}/network/analytical/NVSwitch_8gpus_megatron.yml"
REMOTE_MEMORY="${EXAMPLE_DIR:?}/remote_memory/analytical/no_memory_expansion.json"
COMM_GROUP="${PROJECT_DIR:?}/symbolic_tensor_graph_picasso/generated_attn_8npu/attention_tp8.json"

# start
echo "=========================================================================="
echo "[ASTRA-sim] Megatron-style TP=8 Attention Workload Simulation"
echo "=========================================================================="
echo ""
echo "Configuration:"
echo "  - Model: Qwen3 Attention Layer (dmodel=4096, heads=64, kvheads=4)"
echo "  - Parallelism: Tensor Parallel (TP) = 8"
echo "  - Physical Topology: NVSwitch (Full Crossbar, 8 GPUs)"
echo "  - Collective Algorithm: Ring AllReduce/AllGather"
echo "  - Bandwidth: 400 GB/s per GPU"
echo "  - Latency: 900 ns"
echo ""
echo "  Logical Topology (Single DGX Node):"
echo "              NVSwitch"
echo "         /  |  |  |  |  |  |  \\"
echo "       GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7"
echo ""
echo "  TP Group: [0, 1, 2, 3, 4, 5, 6, 7]"
echo "  Each GPU handles: 64/8 = 8 attention heads"
echo ""
echo "=========================================================================="
echo ""

# run ASTRA-sim
"${ASTRA_SIM:?}" \
    --workload-configuration="${WORKLOAD}" \
    --system-configuration="${SYSTEM:?}" \
    --remote-memory-configuration="${REMOTE_MEMORY:?}" \
    --network-configuration="${NETWORK:?}" \
    --comm-group-configuration="${COMM_GROUP:?}" \
    > simulation_log_megatron_tp8.txt 2>&1

# finalize
echo ""
echo "=========================================================================="
echo "[ASTRA-sim] Simulation Complete!"
echo "=========================================================================="
echo "Log saved to: simulation_log_megatron_tp8.txt"
echo ""
echo "Key Metrics to Check:"
echo "  - Total execution time"
echo "  - Communication overhead"
echo "  - AllReduce/AllGather latency"
echo "  - Network utilization"
echo ""


