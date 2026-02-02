#!/bin/bash
# Generate TP=8, DP=4 Workload (4 H100 nodes with 8 GPUs each)
set -e

cd /home/haotian/waferchip/mesh2D/astra_sim_picasso/symbolic_tensor_graph_picasso

echo "========================================================================"
echo "Generating Workload: TP=8, DP=4 (32 NPUs = 4 H100 Nodes)"
echo "========================================================================"
echo ""
echo "Configuration:"
echo "  - Tensor Parallel (TP): 8 GPUs per node"
echo "  - Data Parallel (DP):   4 nodes"
echo "  - Total NPUs:           32 (8 x 4)"
echo "  - Model:                Qwen3 Attention"
echo "  - dmodel=4096, heads=64, kvheads=4"
echo ""

python main.py \
    --output_dir generated_tp8_dp4/ \
    --output_name attention_tp8_dp4 \
    --model_type moe_attention \
    --tp 8 \
    --dp 4 \
    --num_stacks 1 \
    --dmodel 4096 \
    --head 64 \
    --kvhead 4 \
    --batch 1 \
    --seq 1024

echo ""
echo "========================================================================"
echo "Workload Generated Successfully!"
echo "========================================================================"
echo ""
echo "Files generated in: generated_tp8_dp4/"
ls -lh generated_tp8_dp4/
echo ""

