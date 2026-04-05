#!/bin/bash
# Usage: sudo bash run_ncu_tp.sh <tp>
# Example: sudo bash run_ncu_tp.sh 1
#          sudo bash run_ncu_tp.sh 2
set -e
TP=${1:-1}
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

NCU=/usr/local/cuda-12.4/bin/ncu
PY=/home/haotian/miniconda3/envs/bench/bin/python
SEQ="1024 4096 16384 65536 131072 262144 524288 1048576"

echo "=== Running ncu profiling TP=${TP} ==="
$NCU --set full --target-processes all \
    --launch-skip 8 --launch-count 700 \
    -o "H100_results/attn_scaling_tp${TP}" \
    $PY ncu_attn_profile.py --tp "$TP" --seq $SEQ

echo "=== Done: H100_results/attn_scaling_tp${TP}.ncu-rep ==="
