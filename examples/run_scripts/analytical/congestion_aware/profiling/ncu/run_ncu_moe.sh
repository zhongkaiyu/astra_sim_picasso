#!/bin/bash
# Measure MoE FFN decode utilization (HBM-BW% + TensorCore%) on GPU 0 with ncu.
# Needs sudo (GPU performance counters). Reading the produced .ncu-rep does NOT.
#
# Usage:  sudo bash run_ncu_moe.sh [model] [dtype]
#   sudo bash run_ncu_moe.sh deepseek3 fp8
#   sudo bash run_ncu_moe.sh qwen3-235b fp8
# Default: runs deepseek3 + qwen3-235b in fp8.
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"; cd "$DIR"
NCU=/usr/local/cuda-13.0/bin/ncu
PY=/home/haotian/miniconda3/envs/bench/bin/python
export CUDA_VISIBLE_DEVICES=0          # pin to physical GPU 0 (idle)

# Light, fast metric set (no --set full): exactly what analyze_moe_ncu.py reads.
MET="gpu__time_duration.sum"
MET="$MET,dram__bytes_read.sum"
MET="$MET,dram__bytes_read.sum.pct_of_peak_sustained_elapsed"
MET="$MET,sm__inst_executed.avg.pct_of_peak_sustained_elapsed"
MET="$MET,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed"

run_one () {
  local MODEL=$1 DTYPE=$2
  echo "=== ncu MoE FFN | model=$MODEL dtype=$DTYPE | GPU0 ==="
  # Capture the first ~220 kernels (covers warmup + profiled pass; every expert
  # GEMV shape appears many times — utilization is shape-determined so warmup is
  # representative). Median per weight-size group is taken by analyze_moe_ncu.py.
  "$NCU" --target-processes all \
      --launch-skip 0 --launch-count 220 \
      --metrics "$MET" \
      -f -o "H100_results/moe_decode_${MODEL}_${DTYPE}" \
      "$PY" ncu_moe_decode_profile.py --model "$MODEL" --dtype "$DTYPE"
  echo "=> H100_results/moe_decode_${MODEL}_${DTYPE}.ncu-rep"
}

if [ -n "$1" ]; then
  run_one "$1" "${2:-fp8}"
else
  run_one deepseek3  fp8
  run_one qwen3-235b fp8
fi
echo "DONE. Now (no sudo needed):"
echo "  NCU_PATH=$NCU $PY analyze_moe_ncu.py H100_results/moe_decode_deepseek3_fp8.ncu-rep --model deepseek3"
