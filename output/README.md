# ASTRA-sim Simulation Results Output

This directory contains organized simulation results for different configurations of Qwen3 attention workload.

## Directory Structure

```
output/
??? README.md                           # This file
?
??? mesh2d_tp24/                        # Mesh2D Topology, TP=24
?   ??? logs/
?   ?   ??? simulation_log.txt
?   ??? workloads/
?   ?   ??? attention_tp24.{0-23}.et
?   ?   ??? attention_tp24.json
?   ??? analysis/
?   ?   ??? simulation_summary.md
?   ??? CONFIG_INFO.md
?
??? megatron_tp8/                       # NVSwitch Topology, TP=8
?   ??? logs/
?   ?   ??? simulation_log_megatron_tp8.txt
?   ??? workloads/
?   ?   ??? attention_tp8.{0-7}.et
?   ?   ??? attention_tp8.json
?   ??? analysis/
?   ?   ??? simulation_megatron_tp8_summary.md
?   ??? CONFIG_INFO.md
?
??? comparison/                         # Comparative Analysis
    ??? COMPARISON_SUMMARY.md
```

## Quick Results Summary

| Configuration | Wall Time | Speedup | Compute Util | Comm % |
|--------------|-----------|---------|--------------|--------|
| **Mesh2D TP=24** | 35.3M cycles | 1.0× (baseline) | 13.69% | 96.9% ?? |
| **Megatron TP=8** | 7.4M cycles | **4.76×** ? | 34.33% | 82.2% |

### Key Findings
- ? **Megatron TP=8 is 4.76× faster** than Mesh2D TP=24
- ? NVSwitch single-hop topology significantly reduces communication overhead
- ? TP=8 provides better load balance (8 heads per GPU vs 2.67)
- ?? Both configurations are still communication-bound

## Configuration Comparison

### Mesh2D TP=24
- **Topology**: 2D Mesh (6×4 grid), multi-hop routing
- **Bandwidth**: 100 GB/s per link
- **Parallelism**: 24 GPUs, 2.67 heads per GPU (uneven)
- **Use Case**: Multi-node scaling, custom interconnect

### Megatron TP=8
- **Topology**: NVSwitch (full crossbar), single-hop
- **Bandwidth**: 400 GB/s per GPU (4× higher)
- **Parallelism**: 8 GPUs, 8 heads per GPU (perfect balance)
- **Use Case**: Single DGX node, low latency inference/training

## How to View Results

### View Mesh2D TP=24 Results
```bash
# View simulation log
cat mesh2d_tp24/logs/simulation_log.txt

# View performance analysis
cat mesh2d_tp24/analysis/simulation_summary.md

# View configuration details
cat mesh2d_tp24/CONFIG_INFO.md
```

### View Megatron TP=8 Results
```bash
# View simulation log
cat megatron_tp8/logs/simulation_log_megatron_tp8.txt

# View performance analysis
cat megatron_tp8/analysis/simulation_megatron_tp8_summary.md

# View configuration details
cat megatron_tp8/CONFIG_INFO.md
```

### View Comparison
```bash
# Quick comparison (recommended to read first!)
cat comparison/COMPARISON_SUMMARY.md
```

## Reproduce These Results

### Prerequisites
```bash
# Install dependencies
sudo /opt/venv/astra-sim/bin/pip install tqdm
```

### Step 1: Generate Workloads
```bash
cd ../symbolic_tensor_graph_picasso

# For Mesh2D TP=24
python main.py --output_dir generated_attn_24npu/ --output_name attention_tp24 \
    --model_type moe_attention --tp 24 --num_stacks 1 --dmodel 4096 \
    --head 64 --kvhead 4 --dff 12288 --dvocal 151936 --batch 1 --seq 1024

# For Megatron TP=8
python main.py --output_dir generated_attn_8npu/ --output_name attention_tp8 \
    --model_type moe_attention --tp 8 --num_stacks 1 --dmodel 4096 \
    --head 64 --kvhead 4 --dff 12288 --dvocal 151936 --batch 1 --seq 1024
```

### Step 2: Run Simulations
```bash
cd ..

# Mesh2D TP=24
bash examples/run_scripts/analytical/congestion_aware/Mesh2D_allgather_24npus_6x4.sh

# Megatron TP=8
bash examples/run_scripts/analytical/congestion_aware/Megatron_TP8_attention.sh
```

### Step 3: View Results
```bash
# Check Mesh2D results
grep "sys\[0\]" simulation_log.txt | grep "Wall time"

# Check Megatron results
grep "sys\[0\]" simulation_log_megatron_tp8.txt | grep "Wall time"
```

## Key Metrics Explained

### Wall Time
Total execution time from start to finish (compute + communication).

### GPU Time
Time spent on actual computation (matrix multiplications, activations).

### Comm Time
Time spent on communication (AllReduce, AllGather, ReduceScatter).

### Compute Bound %
Percentage of time where computation is the bottleneck.

### Compute Utilization
How efficiently GPUs are being used for computation.

### Memory Utilization
How much of GPU memory bandwidth is being used.

## Recommendations

### For Production Use
- ? **Use Megatron TP=8** for single-node scenarios
- ? Use NVSwitch/InfiniBand for high-bandwidth interconnect
- ? TP degree should evenly divide number of attention heads

### For Research
- ?? Test different TP degrees (4, 8, 16, 32)
- ?? Compare various network topologies
- ?? Analyze impact of batch size and sequence length

### Next Steps to Improve Performance
1. **Increase Compute Intensity**
   - Larger batch size (1 ? 16-32)
   - More layers (1 ? 80-94)
   - Longer sequences (1024 ? 2048-4096)

2. **Reduce Communication**
   - Mixed precision (FP16/BF16)
   - Gradient accumulation
   - Activation checkpointing

3. **Hardware Optimization**
   - Use H100 with NVLink 4.0 (900 GB/s)
   - Enable GPUDirect RDMA
   - Multi-node with InfiniBand

## Tools Used

- **ASTRA-sim 2.0**: Distributed AI system simulator
- **Symbolic Tensor Graph (STG)**: Workload generator
- **Network Backend**: Analytical (congestion-aware)
- **Workload Format**: Chakra Execution Traces (ET)

## Contact & References

- **ASTRA-sim**: https://astra-sim.github.io/
- **Chakra ET**: https://github.com/mlcommons/chakra
- **STG**: https://github.com/astra-sim/symbolic_tensor_graph

---

**Generated**: 2025-12-18  
**Model**: Qwen3 Attention Layer  
**Configurations**: Mesh2D TP=24 vs Megatron TP=8  
**Result**: Megatron TP=8 is **4.76× faster**

