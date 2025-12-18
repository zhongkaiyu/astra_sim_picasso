# ASTRA-sim Mesh2D 24-NPU AllGather Simulation Results

## ? Simulation Configuration

### Workload
- **Type**: Attention-Only (Qwen3 Model)
- **Model**: moe_attention mode
- **Parallelism**: Tensor Parallel (TP) = 24
- **Model Parameters**:
  - dmodel: 4096
  - heads: 64 (64 heads / 24 NPUs = 2.67 heads per NPU)
  - kvheads: 4
  - num_stacks: 1 (single transformer layer)
  - batch: 1
  - sequence length: 1024

### Physical Network Topology
- **Type**: Mesh2D (2D Grid)
- **Dimensions**: 6 ¡Á 4 = 24 NPUs
- **Bandwidth**: 100 GB/s per link
- **Latency**: 1000 ns per hop
- **Routing**: XY routing with congestion awareness

```
Layout:
     0 --- 1 --- 2 --- 3 --- 4 --- 5
     |     |     |     |     |     |
     6 --- 7 --- 8 --- 9 --- 10--- 11
     |     |     |     |     |     |
     12--- 13--- 14--- 15--- 16--- 17
     |     |     |     |     |     |
     18--- 19--- 20--- 21--- 22--- 23
```

### Collective Algorithm
- **Type**: Ring AllGather
- **Implementation**: Native collective with 4 chunks
- **Scheduling**: LIFO policy
- **Optimization**: localBWAware

## ? Performance Results (NPU 0)

| Metric | Value | Unit | Percentage |
|--------|-------|------|------------|
| **Wall Time** | 35,334,744 | cycles | 100% |
| **GPU Time (Compute)** | 1,229,228 | cycles | 3.48% |
| **Comm Time** | 34,254,490 | cycles | 96.94% |
| **Compute-Comm Overlap** | 148,974 | cycles | 0.42% |

### Key Insights

1. **Communication Dominated**: 96.94% of execution time is spent on communication
   - This is expected for TP=24 with attention layers
   - AllReduce and AllGather operations across 24 NPUs dominate

2. **Low Compute Bound**: 8.987%
   - Indicates the workload is heavily communication-bound
   - Compute operations finish quickly compared to communication

3. **Compute Utilization**: 13.686%
   - Average GPU utilization is low
   - GPU spends most time waiting for communication to complete

4. **Memory Utilization**: 91.382%
   - High memory usage indicates large intermediate tensors
   - Typical for attention layers with seq_len=1024

## ? Analysis

### Communication Pattern
With TP=24:
- Each NPU handles approximately 2-3 attention heads
- QKV projections require AllReduce across TP group
- Output requires AllGather to reconstruct full tensor
- Ring algorithm performs well for uniform distribution

### Mesh2D Performance
- Physical topology: 6¡Á4 grid requires multi-hop communication
- Average hop count between NPUs: ~3-4 hops
- Congestion may occur at central routers
- XY routing provides deterministic paths

### Bottleneck Identification
**Primary Bottleneck**: Communication (96.94%)
- Large data movement across 24 NPUs
- Multi-hop routing in Mesh2D adds latency
- Network bandwidth (100 GB/s) may be saturating

**Potential Optimizations**:
1. Increase network bandwidth (e.g., 200-400 GB/s)
2. Use NVSwitch topology for single-hop communication
3. Reduce TP degree (e.g., TP=8 or TP=16)
4. Increase compute intensity (more layers, larger batch)

## ? Generated Files

```
symbolic_tensor_graph_picasso/generated_attn_24npu/
©À©¤©¤ attention_tp24.0.et - attention_tp24.23.et  (24 workload files)
©¸©¤©¤ attention_tp24.json                          (comm group config)
```

## ? How to Reproduce

```bash
# Step 1: Install dependencies
sudo /opt/venv/astra-sim/bin/pip install tqdm

# Step 2: Generate workload
cd symbolic_tensor_graph_picasso
python main.py \
    --output_dir generated_attn_24npu/ \
    --output_name attention_tp24 \
    --model_type moe_attention \
    --tp 24 \
    --num_stacks 1 \
    --dmodel 4096 \
    --head 64 \
    --kvhead 4 \
    --dff 12288 \
    --dvocal 151936 \
    --batch 1 \
    --seq 1024

# Step 3: Run simulation
cd ..
bash examples/run_scripts/analytical/congestion_aware/Mesh2D_allgather_24npus_6x4.sh

# Step 4: View results
cat simulation_log.txt
```

## ? Notes

- This simulation uses Chakra Execution Traces (ET) format
- Congestion-aware analytical network backend is used
- All 24 NPUs show consistent performance metrics
- Simulation completed successfully without errors

---

**Generated**: 2025-12-17  
**Tool**: ASTRA-sim 2.0 + Symbolic Tensor Graph Generator  
**Workload**: Qwen3 Attention Layer (TP=24)


