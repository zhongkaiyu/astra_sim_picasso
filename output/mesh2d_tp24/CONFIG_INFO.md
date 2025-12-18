# Mesh2D TP=24 Configuration

## Run Date
2025-12-18

## Configuration Files

### Network Configuration
- **File**: `../../examples/network/analytical/Mesh2D_24npus_6x4.yml`
- **Topology**: Mesh2D (6¡Á4 grid)
- **NPUs**: 24
- **Bandwidth**: 100 GB/s per link
- **Latency**: 1000 ns per hop

### System Configuration
- **File**: `../../examples/system/native_collectives/Ring_4chunks.json`
- **Algorithm**: Ring AllGather/AllReduce
- **Chunks**: 4
- **Scheduling**: LIFO

### Workload Configuration
- **Model**: Qwen3 Attention Layer
- **Type**: moe_attention (attention-only)
- **TP Degree**: 24
- **Model Params**:
  - dmodel: 4096
  - heads: 64
  - kvheads: 4
  - num_stacks: 1
  - batch: 1
  - seq: 1024

### Run Script
```bash
bash ../../examples/run_scripts/analytical/congestion_aware/Mesh2D_allgather_24npus_6x4.sh
```

## Directory Structure
```
mesh2d_tp24/
©À©¤©¤ logs/
©¦   ©¸©¤©¤ simulation_log.txt              # Full simulation output
©À©¤©¤ workloads/
©¦   ©À©¤©¤ attention_tp24.0.et - .23.et   # 24 NPU workload traces
©¦   ©¸©¤©¤ attention_tp24.json             # Communication group config
©À©¤©¤ analysis/
©¦   ©¸©¤©¤ simulation_summary.md           # Performance analysis
©¸©¤©¤ CONFIG_INFO.md                      # This file
```

## Key Results
- **Wall Time**: 35,334,744 cycles
- **Compute Time**: 1,229,228 cycles (3.5%)
- **Communication Time**: 34,254,490 cycles (96.9%)
- **Compute Utilization**: 13.69%
- **Status**: Severely communication-bound ??

