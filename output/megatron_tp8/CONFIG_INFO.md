# Megatron TP=8 Configuration

## Run Date
2025-12-18

## Configuration Files

### Network Configuration
- **File**: `../../examples/network/analytical/NVSwitch_8gpus_megatron.yml`
- **Topology**: Switch (NVSwitch full crossbar)
- **NPUs**: 8
- **Bandwidth**: 400 GB/s per GPU
- **Latency**: 900 ns (single-hop)

### System Configuration
- **File**: `../../examples/system/native_collectives/Megatron_TP8.json`
- **Algorithm**: Ring AllGather/AllReduce
- **Chunks**: 8
- **Scheduling**: LIFO

### Workload Configuration
- **Model**: Qwen3 Attention Layer
- **Type**: moe_attention (attention-only)
- **TP Degree**: 8
- **Model Params**:
  - dmodel: 4096
  - heads: 64 (8 heads per GPU)
  - kvheads: 4
  - num_stacks: 1
  - batch: 1
  - seq: 1024

### Run Script
```bash
bash ../../examples/run_scripts/analytical/congestion_aware/Megatron_TP8_attention.sh
```

## Directory Structure
```
megatron_tp8/
©À©¤©¤ logs/
©¦   ©¸©¤©¤ simulation_log_megatron_tp8.txt  # Full simulation output
©À©¤©¤ workloads/
©¦   ©À©¤©¤ attention_tp8.0.et - .7.et      # 8 NPU workload traces
©¦   ©¸©¤©¤ attention_tp8.json               # Communication group config
©À©¤©¤ analysis/
©¦   ©¸©¤©¤ simulation_megatron_tp8_summary.md  # Performance analysis
©¸©¤©¤ CONFIG_INFO.md                       # This file
```

## Key Results
- **Wall Time**: 7,424,768 cycles
- **Compute Time**: 1,455,554 cycles (19.6%)
- **Communication Time**: 6,107,088 cycles (82.2%)
- **Compute Utilization**: 34.33%
- **Speedup vs Mesh2D**: **4.76¡Á** ?

## Advantages
- ? Single-hop communication (NVSwitch)
- ? 4¡Á higher bandwidth (400 vs 100 GB/s)
- ? Perfect load balance (8 heads per GPU)
- ? Fewer communication participants (8 vs 24)

