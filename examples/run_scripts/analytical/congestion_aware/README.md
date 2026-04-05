# Congestion-Aware 仿真配置系统

本目录为 ASTRA-sim Congestion-Aware 分析后端的批量仿真系统，包含配置文件、Trace 生成、仿真执行和结果缓存的完整工作流。

## 目录结构

脚本按功能划分为 **3 大类**：

```
congestion_aware/
├── backend/                     # 【1】Hybrid 数据生成后端
│   ├── roofline/                #     解析 roofline model
│   │   ├── roofline_gqa_calc.py         # GQA 单层 roofline
│   │   ├── deepseek_v3_mla_roofline.py  # MLA (DeepSeek-V3) roofline
│   │   ├── attention.py                 # GQA 单层模型 + trace 生成 (核心)
│   │   ├── hardware_config.py           # 硬件常量 (B200/Ours/Rubin/H100)
│   │   ├── latency_cal.py               # 配置驱动的 latency 估计
│   │   ├── main.py                      # SA utilization 计算主循环
│   │   ├── util.py                      # 生成 {model}_util_{N}.json
│   │   └── utilization_profiles/        # 各模型 utilization JSONs
│   │       ├── qwen3/                   #   Qwen3-235B util_{8..256}.json
│   │       ├── llama4/                  #   Llama4-Maverick
│   │       ├── deepseek3/               #   DeepSeek-V3 MLA
│   │       └── h100/                    #   H100 baseline
│   ├── astrasim_runner/         #     AstraSim trace/config 生成 + 执行
│   │   ├── generate_scaling_{configs,traces,resources}.py
│   │   ├── seq_scale.py                 # GQA seq 扩展批量生成
│   │   ├── batch_run_configs.py, bulk_run_configs.py
│   │   ├── run_from_config.py, run_with_cache.py
│   │   ├── cache_db.py                  # SQLite/CSV 缓存库
│   │   ├── csv_to_sqlite.py, query_db.py
│   │   ├── update_csv_topology.py, test_cache_flow.py
│   │   └── decode_et_to_json.py, print_comm_type.py
│   └── hybrid_merge/            #     Merge AstraSim + roofline + 功耗
│       ├── analyze_comm.py              # AstraSim 日志解析库
│       ├── collect_gqa_data.py, collect_gqa_batch_data.py
│       ├── merge_gqa_results.py         # Hybrid 合并 (AstraSim+roofline)
│       ├── add_rubin_tp2.py             # 注入 Rubin/H100 TP2
│       ├── power_model.py, query_power.py
│       └── gen_report.py, get_util.py
│
├── profiling/                   # 【2】实卡 H100 profiling
│   ├── ncu/                     #     NCU profile 脚本
│   │   ├── ncu_attn_profile{,_dsv3}.py
│   │   ├── ncu_decode_profile{,_dsv3}.py
│   │   ├── ncu_absorbed_mla_profile_dsv3.py
│   │   ├── analyze_ncu_report.py
│   │   └── analyze_attn_scaling.py
│   ├── microbench/              #     PyTorch GPU microbenchmarks
│   │   ├── attention2.py, batch_gemm_1_gemm.py
│   └── H100_results/            #     profiling raw outputs (.ncu-rep 已 gitignore)
│
├── figures/                     # 【3】论文画图
│   ├── paper_figures/           #     fig1-6 完整目录（scripts/data/plots）
│   │   ├── fig1_e2e_latency/
│   │   ├── fig2_energy_power/
│   │   ├── fig3_ablation_study/
│   │   ├── fig4_batch_exploration/
│   │   ├── fig5_time_breakdown/
│   │   └── fig6_design_exploration/
│   └── legacy_plots/            #     早期 plot 脚本
│       ├── plot_hybrid_merged.py, plot_batch_sweep.py
│       ├── plot_gqa_batch_comparison.py, plot_gqa_comparison.py
│       └── plot_power.py
│
├── configs/                     # ASTRA-sim 仿真配置 JSON (78 个)
├── trace_configs/               # Chakra Trace 生成配置
├── reports/                     # 模型结果: {qwen3-235B,llama4,deepseek-v3}/{hybrid,power,...}
├── cache_db_*/                  # 仿真结果 SQLite + CSV 缓存
├── Deppseek-v3.md, SETUP_LOCALE.md, README.md
└── SparseMesh2D_random.sh       # Shell 测试脚本
```

### 端到端数据流

```
[backend/roofline/]                                      [profiling/ncu/]
  util.py ──→ utilization_profiles/{model}/*.json           ncu_*_profile*.py
       │                                                    ↓
       ▼                                              H100_results/*.ncu-rep
  roofline_gqa_calc.py ──→ reports/{model}/roofline/*.json
                                │
[backend/astrasim_runner/]      │
  seq_scale.py ──→ configs/ ──→ bulk_run_configs.py ──→ SQLite cache
                                                │
[backend/hybrid_merge/]                         ▼
  collect_gqa_data.py ──→ reports/{model}/astrasim/*.json
                                │
                                ▼
  merge_gqa_results.py ──→ reports/{model}/hybrid/*.json
       + add_rubin_tp2.py (注入 Rubin/H100 TP2 baseline)
                                │
                                ▼
  power_model.py ──→ reports/{model}/power/*.json

[figures/paper_figures/]
  fig{1-6}/scripts/plot_*.py  ←── 读取 reports/ 和 H100_results/
                 │
                 ▼
  fig{1-6}/plots/*.pdf + *.png
```

---

## 一、Config 命名规则

### 1.1 通信原语测试 (`comm_` 前缀)

以 `comm_` 开头的配置用于测试单一通信原语在不同拓扑上的性能。

**格式**: `comm_{op}_{scale}_{physical}_{logical}[_{device}].json`

| 字段 | 含义 | 示例 |
|------|------|------|
| `op` | 通信操作 | `all_reduce`, `all_gather`, `reduce_scatter`, `all_to_all`, `broadcast`, `scatter`, `gather`, `reduce`, `barrier` |
| `scale` | GPU 规模 | `2gpus`, `4gpus`, `8gpus`, `16gpus`, `tp16` |
| `physical` | 物理拓扑 | `mesh2d`, `fc`, `sparsemesh2d`, `torus` |
| `logical` | 路由算法 | `ring`, `onering`, `direct`, `onering_ring` |
| `device` | 设备型号 (可选) | `HBM4` (省略时默认 H100) |

**示例**:
```
comm_all_reduce_16gpus_mesh2d_ring.json          # 16GPU Mesh2D + Ring, H100
comm_all_reduce_8gpus_mesh2d_onering_HBM4.json   # 8GPU Mesh2D + OneRing, HBM4
comm_tp16_all_reduce_sparsemesh2d_onering_ring.json  # TP16 AllReduce, SparseMesh2D
```

### 1.2 端到端工作负载仿真 (非 `comm_` 前缀)

以 `{par}` 前缀开头的配置用于仿真完整 Transformer 模型推理/训练的端到端延迟。

**格式**: `{par}_{device}_{physical}_{layout}_{logical}_{model}_{operation}_bs{batch}_sl{seq}.json`

| 字段 | 含义 | 示例 |
|------|------|------|
| `par` | 并行策略 + 度数 | `tp16`, `dp16` |
| `device` | 设备型号 | `h100` |
| `physical` | 物理拓扑 | `mesh2d`, `alltoall` |
| `layout` | GPU 布局 | `4x4`, `8x2`, `16gpus` |
| `logical` | 路由算法 | `ring`, `onering`, `direct` |
| `model` | 模型名 | `qwen` |
| `operation` | 工作负载描述 | `decode_bypass_tp16`, `decode_bypass_hmp_fwd`, `decode_bypass_baseline_fwd` |
| `bs{N}` | Batch Size | `bs64` |
| `sl{N}` | Sequence Length | `sl1024` |

**解析函数**: `cache_db._parse_config_name_rules()` 将文件名自动解析为结构化元数据。

**示例**:
```
tp16_h100_mesh2d_4x4_ring_qwen_decode_bypass_tp16_bs64_sl1024.json
│    │     │       │    │    │     └─ operation
│    │     │       │    │    └─ model
│    │     │       │    └─ logical topology
│    │     │       └─ layout
│    │     └─ physical topology
│    └─ device
└─ parallel strategy
```

---

## 二、Config JSON 字段说明

### 2.1 通信原语测试

```json
{
  "workload_dir":  "{PROJECT_DIR}/examples/workloads/all_reduce_scaling/all_reduce_2gpus",
  "workload_base": "all_reduce_2gpus",
  "comm_group":    "{PROJECT_DIR}/.../all_reduce_2gpus.json",
  "system":        "{EXAMPLE_DIR}/system/native_collectives/ring2_H100.json",
  "network":       "{EXAMPLE_DIR}/network/analytical/Mesh2D_2gpus_2x1_H100.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "output_dir":    "{PROJECT_DIR}/output/scaling_test/comm_all_reduce_2gpus_mesh2d_ring",
  "log_file":      "comm_all_reduce_2gpus_mesh2d_ring.log"
}
```

### 2.2 端到端工作负载仿真

```json
{
  "output_dir":    "{PROJECT_DIR}/output_qwen/hmp/tp16_h100_mesh2d_4x4_hmp_fwd_ring",
  "astra_sim":     "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system":        "{EXAMPLE_DIR}/system/native_collectives/ring16_H100.json",
  "network":       "{EXAMPLE_DIR}/network/analytical/Mesh2D_16gpus_4x4_H100.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir":  "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_decode_bypass_hmp_fwd_tph4_tps4",
  "workload_base": "decode_bypass_hmp_fwd_tph4_tps4",
  "log_file":      "simulation_log_tp16_h100_mesh2d_4x4_hmp_fwd_ring.txt"
}
```

### 字段含义

| 字段 | 说明 |
|------|------|
| `output_dir` | 仿真输出目录 (日志、结果) |
| `astra_sim` | AstraSim 二进制路径 (端到端仿真需要，通信原语测试可省略) |
| `system` | 逻辑拓扑 / 集合通信算法配置 (`examples/system/native_collectives/*.json`) |
| `network` | 物理网络拓扑配置 (`examples/network/analytical/*.yml`) |
| `remote_memory` | 远程内存配置 |
| `workload_dir` | Chakra trace 文件所在目录 |
| `workload_base` | trace 文件前缀 (实际文件为 `{workload_base}.{rank}.et`) |
| `comm_group` | 通信组定义文件 (通信原语测试使用) |
| `log_file` | 仿真日志文件名 |

### 占位符

| 占位符 | 展开值 |
|--------|--------|
| `{PROJECT_DIR}` | 项目根目录 (`astra_sim_picasso/`) |
| `{EXAMPLE_DIR}` | `{PROJECT_DIR}/examples` |

---

## 三、Config 分类总览

### 3.1 AllReduce 通信扩展测试 (28 个)

测试不同 GPU 规模 × 不同拓扑组合的 AllReduce 延迟。

| GPU 规模 | 物理拓扑 × 路由 | 设备 |
|----------|-----------------|------|
| 2 / 4 / 8 / 16 | Mesh2D × ring | H100 |
| 2 / 4 / 8 / 16 | Mesh2D × onering | H100 |
| 2 / 4 / 8 / 16 | Mesh2D × ring | HBM4 |
| 2 / 4 / 8 / 16 | Mesh2D × onering | HBM4 |
| 2 / 4 / 8 / 16 | FullyConn × ring | H100 |
| 2 / 4 / 8 / 16 | SparseMesh2D × onering | H100 |
| 2 / 4 / 8 / 16 | Torus2D × ring | H100 |

### 3.2 TP16 通信原语测试 (21 个)

测试 16 GPU TP 下各通信原语在 Mesh2D / FullyConn / SparseMesh2D 上的性能。

| 原语 | Mesh2D ring | FullyConn ring | SparseMesh2D |
|------|:-----------:|:--------------:|:------------:|
| all_reduce | ✓ | ✓ | ✓ |
| all_gather | ✓ | ✓ | |
| reduce_scatter | ✓ | ✓ | |
| reduce_scatter_block | ✓ | ✓ | |
| all_to_all | ✓ | ✓ | |
| broadcast | ✓ | ✓ | |
| scatter | ✓ | ✓ | |
| gather | ✓ | ✓ | |
| reduce | ✓ | ✓ | |
| barrier | ✓ | ✓ | |

### 3.3 Qwen 端到端工作负载仿真 (13 个)

#### TP16 标准模式 (7 个)

| Config | 物理拓扑 | 路由 | workload |
|--------|----------|------|----------|
| `tp16_h100_mesh2d_4x4_ring_qwen_decode_bypass_tp16_*` | Mesh2D 4×4 | ring | `decode_bypass_tp16` |
| `tp16_h100_mesh2d_4x4_onering_qwen_decode_bypass_tp16_*` | Mesh2D 4×4 | onering | `decode_bypass_tp16` |
| `tp16_h100_mesh2d_4x4_direct_qwen_decode_bypass_tp16_*` | Mesh2D 4×4 | direct | `decode_bypass_tp16` |
| `tp16_h100_alltoall_16gpus_ring_qwen_decode_bypass_tp16_*` | AllToAll 16 | ring | `decode_bypass_tp16` |
| `tp16_h100_alltoall_16gpus_direct_qwen_decode_bypass_tp16_*` | AllToAll 16 | direct | `decode_bypass_tp16` |
| `tp16_h100_alltoall_8x2_ring__qwen_decode_bypass_tp16_*` | AllToAll 8×2 | ring | `decode_bypass_tp16` |
| `tp16_h100_alltoall_8x2_direct_qwen_decode_bypass_tp16_*` | AllToAll 8×2 | direct | `decode_bypass_tp16` |

#### HMP Forward-Only 模式 — Ours vs Baseline (6 个)

HMP (Hybrid Model Parallelism) 使用 `tp_h=4, tp_s=4` 将 16 GPU 分为 Head 并行和 Sequence 并行两个维度。

| Config | 策略 | 路由 | workload | 说明 |
|--------|------|------|----------|------|
| `tp16_h100_mesh2d_4x4_ring_qwen_decode_bypass_hmp_fwd_*` | **Ours** | ring | `decode_bypass_hmp_fwd_tph4_tps4` | Softmax Deferred, 合并通信 |
| `tp16_h100_mesh2d_4x4_onering_qwen_decode_bypass_hmp_fwd_*` | **Ours** | onering | `decode_bypass_hmp_fwd_tph4_tps4` | Softmax Deferred, 合并通信 |
| `tp16_h100_mesh2d_4x4_direct_qwen_decode_bypass_hmp_fwd_*` | **Ours** | direct | `decode_bypass_hmp_fwd_tph4_tps4` | Softmax Deferred, 合并通信 |
| `tp16_h100_mesh2d_4x4_ring_qwen_decode_bypass_baseline_fwd_*` | **Baseline** | ring | `decode_bypass_baseline_fwd_tph4_tps4` | 标准 Softmax, 多次通信 |
| `tp16_h100_mesh2d_4x4_onering_qwen_decode_bypass_baseline_fwd_*` | **Baseline** | onering | `decode_bypass_baseline_fwd_tph4_tps4` | 标准 Softmax, 多次通信 |
| `tp16_h100_mesh2d_4x4_direct_qwen_decode_bypass_baseline_fwd_*` | **Baseline** | direct | `decode_bypass_baseline_fwd_tph4_tps4` | 标准 Softmax, 多次通信 |

> **Ours (HMP fwd)**: 利用 Online Softmax 将 `tp_h` 和 `tp_s` 上的部分和延迟到投影之后一次性 AllReduce。
> **Baseline**: 在 GQA kernel 输出处先对 `tp_s` 做 AllReduce (标准 Softmax), 投影后再对 `tp_h` 做 AllReduce + 对 `tp_s` 做 AllGather。

#### DP16 模式 (4 个)

| Config | 物理拓扑 | 路由 | workload |
|--------|----------|------|----------|
| `dp16_h100_mesh2d_4x4_ring_qwen_decode_bypass_*` | Mesh2D 4×4 | ring | `decode_bypass_dp16` |
| `dp16_h100_mesh2d_4x4_direct_qwen_decode_bypass_*` | Mesh2D 4×4 | direct | `decode_bypass_dp16` |
| `dp16_h100_alltoall_16gpus_ring_qwen_decode_bypass_*` | AllToAll 16 | ring | `decode_bypass_dp16` |
| `dp16_h100_alltoall_16gpus_direct_qwen_decode_bypass_*` | AllToAll 16 | direct | `decode_bypass_dp16` |

### 3.4 TP16 GQA-Only 单层仿真 (12 个)

隔离单层 GQA（Grouped Query Attention）通信模式，排除 FFN / Embedding 等公共开销，放大不同并行策略在注意力层的通信差异。

| 设备 | 物理拓扑 × 路由 | 策略 |
|------|----------------|------|
| HBM4 / H100 | Mesh2D 4×4 × onering | Baseline / HMP (Ours) / TP16 |
| H100 | Mesh2D 4×4 × ring | Baseline / HMP (Ours) / TP16 |
| H100 | Mesh2D 4×4 × direct | Baseline / HMP (Ours) / TP16 |

> 详细分析见 [`reports/gqa_analysis_hbm4_onering.md`](reports/gqa_analysis_hbm4_onering.md)

---

## 四、Trace Config (`trace_configs/`)

Trace config 定义如何调用 `symbolic_tensor_graph_picasso/main.py` 生成 Chakra trace。

### 4.1 端到端 Trace Configs

| 文件 | model_type | 并行参数 | 模型参数 |
|------|-----------|----------|----------|
| `qwen_decode_bypass_dp16.json` | `qwen_decode_bypass` | dp=16 | head=64, kvhead=4, stacks=94 |
| `qwen_decode_bypass_dp16_b16.json` | `qwen_decode_bypass` | dp=16 | head=64, kvhead=4, stacks=94, batch=16 |
| `qwen_decode_bypass_hmp_fwd_tph4_tps4.json` | `qwen_decode_bypass_hmp_fwd` | tp_h=4, tp_s=4 | head=64, kvhead=8, stacks=80 |
| `qwen_decode_bypass_baseline_fwd_tph4_tps4.json` | `qwen_decode_bypass_baseline_fwd` | tp_h=4, tp_s=4 | head=64, kvhead=8, stacks=80 |
| `qwen_decode_bypass_tp16.json` | `qwen_decode_bypass_tp16` | tp=16 | head=64, kvhead=8, stacks=80 |

### 4.2 GQA-Only Trace Configs (单层注意力)

| 文件 | model_type | 并行参数 | 模型参数 |
|------|-----------|----------|----------|
| `qwen_gqa_baseline_fwd_tph4_tps4.json` | `qwen_gqa_baseline_fwd` | tp_h=4, tp_s=4 | head=64, kvhead=8, stacks=1 |
| `qwen_gqa_hmp_fwd_tph4_tps4.json` | `qwen_gqa_hmp_fwd` | tp_h=4, tp_s=4 | head=64, kvhead=8, stacks=1 |
| `qwen_gqa_tp16_fwd.json` | `qwen_gqa_tp16_fwd` | tp=16 | head=64, kvhead=8, stacks=1 |

### Trace Config JSON 示例 (HMP)

```json
{
  "output_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_decode_bypass_hmp_fwd_tph4_tps4",
  "output_name": "decode_bypass_hmp_fwd_tph4_tps4",
  "model_type": "qwen_decode_bypass_hmp_fwd",
  "dp": 1, "tp": 1, "tp_h": 4, "tp_s": 4,
  "kvhead": 8, "head": 64, "num_stacks": 80,
  "dmodel": 8192, "batch": 64, "seq": 1024,
  "tpsp": true
}
```

### 支持的 model_type

| model_type | 说明 |
|-----------|------|
| `qwen_decode_bypass_hmp_fwd` | Ours — HMP Forward-Only (Softmax Deferred) |
| `qwen_decode_bypass_baseline_fwd` | Baseline — HMP Forward-Only (标准 Softmax) |
| `qwen_decode_bypass_tp16` | TP16 Decode Bypass |
| `qwen_decode_bypass` | 通用 Decode Bypass |
| `qwen_gqa_hmp_fwd` | GQA-Only: HMP Forward (单层注意力) |
| `qwen_gqa_baseline_fwd` | GQA-Only: Baseline Forward (单层注意力) |
| `qwen_gqa_tp16_fwd` | GQA-Only: TP16 Forward (单层注意力) |
| `qwen_decode_attention` | Decode Attention |
| `qwen_forward_only_bypass` | Forward-Only Bypass |
| `qwen_forward_attention` | Forward Attention |
| `moe` | Mixture of Experts |
| `dense` | Dense 模型 (默认) |

---

## 五、物理拓扑 vs 逻辑拓扑

每个 config 通过两个文件分别指定物理和逻辑拓扑：

| 维度 | 配置文件 | 目录 | 说明 |
|------|---------|------|------|
| **物理拓扑** (`network`) | `*.yml` | `examples/network/analytical/` | 互联图、链路带宽、延迟 |
| **逻辑拓扑** (`system`) | `*.json` | `examples/system/native_collectives/` | 集合通信算法路由 |

### 可用物理拓扑

| 类型 | 文件示例 | 说明 |
|------|---------|------|
| Mesh2D | `Mesh2D_16gpus_4x4_H100.yml` | 4×4 二维网格 |
| FullyConn | `FullyConn_16gpus_H100.yml` | 全连接 (AllToAll) |
| Torus2D | `Torus2D_8gpus_4x2_H100.yml` | 二维环面 |
| SparseMesh2D | `SparseMesh2D_16npus_onering.yml` | 稀疏网格 |
| HBM4 变体 | `Mesh2D_16gpus_4x4_HBM4.yml` | HBM4 带宽/延迟参数 |

### 可用逻辑拓扑 (路由算法)

| 类型 | 文件示例 | 说明 |
|------|---------|------|
| Ring | `ring16_H100.json` | 环形路由 |
| OneRing | `onering16_H100.json` | 单环路由 |
| Direct | `direct16_H100.json` | 直连路由 |
| AllToAll | `DP16_H100_AllToAll.json` | 全对全 |

---

## 六、运行方式

### 6.1 一键生成 Trace + 运行仿真 (推荐)

使用 `--gen-trace` + `--trace-pattern` 自动从 `trace_configs/` 生成 Chakra trace，然后运行仿真：

```bash
cd {PROJECT_DIR}/examples/run_scripts/analytical/congestion_aware

# 生成 HMP + Baseline 的 trace，然后运行 HBM4 Mesh2D 4×4 OneRing 仿真
python3 bulk_run_configs.py \
  --gen-trace \
  --trace-pattern "*tph4_tps4.json" \
  --trace-force \
  --pattern "tp16_hbm4_mesh2d_4x4_onering_qwen_decode_bypass_*_bs64_sl1024.json" \
  --force-run
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--gen-trace` | 启用 trace 生成阶段 |
| `--trace-pattern` | 在 `trace_configs/` 目录下匹配 trace 配置文件 |
| `--trace-force` | 强制重新生成 trace（即使已存在） |
| `--pattern` | 在 `configs/` 目录下匹配仿真配置文件 |
| `--force-run` | 忽略缓存，强制重新运行仿真 |

### 6.2 仅生成 Trace (不运行仿真)

```bash
# 只生成 trace，不运行任何仿真 (用不匹配的 pattern 跳过仿真)
python3 bulk_run_configs.py \
  --gen-trace \
  --trace-pattern "*tph4_tps4.json" \
  --trace-force \
  --pattern "NOMATCH"
```

### 6.3 仅运行仿真 (Trace 已存在)

```bash
# 运行 HBM4 拓扑下的 TP16/HMP/Baseline 三方对比
python3 bulk_run_configs.py \
  --pattern "tp16_hbm4_mesh2d_4x4_onering_qwen_decode_bypass_*_bs64_sl1024.json" \
  --force-run

# 运行 H100 拓扑下的 Ours vs Baseline 对比
python3 bulk_run_configs.py \
  --pattern "tp16_h100_mesh2d_4x4_*_qwen_decode_bypass_*_fwd_*.json" \
  --force-run
```

### 6.4 运行通信原语扩展测试

```bash
python3 bulk_run_configs.py --pattern "comm_all_reduce_*.json"
```

### 6.5 运行特定拓扑

```bash
python3 bulk_run_configs.py --pattern "*mesh2d*ring*.json"
```

### 6.6 查询缓存结果

```bash
python3 query_db.py --db cache_db_v131/results.sqlite --pattern "%hmp%"
```

### 6.7 导出 CSV → SQLite

```bash
python3 csv_to_sqlite.py \
  --csv cache_db_v131/results_cache.csv \
  --db cache_db_v131/results.sqlite
```

### 6.8 完整示例：HBM4 Mesh2D OneRing 三方对比

以下示例演示如何从零开始运行 TP16 / HMP (Ours) / Baseline 在 HBM4 Torus2D 4×4 + OneRing 拓扑上的对比仿真：

```bash
cd {PROJECT_DIR}/examples/run_scripts/analytical/congestion_aware

# Step 1: 生成 HMP 和 Baseline 的 trace
python3 bulk_run_configs.py \
  --gen-trace \
  --trace-pattern "*tph4_tps4.json" \
  --trace-force \
  --pattern "NOMATCH"

# Step 2: 运行三个仿真 (TP16 trace 需已存在)
python3 bulk_run_configs.py \
  --pattern "tp16_hbm4_mesh2d_4x4_onering_qwen_decode_bypass_*_bs64_sl1024.json" \
  --force-run

# 三个 config 文件:
#   tp16_hbm4_mesh2d_4x4_onering_qwen_decode_bypass_tp16_bs64_sl1024.json
#   tp16_hbm4_mesh2d_4x4_onering_qwen_decode_bypass_hmp_fwd_bs64_sl1024.json
#   tp16_hbm4_mesh2d_4x4_onering_qwen_decode_bypass_baseline_fwd_bs64_sl1024.json

# Step 3: 查看结果
python3 query_db.py --db cache_db_v131/results.sqlite --pattern "%hbm4%onering%"
```

各 config 对应的 trace 和硬件配置：

| Config | Trace | 网络拓扑 | 逻辑路由 |
|--------|-------|----------|----------|
| `..._decode_bypass_tp16_...` | `generated_decode_bypass_tp16` | `Torus2D_16gpus_4x4_HBM4.yml` | `onering16_H100.json` |
| `..._decode_bypass_hmp_fwd_...` | `generated_decode_bypass_hmp_fwd_tph4_tps4` | `Torus2D_16gpus_4x4_HBM4.yml` | `onering16_HBM4.json` |
| `..._decode_bypass_baseline_fwd_...` | `generated_decode_bypass_baseline_fwd_tph4_tps4` | `Torus2D_16gpus_4x4_HBM4.yml` | `onering16_HBM4.json` |

---

## 七、结果缓存

仿真结果自动保存到 `cache_db_v131/`:

| 文件 | 说明 |
|------|------|
| `results.sqlite` | SQLite 数据库 (主存储) |
| `results_cache.csv` | CSV 备份 (追加写入) |

### CSV 字段

| 字段 | 说明 |
|------|------|
| `config_name` | 配置文件名 (不含 `.json`) |
| `wall_time` | 端到端墙钟时间 (ns) |
| `gpu_time` | GPU 计算时间 (ns) |
| `comm_time` | 通信时间 (ns) |
| `overlap` | 计算-通信重叠 (ns) |
| `status` | `SUCCESS` / `FAILED: returncode=N` |

---

## 八、新增配置指南

### 添加新的通信原语测试

文件名格式: `comm_{op}_{scale}_{physical}_{logical}[_{device}].json`

```json
{
  "workload_dir":  "{PROJECT_DIR}/tests/rt_template/inputs/workload/comm_primitive_traces_16gpu/{op}",
  "workload_base": "chakra_trace",
  "system":        "{EXAMPLE_DIR}/system/native_collectives/{logical}{N}_{device}.json",
  "network":       "{EXAMPLE_DIR}/network/analytical/{Physical}_{N}gpus_{layout}_{device}.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "output_dir":    "{PROJECT_DIR}/output/scaling_test/comm_{op}_{N}gpus_{physical}_{logical}",
  "log_file":      "comm_{op}_{N}gpus_{physical}_{logical}.log"
}
```

### 添加新的端到端工作负载仿真

文件名格式: `{par}_{device}_{physical}_{layout}_{logical}_{model}_{operation}_bs{batch}_sl{seq}.json`

```json
{
  "output_dir":    "{PROJECT_DIR}/output_qwen/{experiment}/{unique_name}",
  "astra_sim":     "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system":        "{EXAMPLE_DIR}/system/native_collectives/{logical}{N}_{device}.json",
  "network":       "{EXAMPLE_DIR}/network/analytical/{Physical}_{N}gpus_{layout}_{device}.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir":  "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/{trace_dir}",
  "workload_base": "{trace_name}",
  "log_file":      "simulation_log_{unique_name}.txt"
}
```

### 添加新的 Trace Config

```json
{
  "output_dir":   "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/{trace_dir}",
  "output_name":  "{trace_name}",
  "model_type":   "qwen_decode_bypass_hmp_fwd",
  "dp": 1, "tp": 1, "tp_h": 4, "tp_s": 4,
  "kvhead": 8, "head": 64, "num_stacks": 80,
  "dmodel": 8192, "batch": 64, "seq": 1024,
  "tpsp": true
}
```

> **注意**: Trace 生成 (`main.py`) 必须从 `symbolic_tensor_graph_picasso/` 目录执行，因为 CSV 计算图使用相对路径。

---

## 九、通信延迟分析工具 (`analyze_comm.py`)

解析仿真日志中的 `[SIM_COMM]` / `[SIM_COMM_FINISH]` 事件，按 Transformer 层汇总通信延迟，支持多日志横向对比。

### 9.1 基本用法

```bash
cd {PROJECT_DIR}/examples/run_scripts/analytical/congestion_aware

# 分析单个日志
python3 analyze_comm.py <log_file>

# 多日志横向对比
python3 analyze_comm.py <log1> <log2> [<log3> ...]
```

### 9.2 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `logs` (位置参数) | 一个或多个仿真日志文件路径 | 必填 |
| `--npu N` | 只分析 NPU N 的通信事件 | `0` |
| `--layers N` | 显示前 N 层的逐操作详情 | `2` |
| `--layers 0` | 显示所有层的逐操作详情 | — |
| `--output FILE` / `-o FILE` | 同时将报告保存到文件 | — |

### 9.3 输出内容

对每个日志文件输出以下信息：

| 输出模块 | 说明 |
|----------|------|
| **全局指标** | Wall time, GPU time, Comm time (cycles) |
| **逐层详情** | 每层中每个通信操作的类型、数据量 (MB)、通信组 ID、延迟 (us) |
| **逐层汇总** | 所有层的平均/总通信延迟 |
| **按通信类型汇总** | 每种类型 (ALL_GATHER, ALL_REDUCE 等) 的次数、总延迟、平均延迟 |
| **按通信组汇总** | 每个 `comm_group_id` 的次数、总延迟、平均延迟 |
| **横向对比表** (≥2 个日志) | 以 `transformer.0` 层为基准，并排对比各配置的通信操作和延迟 |

### 9.4 示例：TP16 / HMP / Baseline 三方通信对比

脚本会自动检测项目根目录并解析路径中的 `{PROJECT_DIR}` 占位符，因此可以直接复制使用：

```bash
python3 analyze_comm.py \
  {PROJECT_DIR}/output_qwen/hmp/tp16_hbm4_mesh2d_4x4_baseline_fwd_ring/logs/simulation_log_tp16_hbm4_torus_4x4_baseline_fwd_onering.txt \
  {PROJECT_DIR}/output_qwen/hmp/tp16_h100_mesh2d_4x4_hmp_fwd_direct/logs/simulation_log_tp16_hbm4_torus_4x4_decode_bypass_hmp_fwd_onering.txt \
  {PROJECT_DIR}/output_qwen/test/tp16_h100_mesh2d_4x4_decode_bypass_onering/logs/simulation_log_tp16_hbm4_torus_4x4_decode_bypass_onering.txt \
  --layers 2
```

也可使用相对路径（从 `congestion_aware/` 目录运行，项目根在 4 级上层）：

```bash
python3 analyze_comm.py \
  ../../../../output_qwen/hmp/tp16_hbm4_mesh2d_4x4_baseline_fwd_ring/logs/simulation_log_tp16_hbm4_torus_4x4_baseline_fwd_onering.txt \
  ../../../../output_qwen/hmp/tp16_h100_mesh2d_4x4_hmp_fwd_direct/logs/simulation_log_tp16_hbm4_torus_4x4_decode_bypass_hmp_fwd_onering.txt \
  ../../../../output_qwen/test/tp16_h100_mesh2d_4x4_decode_bypass_onering/logs/simulation_log_tp16_hbm4_torus_4x4_decode_bypass_onering.txt \
  --layers 2
```

### 9.5 输出示例 (节选)

```
==========================================================================================
  simulation_log_tp16_hbm4_torus_4x4_baseline_fwd_onering
  Wall=58,949,859 cycles  GPU=40,159,950  Comm=18,868,963
==========================================================================================

  transformer.0  (total comm = 201.3 us, 7 ops)
  Op                                  Type                  Size MB  Group      Latency
  ----------------------------------- ------------------ ---------- ------ ------------
  mha.x                               ALL_GATHER            16.0000     17     24.720 us
  mha.x                               ALL_GATHER            16.0000     21     93.668 us
  mha.attn_kernel.k1                  ALL_TO_ALL             2.0059     21     22.681 us
  mha.attn_kernel.v1                  ALL_TO_ALL             2.0059     21     22.655 us
  mha.attn                            ALL_REDUCE             0.1250     21     19.760 us
  mha.o                               ALL_REDUCE             0.0625     17      7.760 us
  mha.o                               ALL_GATHER             0.0625     21     10.017 us

  --- Per-layer comm summary (94 layers) ---
  Avg per layer: 200.7 us
  Total all layers: 18868.9 us

  --- By comm type ---
  Type                  Count     Total us     Avg us
  ALL_GATHER              282      11998.5     42.548
  ALL_TO_ALL              188       4259.2     22.655
  ALL_REDUCE              188       2611.2     13.890

  --- By comm group ---
   Group  Count     Total us     Avg us
      21    470      15819.8     33.659
      17    188       3049.0     16.218
```

### 9.6 生成报告到文件

```bash
# GQA-Only 三方对比, 保存到 reports/
python3 analyze_comm.py \
  ../../../../output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering.txt \
  ../../../../output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering.txt \
  ../../../../output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering.txt \
  --layers 0 \
  --output reports/gqa_only_hbm4_onering_comparison.txt
```

---

## 十、分析报告 (`reports/`)

分析报告保存在 `reports/` 目录，包含由 `analyze_comm.py --output` 生成的通信详情和手工编写的综合分析。

| 文件 | 说明 |
|------|------|
| `gqa_analysis_hbm4_onering.md` | **综合分析报告**: GQA-Only vs 端到端对照，结论与发现 |
| `gqa_only_hbm4_onering_comparison.txt` | GQA-Only 三方通信详情 (Baseline / HMP / TP16) |
| `e2e_hbm4_onering_comparison.txt` | 端到端 94 层三方通信详情 |

### 报告更新方式

当新增仿真配置或重新运行仿真后，使用以下命令更新报告：

```bash
# 重新生成 GQA-only 分析报告
python3 analyze_comm.py \
  ../../../../output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering/logs/*.txt \
  ../../../../output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering/logs/*.txt \
  ../../../../output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering/logs/*.txt \
  --layers 0 --output reports/gqa_only_hbm4_onering_comparison.txt

# 重新生成端到端分析报告
python3 analyze_comm.py \
  ../../../../output_qwen/hmp/tp16_hbm4_mesh2d_4x4_baseline_fwd_ring/logs/*.txt \
  ../../../../output_qwen/hmp/tp16_h100_mesh2d_4x4_hmp_fwd_direct/logs/*.txt \
  ../../../../output_qwen/test/tp16_h100_mesh2d_4x4_decode_bypass_onering/logs/*.txt \
  --layers 2 --output reports/e2e_hbm4_onering_comparison.txt
```
