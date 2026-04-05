# Ring vs Mesh2D 拓扑对比报告：GQA-Only 单层仿真

> **生成时间**: 2026-02-27
> **设备**: HBM4 | **逻辑路由**: OneRing | **并行度**: tp_h=4, tp_s=4 (16 GPUs)
> **模型参数**: Qwen (head=64, kvhead=8, dmodel=8192, batch=64, seq=1024)

---

## 1. 实验目的

前序实验（`gqa_analysis_hbm4_onering.md`）在 Ring 拓扑下已确认 HMP FWD 在 GQA 层面的通信优势。
本实验通过将网络拓扑从 **Ring（1D）** 切换到 **真正的 Mesh2D（4×4 网格）**，验证：

1. 行方向通信组 g17 `[0,1,2,3]` 和列方向通信组 g21 `[0,4,8,12]` 在 Mesh2D 上的延迟变化
2. 2D 空间局部性对不同并行策略的影响差异
3. TP16 全局 AllReduce 在 Mesh2D 上的路由开销

---

## 2. 网络拓扑文件

### 2.1 Ring 拓扑（标记为 "Torus2D" 但实际为 1D Ring）

**文件**: `examples/network/analytical/Torus2D_16gpus_4x4_HBM4.yml`

```yaml
topology: [ Ring ]
npus_count: [ 16 ]
bandwidth: [ 2000.0 ]  # GB/s
latency: [ 100 ]        # ns
```

物理连接：`0 - 1 - 2 - 3 - 4 - 5 - 6 - 7 - 8 - 9 - 10 - 11 - 12 - 13 - 14 - 15`

### 2.2 Mesh2D 拓扑（真正 4×4 二维网格）

**文件**: `examples/network/analytical/Mesh2D_16gpus_4x4_HBM4.yml`

```yaml
topology: [ Mesh2D ]
npus_count: [ 16 ]
width: 4
height: 4
bandwidth: [ 2000.0 ]  # GB/s (D2D: 2 TB/s)
latency: [ 100 ]        # ns (D2D latency)
```

物理连接：

```
 0 --- 1 --- 2 --- 3
 |     |     |     |
 4 --- 5 --- 6 --- 7
 |     |     |     |
 8 --- 9 --- 10--- 11
 |     |     |     |
 12--- 13--- 14--- 15
```

### 2.3 两种拓扑关键区别

| 特性 | Ring | Mesh2D |
|------|------|--------|
| 拓扑维度 | 1D | 2D（4×4 网格） |
| g17 `[0,1,2,3]` max hop/step | 1 | 1（同行直连） |
| g21 `[0,4,8,12]` max hop/step | **4**（间隔 4 节点） | **1**（同列直连） |
| 16-GPU OneRing max hop/step | 1（链状直连） | **3**（跨行/列需绕路） |
| 带宽/延迟 | 2000 GB/s / 100 ns | 2000 GB/s / 100 ns |

---

## 3. 仿真配置详情

### 3.1 Ring 拓扑配置

#### Ring Baseline FWD

**Config 文件**: `configs/tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_baseline_fwd_bs64_sl1024.json`

```json
{
  "output_dir": "{PROJECT_DIR}/output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering",
  "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system": "{EXAMPLE_DIR}/system/native_collectives/onering16_HBM4.json",
  "network": "{EXAMPLE_DIR}/network/analytical/Torus2D_16gpus_4x4_HBM4.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_gqa_baseline_fwd_tph4_tps4",
  "workload_base": "gqa_baseline_fwd_tph4_tps4",
  "log_file": "simulation_log_tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering.txt"
}
```

**Log 文件**: `output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering.txt`

#### Ring HMP FWD

**Config 文件**: `configs/tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_hmp_fwd_bs64_sl1024.json`

```json
{
  "output_dir": "{PROJECT_DIR}/output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering",
  "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system": "{EXAMPLE_DIR}/system/native_collectives/onering16_HBM4.json",
  "network": "{EXAMPLE_DIR}/network/analytical/Torus2D_16gpus_4x4_HBM4.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_gqa_hmp_fwd_tph4_tps4",
  "workload_base": "gqa_hmp_fwd_tph4_tps4",
  "log_file": "simulation_log_tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering.txt"
}
```

**Log 文件**: `output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering.txt`

#### Ring TP16 FWD

**Config 文件**: `configs/tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_tp16_fwd_bs64_sl1024.json`

```json
{
  "output_dir": "{PROJECT_DIR}/output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering",
  "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system": "{EXAMPLE_DIR}/system/native_collectives/onering16_HBM4.json",
  "network": "{EXAMPLE_DIR}/network/analytical/Torus2D_16gpus_4x4_HBM4.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_gqa_tp16_fwd",
  "workload_base": "gqa_tp16_fwd",
  "log_file": "simulation_log_tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering.txt"
}
```

**Log 文件**: `output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering.txt`

### 3.2 Mesh2D 拓扑配置

#### Mesh2D Baseline FWD

**Config 文件**: `configs/tp16_gqa/tp16_hbm4_real_mesh2d_4x4_onering_qwen_gqa_baseline_fwd_bs64_sl1024.json`

```json
{
  "output_dir": "{PROJECT_DIR}/output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_baseline_fwd_onering",
  "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system": "{EXAMPLE_DIR}/system/native_collectives/onering16_HBM4.json",
  "network": "{EXAMPLE_DIR}/network/analytical/Mesh2D_16gpus_4x4_HBM4.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_gqa_baseline_fwd_tph4_tps4",
  "workload_base": "gqa_baseline_fwd_tph4_tps4",
  "log_file": "simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_baseline_fwd_onering.txt"
}
```

**Log 文件**: `output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_baseline_fwd_onering/logs/simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_baseline_fwd_onering.txt`

#### Mesh2D HMP FWD

**Config 文件**: `configs/tp16_gqa/tp16_hbm4_real_mesh2d_4x4_onering_qwen_gqa_hmp_fwd_bs64_sl1024.json`

```json
{
  "output_dir": "{PROJECT_DIR}/output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_hmp_fwd_onering",
  "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system": "{EXAMPLE_DIR}/system/native_collectives/onering16_HBM4.json",
  "network": "{EXAMPLE_DIR}/network/analytical/Mesh2D_16gpus_4x4_HBM4.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_gqa_hmp_fwd_tph4_tps4",
  "workload_base": "gqa_hmp_fwd_tph4_tps4",
  "log_file": "simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_hmp_fwd_onering.txt"
}
```

**Log 文件**: `output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_hmp_fwd_onering/logs/simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_hmp_fwd_onering.txt`

#### Mesh2D TP16 FWD

**Config 文件**: `configs/tp16_gqa/tp16_hbm4_real_mesh2d_4x4_onering_qwen_gqa_tp16_fwd_bs64_sl1024.json`

```json
{
  "output_dir": "{PROJECT_DIR}/output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_tp16_fwd_onering",
  "astra_sim": "{PROJECT_DIR}/build/astra_analytical/build/bin/AstraSim_Analytical_Congestion_Aware",
  "system": "{EXAMPLE_DIR}/system/native_collectives/onering16_HBM4.json",
  "network": "{EXAMPLE_DIR}/network/analytical/Mesh2D_16gpus_4x4_HBM4.yml",
  "remote_memory": "{EXAMPLE_DIR}/remote_memory/analytical/no_memory_expansion.json",
  "workload_dir": "{PROJECT_DIR}/symbolic_tensor_graph_picasso/et_trace/generated_gqa_tp16_fwd",
  "workload_base": "gqa_tp16_fwd",
  "log_file": "simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_tp16_fwd_onering.txt"
}
```

**Log 文件**: `output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_tp16_fwd_onering/logs/simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_tp16_fwd_onering.txt`

### 3.3 共享资源

| 资源 | 路径 |
|------|------|
| **System (OneRing16 HBM4)** | `examples/system/native_collectives/onering16_HBM4.json` |
| **Remote Memory** | `examples/remote_memory/analytical/no_memory_expansion.json` |
| **Baseline/HMP Workload** | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_baseline_fwd_tph4_tps4/` |
| **HMP Workload** | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_hmp_fwd_tph4_tps4/` |
| **TP16 Workload** | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_tp16_fwd/` |
| **Baseline/HMP Comm Group** | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_baseline_fwd_tph4_tps4/gqa_baseline_fwd_tph4_tps4.json` |
| **HMP Comm Group** | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_hmp_fwd_tph4_tps4/gqa_hmp_fwd_tph4_tps4.json` |
| **TP16 Comm Group** | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_tp16_fwd/gqa_tp16_fwd.json` |

---

## 4. 通信组定义

### 4.1 Baseline / HMP (tp_h=4, tp_s=4)

```
tp_h 维度（行方向, 4 GPU/组）:
  g17: [0, 1, 2, 3]      ← 第 0 行
  g18: [4, 5, 6, 7]      ← 第 1 行
  g19: [8, 9, 10, 11]    ← 第 2 行
  g20: [12, 13, 14, 15]  ← 第 3 行

tp_s 维度（列方向, 4 GPU/组）:
  g21: [0, 4, 8, 12]     ← 第 0 列
  g22: [1, 5, 9, 13]     ← 第 1 列
  g23: [2, 6, 10, 14]    ← 第 2 列
  g24: [3, 7, 11, 15]    ← 第 3 列
```

在 Mesh2D 4×4 上的物理位置：

```
       col0   col1   col2   col3
row0:  [0] ---[1] ---[2] ---[3]     ← g17
        |      |      |      |
row1:  [4] ---[5] ---[6] ---[7]     ← g18
        |      |      |      |
row2:  [8] ---[9] ---[10]---[11]    ← g19
        |      |      |      |
row3:  [12]---[13]---[14]---[15]    ← g20
        ↑      ↑      ↑      ↑
       g21    g22    g23    g24
```

### 4.2 TP16

```
全局维度（16 GPU）:
  g17: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
```

---

## 5. 仿真结果

### 5.1 全局指标对比

| Config | Topology | Wall(ns) | GPU(ns) | Comm(ns) | Comm% |
|--------|----------|-------:|-------:|-------:|------:|
| **Baseline FWD** | Ring | 108,350 | 26,344 | 82,847 | 76.5% |
| **HMP FWD** | Ring | 84,235 | 39,766 | 45,310 | 53.8% |
| **TP16 FWD** | Ring | 110,414 | 83,934 | 26,480 | 24.0% |
| **Baseline FWD** | **Mesh2D** | **62,690** | 26,344 | **37,187** | 59.3% |
| **HMP FWD** | **Mesh2D** | **56,635** | 39,766 | **17,710** | 31.3% |
| **TP16 FWD** | **Mesh2D** | 131,414 | 83,934 | 47,480 | 36.1% |

### 5.2 拓扑切换加速比

| 策略 | Ring Wall(ns) | Mesh2D Wall(ns) | 加速比 | 通信变化 |
|------|----------:|----------:|------:|------:|
| **Baseline FWD** | 108,350 | **62,690** | **1.73×** ↓ | 82,847 → 37,187 (−55%) |
| **HMP FWD** | 84,235 | **56,635** | **1.49×** ↓ | 45,310 → 17,710 (−61%) |
| **TP16 FWD** | 110,414 | 131,414 | **0.84×** ↑ | 26,480 → 47,480 (+79%) |

---

## 6. 逐操作通信时间线对比 (NPU 0, Layer 0)

### 6.1 Baseline FWD

| Op | Type | Group | Dim | Ring Start | Ring End | Ring Dur(ns) | Mesh2D Start | Mesh2D End | Mesh2D Dur(ns) | Mesh/Ring |
|:---|:-----|------:|:----|----------:|---------:|------------:|------------:|-----------:|--------------:|:---------:|
| mha.attn_kernel.k1 | ALL_TO_ALL | g21 | tp_s(列) | 20,978 | 43,633 | 22,655 | 20,978 | 29,833 | **8,855** | **0.39×** ↓ |
| mha.attn_kernel.v1 | ALL_TO_ALL | g21 | tp_s(列) | 43,633 | 66,288 | 22,655 | 29,833 | 38,688 | **8,855** | **0.39×** ↓ |
| mha.attn | ALL_REDUCE | g21 | tp_s(列) | 66,340 | 86,100 | 19,760 | 38,740 | 46,500 | **7,760** | **0.39×** ↓ |
| mha.o | ALL_REDUCE | g17 | tp_h(行) | 90,573 | 98,333 | 7,760 | 50,973 | 58,733 | 7,760 | **1.00×** ≈ |
| mha.o | ALL_GATHER | g21 | tp_s(列) | 98,333 | 108,350 | 10,017 | 58,733 | 62,690 | **3,957** | **0.40×** ↓ |

### 6.2 HMP FWD

| Op | Type | Group | Dim | Ring Start | Ring End | Ring Dur(ns) | Mesh2D Start | Mesh2D End | Mesh2D Dur(ns) | Mesh/Ring |
|:---|:-----|------:|:----|----------:|---------:|------------:|------------:|-----------:|--------------:|:---------:|
| mha.attn_kernel.k1 | ALL_TO_ALL | g21 | tp_s(列) | 20,978 | 43,633 | 22,655 | 20,978 | 29,833 | **8,855** | **0.39×** ↓ |
| mha.attn_kernel.v1 | ALL_TO_ALL | g21 | tp_s(列) | 43,633 | 66,288 | 22,655 | 29,833 | 38,688 | **8,855** | **0.39×** ↓ |

### 6.3 TP16 FWD

| Op | Type | Group | Dim | Ring Start | Ring End | Ring Dur(ns) | Mesh2D Start | Mesh2D End | Mesh2D Dur(ns) | Mesh/Ring |
|:---|:-----|------:|:----|----------:|---------:|------------:|------------:|-----------:|--------------:|:---------:|
| mha.o | ALL_REDUCE | g17 | all 16 | 83,934 | 110,414 | 26,480 | 83,934 | 131,414 | 47,480 | **1.79×** ↑ |

---

## 7. 关键分析：g17 与 g21 在不同拓扑上的行为

### 7.1 g21 `[0,4,8,12]` — 列方向通信

| 拓扑 | OneRing 路径 | Max Hop/Step | 代表延迟 (ALL_TO_ALL) |
|------|-------------|:----------:|----------:|
| **Ring** | 0→4：经 1→2→3→4（4 hop） | **4** | 22,655 ns |
| **Mesh2D** | 0→4：直连（1 hop） | **1** | **8,855 ns** |

**结论**: Mesh2D 上 g21 每步仅需 1 hop（同列直连），Ring 上需 4 hop。Mesh2D 通信延迟降低 **61%**。

### 7.2 g17 `[0,1,2,3]` — 行方向通信（Baseline/HMP）

| 拓扑 | OneRing 路径 | Max Hop/Step | 代表延迟 (ALL_REDUCE) |
|------|-------------|:----------:|----------:|
| **Ring** | 0→1→2→3（直连） | **1** | 7,760 ns |
| **Mesh2D** | 0→1→2→3（同行直连） | **1** | 7,760 ns |

**结论**: g17 在两种拓扑上延迟完全一致——两者都是直连邻居。

### 7.3 g17 `[0..15]` — TP16 全局通信

| 拓扑 | OneRing 16-node 路径 | Max Hop/Step | ALL_REDUCE 延迟 |
|------|---------------------|:----------:|----------:|
| **Ring** | 0→1→2→...→15（链状直连） | **1** | 26,480 ns |
| **Mesh2D** | 0→1→2→3→**?**→4→5→... | **≥3**（如 3→4 需 3→2→1→0→4 或 3→7→4 等） | **47,480 ns** |

**结论**: TP16 的 16-GPU OneRing 在 Mesh2D 上路由效率差，跨行节点不直连（如 3 到 4 需多跳），导致延迟增加 79%。

---

## 8. 综合结论

### 8.1 Mesh2D 拓扑对分组策略的影响

| 策略 | Ring Wall | Mesh2D Wall | Mesh2D 排名 | 分析 |
|------|-------:|-------:|:----------:|------|
| **HMP FWD** | 84,235 | **56,635** | **第 1** | 仅用 g21(列) 通信，Mesh2D 直连优势最大化 |
| **Baseline FWD** | 108,350 | **62,690** | **第 2** | g21(列) 大幅加速，g17(行) 不变 |
| **TP16 FWD** | 110,414 | 131,414 | **第 3** | 16-GPU 全局 OneRing 在 Mesh2D 上路由差 |

### 8.2 策略差距被 Mesh2D 拓扑放大

| 对比 | Ring 差距 | Mesh2D 差距 |
|------|-------:|-------:|
| HMP vs Baseline | 1.29× | 1.11× |
| HMP vs TP16 | 1.31× | **2.32×** |
| Baseline vs TP16 | 1.02× | **2.10×** |

在 Mesh2D 上，HMP 和 Baseline 相对 TP16 的优势从 ~1.3× 拉大到 **~2.1–2.3×**，因为：
- HMP/Baseline 的 4-GPU 行/列分组完美匹配 Mesh2D 4×4 的物理连接
- TP16 的 16-GPU 全局通信在 2D 网格上存在多跳路由惩罚

### 8.3 启示

1. **网络感知并行策略至关重要**：在 2D Mesh 物理拓扑上，将通信组对齐到物理维度（行/列）可以获得 2.5× 的通信加速
2. **HMP 在 Mesh2D 上表现最优**：wall time 56,635 ns，比 TP16 快 2.32×
3. **TP16 不适合 2D Mesh**：16-GPU 全局 OneRing 在 Mesh2D 上产生严重的路由开销

---

## 附件与数据来源

### Config 文件

| 标识 | 路径 |
|------|------|
| Ring Baseline | `configs/tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_baseline_fwd_bs64_sl1024.json` |
| Ring HMP | `configs/tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_hmp_fwd_bs64_sl1024.json` |
| Ring TP16 | `configs/tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_tp16_fwd_bs64_sl1024.json` |
| Mesh2D Baseline | `configs/tp16_gqa/tp16_hbm4_real_mesh2d_4x4_onering_qwen_gqa_baseline_fwd_bs64_sl1024.json` |
| Mesh2D HMP | `configs/tp16_gqa/tp16_hbm4_real_mesh2d_4x4_onering_qwen_gqa_hmp_fwd_bs64_sl1024.json` |
| Mesh2D TP16 | `configs/tp16_gqa/tp16_hbm4_real_mesh2d_4x4_onering_qwen_gqa_tp16_fwd_bs64_sl1024.json` |

### Log 文件

| 标识 | 路径 |
|------|------|
| Ring Baseline | `output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_baseline_fwd_onering.txt` |
| Ring HMP | `output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_hmp_fwd_onering.txt` |
| Ring TP16 | `output_qwen/gqa/tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering/logs/simulation_log_tp16_hbm4_mesh2d_4x4_gqa_tp16_fwd_onering.txt` |
| Mesh2D Baseline | `output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_baseline_fwd_onering/logs/simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_baseline_fwd_onering.txt` |
| Mesh2D HMP | `output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_hmp_fwd_onering/logs/simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_hmp_fwd_onering.txt` |
| Mesh2D TP16 | `output_qwen/gqa/tp16_hbm4_real_mesh2d_4x4_gqa_tp16_fwd_onering/logs/simulation_log_tp16_hbm4_real_mesh2d_4x4_gqa_tp16_fwd_onering.txt` |

### 通信组文件

| 策略 | 路径 |
|------|------|
| Baseline (tp_h=4, tp_s=4) | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_baseline_fwd_tph4_tps4/gqa_baseline_fwd_tph4_tps4.json` |
| HMP (tp_h=4, tp_s=4) | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_hmp_fwd_tph4_tps4/gqa_hmp_fwd_tph4_tps4.json` |
| TP16 | `symbolic_tensor_graph_picasso/et_trace/generated_gqa_tp16_fwd/gqa_tp16_fwd.json` |

### 网络拓扑文件

| 拓扑 | 路径 |
|------|------|
| Ring (标记 Torus2D) | `examples/network/analytical/Torus2D_16gpus_4x4_HBM4.yml` |
| Mesh2D 4×4 | `examples/network/analytical/Mesh2D_16gpus_4x4_HBM4.yml` |

### 关联报告

| 报告 | 说明 |
|------|------|
| [`gqa_analysis_hbm4_onering.md`](gqa_analysis_hbm4_onering.md) | Ring 拓扑下的 GQA-only + 端到端分析 |
| [`gqa_only_hbm4_onering_comparison.txt`](gqa_only_hbm4_onering_comparison.txt) | Ring GQA-only analyze_comm.py 原始输出 |
| [`e2e_hbm4_onering_comparison.txt`](e2e_hbm4_onering_comparison.txt) | Ring 端到端 analyze_comm.py 原始输出 |
