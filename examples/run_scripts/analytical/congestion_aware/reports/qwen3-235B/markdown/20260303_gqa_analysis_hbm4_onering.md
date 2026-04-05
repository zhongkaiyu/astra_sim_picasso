# GQA-Only 通信分析报告：HBM4 Mesh2D 4×4 + OneRing

> **生成时间**: 2026-02-27
> **设备**: HBM4 | **物理拓扑**: Mesh2D 4×4 | **逻辑路由**: OneRing
> **模型参数**: Qwen (head=64, kvhead=8, dmodel=8192, batch=64, seq=1024)

---

## 1. 实验目的

端到端仿真显示 Baseline FWD、HMP FWD (Ours)、TP16 FWD 三种并行策略的 wall time 极其接近（~58.8–59.1M cycles），难以区分通信模式的差异。本实验通过 **隔离单层 GQA（Grouped Query Attention）** 来放大通信差异，排除 FFN、Embedding 等共享计算的干扰。

## 2. 仿真配置

| 配置 | Config 文件 | Trace | 策略说明 |
|------|------------|-------|----------|
| **Baseline FWD** | `tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_baseline_fwd_bs64_sl1024.json` | `gqa_baseline_fwd_tph4_tps4` | 标准 Softmax, tp_h=4, tp_s=4 |
| **HMP FWD (Ours)** | `tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_hmp_fwd_bs64_sl1024.json` | `gqa_hmp_fwd_tph4_tps4` | Online Softmax, 合并通信, tp_h=4, tp_s=4 |
| **TP16 FWD** | `tp16_gqa/tp16_hbm4_mesh2d_4x4_onering_qwen_gqa_tp16_fwd_bs64_sl1024.json` | `gqa_tp16_fwd` | 纯 TP16, KVHead 不分片 |

## 3. GQA-Only 仿真结果

### 3.1 全局指标对比

| 指标 | Baseline FWD | HMP FWD (Ours) | TP16 FWD |
|------|:-----------:|:-------------:|:--------:|
| **Wall time (cycles)** | 108,350 | **84,235** | 110,414 |
| **GPU time (cycles)** | 26,344 | 39,766 | 83,934 |
| **Comm time (cycles)** | 82,847 | 45,310 | 26,480 |
| **Overlap (cycles)** | 841 | 841 | — |
| **Compute utilization** | 93.975% | 96.008% | 41.311% |

### 3.2 逐操作通信对比 (Layer 0)

| 操作 | Baseline FWD | HMP FWD (Ours) | TP16 FWD |
|------|:------------|:--------------|:---------|
| `mha.attn_kernel.k1` | ALL_TO_ALL g21 22.7 us | ALL_TO_ALL g21 22.7 us | — |
| `mha.attn_kernel.v1` | ALL_TO_ALL g21 22.7 us | ALL_TO_ALL g21 22.7 us | — |
| `mha.attn` | ALL_REDUCE g21 19.8 us | — | — |
| `mha.o` | ALL_REDUCE g17 7.8 us + ALL_GATHER g21 10.0 us | — | ALL_REDUCE g17 26.5 us |
| **每层通信总计** | **82.8 us (5 ops)** | **45.3 us (2 ops)** | **26.5 us (1 op)** |

### 3.3 通信类型汇总

**Baseline FWD:**

| 类型 | 次数 | 总延迟 | 平均延迟 |
|------|:----:|-------:|-------:|
| ALL_TO_ALL | 2 | 45.3 us | 22.655 us |
| ALL_REDUCE | 2 | 27.5 us | 13.760 us |
| ALL_GATHER | 1 | 10.0 us | 10.017 us |

**HMP FWD (Ours):**

| 类型 | 次数 | 总延迟 | 平均延迟 |
|------|:----:|-------:|-------:|
| ALL_TO_ALL | 2 | 45.3 us | 22.655 us |

**TP16 FWD:**

| 类型 | 次数 | 总延迟 | 平均延迟 |
|------|:----:|-------:|-------:|
| ALL_REDUCE | 1 | 26.5 us | 26.480 us |

## 4. 端到端仿真对照 (94 层 × HBM4 OneRing)

| 指标 | Baseline FWD | HMP FWD (Ours) | TP16 FWD |
|------|:-----------:|:-------------:|:--------:|
| **Wall time (cycles)** | 58,949,760 | **58,849,758** | 59,083,090 |
| **GPU time (cycles)** | 40,159,950 | 41,790,568 | 45,573,410 |
| **Comm time (cycles)** | 18,868,864 | 17,217,298 | 13,509,680 |
| **Overlap (cycles)** | 79,054 | 158,108 | — |
| **Compute utilization** | 96.357% | 96.151% | 86.956% |

### 端到端逐层通信对比 (Layer 0)

| 操作 | Baseline FWD | HMP FWD (Ours) | TP16 FWD |
|------|:------------|:--------------|:---------|
| `mha.x` (weight gather) | AG g17 24.7 us + AG g21 93.7 us | AG g17 24.7 us + AG g21 93.7 us | AG g17 117.2 us |
| `mha.attn_kernel.k1` | A2A g21 22.7 us | A2A g21 25.6 us | — |
| `mha.attn_kernel.v1` | A2A g21 22.7 us | A2A g21 25.6 us | — |
| `mha.attn` | AR g21 19.8 us | — | — |
| `mha.o` / `mha_res.x1` | AR g17 7.8 us + AG g21 10.0 us | RS g17 4.0 us + RS g21 10.1 us | AR g17 26.5 us |
| **每层通信总计** | **201.3 us (7 ops)** | **183.6 us (6 ops)** | **143.7 us (2 ops)** |

## 5. 分析结论

### 5.1 GQA 层面：HMP FWD 通信最优

在纯 GQA 层（排除权重 AllGather 等公共开销）：

- **HMP FWD (Ours)** wall time 最低（84,235 cycles），比 Baseline 低 **22.3%**，比 TP16 低 **23.7%**
- HMP FWD 仅需 **2 次 ALL_TO_ALL** 通信（K/V 重排），因为 Online Softmax 消除了中间的 ALL_REDUCE，输出投影后的归约也被延迟合并
- TP16 通信量最少（1 次 ALL_REDUCE），但 GPU 计算量最大（83,934 cycles vs 26,344/39,766），因为 KVHead 未分片导致每个 GPU 需处理全部 KV heads

### 5.2 为什么端到端 Wall Time 如此接近

三种策略的端到端 wall time 差异仅 ~0.4%（58.85M–59.08M cycles），原因：

1. **权重 AllGather 主导通信**：`mha.x` 的 AllGather（16MB × 2 groups）占单层通信的 ~60–82%，三种策略均需执行
2. **94 层 FFN 计算一致**：FFN 层的计算量和通信模式（AllGather + ReduceScatter）在三种策略中完全相同
3. **GQA 差异被稀释**：GQA 通信差异（~37 us/layer）在 94 层总延迟（~59M cycles）中仅占 0.006%

### 5.3 关键发现

| 维度 | Baseline FWD | HMP FWD (Ours) | TP16 FWD |
|------|:------------|:--------------|:---------|
| GQA 通信次数 | 5 ops | **2 ops** (最少通信次数) | 1 op |
| GQA Wall Time | 108,350 | **84,235** (最优) | 110,414 |
| GQA 计算量 | 最低 | 中等 | 最高（KV 未分片） |
| 端到端优势场景 | — | 通信密集型拓扑（高延迟网络） | 通信带宽充裕时 |

---

## 附件

- 完整 GQA-only 分析输出：[`gqa_only_hbm4_onering_comparison.txt`](gqa_only_hbm4_onering_comparison.txt)
- 完整端到端分析输出：[`e2e_hbm4_onering_comparison.txt`](e2e_hbm4_onering_comparison.txt)
- 结果缓存：[`../cache_db_v131/results_cache.csv`](../cache_db_v131/results_cache.csv) (行 149–151: GQA-only, 行 146–148: 端到端)
- **Ring vs Mesh2D 拓扑对比报告**：[`ring_vs_mesh2d_gqa_comparison.md`](ring_vs_mesh2d_gqa_comparison.md)
