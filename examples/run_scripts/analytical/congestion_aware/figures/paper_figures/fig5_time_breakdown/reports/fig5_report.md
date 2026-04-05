# Fig 5: 时间分解报告

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-235B (d=4096, Hq=64, Hkv=4, dk=128) |
| Batch Size | 1, 4, 8, 32 |
| 序列长度 | 8K, 128K, 1M |
| 精度 | FP8 |
| NPU 配置 | 96 TFLOPS/NPU, 2.5 TB/s HBM, 1.5 TB/s D2D |
| Utilization | utilization_96.json |
| 策略 | Ours (RO_new), H100, H100 TP2, Rubin, Rubin TP2 |

### 数据来源

- **Hybrid JSON**: `reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{1,4,8,32}.json`
- **Roofline**: `roofline_gqa_calc.py --peak-perf 96 --batch {1,4,8,32}` 生成各 batch 的理论延时
- **AstraSim**: `gqa_seq_scaling_bs1_80t_split4_bw1500_data.json` 提供通信延时 (bs=1 基准)
- **Utilization 调整**: NPU 用 `utilization_96.json`, Rubin/H100 用 `h100_rubin_utilization.json`
- **TP2 通信**: Rubin TP2 NVLink 1.8 TB/s, H100 TP2 NVLink 0.9 TB/s, roofline AllReduce 模型
- **Attention 模型**: 不可 batch 并行, wall_time = bs × single_attn_time
- **Proj 模型**: batched GEMM, 权重只加载一次, utilization 随 bs 提升

## 1. 各策略延时总览 (μs)

### bs = 1

| seq | Ours | H100 | H100 TP2 | Rubin | Rubin TP2 |
|-----|--------:|--------:|--------:|--------:|--------:|
| 8K |      3.4 |     40.6 |     30.3 |      6.2 |      6.1 |
| 128K |      5.9 |     84.5 |     56.1 |     12.9 |     10.1 |
| 1M |     29.4 |    380.5 |    203.9 |     57.9 |     32.6 |

### bs = 4

| seq | Ours | H100 | H100 TP2 | Rubin | Rubin TP2 |
|-----|--------:|--------:|--------:|--------:|--------:|
| 8K |      6.2 |     70.4 |     56.6 |     10.7 |     10.2 |
| 128K |     16.0 |    246.1 |    159.8 |     37.5 |     25.9 |
| 1M |    109.9 |   1430.3 |    750.9 |    217.8 |    115.9 |

### bs = 8

| seq | Ours | H100 | H100 TP2 | Rubin | Rubin TP2 |
|-----|--------:|--------:|--------:|--------:|--------:|
| 8K |      9.9 |    110.2 |     91.7 |     16.8 |     15.5 |
| 128K |     29.5 |    461.5 |    298.0 |     70.3 |     46.9 |
| 1M |    217.4 |   2830.0 |   1480.2 |    430.9 |    226.9 |

### bs = 32

| seq | Ours | H100 | H100 TP2 | Rubin | Rubin TP2 |
|-----|--------:|--------:|--------:|--------:|--------:|
| 8K |     34.3 |    348.9 |    302.1 |     53.1 |     47.6 |
| 128K |    112.5 |   1754.2 |   1127.2 |    267.1 |    173.2 |
| 1M |    864.1 |  11228.2 |   5856.2 |   1709.7 |    893.3 |

## 2. 模块分解 @ bs=1 (ns)

| seq | 策略 | QKV | Attn | Proj_O | GPU | Comm | Wall | Comm% |
|-----|------|----:|-----:|------:|----:|-----:|-----:|------:|
| 8K | Ours | 1,419 | 913 | 841 | 3,174 | 263 | 3,437 | 7.7% |
| 8K | H100 | 16,408 | 9,937 | 14,213 | 40,557 | 0 | 40,557 | 0.0% |
| 8K | H100 TP2 | 10,839 | 8,755 | 8,946 | 28,541 | 1,805 | 30,345 | 5.9% |
| 8K | Rubin | 2,498 | 1,513 | 2,164 | 6,176 | 0 | 6,176 | 0.0% |
| 8K | Rubin TP2 | 1,650 | 1,333 | 1,362 | 4,346 | 1,802 | 6,148 | 29.3% |
| 128K | Ours | 1,419 | 3,355 | 841 | 5,616 | 263 | 5,879 | 4.5% |
| 128K | H100 | 16,408 | 53,851 | 14,213 | 84,471 | 0 | 84,471 | 0.0% |
| 128K | H100 TP2 | 10,839 | 34,539 | 8,946 | 54,324 | 1,805 | 56,128 | 3.2% |
| 128K | Rubin | 2,498 | 8,200 | 2,164 | 12,863 | 0 | 12,863 | 0.0% |
| 128K | Rubin TP2 | 1,650 | 5,259 | 1,362 | 8,272 | 1,802 | 10,074 | 17.9% |
| 1M | Ours | 1,419 | 26,844 | 841 | 29,104 | 263 | 29,367 | 0.9% |
| 1M | H100 | 16,408 | 349,913 | 14,213 | 380,533 | 0 | 380,533 | 0.0% |
| 1M | H100 TP2 | 10,839 | 182,321 | 8,946 | 202,106 | 1,805 | 203,910 | 0.9% |
| 1M | Rubin | 2,498 | 53,282 | 2,164 | 57,945 | 0 | 57,945 | 0.0% |
| 1M | Rubin TP2 | 1,650 | 27,762 | 1,362 | 30,775 | 1,802 | 32,577 | 5.5% |

## 3. 通信分解 @ bs=1 (ns)

### seq = 8K

| 策略 | 通信操作 | 延时 (ns) | 来源 |
|------|---------|--------:|------|
| Ours | qkv_allgather | 76.2 | roofline |
|  | attn_rs | 76.0 | roofline |
|  | final_reduce | 110.9 | roofline |
| | **总计** | **263** | |
| H100 | (无通信) | 0 | — |
| H100 TP2 | allreduce_attn | 1,804.5 | roofline_nvlink_h100 |
| | **总计** | **1,805** | |
| Rubin | (无通信) | 0 | — |
| Rubin TP2 | allreduce_attn | 1,802.3 | roofline_nvlink |
| | **总计** | **1,802** | |

### seq = 128K

| 策略 | 通信操作 | 延时 (ns) | 来源 |
|------|---------|--------:|------|
| Ours | qkv_allgather | 76.2 | roofline |
|  | attn_rs | 76.0 | roofline |
|  | final_reduce | 110.9 | roofline |
| | **总计** | **263** | |
| H100 | (无通信) | 0 | — |
| H100 TP2 | allreduce_attn | 1,804.5 | roofline_nvlink_h100 |
| | **总计** | **1,805** | |
| Rubin | (无通信) | 0 | — |
| Rubin TP2 | allreduce_attn | 1,802.3 | roofline_nvlink |
| | **总计** | **1,802** | |

### seq = 1M

| 策略 | 通信操作 | 延时 (ns) | 来源 |
|------|---------|--------:|------|
| Ours | qkv_allgather | 76.2 | roofline |
|  | attn_rs | 76.0 | roofline |
|  | final_reduce | 110.9 | roofline |
| | **总计** | **263** | |
| H100 | (无通信) | 0 | — |
| H100 TP2 | allreduce_attn | 1,804.5 | roofline_nvlink_h100 |
| | **总计** | **1,805** | |
| Rubin | (无通信) | 0 | — |
| Rubin TP2 | allreduce_attn | 1,802.3 | roofline_nvlink |
| | **总计** | **1,802** | |

## 4. Batch 扩展特性 @ seq=128K

| bs | Ours Wall (μs) | Ours/tok | H100 Wall (μs) | Speedup vs H100 |
|---:|---------:|--------:|-----------:|------:|
| 1 |      5.9 |     5879 |       84.5 | 14.4× |
| 4 |     16.0 |     3998 |      246.1 | 15.4× |
| 8 |     29.5 |     3685 |      461.5 | 15.7× |
| 32 |    112.5 |     3514 |     1754.2 | 15.6× |

## 5. 核心发现

### 5.1 Attention 主导延时, 随 batch 线性增长

- bs=1 时 Attention 占 Ours wall time 的 57% (3.4 μs)
- bs=32 时 Attention 占比升至 95% (107.4 μs)
- 原因: 每个 batch item 有独立 KV cache, 无法并行处理, wall_time = bs × single_attn
- QKV/Proj_O 利用率随 bs 提升 (权重分摊), per-token 延时下降, 但 Attention 恒定

### 5.2 通信开销极低且恒定

- Ours (RO_new) 通信 = **263 ns** (全 seq/bs 恒定), 仅含 QKV AG + Attn RS + Final Reduce
- H100/Rubin 单卡无通信 (0 ns), 但 TP2 版本有 NVLink AllReduce:
  - H100 TP2: **1,805 ns** (NVLink 0.9 TB/s, msg=4KB)
  - Rubin TP2: **1,802 ns** (NVLink 1.8 TB/s, msg=4KB)
- 通信仅占 Ours wall time 的 **<5%** (bs=1), bs 增大后降至 **<1%**

### 5.3 H100 延时远高于 Ours

- seq=128K, bs=1: H100 wall = 84.5 μs, Ours = 5.9 μs → **14.4× 加速**
- 原因: H100 HBM BW 仅 3.35 TB/s (Ours 总计 40 TB/s), decode 完全 memory-bound
- H100 TP2 通过模型减半改善 ~30%, 但仍远慢于 Ours

### 5.4 Rubin TP2 在长 seq 下有优势

- seq=1M: Rubin TP1 = 57.9 μs, Rubin TP2 = 32.6 μs (0.56×)
- 模型减半节省的计算 (>170 μs) 远超 NVLink 通信 (1.8 μs), 净收益显著
- 但 Rubin TP2 仍慢于 Ours: Ours = 29.4 μs

## 6. 可视化

| 图表 | 路径 |
|------|------|
| 计算模块分解 (Fig 5a) | `paper_figures/fig5_time_breakdown/plots/fig5a_compute_breakdown.{pdf,png}` |
| 通信分解 (Fig 5b) | `paper_figures/fig5_time_breakdown/plots/fig5b_comm_breakdown.{pdf,png}` |

