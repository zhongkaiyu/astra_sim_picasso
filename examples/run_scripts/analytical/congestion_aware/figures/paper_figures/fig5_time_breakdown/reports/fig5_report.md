# Fig 5: 时间分解报告

## 实验配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen3-235B (d=4096, Hq=64, Hkv=4, dk=128) |
| Batch Size | 1, 4 |
| 序列长度 | 8K, 128K |
| 精度 | FP8 |
| NPU 配置 | 96 TFLOPS/NPU, 2.5 TB/s HBM, 1.5 TB/s D2D |
| Utilization | utilization_96.json |
| 策略 | Rubin, Rubin TP2, AMMA(TP16), AMMA(HMP), AMMA(HP+RO) |

### 数据来源

- **Hybrid JSON**: `reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{1,4}.json`
- **Roofline**: `roofline_gqa_calc.py --peak-perf 96 --batch {1,4}` 生成各 batch 的理论延时
- **AstraSim**: `gqa_seq_scaling_bs1_80t_split4_bw1500_data.json` 提供通信延时 (bs=1 基准)
- **Utilization 调整**: NPU 用 `utilization_96.json`, Rubin/H100 用 `h100_rubin_utilization.json`
- **TP2 通信**: Rubin TP2 NVLink 1.8 TB/s, H100 TP2 NVLink 0.9 TB/s, roofline AllReduce 模型
- **Attention 模型**: 不可 batch 并行, wall_time = bs × single_attn_time
- **Proj 模型**: batched GEMM, 权重只加载一次, utilization 随 bs 提升

---

## 1. Fig 5a 计算+通信分解 (2×2 面板)

Fig 5a 为 2×2 分组柱状图, 列 = 序列长度 (8K, 128K), 行 = Batch Size (1, 4)。每根柱子从下到上依次为 Proj QKV、Attention、Proj O、Comm 四段。

### 1.1 BS=1, Seq=8K

| 策略 | Proj_QKV (ns) | Attn (ns) | Proj_O (ns) | Compute (ns) | Comm (ns) | Total (ns) | Total (μs) | Comm% |
|------|----------:|------:|--------:|----------:|------:|-------:|-------:|------:|
| Rubin | 2,498.5 | 1,513.1 | 2,164.2 | 6,175.8 | 0.0 | 6,175.8 | 6.18 | 0.0% |
| Rubin TP2 | 1,650.4 | 1,333.2 | 1,362.3 | 4,345.9 | 1,802.3 | 6,148.2 | 6.15 | 29.3% |
| AMMA(TP16) | 1,419.2 | 389.1 | 840.7 | 2,649.0 | 712.0 | 3,361.0 | 3.36 | 21.2% |
| AMMA(HMP) | 1,419.2 | 389.1 | 841.3 | 2,649.6 | 467.3 | 3,116.9 | 3.12 | 15.0% |
| AMMA(HP+RO) | 1,419.2 | 389.1 | 841.3 | 2,649.6 | 263.1 | 2,912.7 | 2.91 | 9.0% |

### 1.2 BS=1, Seq=128K

| 策略 | Proj_QKV (ns) | Attn (ns) | Proj_O (ns) | Compute (ns) | Comm (ns) | Total (ns) | Total (μs) | Comm% |
|------|----------:|------:|--------:|----------:|------:|-------:|-------:|------:|
| Rubin | 2,498.5 | 8,200.0 | 2,164.2 | 12,862.7 | 0.0 | 12,862.7 | 12.86 | 0.0% |
| Rubin TP2 | 1,650.4 | 5,259.3 | 1,362.3 | 8,272.0 | 1,802.3 | 10,074.3 | 10.07 | 17.9% |
| AMMA(TP16) | 1,419.2 | 3,355.4 | 840.7 | 5,615.3 | 2,551.0 | 8,166.3 | 8.17 | 31.2% |
| AMMA(HMP) | 1,419.2 | 3,355.4 | 841.3 | 5,616.0 | 467.3 | 6,083.3 | 6.08 | 7.7% |
| AMMA(HP+RO) | 1,419.2 | 3,355.4 | 841.3 | 5,616.0 | 263.1 | 5,879.1 | 5.88 | 4.5% |

### 1.3 BS=4, Seq=8K

| 策略 | Proj_QKV (ns) | Attn (ns) | Proj_O (ns) | Compute (ns) | Comm (ns) | Total (ns) | Total (μs) | Comm% |
|------|----------:|------:|--------:|----------:|------:|-------:|-------:|------:|
| Rubin | 2,501.1 | 6,052.4 | 2,166.6 | 10,720.1 | 0.0 | 10,720.1 | 10.72 | 0.0% |
| Rubin TP2 | 1,652.2 | 5,332.9 | 1,363.8 | 8,348.8 | 1,802.3 | 10,151.1 | 10.15 | 17.8% |
| AMMA(TP16) | 1,419.3 | 1,556.5 | 846.2 | 3,822.0 | 2,848.0 | 6,670.0 | 6.67 | 42.7% |
| AMMA(HMP) | 1,419.3 | 1,556.5 | 848.7 | 3,824.5 | 969.2 | 4,793.6 | 4.79 | 20.2% |
| AMMA(HP+RO) | 1,419.3 | 1,556.5 | 848.7 | 3,824.5 | 302.4 | 4,126.9 | 4.13 | 7.3% |

### 1.4 BS=4, Seq=128K

| 策略 | Proj_QKV (ns) | Attn (ns) | Proj_O (ns) | Compute (ns) | Comm (ns) | Total (ns) | Total (μs) | Comm% |
|------|----------:|------:|--------:|----------:|------:|-------:|-------:|------:|
| Rubin | 2,501.1 | 32,800.0 | 2,166.6 | 37,467.7 | 0.0 | 37,467.7 | 37.47 | 0.0% |
| Rubin TP2 | 1,652.2 | 21,037.3 | 1,363.8 | 24,053.2 | 1,802.3 | 25,855.5 | 25.86 | 7.0% |
| AMMA(TP16) | 1,419.3 | 13,421.8 | 846.2 | 15,687.3 | 10,204.0 | 25,891.3 | 25.89 | 39.4% |
| AMMA(HMP) | 1,419.3 | 13,421.8 | 848.7 | 15,689.8 | 969.2 | 16,658.9 | 16.66 | 5.8% |
| AMMA(HP+RO) | 1,419.3 | 13,421.8 | 848.7 | 15,689.8 | 302.4 | 15,992.2 | 15.99 | 1.9% |

---

## 2. 通信分解详细表

### 2.1 BS=1, Seq=8K 通信分解 (ns)

| 策略 | 通信操作 | 延时 (ns) |
|------|---------|--------:|
| Rubin | (无通信) | 0.0 |
| Rubin TP2 | allreduce_attn (NVLink AR) | 1,802.3 |
| | **总计** | **1,802.3** |
| AMMA(TP16) | attn_comm (Attn AR) | 312.0 |
| | output_comm (Out 2×AR) | 400.0 |
| | **总计** | **712.0** |
| AMMA(HMP) | qkv_allgather (QKV AG) | 76.2 |
| | attn_comm (Attn AR) | 160.0 |
| | output_ar_comm (Out AR) | 154.1 |
| | output_ag_comm (Out AG) | 77.0 |
| | **总计** | **467.3** |
| AMMA(HP+RO) | qkv_allgather (QKV AG) | 76.2 |
| | attn_rs (Attn RS) | 76.0 |
| | final_reduce (Final Red) | 110.9 |
| | **总计** | **263.1** |

### 2.2 BS=1, Seq=128K 通信分解 (ns)

| 策略 | 通信操作 | 延时 (ns) |
|------|---------|--------:|
| Rubin | (无通信) | 0.0 |
| Rubin TP2 | allreduce_attn (NVLink AR) | 1,802.3 |
| | **总计** | **1,802.3** |
| AMMA(TP16) | attn_comm (Attn AR) | 2,171.0 |
| | output_comm (Out 2×AR) | 380.0 |
| | **总计** | **2,551.0** |
| AMMA(HMP) | qkv_allgather (QKV AG) | 76.2 |
| | attn_comm (Attn AR) | 160.0 |
| | output_ar_comm (Out AR) | 154.1 |
| | output_ag_comm (Out AG) | 77.0 |
| | **总计** | **467.3** |
| AMMA(HP+RO) | qkv_allgather (QKV AG) | 76.2 |
| | attn_rs (Attn RS) | 76.0 |
| | final_reduce (Final Red) | 110.9 |
| | **总计** | **263.1** |

### 2.3 BS=4, Seq=8K 通信分解 (ns)

| 策略 | 通信操作 | 延时 (ns) |
|------|---------|--------:|
| Rubin | (无通信) | 0.0 |
| Rubin TP2 | allreduce_attn (NVLink AR) | 1,802.3 |
| | **总计** | **1,802.3** |
| AMMA(TP16) | attn_comm (Attn AR) | 1,248.0 |
| | output_comm (Out 2×AR) | 1,600.0 |
| | **总计** | **2,848.0** |
| AMMA(HMP) | qkv_allgather (QKV AG) | 79.6 |
| | attn_comm (Attn AR) | 640.0 |
| | output_ar_comm (Out AR) | 166.4 |
| | output_ag_comm (Out AG) | 83.2 |
| | **总计** | **969.2** |
| AMMA(HP+RO) | qkv_allgather (QKV AG) | 79.6 |
| | attn_rs (Attn RS) | 79.1 |
| | final_reduce (Final Red) | 143.7 |
| | **总计** | **302.4** |

### 2.4 BS=4, Seq=128K 通信分解 (ns)

| 策略 | 通信操作 | 延时 (ns) |
|------|---------|--------:|
| Rubin | (无通信) | 0.0 |
| Rubin TP2 | allreduce_attn (NVLink AR) | 1,802.3 |
| | **总计** | **1,802.3** |
| AMMA(TP16) | attn_comm (Attn AR) | 8,684.0 |
| | output_comm (Out 2×AR) | 1,520.0 |
| | **总计** | **10,204.0** |
| AMMA(HMP) | qkv_allgather (QKV AG) | 79.6 |
| | attn_comm (Attn AR) | 640.0 |
| | output_ar_comm (Out AR) | 166.4 |
| | output_ag_comm (Out AG) | 83.2 |
| | **总计** | **969.2** |
| AMMA(HP+RO) | qkv_allgather (QKV AG) | 79.6 |
| | attn_rs (Attn RS) | 79.1 |
| | final_reduce (Final Red) | 143.7 |
| | **总计** | **302.4** |

---

## 3. 策略间加速比总览

### 3.1 AMMA(HP+RO) vs 其他策略 — Total Latency 加速比

| BS | Seq | AMMA(HP+RO) (μs) | Rubin (μs) | Rubin TP2 (μs) | AMMA(TP16) (μs) | AMMA(HMP) (μs) | vs Rubin | vs Rubin TP2 | vs TP16 | vs HMP |
|---:|----:|------:|------:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 8K | 2.91 | 6.18 | 6.15 | 3.36 | 3.12 | 2.12× | 2.11× | 1.15× | 1.07× |
| 1 | 128K | 5.88 | 12.86 | 10.07 | 8.17 | 6.08 | 2.19× | 1.71× | 1.39× | 1.03× |
| 4 | 8K | 4.13 | 10.72 | 10.15 | 6.67 | 4.79 | 2.60× | 2.46× | 1.62× | 1.16× |
| 4 | 128K | 15.99 | 37.47 | 25.86 | 25.89 | 16.66 | 2.34× | 1.62× | 1.62× | 1.04× |

### 3.2 通信开销对比

| BS | Seq | AMMA(HP+RO) Comm | AMMA(TP16) Comm | AMMA(HMP) Comm | Rubin TP2 Comm | TP16/HP+RO | HMP/HP+RO | RubinTP2/HP+RO |
|---:|----:|------:|------:|------:|------:|------:|------:|------:|
| 1 | 8K | 263 ns | 712 ns | 467 ns | 1,802 ns | 2.71× | 1.78× | 6.85× |
| 1 | 128K | 263 ns | 2,551 ns | 467 ns | 1,802 ns | 9.70× | 1.78× | 6.85× |
| 4 | 8K | 302 ns | 2,848 ns | 969 ns | 1,802 ns | 9.43× | 3.21× | 5.97× |
| 4 | 128K | 302 ns | 10,204 ns | 969 ns | 1,802 ns | 33.79× | 3.21× | 5.97× |

---

## 4. 核心发现

### 4.1 Attention 主导延时, 随 seq 和 batch 线性增长

- BS=1, Seq=8K 时 Attention 仅 389 ns, 占 AMMA(HP+RO) 的 13.4%
- BS=4, Seq=128K 时 Attention 增至 13,422 ns, 占 AMMA(HP+RO) 的 83.9%
- Rubin 单卡 Attention 比 AMMA 高 ~4× (bs=1 8K: 1,513 vs 389 ns), 因为 16-NPU 分散了 KV cache

### 4.2 AMMA(HP+RO) 通信开销极低且几乎恒定

- BS=1 时通信仅 **263 ns** (QKV AG 76 + Attn RS 76 + Final Red 111)
- BS=4 时通信仅 **302 ns**, 仅增长 15%
- Comm% 从 BS=1 Seq=8K 的 9.0% 降至 BS=4 Seq=128K 的 1.9%
- 相比之下 AMMA(TP16) 通信随 seq 线性暴涨: Seq=128K BS=4 时达 10,204 ns (39.4%)

### 4.3 AMMA(TP16) 在长序列+大batch下通信瓶颈严重

- BS=4 Seq=128K: AMMA(TP16) Comm = 10,204 ns, 占总延时 39.4%
- 其中 attn_comm = 8,684 ns, 随 seq 线性增长 (Attention AllReduce 传输量 ∝ seq)
- AMMA(HP+RO) 通过 ReduceScatter + Final Reduce 替代 AllReduce, 通信量降低 ~34×

### 4.4 Rubin TP2 通信开销固定但偏高

- Rubin TP2 通信恒为 **1,802 ns** (NVLink AllReduce), 不随 seq/bs 变化
- BS=1 Seq=8K 时 Comm% = 29.3% (通信瓶颈明显)
- 但在长 seq 时计算主导, Comm% 降至 7.0% (BS=4, Seq=128K)

### 4.5 AMMA(HMP) vs AMMA(HP+RO): RO 优化收益

- AMMA(HMP) → AMMA(HP+RO) 通信从 467→263 ns (BS=1) 和 969→302 ns (BS=4)
- RO 优化将 attn_comm (AllReduce) 替换为 attn_rs (ReduceScatter), 并消除了 output_ar/ag_comm
- 总延时收益: 1.03×–1.16× 加速 (通信占比本身不高, 但在短 seq 低 batch 下仍有意义)

---

## 5. 可视化

| 图表 | 路径 |
|------|------|
| 计算模块分解 (Fig 5a) | `paper_figures/fig5_time_breakdown/plots/fig5a_compute_breakdown.{pdf,png}` |
| 通信分解 (Fig 5b) | `paper_figures/fig5_time_breakdown/plots/fig5b_comm_breakdown.{pdf,png}` |
