# Fig 6: 设计空间探索报告

> 模型: Qwen3-235B, seq=64K, 策略: RO_new (hmp_reo_new)
> 原始 Roofline 模型 (不含 utilization 调整)
> HBM BW 固定 2.5 TB/s/NPU, hop_latency=15ns

---

## 数据来源

### 计算流程

```
1. roofline_gqa_calc.py --peak-perf PP --bandwidth 2.5 --link-bw LBW --batch BS --seq 65536
   生成 roofline JSON, 包含每个 NPU 的 compute_ns, memory_ns, comm_ns

2. 从 roofline JSON 中读取 RO_new 策略的各模块延时:
   QKV_total  = max(compute_ns, memory_ns)   // compute_ns = 2*MACs / (PP*1e12), memory_ns = total_mem / (2.5*1e12)
   Attn_total = max(compute_ns, memory_ns)   // memory = KV cache loading
   ProjO_total = max(compute_ns, memory_ns)
   Comm_total = sum(steps * (hop_latency + chunk / LBW))  // chunk 和 BS 成正比
   Wall = QKV + Attn + ProjO + Comm
```

### 参数依赖关系

| 参数 | 影响 compute_ns | 影响 memory_ns | 影响 comm_ns |
|------|:-:|:-:|:-:|
| Compute Power (PP) | 反比于 PP | 无 | 无 |
| D2D Link BW (LBW) | 无 | 无 | 反比于 LBW |
| Batch Size (BS) | 正比于 BS | 正比于 BS (权重不变) | 正比于 BS |
| HBM BW | 无 | 反比于 HBM_BW (固定 2.5) | 无 |

### 脚本路径

| 脚本 | 用途 |
|------|------|
| `sweep_compute_bw.py` | 调用 roofline 扫描 140 个参数组合 |
| `roofline_gqa_calc.py` | 核心 roofline 模型 (compute_ns, memory_ns, comm_ns) |
| `plot_exploration.py` | 可视化: 热力图 + 缩放曲线 |

### 数据文件

```
paper_figures/fig6_design_exploration/data/sweep_compute_bw_batch.json
  140 条记录: 7 Compute x 5 D2D BW x 4 Batch
```

---

## Wall Time 数据表

### bs=1

| Compute \ D2D BW | 0.5 TB/s | 1.0 TB/s | 1.5 TB/s | 2.0 TB/s | 2.5 TB/s |
|---:|--------:|--------:|--------:|--------:|--------:|
| 8T | 18.9 | 18.8 | 18.8 | 18.8 | 18.8 |
| 16T | 10.5 | 10.4 | 10.4 | 10.4 | 10.4 |
| 32T | 6.3 | 6.3 | 6.2 | 6.2 | 6.2 |
| 64T | 4.2 | 4.2 | 4.1 | 4.1 | 4.1 |
| 96T | 3.8 | 3.7 | 3.7 | 3.7 | 3.7 |
| 128T | 3.8 | 3.7 | 3.7 | 3.7 | 3.7 |
| 256T | 3.8 | 3.7 | 3.7 | 3.7 | 3.7 |

### bs=4

| Compute \ D2D BW | 0.5 TB/s | 1.0 TB/s | 1.5 TB/s | 2.0 TB/s | 2.5 TB/s |
|---:|--------:|--------:|--------:|--------:|--------:|
| 8T | 72.0 | 71.9 | 71.9 | 71.9 | 71.8 |
| 16T | 36.2 | 36.1 | 36.1 | 36.1 | 36.1 |
| 32T | 19.0 | 18.9 | 18.9 | 18.9 | 18.9 |
| 64T | 10.6 | 10.5 | 10.5 | 10.5 | 10.5 |
| 96T | 8.9 | 8.8 | 8.8 | 8.8 | 8.8 |
| 128T | 8.9 | 8.8 | 8.8 | 8.8 | 8.8 |
| 256T | 8.9 | 8.8 | 8.8 | 8.8 | 8.8 |

### bs=16

| Compute \ D2D BW | 0.5 TB/s | 1.0 TB/s | 1.5 TB/s | 2.0 TB/s | 2.5 TB/s |
|---:|--------:|--------:|--------:|--------:|--------:|
| 8T | 287.6 | 287.5 | 287.5 | 287.5 | 287.5 |
| 16T | 143.9 | 143.7 | 143.7 | 143.6 | 143.5 |
| 32T | 72.1 | 71.9 | 71.9 | 71.8 | 71.8 |
| 64T | 36.3 | 36.2 | 36.1 | 36.1 | 36.0 |
| 96T | 29.3 | 29.2 | 29.1 | 29.1 | 29.1 |
| 128T | 29.3 | 29.2 | 29.1 | 29.1 | 29.1 |
| 256T | 29.3 | 29.2 | 29.1 | 29.1 | 29.1 |

### bs=32

| Compute \ D2D BW | 0.5 TB/s | 1.0 TB/s | 1.5 TB/s | 2.0 TB/s | 2.5 TB/s |
|---:|--------:|--------:|--------:|--------:|--------:|
| 8T | 575.0 | 574.8 | 574.7 | 574.7 | 574.6 |
| 16T | 287.6 | 287.3 | 287.2 | 287.1 | 287.1 |
| 32T | 144.0 | 143.7 | 143.6 | 143.5 | 143.5 |
| 64T | 72.3 | 72.0 | 71.9 | 71.8 | 71.8 |
| 96T | 57.2 | 56.9 | 56.8 | 56.7 | 56.7 |
| 128T | 57.2 | 56.9 | 56.8 | 56.7 | 56.7 |
| 256T | 57.2 | 56.9 | 56.8 | 56.7 | 56.1 |

---

## 核心发现

### 1. Compute Power 存在饱和点

- **bs=1 时**: 从 8T 到 64T, wall time 从 18.8us 降到 4.1us (4.6x 改善)。但 64T 到 256T 仅从 4.1us 降到 3.7us (1.1x)。
- **饱和原因**: 在 ~64T 后, QKV 和 Proj_O 转为 memory-bound (compute_ns < memory_ns)。memory_ns 由 HBM BW (2.5 TB/s) 决定, 与算力无关。
- **Attention 始终 memory-bound**: KV cache 大小 = Hkv * cache_seq * dk = 2MB (seq=64K, tp_s=4), 加载时间 = 2MB / 2.5TB/s = 839ns, 无论算力多高都不变。
- **设计启示**: 对 bs=1 decode, **算力超过 64T/NPU 无意义**, 应投资 HBM BW。

### 2. D2D BW 对 RO_new 影响极小

- **bs=1**: D2D BW 从 0.5 到 2.5 TB/s, wall time 变化 <0.5% (18.9 vs 18.8 us at 8T)。
- **bs=32**: 影响稍大但仍 <0.5% (575.0 vs 574.6 us at 8T)。
- **原因**: RO_new 的通信消息极小 (QKV AllGather ~576B, Final Reduce ~4KB at bs=1), 完全由固定延时 (hop=15ns) 主导, D2D 带宽项可忽略。
- **设计启示**: 对 RO_new 策略, D2D BW 0.5 TB/s 已足够, **无需高速互联**。这是 RO_new 相比 TP16 的核心架构优势。

### 3. Batch Size 是最强杠杆

- **原因**: 权重加载被 bs 个 token 分摊。bs=1 时权重占 memory_ns 的 99%+, bs=32 时权重被分摊 32 份。
- **但 Attention 不可分摊**: Attn 延时正比于 bs (每个 item 独立 KV cache)。
- **Throughput 提升**: 8T+bs=32 的吞吐 = 32 / 575us = 0.056 tokens/us, vs 8T+bs=1 的 1/18.8us = 0.053 tokens/us → 几乎一样! 因为 Attention 线性增长抵消了权重分摊。
- **高算力+大 batch**: 256T+bs=32 的吞吐 = 32/56.1 = 0.57 tokens/us, 比 8T+bs=1 的 0.053 高 10.7x。

### 4. Compute-Comm 边界分析

- GPU vs Comm 分解 (bs=1, D2D=1.5): 8T 时 Comm 占 1%, 256T 时 Comm 占 7%。
- **通信永远不是瓶颈**: 即使在 256T + D2D 0.5 的极端配置, Comm 也不超过 wall time 的 10%。
- 这是 RO_new 策略的本质优势: 通信操作仅涉及 dk 维度的 AllGather 和 d_model 的 Reduce, 消息大小与 seq 无关。

---

## 可视化

| 图表 | 路径 | 内容 |
|------|------|------|
| Fig 6a | `plots/fig6a_heatmap.{pdf,png}` | 热力图: Compute x D2D BW, 4 个 batch |
| Fig 6b | `plots/fig6b_compute_scaling.{pdf,png}` | 算力缩放: wall time + GPU/Comm 分解 |
| Fig 6c | `plots/fig6c_d2d_scaling.{pdf,png}` | D2D BW 缩放: wall time + comm time |
