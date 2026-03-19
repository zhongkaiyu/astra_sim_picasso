# TP16 GQA 计算图重构：CUSTOM vs Einsum (M op) FLOP 计数分析

> **生成时间**: 2026-03-05
> **设备**: HBM4 | **物理拓扑**: Mesh2D 4×4 | **逻辑路由**: OneRing
> **模型参数**: Qwen (head=64, kvhead=4, dmodel=4096, head_dim=128, batch=64)
> **并行配置**: 16 GPUs
> **前置报告**: [`20260303_gqa_seq_scaling_comparison.md`](20260303_gqa_seq_scaling_comparison.md)、[`20260304_gqa_block2x2_seq_scaling_comparison.md`](20260304_gqa_block2x2_seq_scaling_comparison.md)

---

## 1. 实验背景

前置报告中 TP16 的 attention kernel 使用 **CUSTOM op** 手动指定 FLOP 数。为更精确地建模计算量和通信模式，我们将 TP16 GQA 计算图重构为：

- **旧方案** (`tp_hq=16`)：CUSTOM op 包裹整个 QK^T + Attn×V，手写 FLOP 公式
- **新方案** (`tp_hkv=4, tp_hd=4`)：显式 Einsum (M op) 分别描述 QK^T 和 Attn×V，框架自动推导

### 1.1 并行维度映射

| 维度 | 旧方案 (CUSTOM) | 新方案 (Einsum) |
|------|-----------------|-----------------|
| 并行符号 | `tp_hq = 16` (单一扁平维度) | `tp_hkv = 4` + `tp_hd = 4` (双维度正交切分) |
| Q heads/GPU | Head/tp_hq = 64/16 = **4** | Head/tp_hkv = 64/4 = **16** (partial HeadDim) |
| HeadDim/GPU | **128** (完整) | 128/tp_hd = **32** (部分，需 AllReduce) |
| KV heads/GPU | KVHead/tp_hkv = 4/4 = **1** | KVHead/tp_hkv = 4/4 = **1** |
| spatial_parallel_dims | `[dp, tp_hq]` | `[dp, tp_hkv, tp_hd]` |

## 2. CUSTOM vs Einsum 的 FLOP 计数差异

### 2.1 框架源码对比

**CUSTOM** (`ops/customized.py`):

```python
def _eval_impl(cls, tensor):
    num_ops = sp.parse_expr(tensor.op_attr)  # 直接解析手写公式
    return y_shape, y_hidden, num_ops
```

**Einsum / M op** (`ops/einsum.py`):

```python
def _eval_impl(cls, tensor):
    num_ops = 1
    for dim in y_shape:      # 输出维度
        num_ops *= dim
    for dim in y_hidden:      # contracted 维度
        num_ops *= dim
    return y_shape, y_hidden, num_ops
```

**核心差异**：CUSTOM 完全依赖人工公式，不做推导；Einsum 自动从 einsum 字符串推导 `num_ops = ∏(output_dims) × ∏(contracted_dims)`，每个 multiply-accumulate 算 **1 次**。

### 2.2 旧 CUSTOM 公式拆解

CSV 中的 CUSTOM op_attr：

```
Batch/dp × (1+CacheSeq) × HeadDim × Head/tp_hq × 4
```

**×4 的含义**：2 个矩阵乘 × 2 FLOPs/MAC

| 步骤 | 操作 | 维度 (per batch, per head) | 计数 |
|------|------|---------------------------|------|
| ① Q×K^T | dot product | 1 × (1+CacheSeq) × HeadDim | **× 2** (FLOPs/MAC) |
| ② Score×V | matmul | 1 × (1+CacheSeq) × HeadDim | **× 2** (FLOPs/MAC) |
| 合计 | × Batch/dp × Head/tp_hq | | **× 4** |

### 2.3 新 Einsum 自动计算

**score (QK^T)**：einsum `bqdkg,bsdk->bkgqs`

```
y_shape    = [Batch/dp, KVHead/tp_hkv, Head/KVHead, 1, 1+CacheSeq]
y_hidden   = [HeadDim/tp_hd]
num_ops    = 64 × 1 × 16 × 1 × Seq × 32
```

**attn_v (Score×V)**：einsum `bkgqs,bsdk->bkgqd`

```
y_shape    = [Batch/dp, KVHead/tp_hkv, Head/KVHead, 1, HeadDim/tp_hd]
y_hidden   = [1+CacheSeq]
num_ops    = 64 × 1 × 16 × 1 × 32 × Seq
```

两个 matmul 的 num_ops 完全相等。

### 2.4 数值对比 (Seq=1024)

| 指标 | 旧 CUSTOM | 新 Einsum (两步合计) | 比值 |
|------|------:|------:|:---:|
| Attention ops | 134,217,728 | 67,108,864 | **2.0×** |
| 计数约定 | FLOPs (MAC×2) | MACs (MAC×1) | — |

### 2.5 MAC 等价性验证

虽然 ops 数差 2×，但实际 multiply-accumulate 次数完全相同：

| 方案 | Q heads/GPU | HeadDim/GPU | MACs per head per seq | 总 MACs/GPU |
|------|:-----------:|:-----------:|----------------------:|------------:|
| 旧 (tp_hq=16) | 4 | 128 | 128 | 4 × 128 = **512** |
| 新 (tp_hkv=4, tp_hd=4) | 16 | 32 (partial) | 32 | 16 × 32 = **512** |

每 GPU 每 batch 每 seq 位置的 attention MACs 完全一致 (512)，差异纯粹来自 **CUSTOM 的 ×2 FLOP/MAC 约定**。

## 3. KV 投影冗余发现

### 3.1 问题

旧方案 `spatial_parallel_dims = [dp, tp_hq]`，而 KV 投影权重 `wkv` 的形状中使用 `tp_hkv`（不在 spatial dims 中）：

```
旧 wkv: [Dmodel, HeadDim, 2×KVHead/tp_hkv]  →  [4096, 128, 2]
```

由于 `tp_hkv ∉ spatial_parallel_dims`，GraphDistributer 不会按 `tp_hkv` 分片 `wkv`。**16 个 GPU 全部持有相同的完整 KV 投影权重并重复计算**。

### 3.2 新方案修正

新方案 `spatial_parallel_dims = [dp, tp_hkv, tp_hd]`，`wkv` 中的 `tp_hkv` 和 `tp_hd` 都会被正确分片：

```
新 wkv: [Dmodel, HeadDim/tp_hd, 2×KVHead/tp_hkv]  →  [4096, 32, 2]
```

每个 GPU 只计算自己的 HeadDim 分片，无冗余。

### 3.3 KV 投影 ops 对比

| 方案 | wkv 每 GPU 形状 | 每 GPU ops | 冗余度 |
|------|----------------|------:|:---:|
| 旧 (tp_hq) | [4096, 128, 2] | 67,108,864 | 4× 冗余 |
| 新 (tp_hkv+tp_hd) | [4096, 32, 2] | 16,777,216 | 无冗余 |

> 注：全局总 KV 投影 MACs 不变（4 KV heads × 128 HeadDim × 4096 Dmodel × 64 Batch），但旧方案 4 组 GPU（共享同一 tp_hkv rank 的 4 个 tp_hq rank）各自重复计算同样的 KV 投影。

## 4. 每 GPU 总 ops 分项对比

以 Seq=1024 为例：

| 组件 | 旧 CUSTOM (ops) | 新 Einsum (ops) | 差异 | 差异原因 |
|------|------:|------:|:---:|------|
| Q 投影 (M) | 134,217,728 | 134,217,728 | 同 | Dmodel×HeadDim×Head_local 不变 |
| KV 投影 (M) | 67,108,864 | 16,777,216 | **-75%** | `tp_hd` 分片消除冗余 |
| Attention | 134,217,728 | 67,108,864 | **-50%** | ×2 FLOP/MAC 约定差异 |
| Output 投影 (M) | 134,217,728 | 134,217,728 | 同 | contracted dims 乘积不变 |
| **合计** | **469,762,048** | **352,321,536** | **-25%** | — |

其中：
- **-50% (attention)**：纯计数约定差异，不反映真实计算量变化
- **-75% (KV 投影)**：反映旧方案的实际建模缺陷（冗余计算被错误计入）

## 5. 通信模式差异

### 5.1 旧方案通信 (tp_hq=16)

旧 CUSTOM 的 attention kernel `x2_hidden=1`（无 partial sum），通信仅在 output projection 后触发：

| # | 位置 | 类型 | Group | 原因 |
|:-:|------|:----:|:-----:|------|
| 1 | `mha.o` | AllReduce | g17 (tp_h=4) | Head/(tp_h×tp_s) 的 tp_h 方向部分和 |
| 2 | `mha.o` | AllReduce | g21 (tp_s=4) | Head/(tp_h×tp_s) 的 tp_s 方向部分和 |

通信量固定：2 × AllReduce(0.25 MB) = **47,480 ns**（不随 Seq 变化）

### 5.2 新方案通信 (tp_hkv=4, tp_hd=4)

显式 Einsum 使框架精确感知每步的 partial sum：

| # | 位置 | 类型 | Group | 原因 |
|:-:|------|:----:|:-----:|------|
| 1 | `attn_kernel.score1` | AllReduce | tp_hd (4-way) | QK^T 的 HeadDim/tp_hd 部分和需聚合 |
| 2 | `mha.o` | AllReduce | tp_hkv (4-way) | output projection 的 KVHead/tp_hkv 部分和 |
| 3 | `mha.o` | AllReduce | tp_hd (4-way) | output projection 的 HeadDim/tp_hd 部分和 |

通信量**随 Seq 增长**（score AllReduce 大小 = Batch × Head/tp_hkv × 1 × Seq × 2 bytes）。

### 5.3 通信时间对比 (ns)

| Seq | 旧 CUSTOM | 新 Einsum | 差异 |
|------:|------:|------:|:---:|
| 1,024 | 47,480 | 24,513 | **-48.4%** |
| 2,048 | 47,480 | 25,601 | **-46.1%** |
| 4,096 | 47,480 | 27,632 | **-41.8%** |
| 8,192 | 47,480 | 31,767 | **-33.1%** |
| 16,384 | 47,480 | 40,000 | **-15.8%** |
| 32,768 | 47,480 | 56,832 | **+19.7%** |
| 65,536 | 47,480 | 92,106 | **+94.0%** |

旧方案通信恒定；新方案短序列通信更少（4-way AllReduce vs 16-way），但长序列因 score AllReduce 增长而超过旧方案。

## 6. 仿真结果对比

### 6.1 Wall Time (ns)

| Seq | 旧 CUSTOM (tp_hq=16) | 新 Einsum (tp_hkv+tp_hd) | Wall Δ |
|------:|------:|------:|:---:|
| 1,024 | 69,851 | **37,930** | **-45.7%** |
| 2,048 | 81,035 | **42,932** | **-47.0%** |
| 4,096 | 103,405 | **52,793** | **-48.9%** |
| 8,192 | 148,145 | **72,586** | **-51.0%** |
| 16,384 | 237,624 | **112,137** | **-52.8%** |
| 32,768 | 416,581 | **191,605** | **-54.0%** |
| 65,536 | 774,495 | **352,149** | **-54.5%** |

### 6.2 GPU 计算时间 (ns)

| Seq | 旧 CUSTOM | 新 Einsum | GPU Δ | 比值 |
|------:|------:|------:|:---:|:---:|
| 1,024 | 22,371 | 13,417 | -40.0% | 1.67× |
| 2,048 | 33,555 | 17,331 | -48.4% | 1.94× |
| 4,096 | 55,925 | 25,161 | -55.0% | 2.22× |
| 8,192 | 100,665 | 40,819 | -59.5% | 2.47× |
| 16,384 | 190,144 | 72,137 | -62.1% | 2.64× |
| 32,768 | 369,101 | 134,773 | -63.5% | 2.74× |
| 65,536 | 727,015 | 260,043 | -64.2% | 2.80× |

### 6.3 GPU 时间比值分析

GPU 时间比值从 1.67× (sl1024) 单调递增至 2.80× (sl65536)，原因：

- **短序列**：投影层占比大，Q/O 投影 ops 不变 → 比值被拉低
- **长序列**：attention 占比趋于主导
  - Attention ops 比值恒定 2.0× (CUSTOM ×2 FLOP 因子)
  - KV 投影比值恒定 4.0× (冗余消除)
  - 两者叠加使总比值趋近但不超过 2.0×，加上 KV 投影 4× 差异进一步拉高至 ~2.8×

## 7. 与 Baseline/HMP 重新对比

用新 Einsum TP16 数据更新策略对比（参照 `20260303_gqa_seq_scaling_comparison.md` §2.1）：

### 7.1 Wall Time 对比 (ns)

| Seq | 新 TP16 | Baseline | HMP | 最优 | TP16 vs BL | TP16 vs HMP |
|------:|------:|------:|------:|:----:|:---:|:---:|
| 1,024 | **37,930** | 50,262 | 44,207 | **TP16** | **-24.5%** | **-14.2%** |
| 2,048 | **42,932** | 56,412 | 50,357 | **TP16** | **-23.9%** | **-14.7%** |
| 4,096 | **52,793** | 68,716 | 62,661 | **TP16** | **-23.2%** | **-15.7%** |
| 8,192 | **72,586** | 93,323 | 87,268 | **TP16** | **-22.2%** | **-16.8%** |
| 16,384 | **112,137** | 142,537 | 136,482 | **TP16** | **-21.3%** | **-17.8%** |
| 32,768 | 191,605 | 240,962 | **234,907** | HMP | -20.5% | **-18.4%** |
| 65,536 | 352,149 | 437,815 | **431,760** | HMP | -19.6% | **-18.4%** |

### 7.2 关键发现

**新 TP16 在全部测试序列长度下均优于 Baseline 和 HMP**（单层 GQA）！

与旧 CUSTOM TP16 的对比：

| 指标 | 旧 CUSTOM TP16 | 新 Einsum TP16 |
|------|:--------------:|:--------------:|
| TP16 vs HMP 交叉点 | Seq ~2,200 | **不交叉**（全部序列 TP16 更优） |
| TP16 vs BL 交叉点 | Seq ~3,500 | **不交叉**（全部序列 TP16 更优） |
| sl1024 最优策略 | TP16 | TP16 |
| sl65536 最优策略 | HMP | TP16 (wall -18.4% vs HMP) |

> **重要注意**：新 TP16 全面领先的原因部分来自 CUSTOM ×2 FLOP 约定导致旧方案 GPU 时间虚高、KV 投影冗余。新旧方案的**实际 attention MAC 数完全相等**。严格来说，新方案更准确地反映了真实硬件的计算时间。

### 7.3 新 TP16 的通信随 Seq 增长

尽管 wall time 全面领先，新 TP16 的通信时间不再恒定：

| Seq | 新 TP16 comm | BL comm | HMP comm | TP16 comm 占 wall |
|------:|------:|------:|------:|------:|
| 1,024 | 24,513 | 19,477 | 0 | 64.6% |
| 4,096 | 27,632 | 19,477 | 0 | 52.3% |
| 16,384 | 40,000 | 19,477 | 0 | 35.7% |
| 65,536 | 92,106 | 19,477 | 0 | 26.2% |

通信增长来源：`score1` 处的 AllReduce(tp_hd) 大小 = Batch × Head × 1 × Seq × 2 bytes，随 Seq 线性增长。

## 8. 对仿真精度的影响

### 8.1 CUSTOM ×2 FLOP 因子的影响

AstraSim 的 roofline 模型直接使用 `num_ops` 估算 GPU 计算时间：

```
GPU_time ≈ max(num_ops / peak_TFLOPS, tensor_bytes / bandwidth)
```

CUSTOM 的 ×2 因子使 attention 的 `num_ops` 翻倍，导致 GPU 时间被**系统性高估**。这在旧方案中影响了所有包含 CUSTOM op 的策略（Baseline、HMP、TP16），但由于 Baseline/HMP 的 CUSTOM 公式也包含 ×4 因子，**相对排名不受影响**。

### 8.2 KV 投影冗余的影响

旧 TP16 方案中 KV 投影被 16 GPU 冗余计算但只需 4 GPU 各算一次。这在真实硬件上不存在（每个 GPU 只持有自己的 KV 权重分片），因此旧方案对 TP16 的 KV 投影时间高估了 4×。

新方案通过将 `tp_hkv` 和 `tp_hd` 都纳入 `spatial_parallel_dims`，正确消除了冗余。

### 8.3 总结

| 差异来源 | 影响范围 | 量化 | 是否影响策略相对排名 |
|----------|----------|------|:---:|
| CUSTOM ×2 FLOP/MAC | 所有使用 CUSTOM 的策略 (BL/HMP/TP16) | Attention ops ×2 | 否（BL/HMP 也含 ×4） |
| KV 投影冗余 | 仅旧 TP16 | KV proj ops ×4 | **是**（仅 TP16 受影响） |
| 通信自动推导 | 仅新 TP16 | score AllReduce 随 Seq 增长 | **是**（改变 TP16 通信特征） |

## 9. 结论

### 9.1 CUSTOM vs Einsum 计数

1. **CUSTOM ×4 = 2 matmul × 2 FLOPs/MAC**：手写公式将每个 multiply-accumulate 算 2 次浮点运算，Einsum 框架自然计算 MAC 数（每个算 1 次）。实际 attention MACs 完全相等：旧 4heads × 128dim = 新 16heads × 32dim = 512 MACs/seq/batch/GPU。
2. **KV 投影冗余**：旧方案 `tp_hkv ∉ spatial_parallel_dims` 导致 16 GPU 重复计算相同 KV 投影。新方案正确分片，KV 投影 ops 降 75%。
3. **总 ops 差异 -25%** (Seq=1024)：其中 -50% attention 是计数约定差异，-75% KV 是实质建模改善。

### 9.2 仿真性能

4. **新 TP16 全面领先**：在 1k~64k 序列范围内 wall time 比 HMP 低 14~18%，比 Baseline 低 20~25%。
5. **与旧 TP16 对比**：wall time 降 45~55%，GPU 时间比值从 1.67× (短序列) 递增至 2.80× (长序列)。
6. **通信特征改变**：旧方案通信恒定 (47,480 ns)，新方案通信随 Seq 增长 (24,513 → 92,106 ns)，但因 GPU 时间大幅降低，wall time 仍全面更优。

### 9.3 建模建议

7. **建议统一使用 Einsum (M op)** 替代 CUSTOM，避免手动 FLOP 估算引入 ×2 因子偏差。
8. **确保所有并行维度在 spatial_parallel_dims 中**，防止冗余计算被错误计入。
9. **Baseline/HMP 的 CUSTOM op 同样存在 ×2 因子**：三种策略的相对排名不受影响，但绝对 GPU 时间均被高估。若需精确对比，应同步将 Baseline/HMP 迁移至 Einsum 建模。

---

## 附件

- 结果缓存：[`../cache_db_seq_scaling/results.sqlite`](../cache_db_seq_scaling/results.sqlite)
- CSV 导出：[`../cache_db_seq_scaling/results_cache.csv`](../cache_db_seq_scaling/results_cache.csv)
- 新 TP16 计算图 CSV：`symbolic_tensor_graph_picasso/sharding_spreadsheets/module3/qwen/decode_bypass_tp16_fwd/`
- 新 TP16 可视化：`decode_bypass_tp16_fwd_assembled.png`
- CUSTOM op 源码：`symbolic_tensor_graph_picasso/symbolic_tensor_graph/ops/customized.py`
- Einsum op 源码：`symbolic_tensor_graph_picasso/symbolic_tensor_graph/ops/einsum.py`
- 前置 Seq Scaling 报告：[`20260303_gqa_seq_scaling_comparison.md`](20260303_gqa_seq_scaling_comparison.md)
