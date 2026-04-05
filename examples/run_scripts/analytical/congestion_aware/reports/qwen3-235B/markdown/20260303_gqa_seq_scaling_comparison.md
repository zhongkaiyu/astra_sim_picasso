# GQA 单层 Seq Scaling 对比报告：TP16 vs Baseline vs HMP

> **生成时间**: 2026-03-03
> **设备**: HBM4 | **物理拓扑**: Mesh2D 4×4 | **逻辑路由**: OneRing
> **模型参数**: Qwen (head=64, kvhead=4, dmodel=4096, head_dim=128, batch=64)
> **并行配置**: tp_h=4, tp_s=4 (16 GPUs)
> **测试范围**: Seq = 1024 ~ 15360 (步长 1024) + 65536 (单层 GQA, num_stacks=1)

---

## 1. 实验目的

前期 GQA-only 分析（见 `gqa_analysis_hbm4_onering.md`）在固定 Seq=1024 下对比了三种策略，发现 HMP 通信最优、TP16 计算量最大。本实验扩展序列长度维度（1k→64k），观察 **KV cache 规模增长** 对各策略计算/通信平衡的影响。

核心差异点：
- **Baseline / HMP**: `CacheSeq = ceil(Seq / tp_s)` — KV cache 按序列维度切分到 4 个 GPU
- **TP16**: `CacheSeq = Seq - 1` — 每个 GPU 持有完整 KV cache，不做序列切分

## 2. 仿真结果

### 2.1 Wall Time 对比 (ns)

| Seq | TP16 | Baseline | HMP | 最优 | TP16 vs Baseline | TP16 vs HMP |
|------:|------:|---------:|------:|:----:|:----------------:|:-----------:|
| 1,024 | **38,181** | 50,262 | 44,207 | TP16 | **-24.0%** | **-13.6%** |
| 2,048 | **49,365** | 56,412 | 50,357 | TP16 | **-12.5%** | **-2.0%** |
| 3,072 | 60,551 | 62,564 | **56,509** | HMP | -3.2% | +7.1% |
| 4,096 | 71,735 | 68,716 | **62,661** | HMP | +4.4% | +14.5% |
| 5,120 | 82,921 | 74,868 | **68,813** | HMP | +10.8% | +20.5% |
| 6,144 | 94,105 | 81,020 | **74,965** | HMP | +16.1% | +25.5% |
| 7,168 | 105,291 | 87,171 | **81,116** | HMP | +20.8% | +29.8% |
| 8,192 | 116,475 | 93,323 | **87,268** | HMP | +24.8% | +33.5% |
| 9,216 | 127,659 | 99,475 | **93,420** | HMP | +28.3% | +36.7% |
| 10,240 | 138,845 | 105,625 | **99,570** | HMP | +31.4% | +39.4% |
| 11,264 | 150,029 | 111,777 | **105,722** | HMP | +34.2% | +41.9% |
| 12,288 | 161,215 | 117,929 | **111,874** | HMP | +36.7% | +44.1% |
| 13,312 | 172,399 | 124,081 | **118,026** | HMP | +38.9% | +46.1% |
| 14,336 | 183,584 | 130,233 | **124,178** | HMP | +41.0% | +47.9% |
| 15,360 | 194,768 | 136,385 | **130,330** | HMP | +42.8% | +49.4% |
| 65,536 | 742,825 | 437,815 | **431,760** | HMP | +69.7% | +72.0% |

### 2.2 GPU 计算时间 (ns)

| Seq | TP16 | Baseline | HMP |
|------:|------:|---------:|------:|
| 1,024 | 22,371 | 30,785 | 44,207 |
| 2,048 | 33,555 | 36,935 | 50,357 |
| 3,072 | 44,741 | 43,087 | 56,509 |
| 4,096 | 55,925 | 49,239 | 62,661 |
| 5,120 | 67,111 | 55,391 | 68,813 |
| 6,144 | 78,295 | 61,543 | 74,965 |
| 7,168 | 89,481 | 67,694 | 81,116 |
| 8,192 | 100,665 | 73,846 | 87,268 |
| 9,216 | 111,849 | 79,998 | 93,420 |
| 10,240 | 123,035 | 86,148 | 99,570 |
| 11,264 | 134,219 | 92,300 | 105,722 |
| 12,288 | 145,405 | 98,452 | 111,874 |
| 13,312 | 156,589 | 104,604 | 118,026 |
| 14,336 | 167,774 | 110,756 | 124,178 |
| 15,360 | 178,958 | 116,908 | 130,330 |
| 65,536 | 727,015 | 418,338 | 431,760 |

### 2.3 通信时间 (ns)

| Seq | TP16 | Baseline | HMP |
|------:|------:|---------:|------:|
| 全部 Seq | 15,810 (固定) | 19,477 (固定) | N/A (终端节点优化) |

### 2.5 逐操作通信明细 (Log 验证, NPU=0, Seq=1024)

**Baseline (3 ops, 19,477 ns)**:

| # | 节点 | 类型 | 大小 | Group | 延迟 | 说明 |
|:-:|------|:----:|-----:|:-----:|-----:|------|
| 1 | `mha.attn` | ALL_REDUCE | 0.125 MB | g21 (tp_s) | 7,760 ns | qkv kernel 的 x2_hidden=tp_s，聚合序列维度部分和 |
| 2 | `mha.o` | ALL_REDUCE | 0.0625 MB | g17 (tp_h) | 7,760 ns | output projection 后聚合 Head/tp_h 维度部分和 |
| 3 | `mha.o` | ALL_GATHER | 0.0625 MB | g21 (tp_s) | 3,957 ns | Dmodel/tp_s → Dmodel，恢复完整模型维度 |

**TP16 (2 ops, 15,810 ns)**:

| # | 节点 | 类型 | 大小 | Group | 延迟 | 说明 |
|:-:|------|:----:|-----:|:-----:|-----:|------|
| 1 | `mha.o` | ALL_REDUCE | 0.25 MB | g17 (tp_h) | 7,905 ns | output projection 后聚合 Head/(tp_h*tp_s) 部分和 (行方向) |
| 2 | `mha.o` | ALL_REDUCE | 0.25 MB | g21 (tp_s) | 7,905 ns | output projection 后聚合 Head/(tp_h*tp_s) 部分和 (列方向) |

**HMP (0 ops)**:
- HMP 的 output `o` 是终端节点（单层模式下无后续消费者），通信被优化跳过
- 多层 E2E 模式下 `o` 有后续依赖（残差连接），通信会恢复

### 2.4 CacheSeq 值对比

| Seq | TP16 (`Seq-1`) | Baseline / HMP (`ceil(Seq/4)`) | 倍数 |
|------:|------:|------:|:---:|
| 1,024 | 1,023 | 256 | 4.0× |
| 3,072 | 3,071 | 768 | 4.0× |
| 8,192 | 8,191 | 2,048 | 4.0× |
| 15,360 | 15,359 | 3,840 | 4.0× |
| 65,536 | 65,535 | 16,384 | 4.0× |

## 3. 分析

### 3.1 通信特征

| 策略 | 通信操作 | 通信量随 Seq 变化 | 固定开销 |
|------|----------|:-----------------:|:--------:|
| **TP16** | 2× AllReduce (o: g17 + g21) | **不变** — 仅与 Head/Dmodel 维度相关 | 15,810 ns |
| **Baseline** | 1× AllReduce (attn: g21) + 1× AllReduce (o: g17) + 1× AllGather (o: g21) | **不变** — 仅与 Head/Dmodel 维度相关 | 19,477 ns |
| **HMP** | 通信在终端节点，单层被优化掉 | N/A | 0 ns |

三种策略的 GQA 层内通信量均不依赖 Seq（通信发生在 Head/Dmodel 维度，与序列无关）：

- **Baseline (3 ops, 19,477 ns)**: ①`attn` 处 AllReduce(g21) 聚合 qkv 的 tp_s 部分和 → ②`o` 处 AllReduce(g17) 聚合 tp_h 方向的 head 部分和 → ③`o` 处 AllGather(g21) 将 Dmodel/tp_s 恢复为完整 Dmodel
- **TP16 (2 ops, 15,810 ns)**: `o` 处两步 AllReduce 分别在 g17(tp_h) 和 g21(tp_s) 上聚合 Head/(tp_h*tp_s) 部分和。无中间 AllReduce（因为 TP16 kernel 没有 x2_hidden）
- **HMP (0 ops)**: output `o` 是终端节点（单层无后续消费者），通信被优化跳过。多层 E2E 模式下通信会恢复

**Baseline 比 TP16 多 3,667 ns 的原因**：Baseline 需要额外的 `attn` AllReduce(g21) 来聚合 qkv kernel 的 tp_s 部分和，且 `o` 处需要 AllReduce + AllGather 两步（先归约 tp_h 再收集 tp_s），而 TP16 的 `o` 只需两步 AllReduce（head 维度已按 tp_h*tp_s 完全切分，无需 AllGather）。

### 3.2 计算增长率与 CUSTOM FLOPS 公式

#### CUSTOM op（QKV Attention Kernel）公式

| 策略 | CUSTOM op_attr (FLOPS) | CacheSeq |
|------|------------------------|----------|
| **Baseline** | `Batch/dp × (1+CacheSeq) × HeadDim × Head/tp_h × 4` | ceil(Seq/tp_s) |
| **HMP** | `Batch/dp × (1+CacheSeq) × HeadDim × Head/tp_h × 4` | ceil(Seq/tp_s) |
| **TP16** | `Batch/dp × (1+CacheSeq) × HeadDim × Head/(tp_h×tp_s) × 4` | Seq - 1 |

**乘数 4 的物理含义**：GQA 注意力计算包含两次 matmul，每次的乘加操作计为 2 FLOPS。注意：Einsum (M op) 框架将每个 MAC 计为 1 op（不含 ×2），因此 CUSTOM 报告的 ops 是 Einsum 等价值的 2×。详见 [`20260305_tp16_custom_vs_einsum_analysis.md`](20260305_tp16_custom_vs_einsum_analysis.md)。

| 步骤 | 操作 | 维度 (每 batch, 每 head) | FLOPS |
|------|------|-------------------------|-------|
| ① Q×K^T | `q[1,D_h] × K[1+CS,D_h]^T → score[1,1+CS]` | 1 × (1+CS) × D_h | × 2 |
| ② Score×V | `score[1,1+CS] × V[1+CS,D_h] → ctx[1,D_h]` | 1 × (1+CS) × D_h | × 2 |
| 合计 | × Batch/dp × Head_local | | **× 4** |

其中 Head_local = Head/tp_h (Baseline/HMP) 或 Head/(tp_h×tp_s) (TP16)。

#### 数值验证

代入当前参数（Batch=64, dp=1, HeadDim=128, Head=64, tp_h=4, tp_s=4, 30 TFLOPS）：

| Seq | CacheSeq (BL) | CUSTOM FLOPS (BL) | CUSTOM time (BL) | CacheSeq (TP16) | CUSTOM FLOPS (TP16) | CUSTOM time (TP16) |
|------:|------:|------:|------:|------:|------:|------:|
| 1,024 | 256 | 134,742,016 | 4,491 ns | 1,023 | 134,217,728 | 4,474 ns |
| 4,096 | 1,024 | 537,395,200 | 17,913 ns | 4,095 | 536,870,912 | 17,896 ns |
| 65,536 | 16,384 | 8,590,458,880 | 286,349 ns | 65,535 | 8,589,934,592 | 286,331 ns |

**关键发现：Baseline 和 TP16 的 CUSTOM FLOPS 在相同 Seq 下几乎相等**。这是因为：
- Baseline: (1+Seq/tp_s) × Head/tp_h = (1+Seq/4) × 16
- TP16: (1+Seq-1) × Head/(tp_h×tp_s) = Seq × 4

当 Seq >> 1 时：Seq/4 × 16 = Seq × 4 ✓

#### GPU 时间分项拆解

| 组件 | 含义 | Baseline (Seq=4096) | TP16 (Seq=4096) |
|------|------|------:|------:|
| 固定 matmul | q/kv/o1 权重投影 (不依赖 CacheSeq) | 24,607 ns | 11,185 ns |
| **CUSTOM** | **QKV attention kernel** | **17,913 ns** | **17,896 ns** |
| 其他 | concat(KV cache), identity 等 | 6,719 ns | 26,844 ns |
| 通信 | AllReduce / AllGather | 19,477 ns | 15,810 ns |
| **Wall 合计** | | **68,716 ns** | **71,735 ns** |

- 固定 matmul 差异 (24,607 vs 11,185 ns)：Baseline 的 q 投影 Head/tp_h=16，TP16 的 q 投影 Head/(tp_h×tp_s)=4，少 4×
- "其他"差异 (6,719 vs 26,844 ns)：TP16 的 KV cache 是 Baseline 的 4× (CacheSeq 4095 vs 1024)，concat 数据量成比例增大

#### CUSTOM 对 GPU 增长斜率的贡献

| 指标 | Baseline/HMP | TP16 |
|------|------:|------:|
| GPU 总斜率 | 6.008 ns/Seq | 10.923 ns/Seq |
| CUSTOM 斜率 | 4.369 ns/Seq | 4.369 ns/Seq |
| CUSTOM 占比 | **72.7%** | **40.0%** |
| 其他 (concat等) 斜率 | 1.639 ns/Seq | 6.554 ns/Seq |

**CUSTOM 的 per-Seq 斜率完全相同** (4.369 ns/Seq)，因为每增加 1 个 Seq 单位的 CUSTOM FLOPS 增量一致：
- Baseline: ΔCacheSeq=1/4 × Head/tp_h×4 = 1/4 × 16 × 4 = 16 → × Batch/dp×HeadDim = × 64×128 = 131,072 FLOPS/Seq
- TP16: ΔCacheSeq=1 × Head/(tp_h×tp_s)×4 = 1 × 4 × 4 = 16 → × 64×128 = 131,072 FLOPS/Seq

**1.82× GPU 总斜率差异全部来自非 CUSTOM 部分**（concat/copy KV cache），TP16 的 "其他" 斜率是 Baseline 的 4.0× (= tp_s)，因为 TP16 每层处理 4× 更大的 KV cache 张量。

### 3.3 交叉点分析

**TP16 vs Baseline** — TP16 优势 = (Baseline_wall - TP16_wall)

| Seq | TP16 优势 (ns) | 状态 |
|------:|------:|:----:|
| 1,024 | +12,081 | TP16 胜 |
| 2,048 | +7,047 | TP16 胜 |
| 3,072 | +2,013 | TP16 微弱领先 |
| 4,096 | -3,019 | **Baseline 胜** |
| 8,192 | -23,152 | Baseline 大幅领先 |
| 15,360 | -58,383 | Baseline 大幅领先 |

**TP16 vs Baseline 交叉点 ≈ Seq 3,500**

**TP16 vs HMP** — TP16 优势 = (HMP_wall - TP16_wall)

| Seq | TP16 优势 (ns) | 状态 |
|------:|------:|:----:|
| 1,024 | +6,026 | TP16 胜 |
| 2,048 | +992 | TP16 微弱领先 |
| 3,072 | -4,042 | **HMP 胜** |
| 4,096 | -9,074 | HMP 胜 |
| 8,192 | -29,207 | HMP 大幅领先 |
| 15,360 | -64,438 | HMP 大幅领先 |

**TP16 vs HMP 交叉点 ≈ Seq 2,200**

### 3.4 线性验证与增长斜率

对 16 个数据点（Seq = 1024, 2048, 3072, ..., 15360, 65536）进行线性拟合 `wall(Seq) = a × Seq + b`：

| 策略 | 斜率 a (ns/Seq) | 截距 b (ns) | R² | 最大残差 |
|------|----:|----:|:---:|:---:|
| **TP16** | 10.9227 | 26,996 | **1.0000000000** | ±1.0 ns |
| **Baseline** | 6.0075 | 44,110 | **1.0000000000** | ±1.0 ns |
| **HMP** | 6.0075 | 38,055 | **1.0000000000** | ±1.0 ns |

**逐点残差（代表性采样）**：

| Seq | BL 实测 | BL 预测 | BL 残差 | TP16 实测 | TP16 预测 | TP16 残差 |
|------:|------:|------:|------:|------:|------:|------:|
| 1,024 | 50,262 | 50,261.2 | +0.8 ns | 38,181 | 38,181.1 | -0.1 ns |
| 4,096 | 68,716 | 68,716.1 | -0.1 ns | 71,735 | 71,735.6 | -0.6 ns |
| 8,192 | 93,323 | 93,322.7 | +0.3 ns | 116,475 | 116,474.9 | +0.1 ns |
| 15,360 | 136,385 | 136,384.3 | +0.7 ns | 194,768 | 194,768.6 | -0.6 ns |
| 65,536 | 437,815 | 437,815.1 | -0.1 ns | 742,825 | 742,825.1 | -0.1 ns |

**结果：wall time 对 Seq 严格线性**（所有残差 < ±1 ns，为整数舍入误差）。

**严格线性的原因**：

1. **计算模型**：分析仿真器的 runtime = FLOPS / peak_throughput，而 FLOPS = `a × CacheSeq + b` 对 Seq 线性
2. **通信恒定**：通信量不依赖 Seq（仅与 Head/Dmodel 维度相关），三种策略通信时间分别固定为 15,810 / 19,477 / 0 ns
3. **无拥塞**：单层 GQA 所有通信严格串行（无时间重叠），且同一操作的并行 comm groups 在物理拓扑上无链路交叉（g17 各组在不同行，g21 各组在不同列），congestion-aware 退化为无拥塞情况

**注意**：严格线性是分析仿真器的特性，真实硬件会因以下因素呈现非线性：
- Roofline 效应（小 Seq memory-bound → 大 Seq compute-bound，两段斜率不同）
- Cache/TLB 容量边界（KV cache 超出 L2 时访存延迟跳变）
- Kernel launch 等固定开销在小 Seq 时占比更大

**斜率分析**（详见 §3.2 分项拆解）：

- TP16 总斜率是 Baseline/HMP 的 **1.82×**。其中 CUSTOM 斜率完全相同 (4.369 ns/Seq)，差异全部来自 concat/copy KV cache 操作 (TP16 的 "其他" 斜率 = BL 的 4.0× = tp_s)
- HMP 截距比 Baseline 低 ~6,055 ns = 通信差 (19,477 - 0 = 19,477) 被固定计算差额部分抵消
- Baseline 截距最高 (44,110 ns)：固定 matmul 24,607 ns + 通信 19,477 ns；TP16 截距最低 (26,996 ns)：固定 matmul 11,185 ns + 通信 15,810 ns

### 3.5 Seq 增长下的性能排名

| Seq 范围 | 最优 | 次优 | 最差 |
|----------|:----:|:----:|:----:|
| < ~2,200 | **TP16** | HMP | Baseline |
| ~2,200 – ~3,500 | **HMP** | TP16 | Baseline |
| > ~3,500 | **HMP** | Baseline | TP16 |

## 4. 累积 Decode 延迟分析

> **重要**: 上述数据是单步 decode 延迟（固定 CacheSeq 的一个时刻）。实际自回归解码中，每步的 cache 长度递增：第 t 步 cache 有 t-1 个 token（Seq=t）。生成 N 个 token 的总延迟 = Σ_{t=1}^{N} Latency(Seq=t)。

### 4.1 线性拟合参数

由 §3.4 验证，wall time 对 Seq 严格线性（R²=1.0，残差 <1 ns）。拟合参数：

| 策略 | 斜率 a (ns/Seq) | 截距 b (ns) | 物理含义 |
|------|----:|----:|---|
| **TP16** | 10.9227 | 26,996 | 斜率大：CacheSeq = Seq-1（完整 cache）；截距小：通信仅 15,810 ns |
| **Baseline** | 6.0075 | 44,110 | 斜率小：CacheSeq = Seq/4；截距大：通信 19,477 ns + 基础计算 |
| **HMP** | 6.0075 | 38,055 | 斜率同 Baseline；截距最小：通信 0 ns（终端节点优化） |

累积公式：Total(N) = a × N×(N+1)/2 + b × N（O(N²) 增长）

### 4.2 累积延迟对比

| 生成 N tokens | TP16 (ms) | Baseline (ms) | HMP (ms) | 最优 | TP16 vs HMP |
|------:|------:|------:|------:|:----:|:---:|
| 256 | **7.27** | 11.49 | 9.94 | TP16 | -26.9% |
| 512 | **15.26** | 23.37 | 20.27 | TP16 | -24.7% |
| 1,024 | **33.38** | 48.32 | 42.12 | TP16 | -20.8% |
| 2,048 | **78.21** | 102.94 | 90.54 | TP16 | -13.6% |
| 3,072 | **134.49** | 163.86 | 145.26 | TP16 | -7.4% |
| 4,096 | **202.23** | 231.08 | 206.28 | TP16 | -2.0% |
| ~4,500 | 231.95 | 259.19 | 231.95 | **交叉** | ±0% |
| 5,120 | 281.42 | 304.60 | **273.60** | HMP | +2.9% |
| 8,192 | 587.70 | 562.95 | **513.35** | HMP | +14.5% |
| 15,360 | 1,703.24 | 1,386.24 | **1,293.24** | HMP | +31.7% |
| 65,536 | 25,225.91 | 15,791.97 | **15,395.14** | HMP | +63.9% |

### 4.3 交叉点对比（单步 vs 累积）

| 对比 | 单步交叉点 | 累积交叉点 | 偏移原因 |
|------|:---:|:---:|---|
| TP16 vs HMP | Seq ~2,200 | **~4,500 tokens** | 早期步骤 TP16 优势被累积，推迟交叉 |
| TP16 vs Baseline | Seq ~3,500 | **~7,000 tokens** | 同上，且 Baseline 截距更高 |

累积交叉点比单步交叉点推迟约 **2×**，因为 Σ 运算放大了短序列阶段 TP16 的累积优势。

### 4.4 累积视角下的关键结论

1. **TP16 适用范围比单步分析显示的更大**: 生成 ≤4,500 tokens 时，TP16 累积总延迟最低
2. **O(N²) 增长使长序列差距剧烈放大**: 生成 65k tokens 时 TP16 累积延迟是 HMP 的 1.64×（25.2s vs 15.4s）
3. **实际推理场景**: 若平均输出长度 ~2k tokens，TP16 累积延迟比 HMP 低 13.6%；若输出 ~8k tokens，HMP 反超 14.5%

## 5. 结论

### 5.1 单步视角

1. **短序列 (Seq < 2k)**: TP16 单步最优。通信最低 (15.8k ns)，cache 计算增量有限。
2. **中长序列 (Seq 2k–3.5k)**: HMP 单步最优。兼具 CacheSeq=ceil(Seq/4) 和单层零通信。
3. **长序列 (Seq > 3.5k)**: HMP > Baseline > TP16。TP16 的 CacheSeq=Seq-1 导致 FLOPS 为 HMP 的 4×。

### 5.2 累积视角（实际 decode 总延迟）

4. **TP16 适用范围扩大**: 累积交叉点推迟至 ~4,500 tokens（vs 单步 ~2,200），因早期步骤 TP16 优势被累积放大。
5. **生成 ≤4k tokens**: TP16 总延迟最低（比 HMP 低 2–21%），适合短输出场景。
6. **生成 >4.5k tokens**: HMP 总延迟最低，且优势随输出长度加速扩大（O(N²) 增长）。
7. **生成 65k tokens**: TP16 总延迟 25.2s，HMP 15.4s，差距 **64%**。

### 5.3 通用结论

8. **通信不是瓶颈**: 单层 GQA 通信恒定（不依赖 Seq），性能差异完全由 **计算量** 主导。
9. **斜率决定长期趋势**: TP16 斜率 10.92 ns/Seq 是 HMP/Baseline 6.01 ns/Seq 的 **1.82×**，这是 CacheSeq 定义差异 (Seq-1 vs Seq/4) 的直接体现。
10. **实际部署建议**: 根据目标输出长度分布选择策略 — 短输出 (chatbot, QA) 用 TP16；长输出 (summarization, code gen) 用 HMP。

## 6. 多层 GQA 拥塞实验

### 6.1 实验目的

前述分析均在单层（num_stacks=1）下进行，仿真结果严格线性且无拥塞。为验证多层 GQA 是否产生链路拥塞，实现 GQA-only 的多层堆叠并测试。

### 6.2 实现：GQA 多层连接

修改 `qwen_gqa_only()` 函数（`models/stage1/qwen_model.py`），支持 `num_stacks` 参数：

**连接方式**：`transformer.{i-1}.mha.o` → `transformer.{i}.mha.x`

```
Layer 0:  x → [wq·q, wkv·kv] → [kt,vt] → [kc,vc concat] → [k1,v1] → [qkv CUSTOM] → [attn AR] → [o1 matmul] → [o AR+AG] → o
                                                                                                                               │
Layer 1:  x ← ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
          x → [wq·q, wkv·kv] → [kt,vt] → [kc,vc concat] → [k1,v1] → [qkv CUSTOM] → [attn AR] → [o1 matmul] → [o AR+AG] → o
                                                                                                                               │
Layer 2:  x ← ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
          ...
```

每层的输出 `o` 形状为 `[Batch/dp, 1, Dmodel]`，作为下一层的输入 `x`。通过 `ConnectGraph.apply()` 将 PlaceHolder 节点 `x` 替换为 Identical 节点（直接引用前一层的 `o`）。

### 6.3 测试结果（Baseline, Seq=4096, num_stacks=1~16）

| num_stacks | Wall (ns) | GPU (ns) | Comm (ns) | 每层 Wall | 拥塞增量 |
|----:|------:|------:|------:|------:|:---:|
| 1 | 68,716 | 49,239 | 19,477 | 68,716 | +0.00% |
| 2 | 137,432 | 98,478 | 38,954 | 68,716 | +0.00% |
| 4 | 274,864 | 196,956 | 77,908 | 68,716 | +0.00% |
| 8 | 549,728 | 393,912 | 155,816 | 68,716 | +0.00% |
| 16 | 1,099,456 | 787,824 | 311,632 | 68,716 | +0.00% |

**结果：零拥塞**。Wall time 完美线性缩放，每层恒定 68,716 ns。

### 6.4 零拥塞的根本原因

虽然多层连接已正确工作（comm 随层数等比增长），但所有通信仍然严格串行，无时间重叠：

```
时间轴 →

Layer 0: [COMP qkv]──[AR g21]──[COMP o1]──[AR g17]──[AG g21]──┐
                                                                │ 数据依赖
Layer 1:                                                        └──[COMP qkv]──[AR g21]──[COMP o1]──[AR g17]──[AG g21]──┐
                                                                                                                         │
Layer 2:                                                                                                                 └──[COMP qkv]──...
```

**原因**：Layer i+1 的所有操作都依赖 Layer i 的最终输出 `o`（包含通信），不存在可提前执行的独立操作。

**对比 E2E 模型为什么有拥塞**：E2E 每层包含 activation AllGather（`mha.x = AG(activation)`），发生在 LayerNorm 输出 `(Seq/cp)/(tp_h*tp_s)` 到 GQA 输入 `1/1` 的形状转换处。从 E2E log 验证 Layer 0→1 的时间线：

```
E2E Layer 0: [AG mha.x g17: 24.7us]──[AG mha.x g21: 24.7us]──[COMP]──[A2A k1]──[A2A v1]──[AR attn]──[AR o g17]──[AG o g21: 3.9us]──┐
                                                                                                                                        │ 1225 ns gap
E2E Layer 1: [AG mha.x g17: 24.4us]──[AG mha.x g21: 24.4us]──[COMP]──...                                                             ┘
```

Layer 1 的 mha.x AG 在 Layer 0 完成后才开始（因为 activation 依赖前层输出），但执行时间从 24.72→24.42 us 有 300 ns 差异，暗示 **spatial congestion**（不同 comm group 的 ring 路径在物理链路上的交叉）。

**E2E block2x2 实验中拥塞的真正来源**：不是层间通信时间重叠，而是**同一 AllGather/AllReduce 内部不同 comm group 的 ring 路径共享物理链路**。block2x2 将 g17 从行布局改为棋盘布局，不同 g17 组的 2-hop 路径在中间节点交叉，导致链路竞争（详见 §mesh2d-topology-findings）。

### 6.5 结论

| 场景 | 拥塞? | 原因 |
|------|:---:|------|
| GQA-only × 1 层 | 否 | 层内通信串行 |
| GQA-only × N 层 | 否 | 层间严格数据依赖，通信无时间重叠 |
| E2E × N 层 (原版 row/col) | 微弱 | mha.x AG 16MB 产生 spatial congestion，不同 group 的 ring 在物理拓扑上轻微交叉 |
| E2E × N 层 (block2x2) | **显著** | 棋盘布局导致 g17 组的 2-hop 路径在中间节点严重交叉，comm +17~20% |

GQA-only 无拥塞不是建模简化导致的——当前的权重 PlaceHolder 建模是**物理正确**的（§7 权重容量分析证实权重可完全常驻 GPU）。GQA-only 层间无可并行的独立通信源，这与真实硬件行为一致。

## 7. 权重容量分析：是否需要 Weight AllGather？

### 7.1 分析背景

§6.5 指出，要在 GQA-only 中引入拥塞，需将 weight AllGather 建模为独立通信源。本节估算当前设置下每 GPU 的权重大小，判断权重能否一次性常驻 GPU 显存，从而确认本地 PlaceHolder 建模是否合理。

### 7.2 模型参数

| 参数 | 值 | 说明 |
|------|---:|------|
| Dmodel | 4,096 | 模型隐藏维度 |
| HeadDim | 128 | 每头维度 |
| Head | 64 | Q 注意力头数 |
| KVHead | 4 | KV 注意力头数 (GQA) |
| tp_h | 4 | Head 维度并行度 |
| tp_s | 4 | Seq 维度并行度 |
| 精度 | BF16 | 2 bytes/element |

### 7.3 全模型权重（单层，未切分）

| 权重 | 形状 | 参数量 | BF16 大小 |
|------|------|-------:|----------:|
| wq | Dmodel × HeadDim × Head = 4096 × 128 × 64 | 33.55M | **64 MB** |
| wkv | Dmodel × HeadDim × 2×KVHead = 4096 × 128 × 8 | 4.19M | **8 MB** |
| wo | HeadDim × Head × Dmodel = 128 × 64 × 4096 | 33.55M | **64 MB** |
| **单层合计** | | **71.3M** | **136 MB** |
| **94 层合计** | | **6.7B** | **12.5 GB** |

### 7.4 每 GPU 权重（按策略切分）

#### Baseline (tp_h=4, tp_s=4)

| 权重 | 每 GPU 形状 | 每 GPU 大小 |
|------|-------------|----------:|
| wq | Dmodel × HeadDim × Head/tp_h = 4096 × 128 × 16 | **16 MB** |
| wkv | Dmodel × HeadDim × 2×KVHead/tp_h = 4096 × 128 × 2 | **2 MB** |
| wo | HeadDim × Head/tp_h × Dmodel/tp_s = 128 × 16 × 1024 | **4 MB** |
| **单层合计** | | **22 MB** |
| **94 层合计** | | **2.02 GB** |

#### HMP (tp_h=4, tp_s=4)

| 权重 | 每 GPU 形状 | 每 GPU 大小 |
|------|-------------|----------:|
| wq | 4096 × 128 × 16 | **16 MB** |
| wkv | 4096 × 128 × 2 | **2 MB** |
| wo | HeadDim × Head/tp_h × Dmodel = 128 × 16 × 4096 (未按 tp_s 切分) | **16 MB** |
| **单层合计** | | **34 MB** |
| **94 层合计** | | **3.12 GB** |

#### TP16 (tp_h×tp_s=16)

| 权重 | 每 GPU 形状 | 每 GPU 大小 |
|------|-------------|----------:|
| wq | Dmodel × HeadDim × Head/(tp_h×tp_s) = 4096 × 128 × 4 | **4 MB** |
| wkv | Dmodel × HeadDim × 2×KVHead/tp_h = 4096 × 128 × 2 | **2 MB** |
| wo | HeadDim × Head/(tp_h×tp_s) × Dmodel = 128 × 4 × 4096 | **4 MB** |
| **单层合计** | | **10 MB** |
| **94 层合计** | | **0.92 GB** |

### 7.5 与 KV Cache 的对比

| 资源 | Seq=1k | Seq=4k | Seq=16k | Seq=65k |
|------|-------:|-------:|--------:|--------:|
| Baseline 每 GPU 权重 (94层) | 2.02 GB | 2.02 GB | 2.02 GB | 2.02 GB |
| Baseline 每 GPU KV Cache (94层) | 0.75 GB | 2.94 GB | 11.75 GB | 47.00 GB |
| **权重 + KV 合计** | **2.77 GB** | **4.96 GB** | **13.77 GB** | **49.02 GB** |

KV Cache 计算：每层 = 2 × Batch/dp × CacheSeq × HeadDim × KVHead/tp_h × 2 bytes，其中 CacheSeq = ceil(Seq/tp_s)。

### 7.6 结论

| 问题 | 结论 |
|------|------|
| 权重能否一次性常驻 GPU？ | **能**。最大的 HMP 方案每 GPU 仅 3.12 GB / 94 层，远小于 HBM4 容量 (48–96 GB) 或 H100 容量 (80 GB) |
| 需要 Weight AllGather 吗？ | **不需要**。标准 TP 下每 GPU 永久持有自己的权重分片，直接本地参与 matmul 计算 |
| 显存瓶颈在哪？ | **KV Cache**。Seq=65k 时 KV Cache 需 47 GB，接近 H100 容量上限；权重仅占 2–3 GB |
| Weight AllGather 何时出现？ | 仅在 ZeRO-3（权重进一步切分到 data-parallel ranks）或权重流式加载场景，当前设置（dp=1）不适用 |
| 对拥塞建模的影响？ | 权重常驻 GPU 是**物理事实**，PlaceHolder 建模**正确**。GQA-only 多层零拥塞不是建模简化导致的，而是反映了真实硬件行为 |

---

## 附件

- 结果缓存：[`../cache_db_seq_scaling/results.sqlite`](../cache_db_seq_scaling/results.sqlite)
- CSV 导出：[`../cache_db_seq_scaling/results_cache.csv`](../cache_db_seq_scaling/results_cache.csv)
- CacheSeq 验证 (Seq=1022/1023/1024 结果一致)：见 cache_db_seq_scaling 中 baseline_sl102* 记录
- GQA 固定 Seq 对比报告：[`20260303_gqa_analysis_hbm4_onering.md`](20260303_gqa_analysis_hbm4_onering.md)
- Seq Scaling 生成脚本：[`../seq_scale.py`](../seq_scale.py)
- 多层拥塞测试缓存：[`../cache_db_congestion_test/results.sqlite`](../cache_db_congestion_test/results.sqlite)
- CUSTOM vs Einsum FLOP 计数分析：[`20260305_tp16_custom_vs_einsum_analysis.md`](20260305_tp16_custom_vs_einsum_analysis.md)
