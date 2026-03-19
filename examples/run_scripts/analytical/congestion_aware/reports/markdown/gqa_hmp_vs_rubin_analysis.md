# GQA 单层延迟分析报告

> 模型: Qwen-MoE (D=4096, Hq=64, Hkv=4, dk=128, 1层 GQA decode)
> 硬件: HBM4 Mesh2D 4×4 (16 NPU) vs Rubin 单卡
> 数据来源: AstraSim 仿真 + attention.py 解析模型

---

## Part 1: attention.py 与 AstraSim 对齐验证

attention.py 使用与 AstraSim 完全相同的 Roofline 模型: `time = max(num_ops/peak_perf, tensor_size/bandwidth)`。
其中 `num_ops` = MAC 数, `tensor_size` = 输出元素数 (FP8 下等于字节数)。

### B=64, Seq=65536 对齐结果

| Strategy | Module    | Predicted (ns) | AstraSim (ns) | Error |
|----------|-----------|---------------:|---------------:|------:|
| HMP      | QKV       |       20,132.7 |       20,131.0 | 0.0%  |
| HMP      | Attention |      250,555.1 |      250,554.0 | 0.0%  |
| HMP      | Output    |       17,895.7 |       17,999.0 | 0.6%  |
| HMP      | **GPU**   |  **288,583.4** |  **288,690.0** | 0.0%  |
| Baseline | QKV       |       20,132.7 |       20,131.0 | 0.0%  |
| Baseline | Attention |      250,555.1 |      250,554.0 | 0.0%  |
| Baseline | Output    |        4,473.9 |        4,473.0 | 0.0%  |
| Baseline | **GPU**   |  **275,161.6** |  **275,164.0** | 0.0%  |
| TP16     | QKV       |        5,033.2 |        5,032.0 | 0.0%  |
| TP16     | Attention |      250,539.8 |      250,538.0 | 0.0%  |
| TP16     | Output    |        4,473.9 |        4,473.0 | 0.0%  |
| TP16     | **GPU**   |  **260,046.8** |  **260,043.0** | 0.0%  |

### B=1, Seq=65536 对齐结果

| Strategy | Module    | Predicted (ns) | AstraSim (ns) | Error |
|----------|-----------|---------------:|---------------:|------:|
| HMP      | QKV       |          314.6 |          313.0 | 0.5%  |
| HMP      | Attention |        3,914.9 |        3,912.0 | 0.1%  |
| HMP      | Output    |          279.6 |          279.0 | 0.2%  |
| HMP      | **GPU**   |    **4,509.1** |    **4,505.0** | 0.1%  |
| Baseline | QKV       |          314.6 |          313.0 | 0.5%  |
| Baseline | Attention |        3,914.9 |        3,912.0 | 0.1%  |
| Baseline | Output    |           69.9 |           69.0 | 1.3%  |
| Baseline | **GPU**   |    **4,299.4** |    **4,294.0** | 0.1%  |
| TP16     | QKV       |           78.6 |           69.0 | 13.9% |
| TP16     | Attention |        3,914.7 |        3,912.0 | 0.1%  |
| TP16     | Output    |           69.9 |           69.0 | 1.3%  |
| TP16     | **GPU**   |    **4,063.2** |    **4,058.0** | 0.1%  |

**结论**: GPU 总计误差均 <0.2%，完美对齐。TP16 B=1 的 QKV 13.9% 误差来自 AstraSim 模块归属问题 (Q 投影 69ns 记入 qkv_comp，KV 投影 ~8ns 记入 gpu 但未显式归入 qkv_comp)，不影响总计。

---

## Part 2: HMP CUSTOM Op 重分类

HMP trace 中 Output 模块包含一个 CUSTOM op (AllReduce stub)，其参数:
- `num_ops = 4×B×D`, `tensor_size = B×D`, OI = 4
- Roofline: `perf = min(BW×OI, peak) = min(2.5T×4, 30T) = 10 TMAC/s`

### CUSTOM op 耗时 (Seq=65536)

| Batch | Output 总计 (ns) | O1 Einsum (ns) | CUSTOM (ns) | CUSTOM 占比 |
|------:|-----------------:|---------------:|------------:|------------:|
|     1 |              279 |          279.6 |         1.6 |        0.6% |
|     4 |            1,118 |        1,118.5 |         6.6 |        0.6% |
|    16 |            4,499 |        4,473.9 |        26.2 |        0.6% |
|    64 |           17,999 |       17,895.7 |       104.9 |        0.6% |

CUSTOM op 仅占 ~0.6%，量级可忽略。重分类后 HMP 的 Wall Time 不变，但语义上将 CUSTOM 归入通信开销。

---

## Part 3: 四策略延迟对比

### B=64, Seq=65536 完整 Breakdown

| Strategy | QKV (ns) | Attn (ns) | Output (ns) | CUSTOM (ns) | Comm (ns) | **Wall (ns)** |
|----------|--------:|-----------:|------------:|------------:|----------:|--------------:|
| HMP      |  20,131 |   250,554  |    17,895*  |       105   |         0 |   **288,690** |
| Baseline |  20,131 |   250,554  |     4,473   |           0 |    18,265 |   **293,429** |
| TP16     |   5,032 |   250,538  |     4,473   |           0 |   236,538 |   **496,581** |
| Rubin    |     276 |   195,229  |       245   |           0 |         0 |   **195,750** |

*HMP Output = O1 Einsum(17,895) + CUSTOM(105)

### 各策略相对 Rubin 的 Wall Time 倍率

| Batch | HMP/Rubin | Baseline/Rubin | TP16/Rubin |
|------:|----------:|---------------:|-----------:|
|     1 |     1.47x |          7.37x |     17.37x |
|     4 |     1.47x |          2.42x |      4.92x |
|    16 |     1.47x |          1.59x |      2.72x |
|    64 |     1.47x |          1.50x |      2.54x |

### 可视化

见 `reports/gqa_latency_breakdown_batch_scaling.png` (Batch scaling)
和 `reports/gqa_latency_breakdown_seq_scaling.png` (Seq scaling)
和 `reports/gqa_latency_composition_b64_s65536.png` (百分比分解)

---

## Part 4: Rubin 优于 HMP 的原因分析

### 4.1 核心数据对比 (B=64, Seq=65536)

| 组件      | HMP (ns)  | Rubin (ns) | HMP/Rubin | HMP 占比 | Rubin 占比 |
|-----------|----------:|-----------:|----------:|---------:|-----------:|
| QKV       |    20,131 |        276 |    72.9x  |    7.0%  |      0.1%  |
| Attention |   250,554 |    195,229 |     1.28x |   86.8%  |     99.7%  |
| Output    |    17,999 |        245 |    73.3x  |    6.2%  |      0.1%  |
| **Wall**  |**288,690**| **195,750**|  **1.47x**|          |            |

### 4.2 逐组件分析

#### QKV 投影: HMP 73x 慢于 Rubin

两者的 MAC 数完全相同 (B × Hq_per_npu × dk × D)，性能差异来自硬件:

| 参数          | HMP (每 NPU)  | Rubin (单卡) | 倍率   |
|---------------|:-------------:|:------------:|-------:|
| Peak Compute  | 30 TMAC/s     | 8,750 TMAC/s | 291.7x |
| HBM Bandwidth | 2.5 TB/s      | 22 TB/s      | 8.8x   |

QKV 是 compute-bound (OI = D = 4096 >> 交叉点)，所以 Rubin 291.7x 的算力优势直接转化为 73x 加速 (HMP 有 4 个 NPU 分担，所以 291.7/4 ≈ 73x)。

#### Output 投影: HMP 73x 慢于 Rubin

类似 QKV，compute-bound。但 HMP 还有额外开销：
- HMP 使用 `wo_mode=duplicate`（每 NPU 持有完整 Wo 权重），Output FLOPs = B × 16×128 × 4096
- Baseline 使用 `wo_mode=sharded`，Output FLOPs = HMP 的 1/4
- HMP 选择 4x 的冗余计算来换取 0 通信

#### Attention Kernel: HMP 1.28x 慢于 Rubin (核心差距)

Attention 是 Wall Time 的绝对主体 (HMP 86.8%, Rubin 99.7%)。

**HMP Attention 分解** (4 个串行子操作):

| 子操作           | 类型         | 公式                                          | 耗时 (ns)  |
|------------------|:------------:|-----------------------------------------------|----------:|
| K concat (mem)   | Memory-bound | B×CacheSeq×dk×Hkv / BW = 64×16385×128×1 / 2.5T | ~53,556  |
| V concat (mem)   | Memory-bound | 同上                                          | ~53,556   |
| QK^T einsum      | Compute-bound| B×Hq×CacheSeq×dk / Compute = 64×16×16385×128 / 30T | ~71,721 |
| Score@V einsum   | Compute-bound| 同上                                          | ~71,721   |
| **合计**         |              |                                               |**250,554**|

**Rubin Attention** (Fused Kernel, FlashDecoding):

| 子操作                   | 公式                                            | 耗时 (ns)  |
|--------------------------|------------------------------------------------|----------:|
| KV cache 读取 (mem)      | B×2×Hkv×Seq×dk / BW = 64×2×4×65536×128 / 22T  | ~195,229 |
| QK^T + Score@V (compute) | 2×B×Hq×Seq×dk / Compute = 2×64×64×65536×128 / 8750T | ~61,756 |
| **max(mem, compute)**    |                                                |**195,229**|

### 4.3 根本原因

**Rubin 优于 HMP 1.47x 的三个因素:**

1. **Fused Attention vs 分解操作 (贡献 ~56%)**
   - HMP 将 attention 拆为 4 个串行操作 (2×concat + 2×einsum)，无法重叠 memory 和 compute
   - Rubin 使用 fused kernel (FlashDecoding)，memory 和 compute 可重叠，取 max
   - 差值: 250,554 - 195,229 = **55,325 ns**

2. **Duplicate Wo 冗余计算 (贡献 ~15%)**
   - HMP Output = 17,999 ns (完整 Wo)，等效于 Baseline 的 4× (4,473 ns)
   - 这是 HMP 用冗余计算换 0 通信的设计代价
   - 额外开销: 17,999 - 4,473 = **13,526 ns**

3. **多 NPU 切分的 QKV 效率损失 (贡献 ~21%)**
   - HMP 16 NPU 各自计算 QKV 的 1/4 头，但每个 NPU 算力仅 30 TMAC/s
   - Rubin 单卡 8,750 TMAC/s 处理全部头，等效吞吐 291.7x
   - QKV 差值: 20,131 - 276 = **19,855 ns**

### 4.4 策略总结

| 策略     | 优势                         | 代价                          |
|----------|------------------------------|-------------------------------|
| HMP      | 零网络通信，并行 attention    | 冗余 Wo 计算，分解 attention  |
| Baseline | 共享 Wo，低通信              | 需要 AllReduce (18 μs)       |
| TP16     | 最小单 NPU 计算量            | 巨大通信开销 (237 μs)        |
| Rubin    | Fused attention，极高单卡算力 | 单卡方案，不可扩展            |
