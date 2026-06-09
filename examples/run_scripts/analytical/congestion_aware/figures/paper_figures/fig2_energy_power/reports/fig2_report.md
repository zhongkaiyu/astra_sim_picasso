# Fig 2: 单 Token 能耗分析报告

> 配置: 96T, utilization_96, bs=1, FP8, D2D 1.5TB/s
> 基准: H100 单卡 (700W TDP, 1979 TFLOPS, 3.35 TB/s HBM3)
> 功耗系数 (更新): Ours cmpt_coeff=260, mem_coeff=1021

---

## Qwen3-235B

### 单 Token 能耗 (nJ)

| seq | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|---------:|---------:|---------:|---------:|---------:|
| 4K | 2,919,677 | 8,483,227 | 13,499,910 | 17,890,093 | 21,158,062 |
| 64K | 5,283,334 | 15,070,650 | 20,927,068 | 31,791,464 | 35,975,779 |
| 256K | 13,042,053 | 35,332,121 | 42,886,487 | 74,565,925 | 80,708,733 |
| 1M | 43,983,487 | 116,251,332 | 124,545,627 | 245,399,090 | 250,550,255 |

### 能效提升倍数 (Token/J, baseline = H100)

| seq | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|---------:|---------:|---------:|---------:|---------:|
| 4K | **6.1×** | 2.1× | 1.3× | 1.0× | 0.8× |
| 64K | **6.0×** | 2.1× | 1.5× | 1.0× | 0.9× |
| 256K | **5.7×** | 2.1× | 1.7× | 1.0× | 0.9× |
| 1M | **5.6×** | 2.1× | 2.0× | 1.0× | 1.0× |

### Ours 相对各策略的能耗节省

| seq | vs H100 | vs H100 TP2 | vs Rubin | vs Rubin TP2 |
|-----|---------:|---------:|---------:|---------:|
| 4K | **-84%** | **-86%** | **-66%** | **-78%** |
| 64K | **-83%** | **-85%** | **-65%** | **-75%** |
| 256K | **-83%** | **-84%** | **-63%** | **-70%** |
| 1M | **-82%** | **-82%** | **-62%** | **-65%** |

### 瞬时功耗 @ seq=64K (W)

| 指标 | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|---------:|---------:|---------:|---------:|---------:|
| 总功耗 |      1254 |      1568 |      2527 |       504 |       811 |
| HBM功耗 |       896 |      1178 |      1748 |       380 |       563 |
| 计算功耗 |       104 |         0 |         0 |         0 |         0 |
| TDP |      1440 |      2200 |      4400 |       700 |      1400 |
| TDP 利用率 |     87.1% |     71.2% |     57.4% |     71.9% |     57.9% |
| 延时 (ns) |     4,212 |     9,615 |     8,281 |    63,141 |    44,354 |

## Llama4-Maverick

### 单 Token 能耗 (nJ)

| seq | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|---------:|---------:|---------:|---------:|---------:|
| 4K | 2,755,798 | 8,416,844 | 13,224,851 | 17,742,275 | 21,560,665 |
| 64K | 6,932,101 | 21,591,713 | 28,040,164 | 45,545,003 | 51,183,509 |
| 256K | 20,591,046 | 62,114,628 | 71,268,333 | 131,093,921 | 140,426,600 |
| 1M | 75,238,047 | 223,953,011 | 233,178,751 | 472,760,263 | 479,655,409 |

### 能效提升倍数 (Token/J, baseline = H100)

| seq | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|---------:|---------:|---------:|---------:|---------:|
| 4K | **6.4×** | 2.1× | 1.3× | 1.0× | 0.8× |
| 64K | **6.6×** | 2.1× | 1.6× | 1.0× | 0.9× |
| 256K | **6.4×** | 2.1× | 1.8× | 1.0× | 0.9× |
| 1M | **6.3×** | 2.1× | 2.0× | 1.0× | 1.0× |

### Ours 相对各策略的能耗节省

| seq | vs H100 | vs H100 TP2 | vs Rubin | vs Rubin TP2 |
|-----|---------:|---------:|---------:|---------:|
| 4K | **-84%** | **-87%** | **-67%** | **-79%** |
| 64K | **-85%** | **-86%** | **-68%** | **-75%** |
| 256K | **-84%** | **-85%** | **-67%** | **-71%** |
| 1M | **-84%** | **-84%** | **-66%** | **-68%** |

### 瞬时功耗 @ seq=64K (W)

| 指标 | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|---------:|---------:|---------:|---------:|---------:|
| 总功耗 |      1304 |      1540 |      2459 |       495 |       789 |
| HBM功耗 |       997 |      1150 |      1680 |       371 |       541 |
| 计算功耗 |        52 |         0 |         0 |         0 |         0 |
| TDP |      1440 |      2200 |      4400 |       700 |      1400 |
| TDP 利用率 |     90.5% |     70.0% |     55.9% |     70.6% |     56.4% |
| 延时 (ns) |     5,318 |    14,024 |    11,404 |    92,096 |    64,860 |

## 核心发现 — Ours vs 竞争方案

> 以下分析基于 Fig 2b (Energy Efficiency Improvement, baseline = H100)。
> 功耗系数已更新: cmpt_coeff 394→260, mem_coeff 1120→1021 (降低 Ours 动态功耗, 能效优势增大)。

### 1. Ours vs H100: 5.6–6.6× 能效提升

| 模型 | seq=4K | seq=64K | seq=256K | seq=1M |
|------|-------:|-------:|-------:|-------:|
| Qwen3-235B | **6.1×** | **6.0×** | **5.7×** | **5.6×** |
| Llama4-Maverick | **6.4×** | **6.6×** | **6.4×** | **6.3×** |

- 对应节能 **82–85%** (每 token 能耗仅为 H100 的 15–18%)。
- **根因**: Ours 的 16 个 HBM4 NPU 提供 40 TB/s 聚合带宽 (H100 仅 3.35 TB/s), 同样的 memory-bound 操作完成速度快 **12–16×**, 静态能耗大幅摊薄。
- 功耗系数更新后 (cmpt_coeff 260, mem_coeff 1021), Ours 的动态功耗进一步降低, 优势从此前的 4.8–5.9× 扩大至 5.6–6.6×。
- Llama4 在长 seq 下优势更突出 (6.3–6.6×), 因为 d_model=5120 带来更重的权重加载, HBM 利用率在 64K 时即达 97.6%, Ours 的高 BW 优势被充分发挥。

### 2. Ours vs Rubin: 稳定 2.6–3.1× 能效优势

| 模型 | seq=4K | seq=64K | seq=256K | seq=1M |
|------|-------:|-------:|-------:|-------:|
| Qwen3-235B | **2.9×** | **2.9×** | **2.7×** | **2.6×** |
| Llama4-Maverick | **3.1×** | **3.1×** | **3.0×** | **3.0×** |

- Rubin 相对 H100 始终为 **2.1×** (各 seq 长度下几乎不变), 但 Ours 在此基础上再拉开 **2.6–3.1×** 差距。
- 对应节能 **62–68%** (Ours 每 token 能耗仅为 Rubin 的 32–38%)。
- **延时维度**: Ours 比 Rubin 快 **1.8–2.5×** (近端存算一体减少了数据搬运开销)。
- Rubin 的 2.1× 能效提升主要来自更高的 HBM 带宽 (22 TB/s vs 3.35 TB/s), 但其单 die 架构的带宽天花板限制了进一步提升; Ours 的 16 NPU 分布式架构突破了这一瓶颈。
- Llama4 下优势略大 (3.0–3.1× vs Qwen3 的 2.6–2.9×), 因其 attention head 更多, memory-bound 程度更高。

### 3. Ours vs Rubin TP2: 2.8–4.8× 能效优势, 短 seq 差距显著

| 模型 | seq=4K | seq=64K | seq=256K | seq=1M |
|------|-------:|-------:|-------:|-------:|
| Qwen3-235B | **4.6×** | **4.0×** | **3.3×** | **2.8×** |
| Llama4-Maverick | **4.8×** | **4.0×** | **3.5×** | **3.1×** |

- Rubin TP2 相对 H100 仅有 **1.3× (4K) → 2.0× (1M)** 的能效提升, 短 seq 下甚至不如 Rubin TP1 (1.3× vs 2.1×), 因为双芯片的静态功耗翻倍、通信开销加重。
- **核心洞察**: TP2 的 NVLink 通信 + 双倍静态功耗在短 seq 下是净负担; 只有在 seq>=256K 时, 模型分半带来的延时收益才开始弥补这些开销。Ours 的单系统集成避免了这一困境。
- Ours 在 seq=4K 下比 Rubin TP2 高效 **4.6–4.8×**, 在 seq=1M 下仍保持 **2.8–3.1×** 优势。

### 4. Ours vs H100 TP2: 5.7–7.8× 能效优势

| 模型 | seq=4K | seq=64K | seq=256K | seq=1M |
|------|-------:|-------:|-------:|-------:|
| Qwen3-235B | **7.2×** | **6.8×** | **6.2×** | **5.7×** |
| Llama4-Maverick | **7.8×** | **7.4×** | **6.8×** | **6.4×** |

- H100 TP2 相对 H100 **没有能效提升** (0.8–1.0×): 双卡虽然延时减半, 但功耗翻倍, Token/J 基本持平甚至下降。
- Ours 相对 H100 TP2 的优势甚至超过相对 H100 单卡, 因为 H100 TP2 的能效不升反降。
- **关键结论**: 传统 GPU 堆叠 (TP scaling) 无法改善 memory-bound decode 场景的能效; Ours 的 near-memory 架构才是正解。

### 5. Ours vs NeuPIMs: 1.7–5.1× 能效优势

| 模型 | seq=4K | seq=64K | seq=256K | seq=1M |
|------|-------:|-------:|-------:|-------:|
| Qwen3-235B | **3.7×** | **4.1×** | **4.8×** | **5.1×** |
| Llama4-Maverick | **2.8×** | **2.2×** | **1.9×** | **1.8×** |

- NeuPIMs 功耗 = Ours + 760W (host GPU 固定开销), 瞬时功耗 1366–2292W, 高于 Ours 的 1002–1498W。
- NeuPIMs 延时 = Ours_QKV×2 + Ours_Attn×k + Ours_O×2 + 200ns, 其中 k 取决于 attention head 数量 (Qwen3: 2×16/9, Llama4: 2×5/9)。
- **功耗更高 + 延时更长 → 能耗 (E=P×t) 远大于 Ours**, Token/J 显著低于 Ours。
- **Qwen3 vs Llama4 差异显著**: Qwen3 的 attention 倍数更大 (16/9 vs 5/9), 导致 NeuPIMs 在 Qwen3 上延时惩罚更重, Ours 优势更大 (3.7–5.1× vs 1.8–2.8×)。
- NeuPIMs 相对 H100 的能效仅为 **1.0–1.7×** (Qwen3) / **2.3–3.5×** (Llama4), 在 Qwen3 大 BS 下甚至与 H100 持平 (1.0×)。
- **核心洞察**: 仅在现有 GPU 上附加 PIM 能力 (如 NeuPIMs) 不足以实现高能效 — host GPU 的 760W 固定功耗和串行 bank 访问抵消了大部分近存计算收益。Ours 的全集成 16-NPU 架构无需 host GPU, 从根本上消除了这一开销。

### 6. 跨模型一致性

- Qwen3-235B 和 Llama4-Maverick 呈现高度一致的能效排序和趋势, 证明 Ours 的优势不依赖于特定模型架构。
- Llama4 在中长 seq 下优势更大 (6.3–6.6× vs Qwen3 的 5.6–6.1×), 因为 d_model=5120 (vs 4096) 带来更重的权重加载, Ours 的高 BW 优势更突出; 同时 Llama4 的 HBM 利用率在 64K 时即达 97.6%, 远高于 Qwen3 的 87.7%。
- 两个模型的 Rubin 能效倍数几乎相同 (均为 2.1×), 说明 GPU 架构的能效上限在模型间是稳定的。

### 7. 瞬时功耗与能效的解耦

| 指标 (Qwen3 @ 64K) | Ours | Rubin | Rubin TP2 | H100 | H100 TP2 |
|-----|------:|------:|------:|------:|------:|
| 瞬时总功耗 (W) | 1254 | 1568 | 2527 | 504 | 811 |
| 延时 (ns) | 4212 | 9615 | 8281 | 63,141 | 44,354 |
| TDP 利用率 | 87.1% | 71.2% | 57.4% | 71.9% | 57.9% |

- **Ours 瞬时功耗 (1254W) 是 H100 (504W) 的 2.5×**, 但延时仅为 1/15, 最终 Energy = Power x Time 反而低 **83%**。
- **Rubin TP2 功耗最高** (2527W, 双芯片) 但 TDP 利用率仅 57% — 大量算力空闲。
- **Ours 的 TDP 利用率为 87.1%** (Qwen3) / 90.5% (Llama4), 表明硬件资源被充分利用, 而非简单堆叠后空转。
- 长 seq 下 Ours 功耗从 ~966W (1K) 升至 ~1498W (1M), HBM 利用率从 68% 升至 98%, 接近满载 — 这是 near-memory computing 的理想工作模式: **更高利用率 → 更高能效**。

### 8. 功耗系数更新的影响

| 对比 | 旧系数 (394/1120) | 新系数 (260/1021) | 变化 |
|------|-----|-----|------|
| Ours vs H100 能效 | 4.8–5.9× | 5.6–6.6× | +0.7–0.8× |
| Ours vs Rubin 能效 | 2.3–2.8× | 2.6–3.1× | +0.3× |
| Ours vs Rubin TP2 能效 | 2.4–4.4× | 2.8–4.8× | +0.4× |
| Ours 瞬时功耗 @ 64K | ~1318W | ~1254W | -64W (-5%) |
| Ours TDP 利用率 @ 64K | 91.5% | 87.1% | -4.4pp |

- cmpt_coeff 从 394 降至 260 (-34%), mem_coeff 从 1120 降至 1021 (-9%), 使得 Ours 在相同利用率下的动态功耗更低。
- 能效优势全面提升: vs H100 从 ~5× 提升至 ~6×, vs Rubin 从 ~2.5× 提升至 ~3×。
- 瞬时功耗略有下降 (约 5%), TDP 利用率从 ~91% 降至 ~87%, 但由于能耗 (Power x Time) 同步降低, 能效反而更优。

---

## 数据来源

| 数据 | 来源文件 | 生成脚本 |
|------|---------|---------|
| Ours (RO_new) 延时 | `reports/{model}/hybrid/gqa_hybrid_merged_96T_*_util96.json` | `merge_gqa_results.py` |
| Ours 通信延时 | AstraSim 仿真 + Roofline 解析模型 | `collect_gqa_data.py` + `roofline_gqa_calc.py` |
| Ours 计算延时 | Roofline 模型 (96T peak, 2.5 TB/s BW) + NPU 利用率修正 | `roofline_gqa_calc.py` + `utilization_96.json` |
| Rubin 延时 | Roofline (17500T, 22TB/s) + H100 BW 利用率修正 | `roofline_gqa_calc.py` + `h100_rubin_utilization.json` |
| H100 延时 | Roofline (1979T, 3.35TB/s) + H100 BW 利用率修正 | `roofline_gqa_calc.py` + `h100_rubin_utilization.json` |
| TP2 策略 | TP1 roofline / 2 + TP2 BW util + NVLink AR | `add_rubin_tp2.py` + `h100_tp2_profile.json` |
| 功耗模型 | `P = static_ratio x TDP + dynamic (HBM util + compute util)` | `power_model.py` |
| H100 BW 利用率 | H100 SXM e2e CUDA event profiling (TP=1, TP=2) | `H100_results/attn_scaling_tp1_vs_tp2_report.txt` |
| NPU 利用率 | 分析模型 (compute utilization at 96 TFLOPS) | `utilization_96.json` |

### 功耗模型公式

```
P_total = P_static + P_HBM + P_compute + P_D2D

P_static = static_ratio x TDP                    (TDP 的固定比例)
P_HBM    = mem_coeff x U_mem                      (HBM 带宽利用率驱动)
P_compute = cmpt_coeff x U_cmpt                   (计算利用率驱动)
P_D2D    = D2D_bits x pJ_per_bit / time           (D2D 链路能耗)

Energy_per_token = P_total x wall_time
```

| 架构 | TDP | static_ratio | cmpt_coeff | mem_coeff | 公式 |
|------|-----|-------------|-----------|----------|------|
| Ours (16 NPU) | 1440W | 17.7% | **260** | **1021** | P = 254.9 + 260×U_cmpt + 1021×U_mem + D2D |
| NeuPIMs | — | — | — | — | P_NeuPIMs = P_Ours + 760W (host GPU 固定开销) |
| Rubin | 2200W | 17.7% | 2631 | 1800 | P = 389.4 + 2631×U_cmpt + 1800×U_mem |
| H100 | 700W | 17.7% | 829 | 580 | P = 123.9 + 829×U_cmpt + 580×U_mem |

## 可视化

| 图表 | 路径 |
|------|------|
| 能耗比柱状图 | `.../paper_figures/fig2_energy_power/plots/fig2_energy_4panel.{pdf,png}` |
| 能效柱状图 (Token/J) | `.../paper_figures/fig2_energy_power/plots/fig2b_power_4panel.{pdf,png}` |
| 瞬时功耗柱状图 (W) | `.../paper_figures/fig2_energy_power/plots/fig2c_power_W_4panel.{pdf,png}` |
| 能耗绘图脚本 | `.../paper_figures/fig2_energy_power/scripts/plot_energy.py` |
| 能效绘图脚本 | `.../paper_figures/fig2_energy_power/scripts/plot_power.py` |
| 功耗绘图脚本 | `.../paper_figures/fig2_energy_power/scripts/plot_token_per_watt.py` |

## Figure Caption

Energy efficiency gain (Token/J) normalized to H100 across Qwen3-235B and Llama4-Maverick at batch sizes 1, 4, 16, 32 with sequence lengths from 4K to 1M.
