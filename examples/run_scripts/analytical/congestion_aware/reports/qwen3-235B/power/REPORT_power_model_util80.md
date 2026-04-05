# Qwen3-235B GQA 单层功耗报告 (utilization_80T, bw1500)

> bs=1 decode, FP8, Mesh2D 4×4 block2x2, D2D 1.5TB/s, hop=15ns

## 功率模型

三套公式，每套给出瞬时功率 P (W) = 静态 + 计算动态 + 访存动态：

| 架构 | 公式 | TDP |
|------|------|----:|
| **Ours** (16 cubes) | P = 17.7%×1440 + 394×U_cmpt + 1120×U_mem + D2D | 1440W |
| **Rubin** (1 chip) | P = 17.7%×2200 + 2631×U_cmpt + 1800×U_mem | 2200W |
| **Rubin_TP2** (2× Rubin) | 2 × P_Rubin (per-chip U_mem 来自 H100 TP2 profiling) | 4400W |

利用率来源：
- Ours U_cmpt: NCU profiling (`utilization_80.json`)，per-module 时间加权
- Ours U_mem: `hbm_bytes / (gpu_time × 2.5TB/s)`
- Rubin/TP2 U_mem: H100 BW profiling (`h100_rubin_utilization.json` / `h100_tp2_profile.json`)
- Rubin/TP2 U_cmpt = 0 (bs=1 decode 纯访存受限)

## 功率分解 @ seq=64K

| 策略 | 静态 (W) | 访存 (W) | 计算 (W) | D2D (W) | **总功率 (W)** | TDP占比 |
|------|-------:|-------:|-------:|------:|----------:|------:|
| HMP_RO | 255 | 897 | 105 | 0 | **1,256** | 87% |
| RO_new | 255 | 910 | 169 | 0 | **1,333** | 93% |
| HMP | 255 | 910 | 169 | 0 | **1,333** | 93% |
| TP16 | 255 | 909 | 169 | 14 | **1,347** | 94% |
| Rubin | 389 | 1,178 | 0 | 0 | **1,567** | 71% |
| Rubin_TP2 | 779 | 1,748 | 0 | 0 | **2,527** | 57% |

## 能耗/token @ seq=64K

| 策略 | 功率 (W) | Wall (ns) | Energy (nJ) | vs RO_new |
|------|-------:|--------:|-----------:|----------:|
| **RO_new** | 1,333 | 4,529 | **6.04M** | 1.00× |
| HMP | 1,333 | 4,733 | 6.31M | 1.05× |
| TP16 | 1,347 | 5,595 | 7.53M | 1.25× |
| HMP_RO | 1,256 | 7,659 | 9.62M | 1.59× |
| Rubin | 1,567 | 9,615 | 15.07M | 2.49× |
| Rubin_TP2 | 2,527 | 8,281 | 20.93M | 3.46× |

## 全 seq 能耗趋势 (nJ/token)

| Seq | RO_new | HMP | TP16 | HMP_RO | Rubin | Rubin_TP2 |
|-----|-------:|----:|-----:|-------:|------:|----------:|
| 1K | 3.02M | 3.24M | 3.42M | 6.64M | 8.18M | 13.11M |
| 4K | 3.29M | 3.50M | 3.71M | 6.93M | 8.48M | 13.50M |
| 16K | 3.81M | 4.04M | 4.43M | 7.43M | 9.87M | 15.08M |
| 64K | 6.04M | 6.31M | 7.53M | 9.62M | 15.07M | 20.93M |
| 256K | 14.95M | 15.28M | 22.25M | 18.51M | 35.33M | 42.89M |
| 1M | 50.64M | 50.99M | 80.76M | 54.20M | 116.25M | 124.55M |

## 利用率 @ seq=64K

| 策略 | U_cmpt | U_mem | 来源 |
|------|-------:|------:|------|
| HMP_RO | 26.5% | 80.0% | NCU profiling + hbm_bytes/gpu_time |
| RO_new/HMP | 42.8% | 81.2% | 同上 |
| TP16 | 42.8% | 81.2% | 同上 (utilization_128_2.json for QKV) |
| Rubin | 0% | 65.4% | H100 BW profiling (TP1) |
| Rubin_TP2 | 0% | 48.6% | H100 BW profiling (TP2) |

## 关键发现

1. **RO_new 能效最优** — 功率适中 (1,333W) + wall time 最短 (4,529ns) → 能耗 6.04M nJ，比 Rubin 低 2.5×
2. **Rubin_TP2 功率最高** (2,527W) — 2× Rubin 芯片的静态功耗翻倍 (779W)，访存功率 1.48× (1,748W)
3. **Rubin_TP2 能耗最差** (20.93M nJ) — 虽然 GPU 计算快 (6,479ns)，但通信 (1,802ns) + 高功率 → 总能耗是 RO_new 的 3.5×
4. **Ours 的计算功率贡献有限** — HMP/RO_new 的 U_cmpt=42.8% → 169W，仅占总功率 13%；主要功耗来自访存 (910W, 68%)
5. **HMP_RO 的 Proj_O 拖累** — duplicate Wo 导致 wall time 7,659ns (RO_new 的 1.7×)，功率虽略低 (1,256W) 但能耗高 60%
6. **长 seq 下 TP16 劣势加大** — 256K 时 TP16 能耗 22.25M (RO_new 的 1.49×)，因 AllReduce 通信随 seq 不变但占比增大
7. **Rubin 的访存功率系数远高于 Ours** — 1800W vs 1120W (1.6×)，且 U_mem 也更低 (65% vs 81%)，反映 Rubin 芯片访存功耗特性
