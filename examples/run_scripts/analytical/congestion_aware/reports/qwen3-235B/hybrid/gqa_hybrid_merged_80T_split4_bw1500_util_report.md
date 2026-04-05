# Qwen3-235B GQA Single-Layer Hybrid Estimation Report

> **Roofline**: `reports/qwen3-235B/roofline/gqa_roofline_bs1_80T_bw1500.json`
> **AstraSim**: `reports/qwen3-235B/astrasim/gqa_seq_scaling_bs1_80t_split4_bw1500_data.json`
> **NPU utilization**: `utilization.json` | **Rubin utilization**: `h100_rubin_utilization.json`

## Hardware Configuration

| Parameter | Ours (16 HBM4 NPUs) | Rubin |
|-----------|---------------------|-------|
| Peak Compute | 80.0 TFLOPS/NPU (1280 total) | 17,500 TFLOPS |
| HBM BW | 2.5 TB/s/NPU (40 total) | 22 TB/s |
| D2D Link BW | 1.5 TB/s | N/A (single chip) |
| Topology | Mesh2D 4x4, block2x2 rank remap | Single GPU |

## Utilization Model

**Ours (NPU):** Compute utilization profiling. `adjusted_compute = compute_ns / util`, then `max(adjusted_compute, memory_ns)`. At bs=1: proj_qkv util=2.8%, proj_o util=4.7%.

**Rubin:** H100 e2e HBM BW utilization from real GPU profiling. `adjusted_total = roofline_ns / bw_util`. proj_qkv BW util=68.7%, proj_o BW util=70.5%, attn BW util=4.2%-91.6% by seq.

## Wall Time Comparison (ns)

| Seq | HMP_RO | RO_new | HMP | TP16 | Rubin |
|-----|--------:|--------:|--------:|--------:|--------:|
| 1K | 7,203 | 3,943 | 4,147 | 4,067 | 5,797 |
| 2K | 7,334 | 4,074 | 4,278 | 4,078 | 5,645 |
| 4K | 7,596 | 4,336 | 4,540 | 4,098 | 5,917 |
| 8K | 7,596 | 4,336 | 4,540 | 4,222 | 6,176 |
| 16K | 7,596 | 4,336 | 4,540 | 4,592 | 6,829 |
| 32K | 7,889 | 4,629 | 4,833 | 5,306 | 7,814 |
| 64K | 8,999 | 5,739 | 5,943 | 6,352 | 9,615 |
| 128K | 10,926 | 7,666 | 7,870 | 9,365 | 12,863 |
| 256K | 13,942 | 10,682 | 10,886 | 15,052 | 19,346 |
| 512K | 21,104 | 17,844 | 18,048 | 26,380 | 32,268 |
| 1M | 35,430 | 32,170 | 32,374 | 48,875 | 57,945 |

## Detailed Module Breakdown (ns)

| Seq | Strategy | Proj_QKV | Attn | Proj_O | GPU | Comm | Wall | QKV Src | Attn Src | O Src |
|-----|----------|--------:|-----:|------:|----:|-----:|-----:|---------|----------|-------|
| 1K | HMP_RO | 2,113 | 455 | 4,448 | 7,016 | 187 | 7,203 | roofline+util | roofline+util | roofline+util |
| 1K | RO_new | 2,113 | 455 | 1,112 | 3,680 | 263 | 3,943 | roofline+util | roofline+util | roofline+util |
| 1K | HMP | 2,113 | 455 | 1,112 | 3,680 | 467 | 4,147 | roofline+util | roofline+util | roofline+util |
| 1K | TP16 | 2,113 | 212 | 1,112 | 3,437 | 630 | 4,067 | roofline+util | roofline+util | roofline+util |
| 1K | Rubin | 2,498 | 1,135 | 2,164 | 5,797 | 0 | 5,797 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 2K | HMP_RO | 2,113 | 586 | 4,448 | 7,147 | 187 | 7,334 | roofline+util | roofline+util | roofline+util |
| 2K | RO_new | 2,113 | 586 | 1,112 | 3,811 | 263 | 4,074 | roofline+util | roofline+util | roofline+util |
| 2K | HMP | 2,113 | 586 | 1,112 | 3,811 | 467 | 4,278 | roofline+util | roofline+util | roofline+util |
| 2K | TP16 | 2,113 | 212 | 1,112 | 3,437 | 641 | 4,078 | roofline+util | roofline+util | roofline+util |
| 2K | Rubin | 2,498 | 983 | 2,164 | 5,645 | 0 | 5,645 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 4K | HMP_RO | 2,113 | 848 | 4,448 | 7,409 | 187 | 7,596 | roofline+util | roofline+util | roofline+util |
| 4K | RO_new | 2,113 | 848 | 1,112 | 4,073 | 263 | 4,336 | roofline+util | roofline+util | roofline+util |
| 4K | HMP | 2,113 | 848 | 1,112 | 4,073 | 467 | 4,540 | roofline+util | roofline+util | roofline+util |
| 4K | TP16 | 2,113 | 212 | 1,112 | 3,437 | 661 | 4,098 | roofline+util | roofline+util | roofline+util |
| 4K | Rubin | 2,498 | 1,254 | 2,164 | 5,917 | 0 | 5,917 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 8K | HMP_RO | 2,113 | 848 | 4,448 | 7,409 | 187 | 7,596 | roofline+util | roofline+util | roofline+util |
| 8K | RO_new | 2,113 | 848 | 1,112 | 4,073 | 263 | 4,336 | roofline+util | roofline+util | roofline+util |
| 8K | HMP | 2,113 | 848 | 1,112 | 4,073 | 467 | 4,540 | roofline+util | roofline+util | roofline+util |
| 8K | TP16 | 2,113 | 285 | 1,112 | 3,510 | 712 | 4,222 | roofline+util | roofline+util | roofline+util |
| 8K | Rubin | 2,498 | 1,513 | 2,164 | 6,176 | 0 | 6,176 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 16K | HMP_RO | 2,113 | 848 | 4,448 | 7,409 | 187 | 7,596 | roofline+util | roofline+util | roofline+util |
| 16K | RO_new | 2,113 | 848 | 1,112 | 4,073 | 263 | 4,336 | roofline+util | roofline+util | roofline+util |
| 16K | HMP | 2,113 | 848 | 1,112 | 4,073 | 467 | 4,540 | roofline+util | roofline+util | roofline+util |
| 16K | TP16 | 2,113 | 563 | 1,112 | 3,788 | 804 | 4,592 | roofline+util | roofline+util | roofline+util |
| 16K | Rubin | 2,498 | 2,166 | 2,164 | 6,829 | 0 | 6,829 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 32K | HMP_RO | 2,113 | 1,141 | 4,448 | 7,702 | 187 | 7,889 | roofline+util | roofline+util | roofline+util |
| 32K | RO_new | 2,113 | 1,141 | 1,112 | 4,366 | 263 | 4,629 | roofline+util | roofline+util | roofline+util |
| 32K | HMP | 2,113 | 1,141 | 1,112 | 4,366 | 467 | 4,833 | roofline+util | roofline+util | roofline+util |
| 32K | TP16 | 2,113 | 1,044 | 1,112 | 4,270 | 1,037 | 5,306 | roofline+util | roofline+util | roofline+util |
| 32K | Rubin | 2,498 | 3,151 | 2,164 | 7,814 | 0 | 7,814 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 64K | HMP_RO | 2,113 | 2,251 | 4,448 | 8,812 | 187 | 8,999 | roofline+util | roofline+util | roofline+util |
| 64K | RO_new | 2,113 | 2,251 | 1,112 | 5,476 | 263 | 5,739 | roofline+util | roofline+util | roofline+util |
| 64K | HMP | 2,113 | 2,251 | 1,112 | 5,476 | 467 | 5,943 | roofline+util | roofline+util | roofline+util |
| 64K | TP16 | 2,113 | 1,798 | 1,112 | 5,023 | 1,329 | 6,352 | roofline+util | roofline+util | roofline+util |
| 64K | Rubin | 2,498 | 4,952 | 2,164 | 9,615 | 0 | 9,615 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 128K | HMP_RO | 2,113 | 4,178 | 4,448 | 10,739 | 187 | 10,926 | roofline+util | roofline+util | roofline+util |
| 128K | RO_new | 2,113 | 4,178 | 1,112 | 7,403 | 263 | 7,666 | roofline+util | roofline+util | roofline+util |
| 128K | HMP | 2,113 | 4,178 | 1,112 | 7,403 | 467 | 7,870 | roofline+util | roofline+util | roofline+util |
| 128K | TP16 | 2,113 | 3,589 | 1,112 | 6,814 | 2,551 | 9,365 | roofline+util | roofline+util | roofline+util |
| 128K | Rubin | 2,498 | 8,200 | 2,164 | 12,863 | 0 | 12,863 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 256K | HMP_RO | 2,113 | 7,193 | 4,448 | 13,755 | 187 | 13,942 | roofline+util | roofline+util | roofline+util |
| 256K | RO_new | 2,113 | 7,193 | 1,112 | 10,418 | 263 | 10,682 | roofline+util | roofline+util | roofline+util |
| 256K | HMP | 2,113 | 7,193 | 1,112 | 10,418 | 467 | 10,886 | roofline+util | roofline+util | roofline+util |
| 256K | TP16 | 2,113 | 7,170 | 1,112 | 10,395 | 4,657 | 15,052 | roofline+util | roofline+util | roofline+util |
| 256K | Rubin | 2,498 | 14,683 | 2,164 | 19,346 | 0 | 19,346 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 512K | HMP_RO | 2,113 | 14,356 | 4,448 | 20,917 | 187 | 21,104 | roofline+util | roofline+util | roofline+util |
| 512K | RO_new | 2,113 | 14,356 | 1,112 | 17,581 | 263 | 17,844 | roofline+util | roofline+util | roofline+util |
| 512K | HMP | 2,113 | 14,356 | 1,112 | 17,581 | 467 | 18,048 | roofline+util | roofline+util | roofline+util |
| 512K | TP16 | 2,113 | 14,260 | 1,112 | 17,485 | 8,895 | 26,380 | roofline+util | roofline+util | roofline+util |
| 512K | Rubin | 2,498 | 27,605 | 2,164 | 32,268 | 0 | 32,268 | roofline+H100_util | roofline+H100_util | roofline+H100_util |
| 1M | HMP_RO | 2,113 | 28,681 | 4,448 | 35,243 | 187 | 35,430 | roofline+util | roofline+util | roofline+util |
| 1M | RO_new | 2,113 | 28,681 | 1,112 | 31,906 | 263 | 32,170 | roofline+util | roofline+util | roofline+util |
| 1M | HMP | 2,113 | 28,681 | 1,112 | 31,906 | 467 | 32,374 | roofline+util | roofline+util | roofline+util |
| 1M | TP16 | 2,113 | 28,439 | 1,112 | 31,664 | 17,211 | 48,875 | roofline+util | roofline+util | roofline+util |
| 1M | Rubin | 2,498 | 53,282 | 2,164 | 57,945 | 0 | 57,945 | roofline+H100_util | roofline+H100_util | roofline+H100_util |

## Speedup vs Rubin

| Seq | HMP_RO | RO_new | HMP | TP16 |
|-----|------:|------:|------:|------:|
| 1K | 0.80x | 1.47x | 1.40x | 1.43x |
| 2K | 0.77x | 1.39x | 1.32x | 1.38x |
| 4K | 0.78x | 1.36x | 1.30x | 1.44x |
| 8K | 0.81x | 1.42x | 1.36x | 1.46x |
| 16K | 0.90x | 1.57x | 1.50x | 1.49x |
| 32K | 0.99x | 1.69x | 1.62x | 1.47x |
| 64K | 1.07x | 1.68x | 1.62x | 1.51x |
| 128K | 1.18x | 1.68x | 1.63x | 1.37x |
| 256K | 1.39x | 1.81x | 1.78x | 1.29x |
| 512K | 1.53x | 1.81x | 1.79x | 1.22x |
| 1M | 1.64x | 1.80x | 1.79x | 1.19x |

## Key Findings

### Best Strategy per Seq (Ours only)

- **seq=1K**: Best = RO_new (3,943 ns), Rubin = 5,797 ns, **1.47x speedup**
- **seq=4K**: Best = TP16 (4,098 ns), Rubin = 5,917 ns, **1.44x speedup**
- **seq=16K**: Best = RO_new (4,336 ns), Rubin = 6,829 ns, **1.57x speedup**
- **seq=64K**: Best = RO_new (5,739 ns), Rubin = 9,615 ns, **1.68x speedup**
- **seq=128K**: Best = RO_new (7,666 ns), Rubin = 12,863 ns, **1.68x speedup**
- **seq=256K**: Best = RO_new (10,682 ns), Rubin = 19,346 ns, **1.81x speedup**
- **seq=512K**: Best = RO_new (17,844 ns), Rubin = 32,268 ns, **1.81x speedup**
- **seq=1M**: Best = RO_new (32,170 ns), Rubin = 57,945 ns, **1.80x speedup**

### Crossover Points (Ours vs Rubin)

- **HMP_RO** loses to Rubin starting at seq=1K
- **RO_new** beats Rubin at all seq lengths
- **HMP** beats Rubin at all seq lengths
- **TP16** beats Rubin at all seq lengths

### Utilization Impact

#### NPU (Ours)
- **proj_qkv** (bs=1, util=2.8%): roofline_mem=946 ns -> util_adj=2,113 ns (compute-bound after adjustment, **+123%**)
- **proj_o** (bs=1, util=4.7%): HMP_RO: roofline_mem=3,358 ns -> util_adj=4,448 ns (compute-bound, **+32%**)
- **attn score/attn_v**: memory-bound at all seq for cache_seq (seq/4), util adjustment has limited impact

#### Rubin (H100 BW Util)
- **proj_qkv** (BW util=68.7%): 1,716 ns -> 2,498 ns (+46%)
- **proj_o** (BW util=70.5%): 1,526 ns -> 2,164 ns (+42%)
- **attn** (BW util varies):
  - seq=1K: BW util=4.2%, 48 ns -> 1,135 ns (+2281%)
  - seq=16K: BW util=35.2%, 763 ns -> 2,166 ns (+184%)
  - seq=64K: BW util=61.6%, 3,050 ns -> 4,952 ns (+62%)
  - seq=256K: BW util=83.1%, 12,202 ns -> 14,683 ns (+20%)
  - seq=1M: BW util=91.6%, 48,806 ns -> 53,282 ns (+9%)

### Communication Overhead

**seq=64K:**

| Strategy | GPU (ns) | Comm (ns) | Comm % | Wall (ns) |
|----------|--------:|----------:|-------:|----------:|
| HMP_RO | 8,812 | 187 | 2.1% | 8,999 |
| RO_new | 5,476 | 263 | 4.6% | 5,739 |
| HMP | 5,476 | 467 | 7.9% | 5,943 |
| TP16 | 5,023 | 1,329 | 20.9% | 6,352 |
| Rubin | 9,615 | 0 | 0.0% | 9,615 |

**seq=1M:**

| Strategy | GPU (ns) | Comm (ns) | Comm % | Wall (ns) |
|----------|--------:|----------:|-------:|----------:|
| HMP_RO | 35,243 | 187 | 0.5% | 35,430 |
| RO_new | 31,906 | 263 | 0.8% | 32,170 |
| HMP | 31,906 | 467 | 1.4% | 32,374 |
| TP16 | 31,664 | 17,211 | 35.2% | 48,875 |
| Rubin | 57,945 | 0 | 0.0% | 57,945 |

