# H100 Attention Decode Profiling: TP=1 vs TP=2

**Model**: Qwen3-235B GQA | B=1, seq_q=1 (decode) | fp16
**Method**: Single-GPU ncu profiling with TP-sharded dimensions (no communication)
**Stages**: qkv_proj -> rope -> FlashAttention-2 -> o_proj

## Dimensions

| | TP=1 | TP=2 |
|---|---|---|
| Q heads | 64 | 32 |
| KV heads | 4 | 2 |
| W_qkv | 4096 x 9216 = 76 MB | 4096 x 4608 = 38 MB |
| W_o | 8192 x 4096 = 67 MB | 4096 x 4096 = 34 MB |
| KV cache per seq | 2 x S x 4 x 128 x 2B | 2 x S x 2 x 128 x 2B |

## Per-Stage Latency (us)

| Seq | qkv (TP1) | qkv (TP2) | rope (TP1) | rope (TP2) | attn (TP1) | attn (TP2) | o_proj (TP1) | o_proj (TP2) | total (TP1) | total (TP2) | speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 33.8 | 23.4 | 52.2 | 49.6 | 24.9 | 24.5 | 29.2 | 18.9 | 152.5 | 124.6 | 1.22x |
| 4K | 34.4 | 22.9 | 50.9 | 49.5 | 35.8 | 35.1 | 29.3 | 18.5 | 162.8 | 134.2 | 1.21x |
| 16K | 34.7 | 23.2 | 51.7 | 50.3 | 62.9 | 56.9 | 29.9 | 19.6 | 191.8 | 158.2 | 1.21x |
| 64K | 33.4 | 23.6 | 52.8 | 49.0 | 95.8 | 95.3 | 30.1 | 18.5 | 224.6 | 194.5 | 1.15x |
| 128K | 34.5 | 22.9 | 53.0 | 49.2 | 137.9 | 117.8 | 29.1 | 18.8 | 266.8 | 217.3 | 1.23x |
| 256K | 34.7 | 23.4 | 52.2 | 50.0 | 222.7 | 171.9 | 29.0 | 19.1 | 351.0 | 272.7 | 1.29x |
| 512K | 34.1 | 23.1 | 50.8 | 48.7 | 393.4 | 244.2 | 29.5 | 19.1 | 520.3 | 343.3 | 1.52x |
| 1M | 34.8 | 22.9 | 51.5 | 49.5 | 729.4 | 415.7 | 30.0 | 18.8 | 858.7 | 514.8 | 1.67x |

## HBM Bandwidth Utilization (%)

| Seq | qkv (TP1) | qkv (TP2) | attn (TP1) | attn (TP2) | o_proj (TP1) | o_proj (TP2) |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 69.1 | 51.8 | 4.2 | 2.3 | 71.5 | 56.2 |
| 4K | 68.3 | 52.5 | 15.2 | 8.3 | 71.3 | 57.1 |
| 16K | 67.5 | 51.9 | 35.2 | 20.3 | 69.9 | 54.1 |
| 64K | 69.9 | 51.0 | 61.6 | 44.0 | 69.3 | 57.3 |
| 128K | 68.1 | 52.4 | 74.4 | 58.0 | 71.8 | 56.2 |
| 256K | 68.3 | 51.6 | 83.1 | 65.2 | 72.0 | 56.2 |
| 512K | 68.4 | 51.9 | 88.4 | 83.0 | 70.7 | 55.8 |
| 1M | 67.3 | 52.7 | 91.6 | 87.9 | 69.5 | 56.3 |

## Effective Bandwidth (GB/s)

| Seq | qkv (TP1) | qkv (TP2) | attn (TP1) | attn (TP2) | o_proj (TP1) | o_proj (TP2) |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2310 | 1730 | 142 | 76 | 2390 | 1880 |
| 4K | 2290 | 1760 | 508 | 278 | 2390 | 1910 |
| 16K | 2260 | 1740 | 1180 | 678 | 2340 | 1810 |
| 64K | 2340 | 1710 | 2060 | 1470 | 2320 | 1920 |
| 128K | 2280 | 1760 | 2490 | 1940 | 2400 | 1880 |
| 256K | 2290 | 1720 | 2790 | 2190 | 2410 | 1880 |
| 512K | 2290 | 1740 | 2960 | 2780 | 2360 | 1870 |
| 1M | 2250 | 1760 | 3070 | 2950 | 2330 | 1890 |

## FlashAttention-2 Speedup Detail

| Seq | KV cache TP1 | KV cache TP2 | FA dur TP1 | FA dur TP2 | speedup | BW TP1 | BW TP2 | HBM% TP1 | HBM% TP2 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 2 MB | 1 MB | 24.9 us | 24.5 us | 1.02x | 142 GB/s | 76 GB/s | 4.2% | 2.3% |
| 4K | 8 MB | 4 MB | 35.8 us | 35.1 us | 1.02x | 508 GB/s | 278 GB/s | 15.2% | 8.3% |
| 16K | 34 MB | 17 MB | 62.9 us | 56.9 us | 1.11x | 1180 GB/s | 678 GB/s | 35.2% | 20.3% |
| 64K | 134 MB | 67 MB | 95.8 us | 95.3 us | 1.01x | 2060 GB/s | 1470 GB/s | 61.6% | 44.0% |
| 128K | 268 MB | 134 MB | 137.9 us | 117.8 us | 1.17x | 2490 GB/s | 1940 GB/s | 74.4% | 58.0% |
| 256K | 537 MB | 268 MB | 222.7 us | 171.9 us | 1.30x | 2790 GB/s | 2190 GB/s | 83.1% | 65.2% |
| 512K | 1074 MB | 537 MB | 393.4 us | 244.2 us | 1.61x | 2960 GB/s | 2780 GB/s | 88.4% | 83.0% |
| 1M | 2147 MB | 1074 MB | 729.4 us | 415.7 us | 1.75x | 3070 GB/s | 2950 GB/s | 91.6% | 87.9% |

---

## Core Findings

### 1. TP=2 GEMM: weight halved, but BW utilization drops 15-17pp

qkv_proj and o_proj weights are halved by TP=2, so latency drops ~1.5x (34->23us, 29->19us). However, the smaller matrices cause cuBLAS to select a less efficient kernel (sm80 tile 32x32 instead of sm90), dropping HBM bandwidth utilization from ~69% to ~52%. The effective bandwidth falls from ~2300 GB/s to ~1740 GB/s. This means **TP=2 GEMM gets a 1.5x speedup instead of the theoretical 2x** — the smaller matrix cannot saturate H100's memory subsystem as efficiently.

### 2. FlashAttention: KV cache halved, speedup scales with seq length

FA2 latency is dominated by KV cache reads. With TP=2 halving KV heads (4->2), the KV cache per rank halves. The speedup depends on how much time is spent actually reading KV vs kernel overhead:

- **Short seq (1K-16K)**: ~1.0-1.1x — kernel launch overhead dominates, halving data barely helps
- **Medium seq (64K-256K)**: 1.0-1.3x — transitioning to memory-bound, speedup growing
- **Long seq (512K-1M)**: **1.6-1.75x** — nearly memory-bound, approaching theoretical 2x

At 1M seq, TP=1 FA2 achieves 92% HBM utilization (3070 GB/s), TP=2 achieves 88% (2950 GB/s) — both near peak, confirming pure BW-bound behavior.

### 3. rope is the constant overhead bottleneck

rope takes ~50us regardless of TP or seq length (14 tiny elementwise kernels, dominated by launch overhead). It accounts for 34% of total time at 1K seq but only 10% at 1M. This is a fixed cost that limits TP scaling at short sequences.

### 4. Total compute speedup: 1.2x (short seq) to 1.67x (1M seq)

The gap from theoretical 2x comes from three sources:
- **rope fixed overhead** (~50us, not reduced by TP)
- **GEMM BW efficiency loss** (sm80 kernel at TP=2, -15pp HBM%)
- **FA kernel overhead** (short seq: launch cost dominates over data reduction)

At very long sequences where FA dominates (>80% of total time), TP=2 compute speedup approaches the FA-only speedup of 1.75x. Communication cost (all-reduce after o_proj) would further reduce the net benefit in a real TP=2 deployment.
