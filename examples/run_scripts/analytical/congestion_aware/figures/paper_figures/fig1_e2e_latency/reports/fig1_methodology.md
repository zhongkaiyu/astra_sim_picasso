# Fig 1 计算方法论：单层 Decode 加速比

> 本文档描述 `fig1_speedup_4panel.pdf` 中三个模型（Qwen3-235B、Llama4-Maverick、DeepSeek-V3）各方案延时的计算方法。

---

## 1. 统一 Roofline 框架

所有模型、所有方案（H100 / Rubin / Ours 等）均使用相同的 roofline 流程：

```
每个阶段 (QKV / Attention / ProjO):
    comp_ns = 2 × MACs / peak_perf                    # 计算时间
    mem_ns  = (权重 + 激活) × bytes / HBM_BW           # 访存时间
    stage_ns = max(comp_ns, mem_ns) / utilization      # 取瓶颈，除以利用率
    
wall_ns = QKV_ns + Attn_ns + ProjO_ns + Comm_ns       # 总延时
```

**区别仅在维度、并行切分方式和利用率来源。**

---

## 2. Qwen3-235B / Llama4-Maverick（GQA 模型）

### 2.1 模型参数

| 参数 | Qwen3-235B | Llama4-Maverick |
|------|-----------|-----------------|
| d_model | 4096 | 5120 |
| Hq (Query 头数) | 64 | 40 |
| Hkv (KV 头数) | 4 | 8 |
| d_head | 128 | 128 |
| KV cache/tok | 2 × Hkv × d_head | 2 × Hkv × d_head |

### 2.2 Ours（16-NPU hmp_reo_new 策略）

**并行方式**: tp_h=4（头并行）, tp_hd=4（head-dim 切分）, tp_s=4（序列切分）, 共 16 NPU

**Per-NPU 配置**:

| | Qwen3 | Llama4 |
|---|---|---|
| Hq/NPU | 64/4 = 16 | 40/4 = 10 |
| Hkv/NPU | 4/4 = 1 | 8/4 = 2 |
| d_head (QKV) | 128/4 = 32 | 128/4 = 32 |
| d_head_full (Attn) | 128 | 128 |
| cache_seq | seq/4 | seq/4 |

**各阶段计算**:

#### Proj_QKV
```
权重/NPU: W_Q[d × Hq_npu × dk] + W_K[d × Hkv_npu × dk] + W_V[d × Hkv_npu × dk]
         dk = d_head = 32 (QKV 阶段使用 tp_hd 切分后的 d_head)
MACs/NPU: bs × (Hq_npu + 2×Hkv_npu) × dk × d_model
mem/NPU:  (权重 + 激活) / HBM_BW
QKV_ns = max(comp_ns / sa_util, mem_ns)
```

#### Attention（AllGather 后在 d_head_full=128 上计算）
```
KV cache/NPU: bs × 2 × Hkv_npu × cache_seq × d_head_full
score_ns  = max(score_comp / score_util,  score_mem)
attn_v_ns = max(attn_v_comp / attn_v_util, attn_v_mem)
Attn_ns = score_ns + attn_v_ns
```

#### Proj_O
```
权重/NPU: W_O[d × Hq_npu × dk]
O_ns = max(comp_ns / o_util, mem_ns)
```

#### Communication
```
AstraSim 仿真 mesh2D 拓扑:
  - QKV AllGather (tp_s=4 组内 ring)
  - Attn ReduceScatter (tp_s=4)
  - Final Reduce (16 NPU tree)
```

**SA Utilization**: 从 `qwen3/util_96.json` 或 `llama4/util_96.json` 查表（按 batch 或 effective_seq log2 插值）。

**数据来源**: `reports/{model}/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{N}.json`
- 由 `merge_gqa_results.py` 合并 AstraSim + roofline + utilization 生成

---

### 2.3 H100 单卡

**单卡，无并行切分**:

```
Proj_QKV:
  权重: W_Q[d × Hq × dk] + W_K[d × Hkv × dk] + W_V[d × Hkv × dk]
  (dk = d_head = 128, 不切分)
  QKV_ns = max(comp, mem) / bw_util_qkv

Attention:
  KV cache: bs × 2 × Hkv × seq × d_head
  Attn_ns = max(comp, mem) / bw_util_attn

Proj_O:
  权重: W_O[d × Hq × dk]
  O_ns = max(comp, mem) / bw_util_o

Wall = QKV + Attn + ProjO  (无通信)
```

**硬件**: peak=1979 TFLOPS FP8, HBM=3.35 TB/s

**BW Utilization**: `h100_rubin_utilization.json`
- proj_qkv: 0.687, proj_o: 0.705
- attn_fused: 按 seq 插值 (1024→0.042, ..., 1M→0.916)

### 2.4 Rubin 单卡

与 H100 完全相同的公式，仅替换硬件参数:
- peak=17500 TFLOPS, HBM=22.0 TB/s
- 使用相同 BW utilization

### 2.5 H100 TP2 / Rubin TP2

权重和 KV cache 各减半 + 2-GPU NVLink AllReduce:
```
stage_rf = single_gpu_roofline / 2
stage_ns = stage_rf / bw_util
comm_ns = msg_bytes / NVLink_BW + 2 × latency
Wall = (QKV + Attn + ProjO) + comm
```

### 2.6 NeuPims

基于 Ours 的逐模块延时，按固定倍数放大:
```
NeuPims_wall = Ours_QKV × k_qkv + Ours_Attn × k_attn + Ours_O × k_o + comm_ns

Qwen3:     k_qkv=2.0, k_attn=2×16/9=3.56, k_o=2.0, comm=200ns
Llama4:    k_qkv=2.0, k_attn=2×5/9=1.11,  k_o=2.0, comm=200ns
DeepSeek:  k_qkv=2.0, k_attn=2×16/9=3.56, k_o=2.0, comm=200ns
```

---

## 3. DeepSeek-V3（MLA 模型）

### 3.1 MLA 参数（参考 Deppseek-v3.md 附录 A）

| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 7168 | 隐藏层维度 |
| n_h | 128 | 注意力头数 |
| d_c | 512 | KV 压缩维度 |
| d_c' | 1536 | Query 压缩维度 |
| d_r | 64 | RoPE 维度 |
| KV cache/tok | d_c + d_r = 576 | 压缩表示 |

### 3.2 MLA 使用吸收后 (Absorbed) 权重

按照文档附录 A，decode 阶段使用预计算的吸收后权重，Ours 和 H100 baseline **均使用相同的权重形式**:

**QKV 阶段权重** (Step 1 + Step 2):

| 矩阵 | 维度 | 说明 |
|-------|------|------|
| W_DKV | d_c × d_model = 512×7168 | KV 下投影（共享） |
| W_KR | d_r × d_model = 64×7168 | RoPE Key 投影（共享） |
| W_DQ | d_c' × d_model = 1536×7168 | Query 下投影（共享） |
| W_Q_abs[i] | d_c × d_c' = 512×1536 (×128 头) | 吸收后 Query = (W_UK)^T · W_UQ |
| W_QR[i] | d_r × d_c' = 64×1536 (×128 头) | RoPE Query 投影 |
| **合计** | **128,385,024 elements** | |

**ProjO 阶段权重** (Step 6):

| 矩阵 | 维度 | 说明 |
|-------|------|------|
| W_O_abs[i] | d_model × d_c = 7168×512 (×128 头) | 吸收后输出 = W_O · W_UV |
| **合计** | **469,762,048 elements** | |

**KV Cache**: 每 token 576 elements (c_KV[512] + k_R[64])，在压缩空间中直接计算注意力。

### 3.3 Ours（16-NPU, absorbed form）

**并行方式**: tp_h=4, tp_s=4, 共 16 NPU, 每 NPU 8 heads

**代码**: `mla_roofline()` in `deepseek_v3_mla_roofline.py`

```
Per-NPU QKV:
  down_weights = (d_c + d_r + d_c') × d_model         # 共享下投影 (FP8 = 1B/elem)
  absorbed_weights = 8 × (d_c × d_c' + d_r × d_c')    # W_Q_abs + W_QR (8 heads)
  MACs = bs × [(d_c+d_r+d_c')×d + 8×(d_c+d_r)×d_c']
  QKV_ns = max(comp_ns / sa_util, mem_ns)

Per-NPU Attention:
  cache_seq = seq / tp_s = seq / 4
  KV cache = bs × cache_seq × (d_c + d_r)              # FP8
  MACs = bs × 8 × cache_seq × (2×d_c + d_r)            # score + value
  score_util, attn_v_util 分别查表，取平均
  Attn_ns = max(comp/avg_util, mem)

Per-NPU ProjO:
  weights = 8 × d_model × d_c                          # W_O_abs (8 heads, FP8)
  MACs = bs × 8 × d_model × d_c
  O_ns = max(comp/o_util, mem)

Communication (MLA Option B, 解析模型):
  (1) Score/lse AllReduce (tp_s=4): msg = bs × 8 × 2 × 4B (FP32 stats)
  (2) o_comp AllReduce (tp_s=4):    msg = bs × 8 × d_c (FP8)
  (3) Final head-Reduce (16 NPU):   msg = bs × d_model
  comm_ns = score_ar + ocomp_ar + final_reduce
```

**SA Utilization**: `deepseek3/util_96.json`（proj_qkv, score, attention, proj_o）

### 3.4 H100 单卡（absorbed form）

**代码**: `mla_single_gpu_baseline()` in `deepseek_v3_mla_roofline.py`

```
Proj_QKV (absorbed):
  权重 = (d_c+d_r+d_c')×d + n_h×(d_c+d_r)×d_c'        # 128,385,024 elem × 2B (BF16)
  QKV_ns = max(comp, mem) / bw_util_qkv

Attention:
  KV cache = bs × seq × (d_c+d_r) × 2B                 # 压缩 576 elem/tok, BF16
  MACs = bs × n_h × seq × (2×d_c + d_r)
  Attn_ns = max(comp, mem) / bw_util_attn

Proj_O (absorbed):
  权重 = n_h × d_model × d_c                            # 469,762,048 elem × 2B (BF16)
  O_ns = max(comp, mem) / bw_util_o

Wall = QKV + Attn + ProjO  (无通信)
```

**硬件**: peak=1979T, HBM=3.35 TB/s

**BW Utilization**: `h100_mla_profile.json`
- proj_qkv: 0.687, proj_o: 0.70
- attn_absorbed: 按 seq 插值 (1024→0.52, ..., 262144→0.92)

### 3.5 Rubin 单卡

与 H100 相同公式，替换硬件: peak=17500T, HBM=22.0 TB/s, 使用相同 bw_util。

### 3.6 H100 TP2 / Rubin TP2

```
每阶段 roofline 减半 (权重和 KV cache 各分到 2 GPU)
stage_ns = (roofline / 2) / bw_util
comm = NVLink AllReduce on d_model vector (BF16)
```

---

## 4. 加速比计算

```
speedup = H100_wall_ns / strategy_wall_ns
```

H100 单卡 TP1 为 baseline (speedup = 1.0×)。所有方案的加速比均相对 H100 TP1 计算。

---

## 5. GQA 与 MLA 的 Roofline 流程对比

| 环节 | GQA (Qwen3/Llama4) | MLA (DeepSeek-V3) |
|------|--------------------|--------------------|
| **QKV 权重** | W_Q + W_K + W_V | W_DQ + W_DKV + W_KR + W_Q_abs + W_QR |
| **KV cache/tok** | 2 × Hkv × d_head | d_c + d_r = 576 |
| **Attention 计算** | Q·K^T + α·V (在 d_head 维度) | q_abs·c_KV + q_R·k_R + α·c_KV (在 d_c/d_r 压缩维度) |
| **ProjO 权重** | W_O[d × Hq × d_head] | W_O_abs[n_h × d × d_c] (吸收后) |
| **Roofline 公式** | `max(comp, mem) / util` | `max(comp, mem) / util` (相同) |
| **Ours util 来源** | qwen3 或 llama4 SA util | deepseek3 SA util |
| **H100 util 来源** | h100_rubin_utilization.json | h100_mla_profile.json |
| **通信** | AstraSim 仿真 | 解析模型 (Option B) |

**核心一致性**: 两者使用完全相同的 roofline 公式 `max(comp, mem) / util`，仅权重维度和 KV cache 维度不同。MLA 的维度来源为 Deppseek-v3.md 附录 A 的吸收后形式。

---

## 6. 硬件参数汇总

| 方案 | peak (TFLOPS) | HBM BW (TB/s) | dtype | 互连 |
|------|---:|---:|---:|------|
| Ours (per NPU) | 96 | 2.5 | FP8 (1B) | D2D 1.5 TB/s |
| H100 SXM5 | 1979 | 3.35 | BF16 (2B) | — |
| H100 TP2 | 1979×2 | 3.35×2 | BF16 (2B) | NVLink 0.9 TB/s, 800ns latency |
| Rubin | 17500 | 22.0 | BF16 (2B) | — |
| Rubin TP2 | 17500×2 | 22.0×2 | BF16 (2B) | NVLink 1.8 TB/s, 800ns latency |

---

## 7. 数据文件与生成脚本

### 数据

| 模型 | 数据路径 |
|------|---------|
| Qwen3 | `reports/qwen3-235B/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{1,4,16,32}.json` |
| Llama4 | `reports/llama4/hybrid/gqa_hybrid_merged_96T_bw1500_util96_bs{1,4,16,32}.json` |
| DeepSeek-V3 | `figures/paper_figures/fig1_e2e_latency/data/deepseek_v3_mla_fig1.json` |

### 生成命令

```bash
# MLA 数据 (H100/Rubin/Ours 全部生成)
python3 backend/roofline/deepseek_v3_mla_roofline.py

# GQA 数据 (需 AstraSim，见 project_hybrid_workflow.md)
# python3 backend/hybrid_merge/merge_gqa_results.py ...

# 画图
python3 figures/paper_figures/fig1_e2e_latency/scripts/plot_e2e.py
```

### Utilization 文件

| 用途 | 文件 |
|------|------|
| Ours GQA Qwen3 SA util | `backend/roofline/utilization_profiles/qwen3/util_96.json` |
| Ours GQA Llama4 SA util | `backend/roofline/utilization_profiles/llama4/util_96.json` |
| Ours MLA SA util | `backend/roofline/utilization_profiles/deepseek3/util_96.json` |
| H100/Rubin GQA BW util | `backend/roofline/utilization_profiles/h100/h100_rubin_utilization.json` |
| H100/Rubin MLA BW util | `backend/roofline/utilization_profiles/h100/h100_mla_profile.json` |
