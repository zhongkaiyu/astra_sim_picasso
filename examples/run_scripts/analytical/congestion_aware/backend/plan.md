# Utilization & Power 扩展计划

目标：给 **Rubin / AMMA (Ours) / B200** 三套硬件，统一暴露 13 个函数（`compute_utilization_*` / `memory_utilization_*` / `power_*`），同时覆盖 MLA Attention 三段 (QKV/Attn/ProjO) + MoE 三段 (gating/combine/per_expert)，支持 `batch_size ∈ [1..32]` 与可变 TP。

---

## 1. 结构变化对比

### 变化前（当前仓库）

```
backend/
├── roofline/
│   ├── hardware_config.py          # B200/ours/rubin/H100/hbm4_npu 五套 dict，无 arch tag
│   ├── attention.py                # MODEL_CONFIGS={qwen3, llama4}, make_strategy_config(TP 写死 4×4)
│   ├── roofline_gqa_calc.py        # roofline_qkv/attention/output (GQA) + roofline_mla_qkv/attention/output (MLA)
│   ├── util.py                     # SA-based utilization 生成器
│   └── utilization_profiles/
│       ├── deepseek3/util_{N}.json # AMMA SA compute util (proj_qkv/score/attention/proj_o)
│       └── h100/{h100_mla_profile,rubin_bw_util,tp2_profile}.json  # Rubin/H100 BW util
└── hybrid_merge/
    └── power_model.py              # POWER_COEFFS={ours,rubin,h100}, calc_power_ours/rubin
                                    # 仅服务 hybrid merged JSON，未做单点函数 API
```

**痛点**：
- TP 硬编码 `tp_h=4, tp_s=4` (AMMA) / `TP=1` (Rubin)，无法做 batch sweep。
- 没有 MoE 三段（gating / combine / per-expert）的 roofline。
- `power_model.py` 只能吃合并后的 hybrid JSON，无法独立给 `(mem_util, compute_util) → power` 调用。
- B200 没有 `POWER_COEFFS`。

### 变化后（建议）

```
backend/
├── roofline/
│   ├── hardware_config.py          # ★ 各 dict 加 "arch": "rubin"/"amma"/"b200" 字段
│   ├── attention.py                # ★ MODEL_CONFIGS 新增 deepseek3 (含 n_routed_experts/top_k/d_moe_inter)
│   ├── roofline_gqa_calc.py        # 不动（被 utilization_lib 复用）
│   ├── utilization_lib.py          # ★ 新建：13 个统一 API
│   │   ├── compute/memory_utilization_qkv  (config, bs, hw, TP)
│   │   ├── compute/memory_utilization_o    (config, bs, hw, TP)
│   │   ├── compute/memory_utilization_attention (config, seq_len, hw, TP)
│   │   ├── compute/memory_utilization_gating          (config, bs, hw)
│   │   ├── compute/memory_utilization_combine         (config, bs, hw)
│   │   ├── compute/memory_utilization_moe_per_expert  (config, bs, hw)
│   │   └── power_{qkv,o,attention,gating,combine,moe_per_expert}(mem_util, cmpt_util, hw, flops, bytes)
│   └── utilization_profiles/
│       ├── deepseek3/util_{N}.json (沿用)
│       └── b200/                   # ★ TODO: B200 BW util profile（可暂时复用 rubin_bw_util 或填 0.9）
└── hybrid_merge/
    └── power_model.py              # 内部改为调用 utilization_lib.power_*，去掉重复逻辑

reports/utilization/                # ★ 新建：bs sweep 输出
├── deepseek3_rubin_bs{1..32}.json
└── deepseek3_amma_bs{1..32}.json
```

---

## 2. 13 个函数的数据来源 / 复用映射

| # | 函数 | 复用 | 备注 |
|---|---|---|---|
| 1 | `compute_utilization_qkv` | `roofline_mla_qkv` + AMMA `proj_qkv` SA util / Rubin `proj_qkv` BW util | TP 替换 `n_npus = tp_h*tp_s` |
| 2 | `memory_utilization_qkv` | `roofline_mla_qkv["weight_elems"+"activation_elems"]` / time / peak_BW | 同上 |
| 3 | `compute_utilization_o` | `roofline_mla_output` + `proj_o` util | `d_value_shard = d_value/TP` |
| 4 | `memory_utilization_o` | `roofline_mla_output["weight_elems"]` / time / peak_BW | 同上 |
| 5 | `compute_utilization_attention` | `roofline_mla_attention` + `score`/`attention` util (AMMA) 或 `attn_absorbed` (Rubin) | `cache_seq = seq/TP` |
| 6 | `memory_utilization_attention` | `roofline_mla_attention["kv_cache_elems"]` / time / peak_BW | KV 合一 (d_score = d_c+d_r) |
| 7 | `power_*` (6 个) | `POWER_COEFFS[hw["arch"]]` | AMMA 含 D2D，Rubin/B200 不含 |
| 8 | `compute_utilization_gating` | **新写**：`flops=2·bs·d_model·n_routed_experts` | DeepSeek-V3: `d_model=7168, n_routed_experts=256` |
| 9 | `memory_utilization_gating` | **新写**：weight `d_model·n_routed_experts` | FP8 |
| 10 | `compute_utilization_combine` | **新写**：`bs·top_k·d_model` scatter-add | top_k=8，几乎 0 compute util |
| 11 | `memory_utilization_combine` | **新写**：`bs·top_k·d_model + bs·d_model` | 纯 memory-bound |
| 12 | `compute_utilization_moe_per_expert` | **新写**：三段 GEMM (up/gate/down)，权重 `3·d_model·d_moe_inter` | AMMA: 权重按 16 NPU 分片；Rubin/B200: 整块 |
| 13 | `memory_utilization_moe_per_expert` | 同上，`tokens_per_expert = bs·top_k/n_routed_experts` | 解码 bs=1 时 << 1 expert，需 ceil 或传 expected_tokens |

---

## 3. 需要扩展才能得到的结果（gap 清单）

| 缺什么 | 影响 | 建议处理 |
|---|---|---|
| **B200 `POWER_COEFFS`** | `power_*` 在 B200 上无法算 | 用 B200 spec 估 `static_ratio=0.177, tdp=1000W`；`cmpt_coeff`/`mem_coeff` 参考 H100 系数等比缩放（待定）。 |
| **B200 BW util profile** | `memory_utilization_*` 对 B200 没有实测 cap | 短期：复用 `rubin_bw_util.json` × 0.95；长期：补 B200 NCU 数据。 |
| **DeepSeek-V3 MoE 模型配置** | gating/combine/per_expert 没有现成模型参数 | 在 `attention.py:MODEL_CONFIGS` 加 `deepseek3 = {d_model:7168, n_routed_experts:256, n_shared_experts:1, top_k:8, d_moe_inter:2048}`。 |
| **MoE per-expert SA util** | gating/per_expert 是小 GEMM，沿用 `proj_qkv` SA util 不准 | 短期：复用 proj_qkv util；长期：跑一份 `deepseek3_moe_util_{N}.json`。 |
| **TP-aware MLA 切分** | 现 `roofline_mla_*` 把 `n_npus = tp_h*tp_s` 当总卡数，但 Rubin 经常 TP=1/2 | 加 `TP` 参数显式覆盖 `n_npus`；AMMA 默认 16, Rubin 默认 1, B200 默认 2。 |
| **batch 1..32 sweep 入口** | 当前 fig 数据只在 bs∈{1,4,16,32} 跑 | 新建 `reports/utilization/` driver，跑全 bs 范围。 |
| **D2D 能耗系数 B200** | AMMA 用 0.38 pJ/bit，NVLink 1.3 pJ/bit；B200 互联系数不明 | 暂沿用 NVLink 1.3 pJ/bit。 |

---

## 4. 建议开发顺序

1. `hardware_config.py` 加 `"arch"` 字段 + B200 `POWER_COEFFS` 占位。
2. `attention.py:MODEL_CONFIGS["deepseek3"]` 补 MoE 参数。
3. 新建 `roofline/utilization_lib.py`，先实现 1–7（MLA attention 已有 roofline 函数支撑，最快）。
4. 跑 `tests/verify_against_baseline.py`，确认 1–7 与 `tests/baseline_dataset.json` 1e-4 内一致。
5. 实现 8–13（MoE），先用复用 util，后续补 MoE-specific profile。把新结果追加进 `baseline_dataset.json` 作为后续锚点。
6. 写 driver `roofline/run_utilization_sweep.py`，对每个 (arch, bs∈[1..32], seq∈{...}) 输出 JSON。
7. `power_model.py` 内部切换到 `utilization_lib.power_*`，去重。

## 5. 回归测试

`backend/tests/` 已固化了"现有实现"的输入输出对：

| 文件 | 用途 |
|---|---|
| `tests/gen_baseline.py` | 用 **现有** `roofline_mla_*` + util 配置 + `POWER_COEFFS` 跑一个固定 grid。 |
| `tests/baseline_dataset.json` | 32 cases × 3 kernels = 96 个 ground-truth 数。AMMA/Rubin × bs∈{1,4,16,32} × seq∈{1024, 8192, 65536, 131072}。**重构前后必须保持一致。** |
| `tests/README.md` | 字段说明、AMMA/Rubin convention、如何用 `verify_against_baseline.py` 检验新代码。 |

新 `utilization_lib.py` 落地时，运行 `python3 tests/verify_against_baseline.py` 全绿才能合入。MoE 三段 (gating/combine/per_expert) 因无 baseline 暂不在测试集内，待新实现落地后再追加。
