# Utilization Library — `compute_util` / `bw_util` 来源说明

本目录是 **可独立运行** 的快照，包含统一的 utilization / power API（`roofline/utilization_lib.py`）
及其全部依赖（roofline 内核、SA 利用率生成器、profile 数据、power 系数）。

```
tmp/
├── roofline/
│   ├── utilization_lib.py          # 统一 API：MLA(DeepSeek-V3) + GQA(Qwen3/Llama4)
│   ├── roofline_gqa_calc.py        # roofline 内核（GQA + MLA），算 compute_ns / memory_ns
│   ├── attention.py                # MODEL_CONFIGS + make_strategy_config（per-NPU 切分）
│   ├── hardware_config.py          # 各硬件 peak TFLOPS / HBM 带宽
│   ├── main.py                     # SA(脉动阵列) 利用率解析模型
│   ├── util.py                     # profile 生成器：python3 util.py <model> <num_sa>
│   ├── query_util.py               # ★ 命令行查询：compute_util / bw_util / power
│   └── utilization_profiles/
│       ├── deepseek3/util_<N>.json # AMMA SA compute-util（MLA）
│       ├── qwen3/util_<N>.json     # AMMA SA compute-util（GQA）
│       ├── llama4/util_<N>.json    # AMMA SA compute-util（GQA）
│       └── h100/h100_mla_profile.json  # Rubin/B200 的 BW-util（实测）
└── hybrid_merge/
    └── power_model.py              # POWER_COEFFS（power_* 函数用）
```

> 运行：`cd roofline && python3 -c "import utilization_lib"`。
> `utilization_lib` 会把 `../hybrid_merge` 加进 `sys.path` 找 `power_model`，所以
> 必须保持 `roofline/` 与 `hybrid_merge/` 这对同级目录结构。

---

## 0. 一句话总览

| 硬件 (`hw["arch"]`) | `compute_util` 来源 | `memory_util` (BW-util) 来源 |
|---|---|---|
| `amma` (Ours) | **离线 SA 解析模型** 生成的 profile（`util_<N>.json`），按 `bs`/`seq` 查表 | 由 roofline `memory_ns / realized_ns` **实时算出** |
| `rubin` / `b200` | 0（decode bs=1 是带宽受限，compute 不是瓶颈） | **GPU 实测 profile**（`h100_mla_profile.json` 的 `bw_util`），按 `bs`/`seq` 查表 |

两条路线的分界点：**AMMA 的瓶颈是脉动阵列填充率（compute_util），BW-util 由它反推；
Rubin/B200 的瓶颈是 HBM 带宽（bw_util），直接用实测曲线。**

---

## ⚡ 直接使用（命令行）

> 所有命令在 `tmp/roofline/` 目录下执行（依赖 `../hybrid_merge/power_model.py`，保持目录结构即可）。

```bash
cd tmp/roofline
```

### A. 查询 compute_util / bw_util / power —— `query_util.py`

一条命令直接打印某个算子的 `compute_util`、`bw_util`、`power`。算子 `--op` 取 `qkv | o | attn`。

```bash
# ---- Qwen3 (GQA) ----
python3 query_util.py --model qwen3  --arch amma  --op qkv  --bs 8
python3 query_util.py --model qwen3  --arch amma  --op o    --bs 8
python3 query_util.py --model qwen3  --arch amma  --op attn --seq 16384

# ---- Llama4 (GQA) ----
python3 query_util.py --model llama4 --arch amma  --op qkv  --bs 8
python3 query_util.py --model llama4 --arch amma  --op attn --seq 16384

# ---- DeepSeek-V3 (MLA) ---- 自动走 MLA 路径，无需 --strategy
python3 query_util.py --model deepseek3 --arch amma --op qkv  --bs 8
python3 query_util.py --model deepseek3 --arch amma --op attn --seq 16384

# ---- Rubin / B200（带宽受限：compute_util=0，bw_util 查实测曲线）----
python3 query_util.py --model qwen3 --arch rubin --op qkv  --bs 8
python3 query_util.py --model qwen3 --arch b200  --op attn --seq 16384

# ---- GQA 切分策略（默认 HMP_reo，可换 tp16 / hmp / hmp_reo_new）----
python3 query_util.py --model qwen3 --arch amma --op o --bs 8 --strategy tp16

# ---- 改硬件参数（峰值算力 / 带宽 / SA 数量 / TP）----
python3 query_util.py --model qwen3 --arch amma --op qkv --bs 8 \
        --tflops 50 --bw 1.5 --num-sa 64 --tp 16
```

输出示例：

```
$ python3 query_util.py --model qwen3 --arch amma --op qkv --bs 8
[qwen3 | amma | GQA/HMP_reo] qkv bs=8 TP=16
  compute_util = 0.4157
  bw_util      = 0.8797
  power        = 1261.2 W
```

`--help` 查看全部参数：`python3 query_util.py --help`

### B.（可选）重新生成 AMMA 的 SA compute-util profile —— `util.py`

只有当 `num_sa` 不在已有 profile 列表里、或修改了 SA 模型时才需要。
现成 profile 已覆盖 `num_sa ∈ {8,16,32,64,96,128,256,512,768,1024}`。

```bash
cd tmp/roofline
python3 util.py qwen3     64    # -> qwen3_util_64.json     （GQA）
python3 util.py llama4    64    # -> llama4_util_64.json    （GQA）
python3 util.py deepseek3 64    # -> deepseek3_util_64.json （MLA）

# 生成的文件写在当前目录，确认无误后放进 profile 目录：
mv qwen3_util_64.json utilization_profiles/qwen3/util_64.json
```

### C. 在 Python 里直接调用 API

```python
import utilization_lib as ul
hw = {"arch": "amma", "peak_per_npu_tflops": 50, "bw_per_npu_tbs": 1.5, "num_sa_for_util": 64}

# GQA
cu = ul.compute_utilization_qkv_gqa(ul.QWEN3_DEFAULTS, bs=8, hw=hw, TP=16)   # 0.4157
mu = ul.memory_utilization_qkv_gqa(ul.QWEN3_DEFAULTS, bs=8, hw=hw, TP=16)    # 0.8797
pw = ul.power_qkv_gqa(mu, cu, hw)

# MLA
cu = ul.compute_utilization_qkv(ul.DEEPSEEK_V3_DEFAULTS, bs=8, hw=hw, TP=16) # 0.4854
```

---

## 1. AMMA (Ours) 的 `compute_util` — SA 解析模型

AMMA 用的 `compute_util` **不是实测、也不是 roofline 算的**，而是一个
**脉动阵列(systolic array, SA)填充率模型**离线枚举出来、存成 JSON 的。

### 1.1 生成方式

```bash
cd roofline
python3 util.py qwen3   64     # -> qwen3_util_64.json     （GQA）
python3 util.py llama4  64     # -> llama4_util_64.json    （GQA）
python3 util.py deepseek3 64   # -> deepseek3_util_64.json （MLA）
```

`<num_sa>` = 该 NPU 上脉动阵列的数量（16×16 的 SA）。生成的 JSON 形如：

```json
{
  "proj_qkv": {"1": 0.051962, ..., "64": 0.996648},          // 按 batch size 查
  "score":     {"16": {"utilization": 0.005263, "k_split": 2}, ...},  // 按 seq 查
  "attention": {"16": {"utilization": 0.014286, "k_split": 1}, ...},  // 按 seq 查
  "proj_o":    {"1": 0.061568, ..., "64": 0.99...}            // 按 batch size 查
}
```

生成后放到 `utilization_profiles/<model>/util_<N>.json`，供 `utilization_lib` 查表。

### 1.2 模型本身（`main.py`）

对每个 GEMM `(m,k)@(k,n)`，把输出 tile（`ceil(m/16)×ceil(n/16)`）摊到 `num_sa` 个 16×16 SA 上：

```
compute_util = tile_util × spatial_util × k_util
```

- **`tile_util`** — tile 数能否整除 SA 数（负载均衡）。decode 时 `m=bs` 很小 → tile 很少 → 很多 SA 闲置 → util 低。
- **`spatial_util`** — 边角 tile 不满 16×16 的填充损失。
- **`k_util = (t·k)/(t·k + 2·SA − 1)`** — 脉动阵列流水线填充/排空开销，`k` 越大越接近 1。
- **`k_split ∈ {1,2,3,4}`** — 当输出 tile 太少时沿 K 维切分，让更多 SA 参与；带来 tree-reduction 开销，取使 wall-clock 最小的方案。

各模型/各算子喂进去的 `(m,n,k)`（见 `util.py`）：

| 模型 | 算子 | m | n | k |
|---|---|---|---|---|
| **GQA** (qwen3/llama4) | proj_qkv | `bs` | `(Hq·dh + 2·Hkv·dh)/16` | `d_model` |
| | score   | `R = Hq/Hkv` | `seq/4` | `d_head` |
| | attention | `R` | `d_head` | `seq/4` |
| | proj_o  | `bs` | `d_model` | `Hq·dh/16` |
| **MLA** (deepseek3) | proj_qkv | `bs` | `(Hq+Hkv)·(d_c+d_r)/16` | `d_model` |
| | score   | `n_h` | `seq/16` | `d_c+d_r` |
| | attention | `n_h` | `d_c` | `seq/16` |
| | proj_o  | `bs` | `d_model` | `n_h·d_c/16` |

> 直观结论：decode `bs=1` 时 `proj_qkv` util ≈ 5–6%（qwen3 0.052 / deepseek3 0.061），
> 因为只有 1 行 token、tile 数远少于 SA 数；`bs=64` 时 ≈ 99%（SA 填满）。
> attention 的 util 随 `seq` 增大而升高（K 维变长 → `k_util→1`）。

### 1.3 `utilization_lib` 如何用它

`compute_utilization_*`（amma 分支）= 直接对 profile 按 `bs`/`seq` 做 **log2-线性插值** 查表
（`_interp`）。GQA 走 `_gqa_profile(config["name"], num_sa)`，MLA 走 `_amma_profile(num_sa)`。

```python
hw = {"arch": "amma", "peak_per_npu_tflops": 50, "bw_per_npu_tbs": 1.5, "num_sa_for_util": 64}
ul.compute_utilization_qkv_gqa(ul.QWEN3_DEFAULTS, bs=8, hw=hw, TP=16)   # -> 0.4157
ul.compute_utilization_qkv(ul.DEEPSEEK_V3_DEFAULTS, bs=8, hw=hw, TP=16) # -> 0.4854
```

GQA attention 查表用的 key 是归一化后的 **effective_seq**：

```
effective_seq = cache_seq × dk_local / dk_ref,   cache_seq = seq / tp_s
```

这样 TP16(`dk=32`) 与 HMP(`dk_full=128`) 两条切分路径——总 MAC 相同——会落到同一条 SA 曲线上。

---

## 2. AMMA 的 `memory_util` (BW-util) — roofline 反推

AMMA 的 BW-util **不查表**，而是用 roofline 的 `compute_ns` / `memory_ns` 配合上面的 SA-util
**实时算出**。核心是 “realized time” 公式（`_amma_realized_ns`）：

```
realized_ns = max(compute_ns / compute_util,  memory_ns)
memory_util = memory_ns / realized_ns
```

含义：

- `compute_ns / compute_util` = SA 填充率打折后**实际**的计算耗时。
- 与纯带宽耗时 `memory_ns` 取 max = 真实 wall-clock。
- 这段时间里实际搬了 `memory_ns × peak_bw` 的数据 → BW 利用率 = `memory_ns / realized_ns`。

`compute_ns` / `memory_ns` 来自 `roofline_gqa_calc`：

- GQA：`roofline_qkv` / `roofline_attention` / `roofline_output`
- MLA：`roofline_mla_qkv` / `roofline_mla_attention` / `roofline_mla_output`

它们按 per-NPU 切分（`make_strategy_config` / `make_mla_npu_config`）算出权重+激活字节数、
MAC 数，再除以 `hw["peak_per_npu_tflops"]` / `hw["bw_per_npu_tbs"]`。

> 直观结论：decode 时 compute_util 很低 → `compute_ns/compute_util` 被放大 → 计算时间主导 →
> 这段时间带宽没吃满或刚好吃满，BW-util 通常很高（qwen3 QKV bs=8 ≈ 0.88，proj_o ≈ 1.0）。

---

## 3. Rubin / B200 的 `bw_util` — GPU 实测 profile

Rubin / B200 是**带宽受限**架构，decode bs=1 时 compute 不是瓶颈，所以：

- `compute_util ≡ 0`（所有 `compute_utilization_*` 在 `arch != "amma"` 时返回 0）。
- `memory_util` 直接查 **实测 BW-util 曲线**，不做 roofline 反推。

数据来源（`utilization_profiles/h100/h100_mla_profile.json` 的元信息）：

```
_source       = profiling/H100_results/dsv3_attn_e2e_tp1.txt  (bs=1, 100 iters median)
_hbm_bw_tbs   = 3.35
_dtype        = bfloat16
_description  = DeepSeek-V3 MLA H100 SXM5 BW utilization — absorbed form
```

即在真实 H100 上跑 DeepSeek-V3 MLA attention end-to-end，量到的 HBM 带宽利用率，
按 `proj_qkv` / `attn_absorbed` / `proj_o` 三段分别给出（例：`proj_qkv[bs=1]=0.687`，`proj_o[bs=1]=0.702`）。
norm/rope 等逐元素开销被折进 `proj_qkv`（`absorbed` form）。

`utilization_lib` 里 `b200` 暂时复用 rubin 曲线（`_b200_profile` → `_rubin_profile`，见代码 TODO）。

```python
hw = {"arch": "rubin", "peak_per_npu_tflops": 1000, "bw_per_npu_tbs": 8.0}
ul.compute_utilization_qkv_gqa(cfg, 8, hw, TP=8)   # -> 0.0
ul.memory_utilization_qkv_gqa(cfg, 8, hw, TP=8)    # -> 0.687（查实测曲线）
```

---

## 4. Power

三种硬件统一公式（`power_*` 函数，系数来自 `hybrid_merge/power_model.py` 的 `POWER_COEFFS`）：

```
P_kernel_w = static_w + cmpt_coeff × compute_util + mem_coeff × memory_util
```

`arch → POWER_COEFFS key`：`amma→"ours"`，`rubin→"rubin"`，`b200→"h100"`。

---

## 5. 速查：拿到一个 (compute_util, bw_util) 的完整链路

**Qwen3 / Llama4 (GQA) on AMMA**
1. 生成 SA profile：`python3 util.py qwen3 <num_sa>` → `utilization_profiles/qwen3/util_<N>.json`
2. `compute_util` = 查 profile（`proj_qkv`/`score`/`attention`/`proj_o`）按 `bs`/`effective_seq` 插值
3. `bw_util` = `memory_ns / max(compute_ns/compute_util, memory_ns)`，roofline 由 `roofline_qkv/attention/output` 给出

**DeepSeek-V3 (MLA) on AMMA**
- 同上，profile 用 `deepseek3/util_<N>.json`，roofline 用 `roofline_mla_*`，API 用不带 `_gqa` 后缀的函数。

**任意模型 on Rubin / B200**
1. `compute_util = 0`
2. `bw_util` = 查 `h100_mla_profile.json["bw_util"]`（GPU 实测），按 `bs`/`seq` 插值

---

## 6. API 一览（`utilization_lib.py`）

MLA (DeepSeek-V3)：`compute/memory_utilization_qkv`、`_o`、`_attention`、`_gating`、`_combine`、`_moe_per_expert` + 对应 `power_*`。

GQA (Qwen3/Llama4)：`compute/memory_utilization_qkv_gqa`、`_o_gqa`、`_attention_gqa` + `power_*_gqa`，
均多一个可选参数 `strategy ∈ {"hmp","HMP_reo"(默认),"hmp_reo_new","tp16","rubin"}`。

配置常量：`DEEPSEEK_V3_DEFAULTS`、`QWEN3_DEFAULTS`、`LLAMA4_DEFAULTS`。
