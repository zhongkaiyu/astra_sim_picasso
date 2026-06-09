# backend — AMMA 解耦推理的解析建模工具

这里是论文的核心建模代码:用 **roofline + 实测利用率 derate** 估算解耦架构
（**AMMA attention + LPU FFN**）的单层延迟、TPOT 和功耗,并和 GPU baseline 对照。
所有 `figures/` 里的论文图都由本目录的脚本产出。

代码位于 `examples/run_scripts/analytical/congestion_aware/backend/`。每个模块自成一体,
直接 `python <脚本>.py --help` 就能看参数,下面给最小可跑命令。

## 环境

统一用 bench conda 环境(已装 numpy / torch,profiling 还需 CUDA):

```bash
PY=/home/haotian/miniconda3/envs/bench/bin/python
```

下文命令都以 `$PY` 代表解释器,且**从仓库根目录执行**——`cd` 用从根出发的完整相对路径
(脚本按相对路径找配置,目录位置很重要)。

## 模块一览

| 目录 | 作用 | 入口 |
|---|---|---|
| `backend/roofline/` | 单层 attention roofline / 利用率 / 延迟 | `main.py`, `roofline_gqa_calc.py`, `roofline_v4_calc.py` |
| `backend/lpu_ffn_decode/` | FFN decode + 完整 decode TPOT(ours vs baseline) | `run.py`, `decode_compose.py`, `baseline_compose.py` |
| `backend/hybrid_merge/` | 合并 GQA 结果 + 功耗模型 + 出报告 | `power_model.py`, `query_power.py` |
| `backend/hw_estimation/` | 单 cube 的功耗/面积估算(cacti + yosys) | `run_estimation.py` |
| `backend/baselines/` | NMP(近存计算)对照 | `nmp_compose.py` |
| `backend/tests/` | roofline/功耗的回归测试(锚定数值) | `verify_against_baseline.py` |

> `backend/lpu_ffn_decode/` 自带更详细的 README(LPU 硬件、跨池 link 预置、profiling),
> 跑 FFN/TPOT 前先看那一篇。

## 快速上手

> 所有 `cd` 路径均相对仓库根目录。

### 1. Attention roofline / 利用率（`backend/roofline/`）

```bash
cd examples/run_scripts/analytical/congestion_aware/backend/roofline

# 打印各 batch / seq 下的 attention 利用率表(无参数即可)
$PY main.py

# 单层 GQA roofline:含权重加载,扫 seq 长度,可选模型/硬件
$PY roofline_gqa_calc.py --model qwen3 --batch 1 \
    --seq 4096 32768 131072 --peak-perf 60 --bandwidth 2.5 -o gqa_qwen3.json

# DeepSeek-V4 MLA 注意力 per-layer 延迟
$PY roofline_v4_calc.py
```

`roofline_gqa_calc.py` 常用参数:`--model {qwen3,llama4,...}` `--batch` `--seq`(可多值)
`--peak-perf`(TFLOPS) `--bandwidth`(TB/s) `--link-bw` `-o`(输出 JSON)。

### 2. FFN decode 与完整 TPOT（`backend/lpu_ffn_decode/`）

```bash
cd examples/run_scripts/analytical/congestion_aware/backend/lpu_ffn_decode

# 只算 FFN decode 延迟
$PY run.py --model qwen3-235b --batch 1 --tp 4

# 完整 decode TPOT = AMMA attention + LPU FFN + 跨池传输(ours)
$PY decode_compose.py --model deepseek3 --cl-sweep 2048,8192,32768,131072 \
    -o data/deepseek3_tpot_sweep.json

# 对照 baseline:GPU+GPU / GPU+LPU
$PY baseline_compose.py --model deepseek3 --cl-sweep 2048,8192,32768 --batch 1 \
    -o data/baseline_deepseek3_b1.json
```

### 3. 功耗模型与报告（`backend/hybrid_merge/`）

```bash
cd examples/run_scripts/analytical/congestion_aware/backend/hybrid_merge

# 把已有 power JSON 渲染成格式化表格
$PY query_power.py --power ../reports/qwen3-235B/power/<某个>_power.json
```

`power_model.py` 是功耗系数与 `calc_power_*` 的实现,被 roofline / hybrid_merge 复用;
`collect_gqa_*.py` + `merge_gqa_results.py` 负责把多策略结果合并后再 `query_power.py`。

### 4. 硬件估算（`backend/hw_estimation/`）

依赖 cacti 与 yosys(综合产物 `yosys/` 不入库,会本地重算)。

```bash
cd examples/run_scripts/analytical/congestion_aware/backend/hw_estimation

$PY run_estimation.py --peak              # 估算一个 cube 的功耗/面积(峰值)
$PY run_estimation.py --only A,B          # 只算指定组件
```

## 回归测试

改了 roofline / 功耗公式后,务必跑回归,确认数值仍锚定在 `tests/baseline_dataset.json`:

```bash
cd examples/run_scripts/analytical/congestion_aware/backend

$PY tests/verify_against_baseline.py            # 全绿才算没回归
$PY tests/verify_against_baseline.py --verbose  # 打印每个 case
```

> 基线 `baseline_dataset.json` 是"现有实现"的快照,**只有在故意改动建模口径时**
> 才用 `tests/gen_baseline.py` 重新生成,并逐数 review diff。
