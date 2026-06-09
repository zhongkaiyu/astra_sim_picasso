# LPU (SHIP) FFN Decode Latency Simulation

基于论文 **Bitar et al., "SHIP: SRAM-based Huge Inference Pipelines for Fast LLM
Serving" (MLSys'26)** 的 LPU 方法，估算 **FFN 的 decode 延迟**（TPOT 的 FFN 分量）。

与 `backend/` 下其它脚本**完全解耦**：本目录自成一体，只在目录内部互相 import。

## 文件

| 文件 | 作用 |
|------|------|
| `lpu_config.py` | 参数暴露层：`LPUSpec`（LPUv1/LPX 硬件）/ `FFNModelSpec`（模型 FFN 结构）/ `RunSpec`（并行+batch）+ 模型库 |
| `ffn_decode.py` | FFN 核心：roofline + C2C AllReduce，dense 与 MoE 两条路径 |
| `run.py` | FFN-only CLI：组装参数 → 仿真 → 结构化 JSON |
| `interconnect.py` | 解耦架构的跨池(AMMA↔LPU) activation 传输模型 |
| `decode_compose.py` | **完整 decode TPOT 组合器**：AMMA attention + LPU FFN + 跨池传输 |
| `rubin_ffn_decode.py` | **FFN-on-GPU(Rubin)**：与 ffn_decode 同一 roofline，但跑 Rubin HBM 并用实卡利用率 derate（见下）|
| `baseline_compose.py` | **两条非-AMMA baseline 组合器**：GPU+GPU / GPU+LPU（单层 + TPOT）|
| `data/gpu_ffn_utilization.json` | 实卡 GPU FFN 利用率（dense=H100 ncu；MoE=H100 GPU0 实测 per-model）|
| `data/` | 输出目录 |

## Baselines：GPU+GPU / GPU+LPU（`baseline_compose.py`）

与 "ours"（AMMA+LPU, `decode_compose.py`）对照的两条 baseline，单层严格串行：

| baseline | attention | FFN | 跨池 xfer |
|---|---|---|---|
| **GPU+GPU** | Rubin GPU | Rubin GPU | 无（同器件域）|
| **GPU+LPU** | Rubin GPU | LPU(LPX) | 有（GPU↔LPU 分立芯片，默认 `nvlink_rubin`）|
| ours (exp1) | AMMA | LPU(LPX) | 有（同封装 UCIe）|

```
GPU+GPU : t_layer = t_attn_rubin + t_ffn_rubin
GPU+LPU : t_layer = t_attn_rubin + t_xfer + t_ffn_lpu + t_xfer
```

**FFN-on-Rubin 的核心 = 实卡利用率 derate**（用户需求 1+2）：纯 roofline 假设 100% 效率，
真实 GPU 在 decode 小 GEMV 上只达峰值的一小部分。把每条 roofline 边除以实测占比：
```
t_mem     = weight_bytes / (peak_BW   × TP × mem_util)
t_compute = flops        / (peak_FLOP × TP × compute_util)
```
利用率来自 **实卡 H100 ncu**（`data/gpu_ffn_utilization.json`）：dense gate_up `mem≈0.82`/`compute≈0.045`、
down `mem≈0.77`/`compute≈0.040`（MEM-bound）。方法与 attention 侧的 `h100_rubin_utilization.json`
（`total_ns / bw_util`）完全一致——假设 kernel 效率跨代不变（H100→Rubin）。

> MoE 段已 **实测**（H100 GPU0, FP8, per-model, 2026-06-07）：qwen3 expert `gate_up≈0.43`/`down≈0.28`、
> deepseek3 `gate_up≈0.66`/`down≈0.48`（shared 类似）——远低于 dense 0.82/0.77，因单 expert 权重(6–29MB)
> 喂不饱 HBM。存 `gpu_ffn_utilization.json:moe_by_model`，`FFNUtil(model_name=...)` 自动按模型选。
> dense 段为 fp16 ncu 实测，作 FP8 效率比例代理。

```bash
PY=/home/haotian/miniconda3/envs/bench/bin/python
$PY baseline_compose.py --model deepseek3  --cl-sweep 2048,8192,32768,131072 --batch 1 -o data/baseline_deepseek3_b1.json
$PY baseline_compose.py --model qwen3-235b --cl 8192 --batch 4
$PY baseline_compose.py --model deepseek3  --t-attn-ns 1800   # 用自己的 attention 单层延时
```
可调：`--baselines gpu_gpu,gpu_lpu` `--tp-ffn` `--lpu` `--link`（GPU↔LPU）`--no-attn-util`。

### 实卡 GPU FFN profiling（生成/更新利用率）

| 脚本 | 作用 |
|---|---|
| `microbench/microbench_moe_util.py` | **无需 sudo** 的 MoE 利用率实测：CUDA-graph + 16-distinct-weight pool（绕开 launch overhead + L2），按 ncu 校准(/0.87)。**当前 `moe_by_model` 数据来源**。|
| `ncu/ncu_decode_profile.py` | dense MLP decode（已有 H100 数据 → `gpu_ffn_utilization.json:dense`）|
| `ncu/ncu_moe_decode_profile.py` | MoE FFN decode（gating+top_k experts+shared），`--e2e` / ncu 两模式，fp8/fp16 |
| `ncu/analyze_moe_ncu.py` + `ncu/run_ncu_moe.sh` | 可选：sudo ncu 计数器路线（counter-based 确认），按权重 DRAM 大小分组出 util |

```bash
# 无需 sudo（当前用的）：CUDA-graph 实测 MoE expert GEMV 利用率
CUDA_VISIBLE_DEVICES=0 $PY profiling/microbench/microbench_moe_util.py --model qwen3-235b --dtype fp8
# 可选 sudo ncu 计数器确认（GPU perf counter 需 root）
sudo bash profiling/ncu/run_ncu_moe.sh qwen3-235b fp8
```

## 完整 decode TPOT（`decode_compose.py`，解耦 attention-FFN 架构）

架构：attention 跑 **AMMA**（复用 `roofline/attention.py` 的 hmp/"ours" 模型，只读不改），
FFN 跑 **LPU**（本模块）。每层 4 项**严格串行**：

```
t_layer(CL) = t_attn_AMMA(CL) + t_xfer(AMMA→LPU) + t_ffn_LPU + t_xfer(LPU→AMMA)
TPOT        = num_layers × t_layer(CL)        # 单用户延迟，PP 不改 TPOT
```

`t_xfer` 是解耦独有的跨池搬运开销（hidden state `B×d_model`，每层往返 2 次）。
attention 随 CL 增长、FFN+xfer 平坦 → 存在 CL 交叉点。

```bash
python decode_compose.py --model qwen3-235b --cl 4096 --batch 1
python decode_compose.py --model deepseek3 --cl-sweep 2048,8192,32768,131072 \
       -o data/deepseek3_tpot_sweep.json
python decode_compose.py --model deepseek3 --t-attn-ns 1800   # 直接给 attention 单层延时
```
可调：`--lpu {lpuv1,lpx}` `--tp-ffn` `--tp-h/--tp-hd`(AMMA) `--link`(跨池预置) `--link-bw/--link-lat`(覆盖)。

**跨池 link 预置**（取自 AMMA 论文硬件表，`--link`）：

| 预置 | 带宽 | 延迟 | 场景 |
|---|---|---|---|
| `ucie3_d2d`（默认） | 1500 GB/s | **15 ns** | AMMA 原生 UCIe 3.0 D2D，AMMA↔LPU 同封装 |
| `nvlink_rubin` | 3600 GB/s | 900 ns | 分立芯片，Rubin 级 C2C NVLink |
| `nvlink_h100` | 900 GB/s | 900 ns | 分立芯片，H100 级 NVLink |

> 论文 §7.6：低 batch 下传输延迟由**固定启动延迟**主导，故 `latency` 比 `bandwidth` 关键。
> 用 UCIe(15ns) 时跨池开销可忽略(~1%)，用 NVLink(900ns) 时占 ~35%(61 层 ×2 次固定延迟放大)。

> 注：deepseek3 attention 为 MLA，用 GQA 近似（`num_kv_heads_cache≈5` 模拟压缩 KV），输出标 `approx`。

## 论文 modeling（本模块实现）

1. **memory-bound roofline**：`t = max(FLOPs/peak, bytes/BW)`，FFN decode 几乎总落在 mem 边。
2. **SRAM 带宽主导**：权重驻留片上 SRAM，weight-bound 时间 `= weight_bytes/(sram_bw×TP)`。
3. **MoE expert OI=1**（§6）：小 batch 每 token 独立跑 top_k expert，权重逐 token 重读 → 字节随 `batch×top_k` 线性增长（`expert_mode=per_token`）。
4. **TP AllReduce**（§4.2）：C2C 模型 `diameter×300ns + 2(TP-1)/TP × payload/eff_bw`，`eff_bw` 按 Fig.6a 饱和曲线（32KiB→50%, 80KiB→90%）。
5. **PP 聚合**：`ffn_tpot = num_layers × per_layer`。

## LPU 代次（`--lpu`，可叠加逐字段覆盖）

| 字段 | `lpuv1`（论文 SHIP） | `lpx`（Groq 3 LPX / LP30，下一代） |
|------|------|------|
| `sram_cap_GB` | 0.23 (230 MB) | **0.5 (500 MB)** = 128GB/256 |
| `sram_bw_TBs` | 18.4 | **150** = 40PB/s ÷ 256 |
| `compute_TFLOPs`(FP8) | 188 | **1200** = 315PFLOPS ÷ 256 |
| `c2c_bw_GBs` | 235 (11 ports) | **2500** (96 links×112Gbps) |
| `hop_latency_ns` | 300 | 300 *(LPX 未公布，占位)* |
| `power_W` | 388 | 388 *(LPX 未公布，占位)* |

`lpx` 数据来自 NVIDIA 博客 *Inside NVIDIA Groq 3 LPX*（2026-03，机架/Tray 级聚合值
按 256 chips/机架反推到单颗）。未公布字段（跳延迟/功耗/二分带宽/collective 曲线）
沿用 LPUv1 量级作占位，详见 `lpu_config.py` 的 `LPU_LIBRARY` 注释。出处均在字段注释。

## 用法

```bash
python run.py --model deepseek3 --lpu lpuv1 --tp 16 --batch 1 -o data/deepseek3_tp16_b1.json
python run.py --model deepseek3 --lpu lpx   --tp 16 --batch 1   # 下一代 Groq 3 LPX
python run.py --model gpt-oss-120b --tp 8 --batch 4 --expert-mode batched
# batch sweep
for b in 1 2 4 8 16 32; do
  python run.py --model deepseek3 --batch $b -o data/deepseek3_b$b.json
done
```

## 输出结构

```jsonc
{
  "meta":  { "lpu": {...}, "model": {...}, "run": {...} },   // 全部入参
  "per_layer": {
    "stages": [ {"name","flops","weight_bytes","t_compute_ns","t_mem_ns","t_ns","bound"},
                {"name":"AllReduce","payload_bytes","hops","eff_bw_GBs","t_ns"} ],
    "layer_total_ns": ...,
    "bound_breakdown_ns": {"compute","mem","comm"}
  },
  "decode": { "num_layers", "per_layer_ffn_ns", "ffn_tpot_ns", "ffn_tpot_us",
              "bound_breakdown_per_layer_ns" }
}
```
