#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lpu_config.py — LPU (Groq SHIP) 硬件规格 + 模型 FFN 规格的「参数暴露层」。

本文件只负责【定义参数 + 默认值】，不含任何计算逻辑，方便单独调参。
所有 LPU 参数都来自论文 Bitar et al., "SHIP: SRAM-based Huge Inference
Pipelines for Fast LLM Serving" (MLSys'26)，相关出处在每个字段后标注。

设计原则（与现有 backend 解耦）：
  - 不 import roofline/ 下任何文件；
  - 只用 dataclass 暴露字段，calc 逻辑放在 ffn_decode.py；
  - 默认值取论文 LPUv1（GlobalFoundries 14nm），可被 CLI / config 覆盖。
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any


# ===========================================================================
# 1. LPU 硬件规格（单颗 LPU 视角）
# ===========================================================================
@dataclass
class LPUSpec:
    """
    单颗 LPU 的硬件参数。聚合带宽/算力在 ffn_decode 里按 TP 倍乘。

    默认值为 LPUv1（论文 SHIP 所建模的代次，GF 14nm）。
    下一代 Groq 3 LPX（LP30，NVIDIA Vera Rubin 平台）参数见 LPU_LIBRARY["lpx"]，
    用 `--lpu lpx` 选择，或 get_lpu("lpx")。
    """

    # --- 片上 SRAM（decode 权重 & KV 都驻留在这里，是 FFN decode 的带宽瓶颈） ---
    sram_bw_TBs: float = 18.4
    # 论文 §7.1: LPUv1 内置 230 MB SRAM。Table 2 给出 gpt-oss-120B 典型实例
    # （72 节点 × 8 = 576 LPU）的聚合 SRAM 带宽 10,616 TB/s（已扣权重份额）。
    # 单颗 = 10616 / 576 ≈ 18.4 TB/s。这是「带宽共享」的有效值，非峰值。
    sram_cap_GB: float = 0.23          # 230 MB 片上 SRAM（论文 §7.1）

    # --- 算力（decode FFN 一般 memory-bound，compute 很少成为瓶颈，但仍暴露） ---
    compute_TFLOPs: float = 188.0
    # LPUv1 GF14nm FP16 峰值算力（Groq 公开口径量级，可调）。论文未直接给出，
    # 仅说明 LPU 的 mem:compute 带宽比约为 GPU 的 10×（§3.2 / §4），
    # 对 decode FFN 该值通常不触发 roofline 的 compute 边，按需覆盖。

    # --- C2C 片间互联（TP collective 的成本来源，QuadFour 拓扑） ---
    c2c_bw_GBs: float = 235.0          # 单颗聚合 C2C I/O 带宽（论文 §7.1，11 个逻辑端口）
    c2c_num_ports: int = 11            # 逻辑端口数（论文 §7.1）
    hop_latency_ns: float = 300.0      # 每跳 LPU→LPU 同步传输延迟（论文 §4.1/§4.2）
    bisection_bw_GBs: float = 853.0    # 72-LPU TP partition 的二分带宽（论文 §4.1）

    # --- Collective 带宽饱和曲线（小张量 AllReduce 的有效带宽随 payload 上升） ---
    # 论文 Fig.6a: AllReduce 在 32KiB 达 50% 饱和，80KiB 达 90% 饱和。
    # 用两个拐点做分段线性插值，估算有效带宽占峰值的比例。
    coll_knee_kib: tuple = (32.0, 80.0)     # 拐点处的张量大小（KiB）
    coll_knee_frac: tuple = (0.50, 0.90)    # 对应的带宽饱和比例

    # --- 功耗 ---
    power_W: float = 388.0             # 每颗 LPU 的 provisioned power（论文 §7.1.1）

    # --- 数据类型 ---
    w_bytes: int = 1                   # 权重字节数；FP8=1, FP16=2（SHIP 默认 FP8）
    act_bytes: int = 1                 # 激活字节数（C2C payload 用）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# LPU 代次预置库
#   - "lpuv1": 论文 SHIP 所建模的代次（= LPUSpec 默认值）。
#   - "lpx"  : 下一代 Groq 3 LPX（LP30），数据来自 NVIDIA 博客
#              "Inside NVIDIA Groq 3 LPX ... for the NVIDIA Vera Rubin Platform"
#              (developer.nvidia.com, 2026-03)。博客给的是机架/Tray 级聚合值，
#              单颗(LP30)参数按 256 chips/机架、8 chips/tray 反推，已在注释标注换算。
# ---------------------------------------------------------------------------
LPU_LIBRARY: Dict[str, "LPUSpec"] = {
    "lpuv1": LPUSpec(),   # 即上面的默认 LPUv1

    "lpx": LPUSpec(
        # --- SRAM ---
        sram_bw_TBs=150.0,      # 单颗 150 TB/s（机架 40 PB/s ÷ 256；Tray 1.2 PB/s ÷ 8）
        sram_cap_GB=0.5,        # 单颗 500 MB（机架 128 GB ÷ 256；Tray 4 GB ÷ 8）
        # --- 算力 ---
        compute_TFLOPs=1200.0,  # 单颗 1.2 PFLOP/s FP8（机架 315 PFLOPS ÷ 256；Tray 9.6 PFLOPS ÷ 8）
        # --- C2C 互联 ---
        c2c_bw_GBs=2500.0,      # 单颗 2.5 TB/s 聚合双向（96 links × 112 Gbps；Tray scale-up 20 TB/s ÷ 8）
        c2c_num_ports=96,       # 96 条 C2C link/颗（112 Gbps/link）
        hop_latency_ns=300.0,   # 博客未公布 LPX 跳延迟，沿用 LPUv1 同步 C2C 量级（占位，可调）
        bisection_bw_GBs=2500.0,# 博客未单列 TP partition 二分带宽，暂取单颗聚合 C2C（占位，可调）
        # --- Collective 饱和曲线：LPX 未公布，沿用 LPUv1 Fig.6a 形状（占位） ---
        coll_knee_kib=(32.0, 80.0),
        coll_knee_frac=(0.50, 0.90),
        # --- 功耗：博客未给单颗功耗（仅机架级液冷），占位沿用 LPUv1 量级，按需覆盖 ---
        power_W=388.0,
        # --- 数据类型：博客明确 FP8 ---
        w_bytes=1,
        act_bytes=1,
    ),
}


def get_lpu(name: str) -> "LPUSpec":
    """按代次名取 LPU 规格副本；未命中则报错并列出可选项。"""
    if name not in LPU_LIBRARY:
        raise KeyError(f"未知 LPU 代次 '{name}'，可选: {list(LPU_LIBRARY)}")
    return LPUSpec(**LPU_LIBRARY[name].to_dict())


# ===========================================================================
# 2. 模型 FFN 规格
# ===========================================================================
@dataclass
class FFNModelSpec:
    """
    单层 FFN 的结构参数。dense 与 MoE 用 `is_moe` 切换：
      - dense (SwiGLU 门控)：gate / up / down 三个矩阵，宽度 d_ff；
      - MoE：先 gating(router)，再每 token 选 top_k 个 expert，每 expert 三矩阵，
              宽度 d_moe_inter；可选 shared expert。
    """
    name: str = "deepseek3"

    d_model: int = 7168                # 隐藏维度
    num_layers: int = 61               # 总层数（用于把单层 FFN 累加成整模 TPOT）

    is_moe: bool = True
    gated: bool = True                 # SwiGLU 门控 → 三矩阵(gate/up/down)；否则两矩阵

    # --- dense FFN 专用 ---
    d_ff: int = 18432                  # dense 中间维度（is_moe=False 时使用）

    # --- MoE 专用 ---
    n_experts: int = 256               # 路由 expert 总数
    top_k: int = 8                     # 每 token 激活的 expert 数
    d_moe_inter: int = 2048            # 每个 expert 的中间维度
    n_shared_experts: int = 1          # 共享 expert 数（每 token 必走，0 表示无）

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 3. 并行 / 运行配置
# ===========================================================================
@dataclass
class RunSpec:
    """并行切分与 batch 设置。"""
    TP: int = 16                       # tensor-parallel 切分数（FFN 沿中间维切）
    PP: int = 1                        # pipeline-parallel 级数（仅用于报告聚合，不改单层延迟）
    batch: int = 1                     # decode batch（SRAM 推理偏好小 batch）
    tp_diameter_hops: int = 3          # TP partition 网络直径（论文 §4.1: 72-LPU→3 跳）

    # MoE expert 执行方式：
    #   "per_token" —— 小 batch 下每 token 独立跑其 expert，权重逐 token 重读，
    #                  expert MatMul 的 OI=1（论文 §6 Expert Imbalance 明确指出）；
    #   "batched"   —— 假设 batch 内 token 复用同一 expert 权重（大 batch 上界）。
    expert_mode: str = "per_token"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ===========================================================================
# 4. 预置模型库（可按需扩充；数值为公开规格，FFN 相关字段为主）
# ===========================================================================
MODEL_LIBRARY: Dict[str, FFNModelSpec] = {
    # DeepSeek-V3：细粒度 MoE，256 路由 expert + 1 共享，top-8，moe_inter=2048
    "deepseek3": FFNModelSpec(
        name="deepseek3", d_model=7168, num_layers=61, is_moe=True, gated=True,
        n_experts=256, top_k=8, d_moe_inter=2048, n_shared_experts=1,
    ),
    # DeepSeek-V4-Pro：FFN 结构沿用 V3 细粒度 MoE（报告 §4.2.1，d=7168/61 层），
    # attention 换成 CSA / HCA（roofline/roofline_v4_calc.py）。两条只在 attention 不同，
    # FFN/xfer 列与 deepseek3 完全一致，便于隔离 attention 的影响。
    "deepseek-v4-csa": FFNModelSpec(
        name="deepseek-v4-csa", d_model=7168, num_layers=61, is_moe=True, gated=True,
        n_experts=256, top_k=8, d_moe_inter=2048, n_shared_experts=1,
    ),
    "deepseek-v4-hca": FFNModelSpec(
        name="deepseek-v4-hca", d_model=7168, num_layers=61, is_moe=True, gated=True,
        n_experts=256, top_k=8, d_moe_inter=2048, n_shared_experts=1,
    ),
    # gpt-oss-120B：论文 Fig.3b / Table 2 的参考模型
    "gpt-oss-120b": FFNModelSpec(
        name="gpt-oss-120b", d_model=2880, num_layers=36, is_moe=True, gated=True,
        n_experts=128, top_k=4, d_moe_inter=2880, n_shared_experts=0,
    ),
    # Qwen3-235B-A22B：论文 §6.1 的评测模型
    "qwen3-235b": FFNModelSpec(
        name="qwen3-235b", d_model=4096, num_layers=94, is_moe=True, gated=True,
        n_experts=128, top_k=8, d_moe_inter=1536, n_shared_experts=0,
    ),
    # 一个 dense 对照（Llama3.3-70B 量级）
    "llama3-70b": FFNModelSpec(
        name="llama3-70b", d_model=8192, num_layers=80, is_moe=False, gated=True,
        d_ff=28672,
    ),
}


def get_model(name: str) -> FFNModelSpec:
    """按名取预置模型；未命中则报错并列出可选项。"""
    if name not in MODEL_LIBRARY:
        raise KeyError(f"未知模型 '{name}'，可选: {list(MODEL_LIBRARY)}")
    # 返回副本，避免调用方修改全局库
    return FFNModelSpec(**MODEL_LIBRARY[name].to_dict())
