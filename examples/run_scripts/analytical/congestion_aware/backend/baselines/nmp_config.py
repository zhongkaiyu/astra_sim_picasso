#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nmp_config.py — Duplex / Helios / Stratum 三条 NMP decode baseline 的「参数暴露层」。

设计同 lpu_ffn_decode/lpu_config.py：只定义参数 + 默认值，不含计算逻辑。
所有数值出处见 ../../figures/rebuttal_figures/exp4_newbaseline/plan/HW_CONFIG.md（标了 paper 页码）。

三条 baseline 在 decode 阶段都是「attention + FFN 同址跑在一块近存设备上」的范式：
  t_layer = t_attn_nmp(CL) + t_ffn_nmp     # 无跨池 NIC xfer，NoC 已并入 comm
和 exp1 的 GPU+LPU / AMMA+LPU（attention 池↔FFN 池每层走 NIC）结构不同。

复用方式（见 nmp_compose.py）：
  - FFN  ：把这里的 NMPSpec 直接喂 lpu_ffn_decode.ffn_decode.simulate_layer（它只读
           compute_TFLOPs / sram_bw_TBs / NoC 字段 / w_bytes，故 NMPSpec = LPUSpec 子集即可）。
  - attn ：把 attn_hw（{compute,Bandwidth,...}）喂 roofline/attention.py（GQA）或
           roofline/deepseek_v3_mla_roofline.py（MLA，fig1 路径）。
"""
from __future__ import annotations
import os
import sys
from dataclasses import dataclass, asdict, field
from typing import Dict, Any

# 复用 lpu_ffn_decode 的 LPUSpec 字段约定（ffn_decode 只读这些字段）
_LPU_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lpu_ffn_decode")
sys.path.insert(0, os.path.abspath(_LPU_DIR))
from lpu_config import LPUSpec  # noqa: E402  复用同一 dataclass 形状


# ===========================================================================
# 1. NMP 设备规格（FFN 用）—— 复用 LPUSpec 的字段，全部 FP16
# ===========================================================================
# 说明：sram_bw_TBs / compute_TFLOPs 是「单设备」值，ffn_decode 会按 run.TP（=device 数）倍乘聚合。
#       三者均 FP16 → w_bytes = act_bytes = 2（区别于 exp1 LPU/Rubin 的 FP8=1）。
#       NoC（hop_latency_ns / bisection_bw_GBs / coll_*）填各自片内拓扑，供 FFN 末尾 AllReduce 用。

def _spec(sram_bw_TBs, compute_TFLOPs, c2c_bw_GBs, hop_latency_ns,
          bisection_bw_GBs, sram_cap_GB) -> LPUSpec:
    return LPUSpec(
        sram_bw_TBs=sram_bw_TBs,
        sram_cap_GB=sram_cap_GB,
        compute_TFLOPs=compute_TFLOPs,
        c2c_bw_GBs=c2c_bw_GBs,
        hop_latency_ns=hop_latency_ns,
        bisection_bw_GBs=bisection_bw_GBs,
        coll_knee_kib=(32.0, 80.0),       # 沿用 collective 饱和曲线形状（占位）
        coll_knee_frac=(0.50, 0.90),
        w_bytes=2,                        # FP16
        act_bytes=2,                      # FP16
    )


# --- Stratum tiering：决策 D「按命中率加权」---------------------------------
# Mono3D 内部带宽 19–30 TB/s（slow~19, fast~30 ≈ 1.6×，Stratum Table 1 / §6.2.1）。
# topic 预测把热专家放 fast tier，命中率 p_hot（LLaMA-4 Scout >90% → 默认 0.9，Stratum §3 行488）。
# 只有「被路由的 expert 读取」享受 tiering；attention/gating/shared 不享受。
# 工程近似（不改 ffn_decode）：MoE FFN 字节由 expert 主导(>95%)，故把整段 FFN 有效带宽按命中率加权：
#     eff_bw = 1 / ( p_hot / BW_fast + (1-p_hot) / BW_slow )
# attention 不预测 → 用代表值 BW_attn（取中位 ~25）。
STRATUM_BW_FAST = 30.0
STRATUM_BW_SLOW = 19.0
STRATUM_P_HOT_DEFAULT = 0.9
STRATUM_BW_ATTN = 25.0


def stratum_ffn_eff_bw(p_hot: float = STRATUM_P_HOT_DEFAULT,
                       bw_fast: float = STRATUM_BW_FAST,
                       bw_slow: float = STRATUM_BW_SLOW) -> float:
    """tiering 命中率加权后的 expert-read 有效带宽（TB/s）。"""
    return 1.0 / (p_hot / bw_fast + (1.0 - p_hot) / bw_slow)


# NMP 设备库（FFN 用）。Stratum 的 sram_bw 在 get_nmp() 里按 p_hot 现算。
# ⚠️ hop_latency_ns = **node 内 device↔device** 链路的单跳固定延迟（TP collective 用）。
#   之前误填 5ns（那是 on-die NoC），但 TP=8 的 AllReduce 是**跨 device 走 NVLink/cross-chip**，
#   实测应是数百 ns（NVLink ~600ns、cross-chip ~300ns），不是 5ns。已校准（用户 #3 指正）。
#   c2c_bw = node 内链路带宽：Duplex 900 GB/s NVLink、Helios 600 GB/s NVLink-fusion、Stratum 2500 GB/s cross-chip。
def _nmp_library() -> Dict[str, LPUSpec]:
    return {
        # Duplex Logic-PIM：13.4 TB/s（4×HBM3），107 TFLOPS（5 stack×21.3，8:1），80 GB。node 内 900 GB/s NVLink（HGX）
        "duplex": _spec(sram_bw_TBs=13.4, compute_TFLOPs=107.0, c2c_bw_GBs=900.0,
                        hop_latency_ns=600.0, bisection_bw_GBs=5000.0, sram_cap_GB=80.0),
        # Helios HB-Device：16.4 TB/s，262 TFLOPS（16 PE×512×16-MAC@1GHz，面积反推），80 GB。node 内 600 GB/s NVLink-fusion
        "helios": _spec(sram_bw_TBs=16.4, compute_TFLOPs=262.0, c2c_bw_GBs=600.0,
                        hop_latency_ns=600.0, bisection_bw_GBs=3000.0, sram_cap_GB=80.0),
        # Stratum Mono3D：FFN 带宽 = tiering 加权(见 get_nmp)，128 TFLOPS，32 GB/chip。node 内 cross-chip（更快，~300ns）
        "stratum": _spec(sram_bw_TBs=stratum_ffn_eff_bw(), compute_TFLOPs=128.0,
                         c2c_bw_GBs=2500.0, hop_latency_ns=300.0,
                         bisection_bw_GBs=2500.0, sram_cap_GB=32.0),
    }


def get_nmp(name: str, p_hot: float = STRATUM_P_HOT_DEFAULT) -> LPUSpec:
    """取某条 baseline 的 NMP FFN 规格副本。Stratum 按 p_hot 重算 tiering 有效带宽。"""
    lib = _nmp_library()
    if name not in lib:
        raise KeyError(f"未知 NMP baseline '{name}'，可选: {list(lib)}")
    spec = LPUSpec(**asdict(lib[name]))
    if name == "stratum":
        spec.sram_bw_TBs = stratum_ffn_eff_bw(p_hot)
    return spec


# ===========================================================================
# 2. attention 硬件（attention roofline 用，{compute,Bandwidth,...} dict，仿 hardware_config.py）
# ===========================================================================
# Bandwidth = 单设备 mem 带宽 (TB/s)；compute = 单设备峰值 (TFLOPS, §3 面积/功耗反推的真实值)。
# TP=device 数靠 head 切分聚合，但 latency-relevant TP 受 node 上限封顶（见 §3 NODE_SIZE）。
# Stratum attention 不享受 tiering → 用代表值 BW_attn(~25)。
NMP_ATTN_HW: Dict[str, Dict[str, Any]] = {
    "duplex":  {"compute": 107.0, "Bandwidth": 13.4, "capacity": 80, "power": 1000,
                "num_devices": 1, "device_link_bw": 0.9, "kv_bytes_per_element": 2},
    "helios":  {"compute": 262.0, "Bandwidth": 16.4, "capacity": 80, "power": 1000,
                "num_devices": 1, "device_link_bw": 0.6, "kv_bytes_per_element": 2},
    "stratum": {"compute": 128.0, "Bandwidth": STRATUM_BW_ATTN, "capacity": 32, "power": 1000,
                "num_devices": 1, "device_link_bw": 2.5, "kv_bytes_per_element": 2},
}


def get_attn_hw(name: str) -> Dict[str, Any]:
    if name not in NMP_ATTN_HW:
        raise KeyError(f"未知 NMP baseline '{name}'，可选: {list(NMP_ATTN_HW)}")
    return dict(NMP_ATTN_HW[name])


# ===========================================================================
# 3. 每条 baseline 的并行 / NoC 默认值（决策 C：各用 paper 原生 device 数 = TP）
# ===========================================================================
@dataclass
class NMPRunDefaults:
    tp: int                 # 容量 device 数（= 装下整模需要的总颗数，N）
    tp_lat: int             # latency-relevant TP（= min(N, NODE)）—— 真正进 roofline 的并行宽度
    tp_h: int               # attention head-group 切分（由 tp_lat 分解）
    tp_hd: int              # attention head-dim 切分（tp_h×tp_hd = tp_lat）
    diameter_hops: int      # node 内 NoC 直径（within-node AllReduce 固定延迟项）
    noc: str                # 拓扑名（仅标注）
    num_nodes: int = 1      # ceil(N / NODE)：>1 时 expert 走跨 node EP，付 NIC
    nic_bw_GBs: float = 400.0   # 跨 node link 带宽（Duplex IB / Helios NIC / Stratum cross-chip）
    nic_hop_ns: float = 1500.0  # 跨 node 单跳固定延迟（RDMA/IB 量级）
    dtype: str = "fp16"


# --- 决策 C（修订）：设备数 = max(paper 规则, 容量可行)，逐模型 -------------
# 起因：原来 Duplex=4 是 Mixtral(47B) 那档、对 235B/671B 偏少（且容量装不下），
#       使 Duplex 被人为配成"最差"。改为按各 paper 自己的规则 + 容量下限。
#
# 权重容量 (FP16, 2B/param)：qwen3-235B ≈ 470 GB；deepseek3-671B ≈ 1342 GB。
# 单设备容量：Duplex/Helios 80 GB，Stratum 32 GB/chip。
# paper 规则：
#   - Duplex §VI: 按参数量 Mixtral47B→4 / GLaM143B→8 / Grok1-314B→16 → 235B≈8, 671B≈16
#   - Helios §VI-A: "All instances are set to TP=8"（明确）
#   - Stratum §3.1: Stratum-L = H100 + 6 Mono3D chip（但 eval 仅覆盖 ≤47B；大模型按容量上 XL+）
# 取 N = max(paper, ceil(weight/cap))：
MODEL_WEIGHT_GB = {"qwen3-235b": 470.0, "deepseek3": 1342.0}
DEVICE_CAP_GB = {"duplex": 80.0, "helios": 80.0, "stratum": 32.0}

DEVICE_COUNT: Dict[str, Dict[str, int]] = {
    # Duplex: qwen3 max(param8, cap6)=8; ds3 max(param16, cap17)=18
    "duplex":  {"qwen3-235b": 8,  "deepseek3": 18},
    # Helios: qwen3 max(TP8, cap6)=8; ds3 max(TP8, cap17)=18（TP=8 装不下 ds3 → 容量主导）
    "helios":  {"qwen3-235b": 8,  "deepseek3": 18},
    # Stratum: qwen3 max(L6, cap15)=15; ds3 max(L6, cap42)=44（32GB 小芯片 → 需很多颗）
    "stratum": {"qwen3-235b": 15, "deepseek3": 44},
}

# --- 决策 C-2（新增）：单 token latency 的 TP 受 scale-up node 上限封顶 -------------
# 关键：N（上面）是「装下整模需要的颗数」（容量），但单 token decode 的 latency 只能用
#       一个 scale-up node 内的 TP；超出 node 的设备是 EP/DP（容量+吞吐，不缩短单 token latency）。
#   - Duplex §786-790：node 内 TP（非专家）、node 间 DP、expert 用 EP → node = 8-device NVLink（行 848）。
#   - Helios §2014   ："TP=8 (+EP=8)"，8-device node → node = 8。
#   - Stratum §1137  ：PU-TP(片内 16 PU 已并入 28.4 per-chip BW) + grouped-PU head 注意力，
#                      多 chip 经 cross-chip interconnect scale-up → 取同口径 node = 8 chip（保守，可调）。
# latency-relevant TP = min(N, NODE)。超出部分 = ceil(N/NODE) 个 node，expert 走跨 node EP（付 NIC）。
NODE_SIZE = {"duplex": 8, "helios": 8, "stratum": 8}

# 跨 node link（>8 卡后带宽改变：NVLink → NIC/IB/cross-chip）。出处见 HW_CONFIG §6。
# hop_ns = 单次跨 node 消息固定延迟。校准到 exp1 interconnect.py 的实测 RDMA 量级：
#   nic_cx7(ConnectX-7 NDR 400Gb/s, GPUDirect RDMA 小包) = 2000 ns；nic_cx8(XDR 800Gb/s) = 1800 ns。
#   Duplex 400 GB/s InfiniBand、Helios 8×200 Gbps NIC → 都取 2000 ns（IB/RDMA 小包）。
#   Stratum cross-chip interconnect 是 NVLink-class（非 NIC）→ 取 ~700 ns（sub-µs，仍远大于片内）。
CROSS_NODE_NIC = {
    "duplex":  {"bw_GBs": 400.0,  "hop_ns": 2000.0},   # 400 GB/s InfiniBand（行 850-852），RDMA 小包 ~2µs
    "helios":  {"bw_GBs": 200.0,  "hop_ns": 2000.0},   # 8×200 Gbps NIC = 200 GB/s（行 2021-2022）
    "stratum": {"bw_GBs": 900.0,  "hop_ns": 700.0},    # cross-chip interconnect（NVLink-class，行 562/473）
}
# NoC 拓扑（diameter 是 N 的函数；NoC <1% 影响，量级即可）
NOC_TOPO = {"duplex": "bank-bundle", "helios": "mesh", "stratum": "ring-bidir"}

# --- per-device 功耗（W）：throughput/W 图用。⚠️ 口径不齐，详见 HW_CONFIG §8 ---
#   Stratum 145 = 整栈峰值含 DRAM（paper L1540，最干净）；
#   Helios  207 = Fig.16b per-PE 12.94W ×16 PE/device（SRAM1.25+Matrix3.44+Vec1.73+NoC0.04+Ctrl1.07+DRAM5.41）；
#   Duplex  ~150 = 估算（5×HBM3 stack + Logic-PIM，paper 未给绝对功耗，仅相对能耗 −42%）。标 estimated。
DEVICE_POWER_W = {"duplex": 150.0, "helios": 207.0, "stratum": 145.0}
DEVICE_POWER_SRC = {"duplex": "estimated (no paper abs)", "helios": "Fig.16b ×16PE",
                    "stratum": "L1540 full-stack peak"}


def _gqa_split(N: int):
    """把设备数 N 分解成 (tp_h, tp_hd)：tp_h 取 ≤4 的最大因子（GQA Hkv 上限），其余给 tp_hd。
    MLA 路径不用这两个（只用总 N）；GQA attention 经验上随总 N≈线性扩展（已实测）。"""
    tp_h = max((d for d in (4, 3, 2, 1) if N % d == 0), default=1)
    return tp_h, max(1, N // tp_h)


def _diameter(name: str, n_within: int) -> int:
    """node 内 device↔device 网络直径（TP collective 的固定延迟跳数）。
    注意：这是**跨 device** 的 scale-up 网络（NVLink/cross-chip），不是 on-die NoC。
    Duplex/Helios 走 NVSwitch/NVLink-fusion 类**交换式** scale-up → 近全连接，直径 1；
    Stratum 多 chip cross-chip ring → 双向 ring，直径 ≈ n/2（但 n_within=8 → 取 2 量级，避免过惩罚）。"""
    if name == "stratum":
        return max(2, n_within // 4)        # 双向 ring（8 chip → 2）
    return 1                                # NVSwitch / NVLink-fusion 交换式，单跳


def get_run_defaults(name: str, model_name: str) -> NMPRunDefaults:
    if name not in DEVICE_COUNT:
        raise KeyError(f"未知 NMP baseline '{name}'，可选: {list(DEVICE_COUNT)}")
    if model_name not in DEVICE_COUNT[name]:
        raise KeyError(f"模型 '{model_name}' 无设备数配置，可选: {list(DEVICE_COUNT[name])}")
    N = DEVICE_COUNT[name][model_name]
    node = NODE_SIZE[name]
    tp_lat = min(N, node)                   # 单 token latency 的 TP（node 封顶）
    num_nodes = -(-N // node)               # ceil(N/node)：超出 node 的设备走 EP/DP
    tp_h, tp_hd = _gqa_split(tp_lat)        # head 切分按 tp_lat（不是 N）
    nic = CROSS_NODE_NIC[name]
    return NMPRunDefaults(tp=N, tp_lat=tp_lat, tp_h=tp_h, tp_hd=tp_hd,
                          diameter_hops=_diameter(name, tp_lat), noc=NOC_TOPO[name],
                          num_nodes=num_nodes, nic_bw_GBs=nic["bw_GBs"],
                          nic_hop_ns=nic["hop_ns"])


# 人类可读标签（绘图/HTML 用）
NMP_LABELS = {"duplex": "Duplex", "helios": "Helios", "stratum": "Stratum"}
NMP_MEM_TECH = {
    "duplex":  "HBM3 + 4×TSV (Logic-PIM)",
    "helios":  "Hybrid-Bonding 4-die 3D-DRAM",
    "stratum": "Monolithic 3D DRAM (tiered)",
}
