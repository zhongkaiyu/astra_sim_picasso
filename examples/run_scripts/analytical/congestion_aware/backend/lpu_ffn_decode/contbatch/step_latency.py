#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step_latency.py — system 层的「单步延迟」供给。

把 exp1 的三个 composer 封装成一个统一接口：
    t_step_ns(setting, B, mean_seq)
给定一个 decode step 里 B 条 request 的（平均）上下文长，返回这一步（每条 request
各吐 1 个 token）的【整模 TPOT 延迟】(ns)。这正是 continuous-batch 调度器每一步要的量。

============================================================================
复用 exp1（backend/lpu_ffn_decode/，零改动）
----------------------------------------------------------------------------
  - "amma_lpu" -> decode_compose.compose_decode      (AMMA attn + LPU FFN, ours)
  - "gpu_gpu"  -> baseline_compose.compose_decode     (Rubin attn + Rubin FFN)
  - "gpu_lpu"  -> baseline_compose.compose_decode     (Rubin attn + LPU  FFN)
三者的 per-step 延迟 = num_layers × (t_attn(B,seq) + 2·t_xfer(B) + t_ffn(B))。

异构 CL 的处理（关键近似）
----------------------------------------------------------------------------
一个 step 里 B 条 request 的上下文长各不相同 {CL_i}。roofline attention 的
KV-read 项对 (B×seq) 线性（KV 字节 ∝ ΣCL_i），而 projection / FFN / xfer 只依赖 B。
故调 composer 时取 batch=B, seq=mean(CL_i)：ΣCL_i 被精确还原、权重摊销正确，
唯一近似是 max(compute,mem) 拐点在均值处而非逐 request 评估。

为避免百万步实时调用，预计算 (B, seq) 表：B 取整数 1..b_max，seq 取 log 网格，
查询时 B 直接索引、seq 在 log 轴线性插值。一次建表 ~ b_max×|seq_grid|×3 次 compose，
全部是纯算术，毫秒级。
============================================================================
"""
from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, _BACKEND)

from lpu_config import RunSpec, get_model, get_lpu          # noqa: E402
from interconnect import get_link                            # noqa: E402
import decode_compose as dc                                  # noqa: E402
import baseline_compose as bc                                # noqa: E402
import rubin_ffn_decode as rf                                # noqa: E402

SETTINGS = ["gpu_gpu", "gpu_lpu", "amma_lpu", "amma_lpu_ideal",
            "amma_lpu_ec", "amma_lpu_ideal_ec"]
SETTING_LABEL = {
    "gpu_gpu":  "GPU+GPU",
    "gpu_lpu":  "GPU+LPU",
    "amma_lpu": "AMMA+LPU (ours)",
    "amma_lpu_ideal": "AMMA+LPU (ideal)",
    "amma_lpu_ec": "AMMA+LPU (EC)",
    "amma_lpu_ideal_ec": "AMMA+LPU (EC, ideal)",
}

# 与 exp1 figures/gen_baseline_data.py 完全一致的并行宽度配置。
#   attention TP = tp_h × tp_hd ; FFN TP = tp_ffn
#   "link": None → 用 table 的默认链路；否则覆盖（amma_lpu_ideal 走理想链路 nic_ideal）。
# amma_lpu_ideal = 与 amma_lpu 完全相同（16×AMMA-NPU attn + 16×LPU FFN），唯一差别是
# 跨池链路换成 nic_ideal(300ns/235GB/s)，隔离 disaggregation 的 NIC xfer 惩罚（紧耦合上界）。
# npu_compute: 每 NPU 算力 (TFLOPS)。96 = "Ours"；384 = "Ours_EC"(Enhanced-Compute ×4)。
SETTING_CFG = {
    "gpu_gpu":           {"tp_h": 1, "tp_hd": 1, "tp_ffn": 1,  "attn_hw": "1×Rubin",     "ffn_hw": "1×Rubin", "link": None},
    "gpu_lpu":           {"tp_h": 1, "tp_hd": 1, "tp_ffn": 16, "attn_hw": "1×Rubin",     "ffn_hw": "16×LPU",  "link": None},
    "amma_lpu":          {"tp_h": 4, "tp_hd": 4, "tp_ffn": 16, "attn_hw": "16×AMMA-NPU",      "ffn_hw": "16×LPU", "link": None,       "npu_compute": 96.0},
    "amma_lpu_ideal":    {"tp_h": 4, "tp_hd": 4, "tp_ffn": 16, "attn_hw": "16×AMMA-NPU",      "ffn_hw": "16×LPU", "link": "nic_ideal","npu_compute": 96.0},
    "amma_lpu_ec":       {"tp_h": 4, "tp_hd": 4, "tp_ffn": 16, "attn_hw": "16×AMMA-NPU (EC)", "ffn_hw": "16×LPU", "link": None,       "npu_compute": 384.0},
    "amma_lpu_ideal_ec": {"tp_h": 4, "tp_hd": 4, "tp_ffn": 16, "attn_hw": "16×AMMA-NPU (EC)", "ffn_hw": "16×LPU", "link": "nic_ideal","npu_compute": 384.0},
}

# 路由到对应 composer 时，把 ideal/EC 都当作 amma（只是 link/算力不同）。
_COMPOSER_KIND = {"gpu_gpu": "gpu", "gpu_lpu": "gpu",
                  "amma_lpu": "amma", "amma_lpu_ideal": "amma",
                  "amma_lpu_ec": "amma", "amma_lpu_ideal_ec": "amma"}

# decode 时上下文长的 log 网格（覆盖 prompt 64K 量级 + 长尾到 ~1.2M）。
DEFAULT_SEQ_GRID = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
                    131072, 262144, 524288, 1048576, 1200000]


def _compose_tpot_ns(setting: str, model, lpu, gpu, util, link,
                     B: int, seq: int) -> float:
    """调对应 composer，返回该 (setting,B,seq) 的单步整模 TPOT(ns)。"""
    cfg = SETTING_CFG[setting]
    # trace：MoE FFN 权重读 = 真实 distinct 激活专家数（由 expert-selection trace 标定，
    # 见 contbatch/data/expert_activation.json），并在 decode 区强制 memory-bound。
    # 比解析上界 min(B·top_k, n_experts) 低很多（专家选择有重合/偏斜），更贴近主流观测。
    run = RunSpec(TP=cfg["tp_ffn"], batch=B, expert_mode="trace")
    kind = _COMPOSER_KIND[setting]
    if kind == "amma":      # amma_lpu / _ideal / _ec：同 composer，仅 link 与 npu_compute 不同
        r = dc.compose_decode(model, lpu, run, link, B, seq,
                              cfg["tp_h"], cfg["tp_hd"],
                              npu_compute=cfg.get("npu_compute", 96.0))
    elif kind == "gpu":     # gpu_gpu / gpu_lpu
        r = bc.compose_decode(setting, model, run, link, B, seq,
                              cfg["tp_h"], cfg["tp_hd"],
                              lpu=lpu, gpu=gpu, util=util)
    else:
        raise ValueError(setting)
    return r["decode"]["tpot_ns"]


class StepLatencyTable:
    """三种 setting 的 (B, seq) → 单步 TPOT(ns) 查找表 + log 插值。"""

    def __init__(self, model_key: str = "deepseek3", link: str = "nic_cx7",
                 lpu_name: str = "lpx", b_max: int = 32,
                 settings: List[str] = None, seq_grid: List[int] = None):
        self.model_key = model_key
        self.link_name = link
        self.lpu_name = lpu_name
        self.b_max = int(b_max)
        self.settings = list(settings) if settings else list(SETTINGS)
        self.seq_grid = sorted(seq_grid) if seq_grid else list(DEFAULT_SEQ_GRID)
        self._log_seq = [math.log(s) for s in self.seq_grid]

        model = get_model(model_key)
        self.num_layers = model.num_layers
        lpu = get_lpu(lpu_name)
        gpu = rf.RUBIN

        # table[setting][B] = list aligned with seq_grid，存 tpot_ns
        self.table: Dict[str, Dict[int, List[float]]] = {}
        self.setting_link: Dict[str, str] = {}
        for s in self.settings:
            link_name = SETTING_CFG[s].get("link") or self.link_name   # 每 setting 自己的链路
            self.setting_link[s] = link_name
            link_spec = get_link(link_name)
            util = rf.FFNUtil(model_name=model_key) if _COMPOSER_KIND[s] == "gpu" else None
            self.table[s] = {}
            for B in range(1, self.b_max + 1):
                self.table[s][B] = [
                    _compose_tpot_ns(s, model, lpu, gpu, util, link_spec, B, sq)
                    for sq in self.seq_grid
                ]

    def t_step_ns(self, setting: str, B: int, mean_seq: float) -> float:
        """单步整模 TPOT(ns)。B 取整数索引；seq 在 log 轴分段线性插值。"""
        B = max(1, min(int(round(B)), self.b_max))
        row = self.table[setting][B]
        x = math.log(max(mean_seq, self.seq_grid[0]))
        xs = self._log_seq
        if x <= xs[0]:
            return row[0]
        if x >= xs[-1]:
            return row[-1]
        # 线性扫描（网格很短）
        for i in range(len(xs) - 1):
            if xs[i] <= x <= xs[i + 1]:
                t = (x - xs[i]) / (xs[i + 1] - xs[i])
                return row[i] + t * (row[i + 1] - row[i])
        return row[-1]

    def grid_tpot_us(self, setting: str, B: int) -> List[Tuple[int, float]]:
        """返回某 setting/B 在 seq_grid 上的 (seq, TPOT_us)，给画图/报告用。"""
        return [(sq, self.table[setting][B][i] / 1e3)
                for i, sq in enumerate(self.seq_grid)]

    def meta(self) -> dict:
        return {
            "model": self.model_key, "link": self.link_name, "lpu": self.lpu_name,
            "b_max": self.b_max, "num_layers": self.num_layers,
            "settings": self.settings, "setting_cfg": SETTING_CFG,
            "setting_link": getattr(self, "setting_link", {}),
            "seq_grid": self.seq_grid,
        }
