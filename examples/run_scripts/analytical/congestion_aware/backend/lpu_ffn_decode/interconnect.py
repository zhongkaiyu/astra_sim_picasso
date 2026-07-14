#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
interconnect.py — 解耦架构下「跨池 activation 传输」的延迟模型。

背景（attention-FFN disaggregation，论文 §7.3）：
  attention 跑在 AMMA 池、FFN 跑在 LPU 池，是两套物理不同的硬件。
  每个 decode layer 内，hidden state 必须经线缆在两池间往返一次：
      ① AMMA → LPU（attention 输出送去算 FFN）
      ② LPU → AMMA（FFN 输出送回，供下一层 attention）
  这两次传输是【串行】插在 t_attn 与 t_ffn 之间的真实开销。

本文件只建模这一项，与 ffn_decode / attention 完全解耦。
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class LinkSpec:
    """
    AMMA↔LPU 跨池互联链路参数。默认取 AMMA 论文的 UCIe 3.0 D2D 链路
    （AMMA 自身片内 cube↔cube 互联），是与该架构最一致的选择。

    数据出处：Yu, Ye et al., "AMMA: A Multi-Chiplet Memory-Centric
    Architecture for Low-Latency 1M Context Attention Serving" 硬件配置表：
      - D2D UCIe 3.0  : BW=1500 GB/s, latency=15 ns/hop（adapter 4 + PHY 10 + 传播 1）
      - C2C NVLink    : Rubin BW=3600 GB/s / H100 BW=900 GB/s, latency=900 ns
    论文 §7.6 指出：低 batch(1–32) 下传输量小，延迟由【固定启动延迟】主导，
    故 latency 比 bandwidth 更关键 —— 选链路时优先看 latency。
    """
    interpool_bw_GBs: float = 1500.0      # 单向带宽 GB/s（UCIe 3.0 D2D）
    interpool_latency_ns: float = 15.0    # 单次传输固定延迟 ns（UCIe 3.0 D2D）
    name: str = "ucie3_d2d"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# AMMA 论文里的真实链路预置（用 get_link 选择）：
LINK_LIBRARY: Dict[str, "LinkSpec"] = {
    # AMMA 片内 D2D（默认，与 AMMA 架构一致；若 AMMA 与 LPU 同封装则最贴切）
    "ucie3_d2d":  LinkSpec(interpool_bw_GBs=1500.0, interpool_latency_ns=15.0,  name="ucie3_d2d"),
    # 芯片间 NVLink（若 AMMA 与 LPU 是分立芯片，用 Rubin 级 C2C）
    "nvlink_rubin": LinkSpec(interpool_bw_GBs=3600.0, interpool_latency_ns=900.0, name="nvlink_rubin"),
    "nvlink_h100":  LinkSpec(interpool_bw_GBs=900.0,  interpool_latency_ns=900.0, name="nvlink_h100"),
    # 网卡(NIC/MIC) 跨主机/跨板传输 —— GPU↔LPU 分立、走 RDMA NIC 时的真实链路。
    #   关键：decode 下 payload 极小(B×d_model×1B，~几 KB)，延迟完全由【NIC 固定延迟】主导，
    #   带宽几乎不起作用。NIC 延迟(µs 级)比 NVLink(900ns)/UCIe(15ns)高一个量级，
    #   而跨池传输每层往返 2 次、再 ×num_layers 放大，故对 disaggregated decode 影响巨大。
    "nic_cx7":  LinkSpec(interpool_bw_GBs=50.0,  interpool_latency_ns=2000.0, name="nic_cx7"),   # ConnectX-7 NDR 400Gb/s, GPUDirect RDMA 小包 ~2µs
    "nic_cx8":  LinkSpec(interpool_bw_GBs=100.0, interpool_latency_ns=1800.0, name="nic_cx8"),   # ConnectX-8 XDR 800Gb/s, ~1.8µs
    # 理想化跨池链路（AMMA+LPU ideal）：固定延迟 300ns、带宽 235 GB/s。
    #   代表 AMMA 与 LPU 紧耦合/近封装（如片间 die-to-die over advanced packaging）时的乐观上界，
    #   延迟远低于 RDMA NIC(2µs)，用于展示 disaggregation 的 xfer 开销在理想互联下被压到多小。
    "nic_ideal": LinkSpec(interpool_bw_GBs=235.0, interpool_latency_ns=300.0, name="nic_ideal"),
}


def get_link(name: str) -> "LinkSpec":
    """按名取链路预置副本；未命中则报错并列出可选项。"""
    if name not in LINK_LIBRARY:
        raise KeyError(f"未知链路 '{name}'，可选: {list(LINK_LIBRARY)}")
    spec = LINK_LIBRARY[name]
    return LinkSpec(**spec.to_dict())


def transfer_time(num_bytes: float, link: LinkSpec, direction: str) -> Dict[str, Any]:
    """
    单次跨池传输延迟 = 固定延迟 + payload/带宽。

    参数:
      num_bytes : 传输字节数（decode 下通常是 hidden state，B × d_model × act_bytes）
      direction : "a2f"(AMMA→LPU) 或 "f2a"(LPU→AMMA)，仅用于标注
    返回: 结构化 dict，t_ns 为该次传输耗时（纳秒）。
    """
    lat = link.interpool_latency_ns * 1e-9
    bw_term = num_bytes / (link.interpool_bw_GBs * 1e9)
    t = lat + bw_term
    return {
        "name": f"xfer_{direction}",
        "direction": direction,
        "bytes": num_bytes,
        "t_lat_ns": lat * 1e9,
        "t_bw_ns": bw_term * 1e9,
        "t_ns": t * 1e9,
    }
