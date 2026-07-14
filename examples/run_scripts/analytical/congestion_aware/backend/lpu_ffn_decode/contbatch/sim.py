#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sim.py — continuous-batch（迭代级 / in-flight batching）调度仿真。

调度策略（三种 setting 共用，这是公平对比的前提，跟 vLLM/TGI 一致）：
  - B_max = 最大并发 request 数（暴露参数，默认 32）；
  - FCFS 等待队列；只在【迭代边界】做调度决策；
  - 每一步：把已到达的 request 补进 active 集合直到 B_max，跑一次 decode step，
    所有 active request 各吐 1 个 token，吐满 output_len 的 evict，循环。

单步延迟来自 step_latency.StepLatencyTable（封装 exp1 三个 composer）：
    dt = t_step_ns(setting, B=|active|, mean_seq=mean(CL_i))
其中 CL_i = prompt_len_i + generated_i（decode 中逐步 +1）。

指标：
  - avg_tpot_us：按 token 加权的平均 TPOT = Σ(B·dt)/Σ B（= 所有输出 token 的平均吐字延迟）；
  - per-request TPOT 的 p50/p99（request 粒度）；
  - mean_batch（时间加权平均 batch）、throughput（tok/s）、queue/TTFT（排队延迟，不计入 TPOT）。
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from workload import Request
from step_latency import StepLatencyTable


def simulate(setting: str, reqs: List[Request], table: StepLatencyTable,
             b_max: int = 32, record_timeline: bool = False,
             timeline_points: int = 400,
             sched: str = "greedy", lam_rps: float = None,
             out_mean: float = 256.0, seq_ref: float = 65536.0) -> Dict:
    """对单个 setting 跑迭代级仿真。reqs 会被复制，原列表不被修改。

    record_timeline=True 时额外返回 timeline：随时间的 (waiting 队列长度, active batch)，
    用来看「随 λ 变大 request 的积累」——过载的 setting 其 waiting 会线性堆积。

    sched:
      - "greedy"     ：每步把 active 贪婪填到 b_max（throughput 优先，但 attn∝B 时大 batch
                       会让 per-token 延时爆炸）。
      - "throughput" ：吞吐感知——只批到「刚好能撑住到达的 token 速率」的最小 batch
                       B*（throughput(B*) ≥ λ·out_mean），再以 b_max 封顶。低负载用小 batch
                       保延时，高负载才升到 b_max 提吞吐；不做无谓的超批。
    """
    # 深拷贝运行时状态（同一份 workload 要喂多个 setting）
    pool = sorted(
        [Request(r.rid, r.arrival_ns, r.prompt_len, r.output_len) for r in reqs],
        key=lambda r: r.arrival_ns,
    )
    n = len(pool)
    ai = 0                       # 下一个未到达 request 的指针
    waiting: List[Request] = []  # 已到达、未纳入 active（FCFS）
    active: List[Request] = []   # 正在 decode
    done: List[Request] = []

    t = 0.0                      # 仿真时钟 (ns)
    wall = 0.0                   # 总 busy 时间（Σ dt）
    total_token_time = 0.0       # Σ(B·dt)，token 加权 TPOT 分子
    total_tokens = 0             # Σ B
    n_steps = 0
    batch_time_hist: Dict[int, float] = {}   # batch 大小 → 累计占用时间(ns)
    insys_time = 0.0          # ∫(waiting+active) dt，时间加权在系统内 request 数
    wait_time = 0.0           # ∫ waiting dt，时间加权 backlog（未被服务的积压）
    backlog_peak = 0          # waiting 队列峰值
    tl_t: List[float] = []; tl_wait: List[int] = []; tl_act: List[int] = []

    # 吞吐感知：算「撑住到达 token 速率」的最小 batch B*，再以 b_max 封顶。
    fill_cap = b_max
    if sched == "throughput" and lam_rps:
        tok_rate_per_ns = lam_rps * out_mean / 1e9        # 到达 token 速率 (tokens/ns)
        B_star = b_max
        for B in range(1, b_max + 1):
            if B / table.t_step_ns(setting, B, seq_ref) >= tok_rate_per_ns:
                B_star = B
                break
        fill_cap = min(b_max, B_star)

    def admit():
        # 已到达的进 waiting；再 FCFS 填 active 到 fill_cap（≤ b_max）
        nonlocal ai
        while ai < n and pool[ai].arrival_ns <= t + 1e-9:
            waiting.append(pool[ai]); ai += 1
        while len(active) < fill_cap and waiting:
            r = waiting.pop(0)
            r.admit_ns = t
            active.append(r)

    while True:
        admit()
        if not active:
            if ai >= n and not waiting:
                break                      # 全部完成
            # active 空但还有未到达的：时钟跳到下一个到达时刻
            t = max(t, pool[ai].arrival_ns)
            continue

        B = len(active)
        nwait = len(waiting)
        backlog_peak = max(backlog_peak, nwait)
        if record_timeline:
            tl_t.append(t); tl_wait.append(nwait); tl_act.append(B)
        mean_seq = sum(r.prompt_len + r.generated for r in active) / B
        dt = table.t_step_ns(setting, B, mean_seq)

        # 这一步内每条 active 各产 1 token
        for r in active:
            r.generated += 1
            r.sum_step_ns += dt
        total_token_time += B * dt
        total_tokens += B
        batch_time_hist[B] = batch_time_hist.get(B, 0.0) + dt
        insys_time += (nwait + B) * dt
        wait_time += nwait * dt
        wall += dt
        n_steps += 1
        t += dt

        # evict 完成的
        still: List[Request] = []
        for r in active:
            if r.generated >= r.output_len:
                r.finish_ns = t
                done.append(r)
            else:
                still.append(r)
        active = still

    # ---- 汇总指标 ----
    tpot_req_us = np.array([r.tpot_ns / 1e3 for r in done if r.generated > 0])
    queue_us = np.array([r.queue_ns / 1e3 for r in done])
    # 端到端：从到达到生成完最后一个 token 的总处理时间（含排队 + 全部 decode 步）
    e2e_us = np.array([(r.finish_ns - r.arrival_ns) / 1e3 for r in done if r.finish_ns >= 0])
    mean_batch = (total_token_time / wall) if wall > 0 else 0.0   # 时间加权平均 batch
    avg_tpot_us = (total_token_time / total_tokens / 1e3) if total_tokens else 0.0
    throughput = (total_tokens / (wall / 1e9)) if wall > 0 else 0.0
    mean_in_system = (insys_time / wall) if wall > 0 else 0.0     # 平均在系统内 request 数
    mean_backlog = (wait_time / wall) if wall > 0 else 0.0        # 平均积压（waiting）

    timeline = None
    if record_timeline and tl_t:
        # 按时间轴等距下采样到 ~timeline_points 个点
        k = max(1, len(tl_t) // max(1, timeline_points))
        timeline = {
            "t_s": [tl_t[i] / 1e9 for i in range(0, len(tl_t), k)],
            "waiting": [tl_wait[i] for i in range(0, len(tl_t), k)],
            "active": [tl_act[i] for i in range(0, len(tl_t), k)],
            "in_system": [tl_wait[i] + tl_act[i] for i in range(0, len(tl_t), k)],
        }

    return {
        "setting": setting,
        "b_max": b_max,
        "avg_tpot_us": avg_tpot_us,                       # ← 主指标：token 加权平均 TPOT
        "tpot_p50_us": float(np.percentile(tpot_req_us, 50)) if len(tpot_req_us) else 0.0,
        "tpot_p99_us": float(np.percentile(tpot_req_us, 99)) if len(tpot_req_us) else 0.0,
        "mean_batch": mean_batch,
        "max_batch_seen": max(batch_time_hist) if batch_time_hist else 0,
        "throughput_tok_s": throughput,
        "queue_mean_us": float(queue_us.mean()) if len(queue_us) else 0.0,
        "queue_p99_us": float(np.percentile(queue_us, 99)) if len(queue_us) else 0.0,
        "e2e_mean_us": float(e2e_us.mean()) if len(e2e_us) else 0.0,       # 端到端平均处理时间（到结束）
        "e2e_p50_us": float(np.percentile(e2e_us, 50)) if len(e2e_us) else 0.0,
        "e2e_p99_us": float(np.percentile(e2e_us, 99)) if len(e2e_us) else 0.0,
        "mean_in_system": mean_in_system,        # 平均在系统内 request 数（waiting+active）
        "mean_backlog": mean_backlog,            # 平均积压队列（waiting）
        "backlog_peak": backlog_peak,            # 积压峰值
        "n_requests": len(done),
        "n_steps": n_steps,
        "total_tokens": total_tokens,
        "busy_time_s": wall / 1e9,
        "batch_time_frac": {int(b): v / wall for b, v in sorted(batch_time_hist.items())} if wall > 0 else {},
        "timeline": timeline,
    }
