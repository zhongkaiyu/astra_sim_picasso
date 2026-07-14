#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
workload.py — continuous-batch 仿真的输入：request 流。

统一内部表示：一条 request = (arrival_ns, prompt_len, output_len)，sim 只认 List[Request]。
两条来源最后都产出这同一种东西，sim 完全不区分来源：

  ① 合成 synth_workload(spec)：
       - 到达：Poisson（指数 inter-arrival，率 λ req/s），可设 seed 复现；
       - prompt_len / output_len：各从分布抽（默认 lognormal，按均值参数化）。
  ② trace load_trace(path)：读 JSON，每条 {arrival_s|arrival_ns, prompt_len, output_len}。

桥梁 dump_trace()：合成结果可落盘成 trace 格式 → 今天合成扫参、明天换真实 trace 零改动。

公平性：三种 setting 必须跑【逐条相同】的 request 流。做法是同一个 seed（合成）或
同一个 trace 文件生成一份 List[Request]，再分别喂给三种 setting 的 sim（见 sim.py）。

口径：decode-only。prompt_len 只作为初始 KV 长度 P_i（CL_i = P_i + 已生成数），
prefill 延迟不在范围内（exp1 是 decode 模型）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Request:
    rid: int
    arrival_ns: float          # 到达时刻 (ns)
    prompt_len: int            # 初始 KV 长度 P_i
    output_len: int            # 要 decode 的 token 数 L_i（>=1）
    # ---- 运行时状态（sim 填充，不属于 workload 定义）----
    admit_ns: float = -1.0     # 被纳入 active batch 的时刻
    finish_ns: float = -1.0    # 生成完最后一个 token 的时刻
    generated: int = 0         # 已生成 token 数
    sum_step_ns: float = 0.0   # 该 request 经历的所有 step 延迟之和

    @property
    def tpot_ns(self) -> float:
        return self.sum_step_ns / self.generated if self.generated else 0.0

    @property
    def queue_ns(self) -> float:
        return max(self.admit_ns - self.arrival_ns, 0.0) if self.admit_ns >= 0 else 0.0


# ---------------------------------------------------------------------------
# 分布工具：按【均值】参数化 lognormal（更直观），sigma 为对数标准差(CV 控制)。
# ---------------------------------------------------------------------------
def _sample_lengths(rng: np.random.Generator, n: int, mean: float,
                    sigma_log: float, lo: int, hi: int) -> np.ndarray:
    """lognormal，期望 = mean；clip 到 [lo, hi] 并取整。"""
    mu = math_log_mean(mean, sigma_log)
    x = rng.lognormal(mean=mu, sigma=sigma_log, size=n)
    x = np.clip(np.round(x), lo, hi)
    return x.astype(int)


def math_log_mean(mean: float, sigma_log: float) -> float:
    """由目标期望 mean 与对数标准差 sigma 反推底层正态的 mu： E=exp(mu+sigma^2/2)。"""
    return float(np.log(max(mean, 1e-9)) - 0.5 * sigma_log ** 2)


@dataclass
class WorkloadSpec:
    n_requests: int = 2000
    arrival_rate_rps: float = 20.0     # λ（Poisson）
    prompt_mean: float = 65536.0       # prompt 长度均值（默认 64K）
    prompt_sigma_log: float = 0.6
    prompt_lo: int = 256
    prompt_hi: int = 1048576
    output_mean: float = 256.0         # output 长度均值
    output_sigma_log: float = 0.7
    output_lo: int = 1
    output_hi: int = 8192
    seed: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def synth_workload(spec: WorkloadSpec) -> List[Request]:
    """合成一份 request 流（Poisson 到达 + lognormal 长度）。"""
    rng = np.random.default_rng(spec.seed)
    n = spec.n_requests
    # Poisson 到达：指数 inter-arrival，累加成绝对到达时刻 (ns)
    rate_per_ns = spec.arrival_rate_rps / 1e9
    inter = rng.exponential(scale=1.0 / max(rate_per_ns, 1e-30), size=n)
    arrivals = np.cumsum(inter)
    prompts = _sample_lengths(rng, n, spec.prompt_mean, spec.prompt_sigma_log,
                              spec.prompt_lo, spec.prompt_hi)
    outputs = _sample_lengths(rng, n, spec.output_mean, spec.output_sigma_log,
                              spec.output_lo, spec.output_hi)
    return [Request(rid=i, arrival_ns=float(arrivals[i]),
                    prompt_len=int(prompts[i]), output_len=int(outputs[i]))
            for i in range(n)]


def reseed_arrivals(reqs: List[Request], arrival_rate_rps: float,
                    seed: int) -> List[Request]:
    """保持每条 request 的长度不变，仅按新 λ 重抽到达时刻。

    用于「扫 λ」：长度由 spec.seed 固定 → 不同负载下 request 内容逐条一致，
    只有到达密度变化，三种 setting 的对比始终公平、可复现。
    """
    rng = np.random.default_rng(seed)
    n = len(reqs)
    rate_per_ns = arrival_rate_rps / 1e9
    inter = rng.exponential(scale=1.0 / max(rate_per_ns, 1e-30), size=n)
    arrivals = np.cumsum(inter)
    out = []
    for i, r in enumerate(reqs):
        out.append(Request(rid=r.rid, arrival_ns=float(arrivals[i]),
                           prompt_len=r.prompt_len, output_len=r.output_len))
    return out


# ---------------------------------------------------------------------------
# trace 互通：dump / load（同一内部表示）
# ---------------------------------------------------------------------------
def dump_trace(reqs: List[Request], path: str, meta: Optional[dict] = None) -> None:
    obj = {
        "meta": meta or {},
        "requests": [
            {"rid": r.rid, "arrival_ns": r.arrival_ns,
             "prompt_len": r.prompt_len, "output_len": r.output_len}
            for r in reqs
        ],
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_trace(path: str) -> List[Request]:
    """读 trace JSON。每条接受 arrival_ns 或 arrival_s。"""
    with open(path) as f:
        obj = json.load(f)
    rows = obj["requests"] if isinstance(obj, dict) else obj
    reqs = []
    for i, d in enumerate(rows):
        if "arrival_ns" in d:
            a = float(d["arrival_ns"])
        elif "arrival_s" in d:
            a = float(d["arrival_s"]) * 1e9
        else:
            raise KeyError("trace 条目需含 arrival_ns 或 arrival_s")
        reqs.append(Request(rid=int(d.get("rid", i)), arrival_ns=a,
                            prompt_len=int(d["prompt_len"]),
                            output_len=int(max(1, d["output_len"]))))
    reqs.sort(key=lambda r: r.arrival_ns)
    return reqs


def build_workload(spec: Optional[WorkloadSpec] = None,
                   trace_path: Optional[str] = None) -> List[Request]:
    """统一出口：给了 trace_path 就读 trace，否则按 spec 合成。"""
    if trace_path:
        return load_trace(trace_path)
    if spec is None:
        raise ValueError("需要 spec 或 trace_path 其一")
    return synth_workload(spec)


def workload_stats(reqs: List[Request]) -> dict:
    a = np.array([r.arrival_ns for r in reqs])
    p = np.array([r.prompt_len for r in reqs])
    o = np.array([r.output_len for r in reqs])
    span_s = (a.max() - a.min()) / 1e9 if len(a) > 1 else 0.0
    return {
        "n": len(reqs),
        "arrival_span_s": span_s,
        "eff_rate_rps": (len(reqs) / span_s) if span_s > 0 else 0.0,
        "prompt_mean": float(p.mean()), "prompt_p50": float(np.median(p)),
        "prompt_p99": float(np.percentile(p, 99)),
        "output_mean": float(o.mean()), "output_p50": float(np.median(o)),
        "total_output_tokens": int(o.sum()),
    }
