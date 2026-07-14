# contbatch — continuous-batch decode serving (system layer over exp1)

A system-level scheduling layer on top of exp1's per-step decode TPOT. Compares
**GPU+GPU / GPU+LPU / AMMA+LPU** under one identical continuous-batching policy and
the same request stream, to get **average TPOT**.

## What it models
- **Scheduler** (`sim.py`): iteration-level / in-flight batching like vLLM/TGI.
  `B_max` = max concurrent requests (exposed, default 32). FCFS admit at iteration
  boundaries; every active request emits 1 token per step; evict on `output_len`.
- **Per-step latency** (`step_latency.py`): reuses exp1 composers unchanged —
  `decode_compose` (AMMA+LPU), `baseline_compose` (GPU+GPU / GPU+LPU). Builds a
  `(B, seq)` table and log-interpolates. `t_step = num_layers × (t_attn(B,seq) +
  2·t_xfer(B) + t_ffn(B))`.
- **Heterogeneous CL**: a step holds B requests with different CLs. Attention KV-read
  is linear in `B×seq` (∝ ΣCL_i); projections/FFN depend on B only. So call the
  composer with `batch=B, seq=mean(CL_i)` — exact for linear terms, approximate only
  at the compute/mem crossover.
- **Workload** (`workload.py`): unified `List[Request(arrival, prompt_len, output_len)]`.
  Synthetic (Poisson λ + lognormal lengths, seeded) **or** trace JSON; `dump_trace`
  bridges the two. Same stream feeds all 3 settings → fair comparison.
- **Metric**: `avg TPOT = Σ(B·t_step)/Σ B` (token-weighted), plus P50/P99, mean batch,
  throughput. Queue wait is TTFT, **not** TPOT.

## Run
```bash
PY=/home/haotian/miniconda3/envs/bench/bin/python
# main λ sweep at B_max=32, deepseek3, prompt~64K
$PY run_contbatch.py --n-requests 1500 -o data/contbatch_sweep.json
# extended experiments: accumulation timeline / B_max sweep / length sweep
$PY gen_experiments.py -o data/contbatch_experiments.json
# plots + HTML (from figures/.../exp5_contbatch/scripts)
$PY plot_contbatch.py   --data <sweep.json> --out ../plots/contbatch_overview.png
$PY plot_experiments.py --data <experiments.json> --out ../plots/contbatch_experiments.png
$PY gen_system_html.py  --data <sweep.json> --plot ../plots/contbatch_overview.png \
   --exp-data <experiments.json> --exp-plot ../plots/contbatch_experiments.png \
   --out ../system_overview.html
```

## Extended experiments (`gen_experiments.py`)
- **accumulation**: `sim` timeline of `waiting+active` over time. AMMA+LPU has a lower
  throughput ceiling → overloads earlier: at λ=5 GPU+LPU stable (backlog 0) while
  AMMA+LPU piles up (peak backlog 139); at λ=12 both overload but AMMA far steeper (peak 900).
- **B_max sweep** @ saturation: avg TPOT vs B_max∈{1..64}. B_max=1 AMMA fastest (2.4 ms);
  B_max=64 AMMA 52.7 ms vs GPU+LPU 18.3 ms — throughput bought with latency, steepest for AMMA.
- **length sweep** @ B_max=32, light vs saturated: AMMA wins 4K–64K at light load, but
  **collapses at 1024K (364 ms vs GPU+LPU 7.5 ms)** — the known deepseek3-MLA compute-bound-
  at-1M artifact (use DeepSeek-V4 CSA/HCA for long context).

## Settings
`gpu_gpu`, `gpu_lpu`, `amma_lpu` (ours), and **`amma_lpu_ideal`** — identical to
`amma_lpu` but cross-pool link = `nic_ideal` (300 ns / 235 GB/s) instead of `nic_cx7`
(2 µs / 50 GB/s); isolates the disaggregation NIC penalty (same definition as exp1).

## Key result (deepseek3, prompt~64K, B_max=32)
AMMA+LPU's speedup is a **low-load / small-batch** effect. Continuous batching erodes
and **inverts** it: at λ=0.5 (batch≈1.3) AMMA+LPU is fastest (2.7 ms, beats GPU+LPU);
by λ≥5 (batch→32) it is ~2.5× slower than GPU+LPU. Mechanism: AMMA attention goes
compute-bound as batch grows (16 NPUs × 96 TFLOPS), while Rubin attention scales with
batch. Crossover at B≈2–3 (see `plot_contbatch.py` panel c).

**Why ours loses past λ=1.5 = attention, not the NIC.** Ours and GPU+LPU share the same
LPU FFN and same nic_cx7; the only difference is attention HW, which is >95% of TPOT.
AMMA attn 2032→25550 µs (B=1→32, 12.6×) vs Rubin attn 2585→9378 µs (3.6×). The ideal
link (`amma_lpu_ideal`) only removes the ~1–3% NIC xfer — it extends the low-batch win
(λ=0.5: 2674→2422) but at B=32 is still ~2.4× slower than GPU+LPU, proving the bottleneck
is attention compute.

## Boundary / caveats
- decode-only (no prefill); prompt = initial KV. Inherits all exp1 caveats
  (LPU FFN = 100% roofline / optimistic; GPU FFN measured-util derate; link = nic_cx7).
- GPU+GPU uses 1×Rubin FFN (FFN-bound) → low throughput ceiling, near-saturated
  across this sweep.
