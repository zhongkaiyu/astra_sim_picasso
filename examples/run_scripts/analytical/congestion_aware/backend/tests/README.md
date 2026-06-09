# Regression tests for `utilization_lib.py`

This directory pins the behaviour of the existing roofline + power code so the
upcoming `backend/roofline/utilization_lib.py` can be verified against it.

## Files

| File | Role |
|---|---|
| `gen_baseline.py` | Runs the **existing** code (`roofline_mla_*`, `POWER_COEFFS`, utilization profiles) over a fixed grid and snapshots the results. |
| `baseline_dataset.json` | Committed snapshot. **Do not regenerate** unless the source-of-truth code intentionally changes. |
| `verify_against_baseline.py` | (TODO, after `utilization_lib.py` lands) Re-runs the same grid through the new API and asserts every field is within `1e-4` relative tolerance of the baseline. |

## Grid

- `arch ∈ {amma, rubin}`  (B200 excluded — no `POWER_COEFFS` yet)
- `batch_size ∈ {1, 4, 16, 32}`
- `seq_len ∈ {1024, 8192, 65536, 131072}`
- `TP`: AMMA=16 (`tp_h=4, tp_s=4`), Rubin=1
- AMMA SA-util profile: `util_64.json`

→ **32 test cases × 3 kernels (qkv / attention / proj_o) = 96 assertions**.

## What each case records

```jsonc
{
  "id": "amma_bs1_seq1024",
  "inputs": { "arch", "bs", "seq_len", "TP", "peak_per_npu_tflops", "bw_per_npu_tbs", "cache_seq" },
  "qkv":       { "flops", "bytes", "compute_ns_ideal", "memory_ns", "profile_util",
                 "expected_realized_time_ns",
                 "expected_compute_util", "expected_memory_util",
                 "expected_power_w" },
  "attention": { ... same shape ... },
  "proj_o":    { ... same shape ... }
}
```

## Conventions captured

| Arch | `compute_util` | `memory_util` | `realized_time_ns` |
|---|---|---|---|
| AMMA  | SA profile util (efficiency cap) | `memory_ns / realized_time` | `max(compute_ns_ideal / profile_util, memory_ns)` |
| Rubin | `0` (decode is BW-bound) | BW profile util | `total_ns_ideal / bw_profile_util` |

Power: `P = static_w + cmpt_coeff · U_cmpt + mem_coeff · U_mem`  
(D2D omitted in the baseline — see `_unresolved` in the JSON.)

## Not covered

- `gating`, `combine`, `moe_per_expert` — no existing implementation, will be
  designed from scratch and **added** to `baseline_dataset.json` once the new
  formulas are agreed and committed.
- B200 — needs `POWER_COEFFS` + BW util profile.
- AMMA D2D power contribution.

## How to regenerate (only if source-of-truth code changes)

```bash
cd backend
python3 tests/gen_baseline.py
git diff tests/baseline_dataset.json   # review every changed number
```

## How to use after `utilization_lib.py` lands

```python
import json, math
from roofline.utilization_lib import (
    compute_utilization_qkv, memory_utilization_qkv, power_qkv,
    compute_utilization_attention, memory_utilization_attention, power_attention,
    compute_utilization_o, memory_utilization_o, power_o,
)
from roofline.hardware_config import ours_config, rubin_config

baseline = json.load(open("tests/baseline_dataset.json"))
hw_map = {"amma": ours_config, "rubin": rubin_config}
TOL = 1e-4

for case in baseline["test_cases"]:
    arch = case["inputs"]["arch"]
    hw = hw_map[arch]
    bs = case["inputs"]["bs"]
    seq = case["inputs"]["seq_len"]
    TP = case["inputs"]["TP"]
    # Use the MLA model config (deepseek_v3); cfg construction is
    # implementation-specific to utilization_lib.
    cfg = ...  # build deepseek_v3 MLA config

    cu = compute_utilization_qkv(cfg, bs, hw, TP)
    mu = memory_utilization_qkv(cfg, bs, hw, TP)
    p  = power_qkv(mu, cu, hw)
    assert math.isclose(cu, case["qkv"]["expected_compute_util"], rel_tol=TOL)
    assert math.isclose(mu, case["qkv"]["expected_memory_util"], rel_tol=TOL)
    assert math.isclose(p,  case["qkv"]["expected_power_w"],     rel_tol=TOL)
    # ... attention, proj_o ...
```
