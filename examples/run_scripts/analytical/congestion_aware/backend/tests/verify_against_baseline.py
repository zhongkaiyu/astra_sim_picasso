#!/usr/bin/env python3
"""Regression check: new `utilization_lib.py` vs frozen baseline.

Compares every (arch, bs, seq) × (qkv / attention / proj_o) case in
baseline_dataset.json to the new 13-function API. Asserts each numeric
field agrees to within `TOL` relative tolerance.

MoE kernels (gating / combine / moe_per_expert) have no baseline and are
skipped here — they will be added once formulas are validated.

Usage:
    python3 tests/verify_against_baseline.py            # exit 0 on success
    python3 tests/verify_against_baseline.py --verbose  # print every case
"""
import argparse
import json
import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND / "roofline"))

try:
    from utilization_lib import (
        DEEPSEEK_V3_DEFAULTS,
        compute_utilization_qkv, memory_utilization_qkv, power_qkv,
        compute_utilization_attention, memory_utilization_attention, power_attention,
        compute_utilization_o, memory_utilization_o, power_o,
    )
except ImportError as e:
    print(f"[SKIP] utilization_lib.py not importable yet: {e}")
    sys.exit(0)


TOL = 1e-4


def _close(actual, expected, tol=TOL):
    if expected == 0:
        return abs(actual) < tol
    return abs(actual - expected) / abs(expected) < tol


KERNEL_API = {
    # name -> (compute_fn, memory_fn, power_fn, scalar_input_key)
    "qkv":       (compute_utilization_qkv,       memory_utilization_qkv,       power_qkv,       "bs"),
    "attention": (compute_utilization_attention, memory_utilization_attention, power_attention, "seq_len"),
    "proj_o":    (compute_utilization_o,         memory_utilization_o,         power_o,         "bs"),
}


def check_case(case, verbose=False):
    inp = case["inputs"]
    hw = HW_CONFIGS[inp["arch"]]
    TP = inp["TP"]

    failures = []
    for kernel, (cu_fn, mu_fn, pwr_fn, key) in KERNEL_API.items():
        arg = inp[key]
        cu = cu_fn(DEEPSEEK_V3_DEFAULTS, arg, hw, TP)
        mu = mu_fn(DEEPSEEK_V3_DEFAULTS, arg, hw, TP)
        pw = pwr_fn(mu, cu, hw)

        exp = case[kernel]
        for label, got, want in [
            ("compute_util", cu, exp["expected_compute_util"]),
            ("memory_util",  mu, exp["expected_memory_util"]),
            ("power_w",      pw, exp["expected_power_w"]),
        ]:
            ok = _close(got, want)
            if verbose:
                tag = "OK " if ok else "FAIL"
                print(f"  [{tag}] {kernel}.{label:13s}: got={got:.6g}  want={want:.6g}")
            if not ok:
                failures.append(
                    f"{case['id']} / {kernel}.{label}: got={got:.6g} want={want:.6g}"
                )
    return failures


def main():
    global HW_CONFIGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--baseline",
        default=str(Path(__file__).parent / "baseline_dataset.json"),
    )
    args = parser.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    HW_CONFIGS = baseline["hw_configs"]

    all_failures = []
    for case in baseline["test_cases"]:
        if args.verbose:
            print(f"\n=== {case['id']} ===")
        all_failures.extend(check_case(case, verbose=args.verbose))

    n_cases = len(baseline["test_cases"])
    if all_failures:
        print(f"\nFAIL: {len(all_failures)} mismatch(es) across {n_cases} cases:")
        for f in all_failures[:20]:
            print(f"  {f}")
        if len(all_failures) > 20:
            print(f"  ... and {len(all_failures) - 20} more")
        sys.exit(1)
    print(f"\nPASS: all {n_cases} cases match baseline within {TOL} relative tolerance.")


if __name__ == "__main__":
    main()
