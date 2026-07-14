# Decouple analytical backend from ASTRA-sim — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract `congestion_aware/` into a standalone repo where ASTRA-sim is an external optional tool, add a `--comm-model {roofline,astrasim}` flag so the paper figures regenerate purely from our backend, and add a test bounding roofline-vs-astrasim comm error.

**Architecture:** Fresh copy of `congestion_aware/` becomes a new git repo at `/home/haotian/amma-decode-modeling/`. `merge_gqa_results.py` gains a comm-model switch: `roofline` mode composes comm from the roofline JSON's own per-strategy `comm` dict (already computed by `calc_strategy`) and uses roofline util-adjusted compute only; `astrasim` mode is the unchanged current behaviour. A single `astrasim/run_astrasim.py` wrapper orchestrates the external ASTRA-sim toolchain. A pytest validates roofline-mode comm against committed astrasim-mode hybrid JSONs.

**Tech Stack:** Python 3.10+ stdlib (json, argparse, pathlib, math, bisect), pytest, matplotlib/numpy/scipy (figures only). No new runtime deps.

## Global Constraints

- Source tree stays untouched except the checkpoint commit; all code changes happen in the new repo `/home/haotian/amma-decode-modeling/`.
- `--comm-model` **default is `roofline`**; `astrasim` mode must reproduce current numbers bit-for-bit (regression).
- Do not alter `add_rubin_tp2.py`, `power_model.py`, or any figure plot script — they read the hybrid JSON regardless of comm source.
- Comm-model validation tolerance: **per-strategy aggregated comm relative error < 0.25** (provisional; user finalizes after seeing deltas). Per-op deltas are printed, not gated.
- Drop `backend/tmp/` in the copy. `.gitignore` excludes `*.ncu-rep`, `build/`, `output*/`, `__pycache__/`, `*.pyc`.
- Any `git push` / GitHub repo creation is **user-gated** — stop and ask before pushing.

---

### Task 0: Checkpoint the source repo (user-gated push)

**Files:**
- Modify (commit): everything currently uncommitted under `.../astra_sim_picasso/examples/run_scripts/analytical/congestion_aware/`

**Interfaces:**
- Produces: a clean `astra_sim_picasso` working tree at a known commit on `features/latest-updates`, so the extraction in Task 1 captures the latest state.

- [ ] **Step 1: Review what will be committed**

Run: `cd /home/haotian/waferchip/mesh2D/astra_sim_picasso && git status -s examples/run_scripts/analytical/congestion_aware`
Expected: ~20 modified/untracked entries (backend `.py`, new `contbatch/`, new fig1 plots). Confirm no stray large binaries (`*.ncu-rep`, `build/`).

- [ ] **Step 2: Stage and commit the congestion_aware changes**

```bash
cd /home/haotian/waferchip/mesh2D/astra_sim_picasso
git add examples/run_scripts/analytical/congestion_aware
git commit -m "Checkpoint congestion_aware backend before standalone extraction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 3: Verify the commit is scoped and tree is clean**

Run: `git show --stat --oneline HEAD | head -5 && git status -s examples/run_scripts/analytical/congestion_aware`
Expected: the new commit lists only congestion_aware paths; `git status` for that path is empty.

- [ ] **Step 4: Push (USER-GATED)**

STOP. Ask the user to confirm the push. Only on confirmation:
```bash
git push origin features/latest-updates
```
Expected: push succeeds (branch already tracks `origin/features/latest-updates`).

---

### Task 1: Scaffold the standalone repo (fresh extract)

**Files:**
- Create: `/home/haotian/amma-decode-modeling/` (copy of `congestion_aware/`, minus `backend/tmp/`)
- Create: `/home/haotian/amma-decode-modeling/.gitignore`
- Create: `/home/haotian/amma-decode-modeling/requirements.txt`

**Interfaces:**
- Produces: a self-contained tree whose figures render offline; the root path `NEWREPO=/home/haotian/amma-decode-modeling` used by all later tasks.

- [ ] **Step 1: Copy the tree, excluding cruft**

```bash
SRC=/home/haotian/waferchip/mesh2D/astra_sim_picasso/examples/run_scripts/analytical/congestion_aware
NEWREPO=/home/haotian/amma-decode-modeling
rsync -a --exclude='backend/tmp/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='*.ncu-rep' --exclude='build/' --exclude='output*/' \
  "$SRC"/ "$NEWREPO"/
```

- [ ] **Step 2: Add .gitignore**

Create `/home/haotian/amma-decode-modeling/.gitignore`:
```gitignore
__pycache__/
*.pyc
*.ncu-rep
build/
output*/
.DS_Store
/tmp/
```

- [ ] **Step 3: Add requirements.txt**

Create `/home/haotian/amma-decode-modeling/requirements.txt`:
```text
matplotlib>=3.5
numpy>=1.21
scipy>=1.7
pytest>=7.0
```

- [ ] **Step 4: git init + initial commit**

```bash
cd /home/haotian/amma-decode-modeling
git init -q
git add -A
git commit -q -m "Initial import: analytical decode-modeling backend + figures

Extracted from astra_sim_picasso/examples/run_scripts/analytical/congestion_aware.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 5: Smoke-test a pure-python figure (no astra-sim, no backend import)**

Run: `cd /home/haotian/amma-decode-modeling && python3 figures/paper_figures/fig3_ablation_study/scripts/plot_ablation.py`
Expected: exits 0 and writes `figures/paper_figures/fig3_ablation_study/plots/fig3_ablation.pdf` (mtime just now). Confirms the extracted tree is functional.

---

### Task 2: Add `--comm-model {roofline,astrasim}` to merge_gqa_results.py

**Files:**
- Modify: `/home/haotian/amma-decode-modeling/backend/hybrid_merge/merge_gqa_results.py`
- Test: `/home/haotian/amma-decode-modeling/backend/hybrid_merge/tests/test_merge_comm_model.py`

**Interfaces:**
- Consumes: roofline JSON `strategies[strat].comm` = `{qkv_allgather, attn_rs, final_reduce, output_ar, output_ag, total_comm_ns}` (each op dict has `time_ns`, plus size fields like `per_cube_chunk_elems`, `total_elems`, `msg_elems`, `msg_elems_per_step`); roofline entry `strategies[strat].data[i]` keyed by `seq` with `Proj_QKV/attention/Proj_O` sub-dicts.
- Produces: `build_comm_breakdown_roofline(strategy, rf_strat) -> {ops, total_comm_ns, total_d2d_bytes_all_cubes}`; `merge_entry(strategy, as_entry, rf_entry, rf_strat, util_data=None, comm_model="astrasim")` (new param, `as_entry` may be `None`); hybrid JSON `metadata.comm_model` field.

- [ ] **Step 1: Write the failing test**

Create `backend/hybrid_merge/tests/test_merge_comm_model.py`:
```python
import json, subprocess, sys, glob
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]          # /home/haotian/amma-decode-modeling
MERGE = REPO / "backend" / "hybrid_merge" / "merge_gqa_results.py"


def _find_inputs():
    """Pick one (roofline, astrasim) pair that share a bs tag, for qwen3-235B."""
    rf = sorted(glob.glob(str(REPO / "reports/qwen3-235B/roofline/gqa_roofline_bs1_*bw1500.json")))
    as_ = sorted(glob.glob(str(REPO / "reports/qwen3-235B/astrasim/gqa_seq_scaling_bs1_*data*.json")))
    assert rf and as_, "expected committed roofline + astrasim JSON for qwen3-235B bs1"
    return rf[0], as_[0]


def _run_merge(tmp, extra):
    rf, as_ = _find_inputs()
    out = tmp / "hybrid.json"
    cmd = [sys.executable, str(MERGE), "--roofline-data", rf, "-o", str(out)] + extra
    subprocess.run(cmd, check=True, cwd=REPO)
    return json.load(open(out))


def test_roofline_mode_needs_no_astrasim(tmp_path):
    """--comm-model roofline runs with NO --astrasim-data and tags the source."""
    d = _run_merge(tmp_path, ["--comm-model", "roofline"])
    assert d["metadata"]["comm_model"] == "roofline"
    # hmp_reo_new comm is already pure roofline; must be present and > 0
    hrn = d["strategies"]["hmp_reo_new"]["data"]
    assert hrn and all(e["comm_total_ns"] > 0 for e in hrn)


def test_astrasim_mode_still_requires_astrasim(tmp_path):
    """--comm-model astrasim without --astrasim-data exits non-zero."""
    rf, _ = _find_inputs()
    r = subprocess.run([sys.executable, str(MERGE), "--roofline-data", rf,
                        "--comm-model", "astrasim", "-o", str(tmp_path/"x.json")],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0
    assert "astrasim" in (r.stderr + r.stdout).lower()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /home/haotian/amma-decode-modeling && python3 -m pytest backend/hybrid_merge/tests/test_merge_comm_model.py -q`
Expected: FAIL — `--comm-model` is not a known argument (argparse error) / current merge requires `--astrasim-data`.

- [ ] **Step 3: Add the roofline-only comm breakdown builder**

In `merge_gqa_results.py`, add after `build_comm_breakdown` (after line ~348):
```python
def build_comm_breakdown_roofline(strategy: str, rf_strat: dict) -> dict:
    """Roofline-only comm breakdown, mirroring calc_strategy's own composition
    (roofline_gqa_calc.calc_strategy lines 746-750):
      hmp_reo_new: qkv_allgather + attn_rs + final_reduce
      others:      qkv_allgather + final_reduce + output_ar + output_ag
    All op times come from the roofline JSON's comm dict; no astrasim input.
    """
    rf_comm = rf_strat.get("comm", {})
    if strategy == "hmp_reo_new":
        keys = ["qkv_allgather", "attn_rs", "final_reduce"]
    else:
        keys = ["qkv_allgather", "final_reduce", "output_ar", "output_ag"]

    ops = []
    for key in keys:
        op = rf_comm.get(key, {})
        t = op.get("time_ns", 0)
        if t <= 0:
            continue
        ops.append({
            "name": key,
            "source": "roofline",
            "time_ns": round(t, 2),
            "d2d_bytes_all_cubes": op.get("d2d_bytes_all_cubes", 0),
            "group": op.get("group", ""),
        })
    total_comm_ns = sum(op["time_ns"] for op in ops)
    total_d2d = sum(op.get("d2d_bytes_all_cubes", 0) for op in ops)
    return {
        "ops": ops,
        "total_comm_ns": round(total_comm_ns, 2),
        "total_d2d_bytes_all_cubes": total_d2d,
    }
```

- [ ] **Step 4: Make `merge_entry` comm-model aware and allow `as_entry=None`**

In `merge_gqa_results.py`, change the `merge_entry` signature and its comm/compute wiring:
```python
def merge_entry(strategy: str, as_entry: dict | None, rf_entry: dict,
                rf_strat: dict, util_data: dict | None = None,
                comm_model: str = "astrasim") -> dict:
    """Merge a single (strategy, seq) pair. In roofline mode as_entry may be None."""
    mods = (as_entry or {}).get("modules", {})
    as_qkv = mods.get("qkv_comp", 0)
    as_attn = mods.get("attn_comp", 0)
    as_output = mods.get("output_comp", 0)
```
Keep the existing roofline util-adjust block unchanged. Then replace the comm-breakdown call (currently lines ~431-434) with:
```python
    rf_batch = rf_entry.get("batch", 1)
    if comm_model == "roofline":
        comm_bd = build_comm_breakdown_roofline(strategy, rf_strat)
    else:
        comm_bd = build_comm_breakdown(strategy, mods, rf_entry, rf_strat,
                                       comm_batch_scale=rf_batch)
    comm_total = comm_bd["total_comm_ns"]
    hybrid_wall = hybrid_gpu + comm_total
```
Also fix the `original_astrasim` block to tolerate `as_entry=None`:
```python
        "original_astrasim": {
            "wall_ns": (as_entry or {}).get("wall_ns", 0),
            "gpu_ns": (as_entry or {}).get("gpu_ns", 0),
            "comm_ns": (as_entry or {}).get("comm_ns", 0),
        },
    }
```
And change `"seq": as_entry["seq"]` (two occurrences: the return dict and the top) to `"seq": rf_entry["seq"]` so it works when `as_entry is None` (roofline entries carry `seq`).

- [ ] **Step 5: Rewire `main()` — flag, optional astrasim, roofline-driven iteration**

In `main()`, replace the `--astrasim-data` argument and add the flag:
```python
    parser.add_argument("--astrasim-data", default=None,
                        help="collect_gqa_data.py output JSON (required only for "
                             "--comm-model astrasim)")
    parser.add_argument("--comm-model", choices=["roofline", "astrasim"],
                        default="roofline",
                        help="roofline: comm from analytical roofline model (no "
                             "astra-sim needed). astrasim: mesh2D congestion-aware "
                             "comm from --astrasim-data (current behaviour).")
```
After `args = parser.parse_args()`, add the guard and conditional load:
```python
    if args.comm_model == "astrasim" and not args.astrasim_data:
        print("[ERROR] --comm-model astrasim requires --astrasim-data",
              file=sys.stderr)
        sys.exit(2)
    astrasim = load_json(args.astrasim_data) if args.astrasim_data else {}
```
Replace the merged-strategy loop body so seqs come from roofline in roofline mode:
```python
    for strat in merged_strategies:
        if strat not in rf_lookup:
            print(f"[WARN] Strategy '{strat}' not in roofline data, skipping",
                  file=sys.stderr)
            continue
        if args.comm_model == "astrasim" and strat not in as_lookup:
            print(f"[WARN] Strategy '{strat}' not in AstraSim data, skipping",
                  file=sys.stderr)
            continue

        rf_strat = rf_strategies.get(strat, {})
        seqs = (sorted(rf_lookup[strat]) if args.comm_model == "roofline"
                else sorted(as_lookup[strat]))
        merged_entries = []
        for seq in seqs:
            if seq not in rf_lookup[strat]:
                continue
            as_entry = (as_lookup[strat].get(seq) if args.comm_model == "astrasim"
                        else None)
            strat_util = (tp16_util_data if strat == "tp16" and tp16_util_data
                          else util_data)
            entry = merge_entry(strat, as_entry, rf_lookup[strat][seq], rf_strat,
                                util_data=strat_util, comm_model=args.comm_model)
            merged_entries.append(entry)
        result["strategies"][strat] = {"data": merged_entries}
```
Add to the `meta` dict (near line ~563): `"comm_model": args.comm_model,`.

- [ ] **Step 6: Run the new test to verify it passes**

Run: `cd /home/haotian/amma-decode-modeling && python3 -m pytest backend/hybrid_merge/tests/test_merge_comm_model.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Commit**

```bash
cd /home/haotian/amma-decode-modeling
git add backend/hybrid_merge/merge_gqa_results.py backend/hybrid_merge/tests/test_merge_comm_model.py
git commit -q -m "merge: add --comm-model roofline|astrasim (default roofline)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Validation test — roofline comm vs astrasim comm

**Files:**
- Create: `/home/haotian/amma-decode-modeling/tests/test_comm_model_vs_astrasim.py`

**Interfaces:**
- Consumes: committed `reports/*/hybrid/*.json` (astrasim-mode, has per-(strategy,seq) `comm_total_ns`); `merge_gqa_results` roofline mode (Task 2).
- Produces: a gated assertion on per-strategy aggregated comm relative error (< 0.25) plus a printed per-op/per-strategy delta table.

- [ ] **Step 1: Write the test (fails until Task 2 is in place; it is, so it should pass — write it to assert real behaviour)**

Create `tests/test_comm_model_vs_astrasim.py`:
```python
"""Bound the error between roofline-computed comm and astra-sim comm.

For each committed hybrid JSON (produced in astrasim mode), recompute the same
model/bs in roofline mode and compare comm_total_ns per (strategy, seq).
Prints a per-strategy delta table; gates on aggregated relative error.
"""
import json, glob, subprocess, sys, re
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[1]
MERGE = REPO / "backend" / "hybrid_merge" / "merge_gqa_results.py"
TOL = 0.25   # provisional; per Global Constraints

def _roofline_for(model_dir: Path, bs_tag: str):
    m = sorted(model_dir.glob(f"roofline/gqa_roofline_{bs_tag}_*bw1500.json"))
    return m[0] if m else None

def _cases():
    for hyb in sorted(REPO.glob("reports/*/hybrid/gqa_hybrid_merged_*_bs*.json")):
        mt = re.search(r"_(bs\d+)\.json$", hyb.name)
        if not mt:
            continue
        model_dir = hyb.parents[1]
        rf = _roofline_for(model_dir, mt.group(1))
        if rf:
            yield hyb, rf

def test_roofline_comm_within_tolerance(tmp_path, capsys):
    cases = list(_cases())
    if not cases:
        pytest.skip("no matching hybrid+roofline JSON pairs found")
    worst = 0.0
    rows = []
    for hyb_path, rf_path in cases:
        astr = json.load(open(hyb_path))
        out = tmp_path / (hyb_path.stem + "_roofline.json")
        subprocess.run([sys.executable, str(MERGE), "--roofline-data", str(rf_path),
                        "--comm-model", "roofline", "-o", str(out)],
                       check=True, cwd=REPO)
        rl = json.load(open(out))
        for strat, sd in astr.get("strategies", {}).items():
            rl_sd = {e["seq"]: e for e in rl.get("strategies", {}).get(strat, {}).get("data", [])}
            a_sum = r_sum = 0.0
            for e in sd.get("data", []):
                a = e.get("comm_total_ns", 0)
                r = rl_sd.get(e["seq"], {}).get("comm_total_ns", 0)
                a_sum += a; r_sum += r
            if a_sum > 0:
                rel = abs(r_sum - a_sum) / a_sum
                worst = max(worst, rel)
                rows.append((hyb_path.parent.parent.name, strat, a_sum, r_sum, rel))
    with capsys.disabled():
        print(f"\n{'model':>12} {'strategy':>12} {'astrasim_ns':>14} {'roofline_ns':>14} {'rel_err':>8}")
        for model, strat, a, r, rel in sorted(rows, key=lambda x: -x[4]):
            print(f"{model:>12} {strat:>12} {a:>14.1f} {r:>14.1f} {rel:>7.1%}")
        print(f"worst aggregated relative error = {worst:.1%} (tol {TOL:.0%})")
    assert worst <= TOL, f"roofline comm deviates {worst:.1%} > {TOL:.0%} from astra-sim"
```

- [ ] **Step 2: Run and inspect the delta table**

Run: `cd /home/haotian/amma-decode-modeling && python3 -m pytest tests/test_comm_model_vs_astrasim.py -q -s`
Expected: the per-strategy delta table prints. If `worst > 0.25`, the test FAILS — capture the printed worst value and report it to the user to set the final tolerance (do NOT silently raise TOL).

- [ ] **Step 3: Commit**

```bash
cd /home/haotian/amma-decode-modeling
git add tests/test_comm_model_vs_astrasim.py
git commit -q -m "test: bound roofline vs astra-sim comm error (tol 0.25, provisional)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Unified ASTRA-sim wrapper `astrasim/run_astrasim.py`

**Files:**
- Create: `/home/haotian/amma-decode-modeling/astrasim/run_astrasim.py`
- Create: `/home/haotian/amma-decode-modeling/astrasim/README.md`
- Reuse (import, do not fork): `backend/astrasim_runner/cache_db.py` (`load_config`, `parse_log_metrics`), `backend/astrasim_runner/run_from_config.py` (binary launch), `backend/hybrid_merge/analyze_comm.py` (`parse_log`), `backend/hybrid_merge/collect_gqa_data.py` (`collect_module_times`).

**Interfaces:**
- Consumes: env vars `ASTRASIM_BIN` (compiled `AstraSim_Analytical_Congestion_Aware`), `STG_HOME` (`symbolic_tensor_graph_picasso` for Chakra trace gen); a run-config JSON (same schema documented in `README.md` §2).
- Produces: CLI `python astrasim/run_astrasim.py --config <cfg.json> [--gen-trace] [--out reports/<model>/astrasim/<name>_data.json]`; on success writes an astrasim-schema JSON (`{strategy: {data: [{seq, wall_ns, gpu_ns, comm_ns, modules{...}}]}}`) consumable by `merge_gqa_results.py --comm-model astrasim`.

- [ ] **Step 1: Read the reused sources and record exact signatures**

Run: read `backend/astrasim_runner/cache_db.py`, `backend/astrasim_runner/run_from_config.py`, `backend/hybrid_merge/analyze_comm.py`, `backend/hybrid_merge/collect_gqa_data.py`.
Record: `load_config(path)` return keys; how `run_from_config.main` builds the AstraSim CLI (which config fields → which `--*-configuration` flags); `parse_log(log_path, npu=0)` return type; `collect_module_times(summary)` output dict. These are the exact glue points for Step 3.

- [ ] **Step 2: Write a smoke test for config parsing + external-tool detection**

Create `astrasim/tests/test_run_astrasim_smoke.py`:
```python
import os, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
WRAP = REPO / "astrasim" / "run_astrasim.py"

def test_errors_clearly_when_astrasim_bin_missing(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"workload_dir":"x","workload_base":"y","system":"s",'
                   '"network":"n","remote_memory":"r","output_dir":"o",'
                   '"log_file":"l.txt"}')
    env = dict(os.environ); env.pop("ASTRASIM_BIN", None)
    r = subprocess.run([sys.executable, str(WRAP), "--config", str(cfg)],
                       capture_output=True, text=True, env=env, cwd=REPO)
    assert r.returncode != 0
    assert "ASTRASIM_BIN" in (r.stderr + r.stdout)
```

- [ ] **Step 3: Write `run_astrasim.py` — one entry orchestrating the external toolchain**

Create `astrasim/run_astrasim.py`. Structure (fill the reused calls per Step 1 signatures):
```python
#!/usr/bin/env python3
"""Unified ASTRA-sim wrapper: config -> (optional trace-gen) -> external binary
-> parse log -> astrasim-schema JSON. External tools are located via env vars:
  ASTRASIM_BIN : path to AstraSim_Analytical_Congestion_Aware
  STG_HOME     : path to symbolic_tensor_graph_picasso (Chakra trace generator)
The pure-backend path (merge --comm-model roofline) never needs this script.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "backend" / "astrasim_runner"))
sys.path.insert(0, str(REPO / "backend" / "hybrid_merge"))
import cache_db            # load_config, parse_log_metrics
import analyze_comm        # parse_log
import collect_gqa_data    # collect_module_times


def _require(env_key: str) -> str:
    val = os.environ.get(env_key)
    if not val or not Path(val).exists():
        print(f"[ERROR] {env_key} not set or path missing: {val!r}. "
              f"Point it at your external ASTRA-sim install.", file=sys.stderr)
        sys.exit(3)
    return val


def main() -> int:
    ap = argparse.ArgumentParser(description="Run external ASTRA-sim from a config "
                                             "and emit astrasim-schema JSON.")
    ap.add_argument("--config", required=True, help="run-config JSON (see README §2)")
    ap.add_argument("--gen-trace", action="store_true",
                    help="regenerate the Chakra trace via $STG_HOME first")
    ap.add_argument("--out", default=None, help="astrasim JSON output path")
    args = ap.parse_args()

    astrasim_bin = _require("ASTRASIM_BIN")
    cfg = cache_db.load_config(args.config)          # {PROJECT_DIR}/{EXAMPLE_DIR} resolved

    if args.gen_trace:
        stg = _require("STG_HOME")
        # build + run the trace-gen command from cfg's trace sub-config (per Step 1)
        # subprocess.run([...], cwd=stg, check=True)

    # launch the binary (reuse run_from_config's CLI assembly, but with astrasim_bin
    # taken from ASTRASIM_BIN rather than the config's hardcoded path)
    # subprocess.run([...], check=True)

    log_path = Path(cfg["output_dir"]) / "logs" / cfg["log_file"]
    summary = analyze_comm.parse_log(str(log_path))
    modules = collect_gqa_data.collect_module_times(summary)
    metrics = cache_db.parse_log_metrics(str(log_path))
    record = {"seq": cfg.get("seq"), **metrics, "modules": modules}

    out = Path(args.out) if args.out else Path("reports/astrasim_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(record, open(out, "w"), indent=2)
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```
Also **fix the relocation path bug**: in `backend/astrasim_runner/run_from_config.py` (and `batch_run_configs.py`), the project-root is computed as `script_dir/'../../../..'`; after extraction to the new repo root the correct root is the repo root. Change those to resolve via `Path(__file__).resolve().parents[2]` (backend/astrasim_runner → repo root) so `{PROJECT_DIR}` templating works from the new layout. Verify no other file hard-codes the old astra_sim_picasso path.

- [ ] **Step 4: Write astrasim/README.md**

Create `astrasim/README.md` documenting: the two run modes (pure backend vs astra-sim), the required env vars (`ASTRASIM_BIN`, `STG_HOME`), the run-config schema (copy the relevant table from the top-level `README.md` §2), and an end-to-end example that reproduces one committed `reports/*/astrasim/*_data.json`.

- [ ] **Step 5: Run the smoke test**

Run: `cd /home/haotian/amma-decode-modeling && python3 -m pytest astrasim/tests/test_run_astrasim_smoke.py -q`
Expected: PASS — missing `ASTRASIM_BIN` yields a clear non-zero error.

- [ ] **Step 6: Commit**

```bash
cd /home/haotian/amma-decode-modeling
git add astrasim/ backend/astrasim_runner/run_from_config.py backend/astrasim_runner/batch_run_configs.py
git commit -q -m "astrasim: unified run_astrasim.py wrapper for external ASTRA-sim

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Top-level README, full repro verification, publish (user-gated)

**Files:**
- Create/Modify: `/home/haotian/amma-decode-modeling/README.md`

**Interfaces:**
- Produces: a documented repo that reproduces figures offline; a pushed GitHub repo (user-gated).

- [ ] **Step 1: Write the top-level README (two run modes)**

Rewrite `README.md` to lead with: (a) **pure-backend mode** — regenerate any figure with no astra-sim (`merge_gqa_results.py --comm-model roofline` → figs), and (b) **astra-sim mode** — `astrasim/run_astrasim.py` + `--comm-model astrasim` for congestion-aware comm. Keep the existing config/topology reference sections. Note the two validation tests.

- [ ] **Step 2: Regression — astrasim mode reproduces a committed hybrid**

Run `merge_gqa_results.py --comm-model astrasim` on one committed (roofline, astrasim) input pair and diff `comm_total_ns`/`hybrid_wall_ns` against the committed hybrid JSON.
Expected: identical numbers (this is the regression guarantee).

- [ ] **Step 3: End-to-end pure-backend figure regen (representative set)**

Run, from the repo root, one figure per pipeline and confirm fresh outputs:
```bash
python3 figures/paper_figures/fig1_e2e_latency/scripts/plot_e2e.py
python3 figures/paper_figures/fig5_time_breakdown/scripts/plot_breakdown.py
python3 figures/rebuttal_figures/exp1_e2e_decode/scripts/gen_decode_data.py && \
  python3 figures/rebuttal_figures/exp1_e2e_decode/scripts/plot_decode_breakdown.py
```
Expected: each exits 0 and writes its `plots/*.pdf` (mtime just now). fig5 exercises `tp16`/`hmp` — confirms roofline-mode hybrid feeds them.

- [ ] **Step 4: Run the full test suite**

Run: `cd /home/haotian/amma-decode-modeling && python3 -m pytest -q`
Expected: all green (comm-model, comm-vs-astrasim within tolerance, wrapper smoke, plus existing `backend/tests/verify_against_baseline.py`).

- [ ] **Step 5: Commit README + any regen'd artifacts**

```bash
cd /home/haotian/amma-decode-modeling
git add -A
git commit -q -m "docs: dual-mode README + verified offline figure reproduction

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Create GitHub repo and push (USER-GATED)**

STOP. Ask the user for the repo name/owner and confirm publishing. Only then:
```bash
cd /home/haotian/amma-decode-modeling
gh repo create <owner>/<name> --private --source=. --remote=origin --push
```
Expected: repo created, `main` pushed.

---

## Self-Review

**Spec coverage:**
- §4 layout → Task 1 (extract, drop tmp/, .gitignore). ✓
- §5 dual comm mode → Task 2 (flag, roofline breakdown, optional astrasim, metadata). ✓
- §6 wrapper → Task 4 (unified entry, env vars, path-bug fix). ✓
- §7 validation test → Task 3 (per-strategy tolerance + per-op print). ✓ `backend/tests/verify_against_baseline.py` kept → Task 5 Step 4. ✓
- §8 migration (checkpoint → fresh extract → changes → push) → Task 0, 1, 5. ✓
- §9 deepseek_v3_mla `.pyc` gap → not regenerated; figures use vendored JSON (carried in Task 1). Noted, no task needed. ✓
- §10 success criteria → Task 5 Steps 2–4 verify offline repro + both modes + tests. ✓

**Placeholder scan:** Task 4 Step 3 leaves the trace-gen/binary-launch subprocess lines as commented scaffolds intentionally gated behind Step 1 (reading exact reused signatures) — this is a read-then-assemble step, not a "figure it out" placeholder; the reused function names and the glue flow are specified. All other steps carry runnable code/commands.

**Type consistency:** `build_comm_breakdown_roofline(strategy, rf_strat)` return shape matches `build_comm_breakdown` (`ops/total_comm_ns/total_d2d_bytes_all_cubes`) so downstream `merge_entry`/summary code is unchanged. `merge_entry` new params (`as_entry: dict | None`, `comm_model`) are threaded from `main`. `comm_total_ns` field name consistent across Tasks 2/3/5.
