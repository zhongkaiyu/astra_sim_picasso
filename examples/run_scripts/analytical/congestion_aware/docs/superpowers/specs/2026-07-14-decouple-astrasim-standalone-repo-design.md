# Decouple the analytical backend from ASTRA-sim into a standalone repo (dual comm mode)

Date: 2026-07-14
Status: Design — awaiting user review before writing the implementation plan.

## 1. Context

The analytical modeling backend + paper/rebuttal figures currently live **nested inside**
the ASTRA-sim fork `zhongkaiyu/astra_sim_picasso` at
`examples/run_scripts/analytical/congestion_aware/`. A 21-agent read-only mapping
(2026-07-14) established the call chain and the exact ASTRA-sim coupling boundary.

**Two pipelines feed the figures** (`roofline/` is shared):
- **Pipeline A → `paper_figures/fig1–6`** uses ASTRA-sim.
- **Pipeline B → `rebuttal_figures/exp1–7`** is 100% ASTRA-sim-free (pure analytical `lpu_ffn_decode`/`roofline`, or H100/NCU profiling for exp2/exp6, or Yosys/CACTI for exp7).

**ASTRA-sim coupling boundary (verified):**
1. `backend/astrasim_runner/` (9 files) — pure ASTRA-sim driver: config → Chakra trace (via external `symbolic_tensor_graph_picasso`) → ASTRA-sim binary → SQLite/text log.
2. `backend/hybrid_merge/{analyze_comm, collect_gqa_data, collect_gqa_batch_data}` — parse ASTRA-sim logs → `reports/{model}/astrasim/*_data.json`.
3. `backend/hybrid_merge/merge_gqa_results.py` — **`--astrasim-data` is `required=True`**; merges ASTRA-sim + roofline → hybrid JSON. This is the only entry point, so there is currently **no roofline-only path**.
4. Downstream (`power_model.py`, `add_rubin_tp2.py`, fig1–6) consume the hybrid JSON → indirectly ASTRA-sim-dependent.

**Key enabling fact:** in `merge_gqa_results.build_comm_breakdown`, the `hmp_reo_new` ("Ours")
strategy comm is **already entirely roofline** (`qkv_allgather` + `attn_rs` + `final_reduce`,
all `source: roofline`). ASTRA-sim only contributes (a) the **baseline/HMP `attn_comm`** op and
(b) the compute term via `max(astrasim_compute, roofline)`. So a roofline-only mode is mostly
**wiring + adding a roofline `attn_comm` for baseline/HMP**, not new modeling from scratch.

## 2. Goals

- Extract `congestion_aware/` into a **standalone new GitHub repo**, no longer nested in ASTRA-sim.
- Make ASTRA-sim an **external, optional tool** invoked by a **single wrapper script**.
- Add a **`--comm-model {roofline,astrasim}` flag** so fig1–6 can be regenerated either purely
  from our backend (roofline comm, default) or with ASTRA-sim congestion-aware comm.
- Add **validation tests** bounding the error between roofline-computed comm and ASTRA-sim comm.
- The repo reproduces **all figures offline** from vendored data, with no ASTRA-sim install required.

## Non-goals / deferred

- **Figure/script pruning** — carry everything over as-is this round; prune after the user
  supplies the "keep these figures" list (a separate task).
- Recovering `backend/roofline/deepseek_v3_mla_roofline.py` source (only `.pyc` exists); the
  dependent data JSON is vendored so figures still render. Flagged as a known gap.
- Changing any figure's scientific content or the roofline/decode models themselves.

## 3. Decisions (locked with the user)

| # | Decision | Choice |
|---|---|---|
| 1 | Target structure | **Standalone new repo**; ASTRA-sim = external optional tool via wrapper |
| 2 | fig1–6 comm | **Add roofline-only comm path + `--comm-model` flag** (true dual mode) |
| 3 | Validation | **Add tests** bounding roofline-comm vs ASTRA-sim-comm error |
| 4 | Wrapper granularity | **One unified entry** `astrasim/run_astrasim.py`; binary/build/trace-gen stay external |
| 5 | git history | **Fresh start** (single initial commit); old history stays in astra_sim_picasso |
| 6 | Pruning timing | **Carry all now**, prune later |
| 7 | Checkpoint | **Commit the current 40 changes to astra_sim_picasso first**, then extract |

## 4. Target repo layout

Working name `amma-decode-modeling` (user may rename).

```
amma-decode-modeling/
├── backend/
│   ├── roofline/                 # core; calc_strategy already emits comm model (unchanged math)
│   ├── lpu_ffn_decode/ (+contbatch/)
│   ├── hybrid_merge/             # merge_gqa_results gains --comm-model
│   ├── baselines/  hw_estimation/
│   └── tests/                    # existing roofline regression (baseline_dataset.json)
├── astrasim/                     # NEW unified wrapper for the external ASTRA-sim tool
│   ├── run_astrasim.py           # config → external trace-gen + binary → parse log → astrasim JSON
│   └── README.md                 # env vars: ASTRASIM_BIN, STG_HOME
├── figures/                      # paper_figures + rebuttal_figures (+ their data/)
├── reports/                      # vendored model-result JSONs (roofline/astrasim/hybrid/power)
├── tests/                        # NEW: roofline-comm vs astrasim-comm validation
│   └── test_comm_model_vs_astrasim.py
├── docs/                         # this spec + methodology notes
├── README.md                     # documents the two run modes
├── requirements.txt
└── .gitignore                    # *.ncu-rep, build/, output*/, __pycache__, *.pyc
```

Dropped in the copy: `backend/tmp/` (stale duplicate of roofline/hybrid_merge).

## 5. Dual comm mode (core code change)

`backend/hybrid_merge/merge_gqa_results.py`:
- `--astrasim-data`: `required=True` → **optional**.
- New `--comm-model {roofline,astrasim}`, **default `roofline`**.
- **roofline mode:** comm entirely from `calc_strategy`'s analytical collectives; for baseline/HMP,
  add a roofline `attn_comm` (the one op currently ASTRA-sim-sourced). Compute term uses roofline
  util-adjusted only (drop the `max(astrasim_compute, …)`). No ASTRA-sim input needed.
- **astrasim mode:** current behaviour (requires `--astrasim-data`); unchanged numerically.
- Record `comm_source` / `comm_model` in the hybrid JSON `metadata`.
- Downstream (`add_rubin_tp2.py`, `power_model.py`, all fig1–6 scripts) **unchanged** — they read
  the hybrid JSON regardless of how comm was produced.

Result: fig1–6 regenerate from pure backend (default) or from ASTRA-sim (`--comm-model astrasim`).

## 6. ASTRA-sim wrapper `astrasim/run_astrasim.py`

Consolidate into one CLI entry: config parse/templating (`cache_db`) + trace-gen call + binary
launch (`run_from_config`) + log parse (`analyze_comm` / `collect_gqa_data`) → `astrasim JSON`.
- External deps located via env vars: `ASTRASIM_BIN` (compiled binary), `STG_HOME`
  (`symbolic_tensor_graph_picasso` trace generator); Chakra protobuf via the trace-gen repo.
- Clear error if ASTRA-sim is absent; the pure-backend path never needs it.
- **Fix the relocation path bugs** surfaced during the move (`run_from_config.py` /
  `batch_run_configs.py` compute the wrong project root after the directory move).
- Internals may stay modular under `astrasim/`; the point is a single documented entry point.

## 7. Validation test `tests/test_comm_model_vs_astrasim.py`

- Load each committed `reports/*/astrasim/*_data.json` (ASTRA-sim comm ground truth).
- For matching (model, strategy, seq, bs), compute roofline comm via `calc_strategy`.
- **Print a per-op comparison table** (roofline vs ASTRA-sim, absolute + relative delta).
- **Assert per-strategy aggregated comm relative error < tolerance.** Default **25%**; the user
  finalizes the number after inspecting the real deltas. Rationale: ASTRA-sim adds mesh2D
  congestion/routing over idealized ring/tree collectives, so some ops (e.g. AllToAll under
  congestion) may diverge more — hence per-op report + aggregate gate.
- Keep the existing `backend/tests/verify_against_baseline.py` (roofline internal regression vs
  `baseline_dataset.json`).

## 8. Migration plan

1. **Checkpoint:** commit the current 40 uncommitted changes in `astra_sim_picasso`
   (`features/latest-updates`), including new `contbatch/` and new fig1 plots; push. (User confirms push.)
2. **Extract fresh:** copy `congestion_aware/` into the new repo (root = former `congestion_aware/`),
   dropping `backend/tmp/`; add `.gitignore`; single initial commit. No history carry-over.
3. **Apply the code changes** (dual comm mode, wrapper, tests) in the new repo.
4. **Create the empty GitHub repo** and push. (User confirms remote + name + push.)

## 9. Risks / open items

- `deepseek_v3_mla_roofline.py` source missing (only `.pyc`) — fig1 MLA row / exp3 / exp4 depend on
  its vendored JSON; regeneration blocked until source is recovered. Not on this task's critical path.
- Roofline vs ASTRA-sim comm error may exceed 25% for congestion-sensitive ops; the test surfaces
  this and the tolerance is set by the user afterward.
- Stale doc drift (README §6, captions) carried over as-is; cleaned during the later pruning task.

## 10. Success criteria

- New standalone repo builds and **reproduces every figure offline** with pure Python (matplotlib),
  no ASTRA-sim install (using vendored `reports/` + figure `data/`).
- `merge_gqa_results.py --comm-model roofline` regenerates fig1–6 hybrid JSON with **no** ASTRA-sim
  input; `--comm-model astrasim` reproduces the current numbers.
- `astrasim/run_astrasim.py` runs end-to-end against an external ASTRA-sim install and reproduces a
  committed `reports/*/astrasim/*_data.json`.
- `tests/test_comm_model_vs_astrasim.py` passes at the agreed tolerance and prints the per-op table.
- `backend/tests/verify_against_baseline.py` stays green.
