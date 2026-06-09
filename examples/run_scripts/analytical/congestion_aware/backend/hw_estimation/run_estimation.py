#!/usr/bin/env python3
"""
AMMA HMP_REO Per-Cube 4nm Hardware Estimation Pipeline
======================================================

This script estimates power & area of one AMMA cube (× 16 for full chip),
decomposed into 10 components A-J defined in PLAN.md §2.

Per-component decoupling principle
-----------------------------------
Each component has:
  - A unique code letter (A-J)
  - A tool tag ("yosys" / "cacti" / "reference" / "derived")
  - Independent input file(s) and instance count
  - A standardized output: {area_mm2, dynamic_w, leakage_w, total_w}
Adding/removing/changing one component does NOT touch the others.

CACTI 22nm → 4nm scaling (PLAN.md §4)
--------------------------------------
  Area:    × 0.129
  Energy:  × 0.229   (per-access dynamic energy)
  Leakage: × 0.311
  Delay:   × 0.436

Yosys 4nm GE-based estimation
------------------------------
  Density:    ≈ 5 M GE/mm²       → area  ≈ cells / 5e6  mm²  (assume 1 cell ≈ 1 GE)
  Dynamic:    ≈ 0.5 µW/GE/MHz    → P_dyn ≈ cells × 0.5e-3 × f_MHz × U  mW
  Leakage:    ≈ 0.01 µW/GE       → P_lkg ≈ cells × 0.01e-3  mW

Usage
-----
  python run_estimation.py [--cacti PATH] [--yosys PATH] [--only A,B,...]
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import asap7_gate_model as gate   # physical gate-level area+power from ASAP7 .lib

# ============================================================================
# Paths
# ============================================================================
HERE         = Path(__file__).resolve().parent
CACTI_INPUT  = HERE / "cacti" / "input"
CACTI_OUTPUT = HERE / "cacti" / "output"
YOSYS_INPUT  = HERE / "yosys" / "input"
YOSYS_OUTPUT = HERE / "yosys" / "output"
ARCH_FILE    = HERE / "architecture.json"
RESULT_FILE  = HERE / "estimation_results.json"

# ============================================================================
# Global Constants (PLAN.md §4 + §1)
# ============================================================================

# --- CACTI 22nm → 4nm composite scaling factors ---
SCALE_AREA    = 0.129
SCALE_ENERGY  = 0.229
SCALE_LEAKAGE = 0.311
SCALE_DELAY   = 0.436

# --- Yosys 4nm GE constants (calibrated to industry data for 4nm CMOS) ---
# Yosys "generic synth" cell ≈ 2-3 GE on average; the dynamic constant below
# assumes ~10% switching activity baked in. Multiply by component utilization U
# to scale further by workload intensity.
YOSYS_GE_DENSITY_PER_MM2     = 5e6        # 5 M GE/mm² (4nm logic density)
YOSYS_DYN_UW_PER_GE_PER_MHZ  = 0.05e-3    # 0.05 nW/GE/MHz at 4nm (typical α=10%)
YOSYS_LKG_UW_PER_GE          = 0.01e-3    # 10 pW/GE (4nm FinFET/GAA)

# --- ASAP7 PDK integration (real standard-cell synthesis) ---
ASAP7_LIB_DIR = "/home/haotian/waferchip/asap7/asap7sc7p5t_28/LIB/NLDM"
ASAP7_LIBS = [
    f"{ASAP7_LIB_DIR}/asap7sc7p5t_AO_RVT_TT_nldm_211120.lib",
    f"{ASAP7_LIB_DIR}/asap7sc7p5t_INVBUF_RVT_TT_nldm_220122.lib",
    f"{ASAP7_LIB_DIR}/asap7sc7p5t_OA_RVT_TT_nldm_211120.lib",
    f"{ASAP7_LIB_DIR}/asap7sc7p5t_SIMPLE_RVT_TT_nldm_211120.lib",
    f"{ASAP7_LIB_DIR}/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib",
]
ASAP7_SEQ_LIB = f"{ASAP7_LIB_DIR}/asap7sc7p5t_SEQ_RVT_TT_nldm_220123.lib"  # for dfflibmap
USE_ASAP7 = all(os.path.isfile(f) for f in ASAP7_LIBS)

# Leakage temperature factor: ASAP7 .lib is characterized at TT/25°C; real
# operation is ~85-105°C where sub-threshold leakage is ~5-10× higher.
LEAKAGE_TEMP_FACTOR = 8.0   # 25°C → ~90°C operating (documented knob)

# Base switching-activity toggle rate (fraction of nodes toggling per clock at
# FULL throughput). Effective α = TOGGLE_RATE × workload_utilization.
TOGGLE_RATE_FULL = 0.15

# 7nm → 4nm scaling — TSMC N7→N4 researched factors (2026-05, see RESULT_ASAP7 §6)
# KEY INSIGHT: "4nm" is marketing — TSMC N4 is a refined/design-rule-compatible
# member of the N5 family (+6% density via N4P), NOT a true new node. The real
# generational jump is N7→N5. ASAP7 is predictive 7nm, so N7→N4 = N7→N5 × N5→N4.
#
# Per-metric factors (multiply 7nm value to get 4nm):
#   LOGIC area: measured N7→N5 density 1.518× (Angstronomics A14/A15 teardowns,
#               NOT TSMC's 1.84× marketing) × N4P 1.06× = 1.609× → area 1/1.609 = 0.62
#   DYNAMIC pwr: N7→N5 −30% iso-perf (TSMC) + N4P 22% efficiency → ~0.68× iso-freq
#   LEAKAGE:    Vdd plateaued ~0.75-0.8V across N7/N5/N4 (Dennard dead) → leakage
#               does NOT track area; conservative ~0.80×
#   DELAY:      N7→N5 ~15% faster iso-power + N4P 11% → ~0.80×
SCALE_7NM_TO_4NM = {
    "area_logic": 0.62,   # 1 / (1.518 × 1.06);  Angstronomics + N4P
    "area_sram":  0.74,   # 1 / 1.35; SRAM scaling STALLED (ISSCC2020, vs 1.84× logic)
    "area":       0.62,   # default = logic (used by A/C/D/E/F which are logic)
    "power":      0.68,   # −32% iso-freq dynamic (TSMC −30% + N4P)
    "leakage":    0.80,   # conservative; Vdd flat removes Dennard leakage scaling
    "delay":      0.80,   # N7→N5 15% + N4P 11%
}

# --- Clock + utilization defaults ---
CLOCK_GHZ           = 2.0       # AMMA SAs operate at 2 GHz
DECODE_UTILIZATION  = 0.50      # avg activity during decode attention
N_CUBES             = 16        # 4×4 chip

# ============================================================================
# COMPONENT REGISTRY  (PLAN.md §2 — components A through J)
# Each entry is self-contained. To add a component, add a new row.
# ============================================================================

COMPONENTS = {
    # -----------------------------------------------------------------------
    "A": {
        "name":           "MAC arrays (96× 16×16 systolic arrays @ 2 GHz)",
        "tool":           "yosys",
        "rtl_file":       "sa_16x16.v",
        "top_module":     "sa_16x16",
        "instances":      96,                 # 12 cores × 8 SAs
        "utilization":    0.30,               # decode is mem-bound; avg MAC util ~30%
        "clock_ghz":      2.0,
        "layout_utilization": 0.55,           # datapath placement density
        "budget_w":       4.50,               # PLAN §视角①
        # PHYSICAL area model (PLAN_ASAP7_REALISTIC §2): ASAP7 synth is gate-limited.
        #   rtl_completeness 3.7 = pipeline 2.5 (1→3-4 stages for 2GHz) × datapath 1.3
        #     (full INT8 sat/round/accum) × DFT 1.15 (scan)
        #   physical_overhead 1.8 = wire 1.4 × clk/buffer 1.2 × power-grid 1.07
        #   (÷util already applied via layout_utilization=0.55)
        "rtl_completeness_factor":  3.7,
        "physical_overhead_factor": 1.8,
        # Power also scales with the SWITCHING part of RTL completeness:
        #   pipeline registers (2.5) × full datapath (1.3) = 3.25; DFT (1.15) EXCLUDED
        #   (scan is idle in normal operation). Pipeline FFs toggle every cycle.
        "power_rtl_factor": 3.25,
        "notes":          "96 SAs × 256 MAC = 24576 MAC. Area: synth×rtl(3.7)×phys(1.8). Power: ASAP7 gate × power_rtl(3.25, pipeline+datapath, DFT excl).",
    },

    # -----------------------------------------------------------------------
    "B": {
        "name":           "SRAM buffers (3 MB/cube: InBufA + InBufB + OutBuf)",
        "tool":           "cacti",
        "cacti_cfgs":     ["inbufA_aggregated.cfg",   # 1.2 MB (12.8 KB × 96 SA)
                           "inbufB_aggregated.cfg",   # 1.2 MB
                           "outbuf_aggregated.cfg"],  # 0.6 MB
        "instances":      1,                  # macros are pre-aggregated per cube
        # Models 96-way distributed-macro reality where each SA reads its own slice;
        # ~50 G-accesses/sec per macro × 2 GHz = 100 GHz aggregate effective rate
        "access_rate_ghz": 100.0,
        "utilization":    0.50,               # avg during decode (some cycles idle)
        # CACTI assumes ideal macro packing; real ASIC has periphery routing + decoder + banking
        # overhead. Padding factor ~2× brings area from CACTI 0.46 → ~1 mm² per cube.
        "area_padding_factor": 2.0,
        # PHYSICAL area model (PLAN_ASAP7_REALISTIC §1): tiny multiported double-
        # buffered buffers are FF register files, not dense SRAM macros. Area from
        # ASAP7 DFF cell × 25.2 Mbit × overhead. (This is why H100 blended SRAM
        # density is ~4 mm²/MB, not the 0.6-0.9 of pure 6T SRAM.) Power kept from CACTI.
        "area_model":      "register_file",
        "rf_total_mb":     3.0,
        "rf_overhead":     2.0,            # R/W mux + decode + double-buffer + routing
        "budget_w":       3.75,
        "notes":          "Per-SA 32 KB × 96 SA = 3 MB; area = FF register-file (ASAP7 DFF), power from CACTI",
    },

    # -----------------------------------------------------------------------
    "C": {
        "name":           "Two-level crossbar (12× local 8×8 + 1× global 12×12)",
        "tool":           "yosys",
        "rtl_file":       "xbar_two_level.v",
        "top_module":     "xbar_two_level",
        "instances":      1,                  # the top wraps all 13 sub-xbars
        "utilization":    0.40,
        "clock_ghz":      2.0,
        "layout_utilization": 0.55,           # crossbar wiring congestion
        "budget_w":       1.50,
        # PHYSICAL area model (PLAN_ASAP7_REALISTIC §3): crossbar is WIRE-dominated.
        #   A_mux  = ASAP7 synth (gate logic, ~0.02 mm²)
        #   A_wire = analytical global-interconnect: 128-bit × 12 ports × 2 (bidir),
        #            die-spanning ~11 mm, M4 pitch ~130 nm, fill 0.45 (track-limited)
        "wire_area_model": {"n_bits": 128, "n_ports": 12, "span_mm": 11.0,
                            "pitch_nm": 130.0, "fill": 0.45},
        "notes":          "Replaces generic 5-port router; broadcasts query + collects partial sums",
    },

    # -----------------------------------------------------------------------
    "D": {
        "name":           "D2D UCIe MAC-layer control (4 ports/cube)",
        "tool":           "yosys",
        "rtl_file":       "d2d_link_ctrl.v",
        "top_module":     "d2d_link_ctrl",
        "instances":      4,                  # 4 D2D ports per cube (N/S/E/W)
        "utilization":    0.25,
        "clock_ghz":      2.0,
        "layout_utilization": 0.60,
        "budget_w":       0.45,
        # First-principles: UCIe MAC layer (CRC, credit, framing, retry) ~0.5 mm²/port at 4nm
        "area_override_mm2": 2.0,
        "area_override_source": "4 ports × ~0.5 mm²/port (UCIe MAC layer + retry/CRC at 4nm)",
        "notes":          "UCIe 3.0 protocol layer; PHY (G) is separate analog",
    },

    # -----------------------------------------------------------------------
    "E": {
        "name":           "DMA engine + instruction frontend (per core)",
        "tool":           "yosys",
        "rtl_files":      ["dma_engine.v", "instr_frontend.v"],
        "top_modules":    ["dma_engine", "instr_frontend"],
        "instances":      12,                 # 12 cores per cube
        "utilization":    0.50,               # constant data movement in decode
        "clock_ghz":      2.0,
        "layout_utilization": 0.55,           # control logic, lower density
        "budget_w":       1.20,
        # First-principles: per-core control (DMA + IMEM + decode + scoreboard) ~1 mm²/core
        "area_override_mm2": 12.0,
        "area_override_source": "12 cores × ~1 mm²/core (DMA + IMEM 64 instr + decoder + scoreboard at 4nm)",
        "notes":          "DMA prefetches tile while SA computes (double-buffer)",
    },

    # -----------------------------------------------------------------------
    "F": {
        "name":           "Vector unit (softmax / layer-norm, per core)",
        "tool":           "yosys",
        "rtl_file":       "vector_unit.v",
        "top_module":     "vector_unit",
        "instances":      12,
        "utilization":    0.20,               # only active post-SA output
        "clock_ghz":      2.0,
        "layout_utilization": 0.60,
        "budget_w":       0.75,
        # First-principles: FP16 16-lane SIMD (EXP+reduce+reciprocal) ~0.75 mm² × 12 cores
        "area_override_mm2": 9.0,
        "area_override_source": "12 cores × 0.75 mm²/vector unit (16-lane FP16 SIMD pipeline at 4nm)",
        "notes":          "FP16 exp + reduce + reciprocal; reads from OutBuf in pipeline",
    },

    # -----------------------------------------------------------------------
    "G": {
        "name":           "D2D PHY (UCIe SerDes, 4 ports × 1.5 TB/s)",
        "tool":           "reference",
        "reference": {
            "power_w_per_cube": 1.50,
            "energy_pj_per_bit": 0.38,        # from PLAN §1
            # UCIe-Std PHY published density ~0.5 mm²/Tbps (bidirectional aggregate).
            # 4 ports × 1.5 TB/s = 6 TB/s → ~3 mm² (was 0.6 mm², under-counted by 5×).
            "area_mm2_per_cube": 3.0,
            "rationale":        "AMMA: 0.38 pJ/bit; UCIe-Std PHY density ~0.5 mm²/Tbps × 6 TB/s = 3 mm²",
        },
        "budget_w":       1.50,
        "notes":          "Analog/mixed-signal; PHY block on the base logic die",
    },

    # -----------------------------------------------------------------------
    "H": {
        "name":           "HBM4 stack + PHY",
        "tool":           "reference",
        "reference": {
            "power_w_per_cube":  75.0,         # AMMA Table 1
            "capacity_gb":       36,
            "bandwidth_tbs":     2.75,
            # 3D-stacked HBM4 footprint: 11 mm × 11 mm = 121 mm² per die (user-specified)
            "area_mm2_per_cube": 121.0,
            "area_breakdown_mm2": {
                "single_die_dimensions": "11 mm × 11 mm",
                "single_die_area":   121.0,
                "stack_height":      "12-Hi (12 DRAM dies + 1 base logic die, all 11×11)",
                "package_footprint": 121.0,    # = single die area (3D stacked share XY)
                "total_silicon_per_cube_mm2": 121.0 * 13,  # 12 DRAM + 1 base logic
            },
            "rationale":         ("Power from AMMA Table 1 (75 W/cube). "
                                  "Die size 11×11 mm user-specified."),
        },
        "budget_w":       75.0,
        "notes":          "DRAM 3D-stacked atop base logic die (which holds HBM PHY + AMMA compute). Package footprint = single die area; total silicon = 13 × die area.",
    },

    # -----------------------------------------------------------------------
    "I": {
        "name":           "Clock distribution network",
        "tool":           "derived",
        "method":         "fraction_of_logic_dynamic",
        "fraction":       0.12,               # 12% of A+C+D+E+F dynamic
        "depends_on":     ["A", "C", "D", "E", "F"],
        "budget_w":       1.80,
        "notes":          "Empirical: clock tree ≈ 10–15% of dynamic logic power",
    },

    # -----------------------------------------------------------------------
    "J": {
        "name":           "Total leakage (logic + SRAM)",
        "tool":           "derived",
        "method":         "sum_of_leakage",
        "depends_on":     ["A", "B", "C", "D", "E", "F"],
        "budget_w":       1.00,
        "notes":          "Aggregates leakage already estimated in A–F (no double-count)",
    },

    # -----------------------------------------------------------------------
    # NEW (P0): HBM PHY + memory controller — resides on the base logic die
    # alongside AMMA compute. Power is already bundled in H's 75 W per cube;
    # we set K's power to 0 to avoid double-counting, but add 25 mm² area.
    "K": {
        "name":           "HBM PHY + memory controller (base logic die)",
        "tool":           "reference",
        "reference": {
            "area_mm2_per_cube": 25.0,
            "power_w_per_cube":  0.0,
            "rationale":         ("Standard HBM4 base-die logic region (~20% of 121 mm² die). "
                                  "Power bundled in H (75 W/cube) to avoid double-count. "
                                  "Area from architecture.json::base_logic_die.area_budget_mm2."),
        },
        "budget_w":       0.0,
        "notes":          "On-die alongside AMMA compute (A–G); counted in logic-die area total",
    },

    # NEW (P3): Die-level overhead — IO ring + decoupling cap + clock physical
    "L": {
        "name":           "Die-level overhead (IO ring + decap + clock physical)",
        "tool":           "derived",
        "method":         "fraction_of_logic_die_area",
        "fraction":       0.07,                   # 7% of logic die area
        "depends_on":     ["A", "B", "C", "D", "E", "F", "G", "K"],
        "budget_w":       0.0,
        "notes":          "Empirical: IO ring ~3%, decap ~3%, clock-tree physical area ~1%",
    },
}


# ============================================================================
# Standard Output Schema
# ============================================================================

def make_result(code, name, tool, area_mm2=0.0, dynamic_w=0.0, leakage_w=0.0,
                instances=1, budget_w=0.0, raw=None, error=None):
    """All components return this dict shape."""
    total = dynamic_w + leakage_w
    return {
        "code":               code,
        "name":               name,
        "tool":               tool,
        "instances_per_cube": instances,
        "area_mm2_per_cube":  round(area_mm2, 6),
        "dynamic_w_per_cube": round(dynamic_w, 4),
        "leakage_w_per_cube": round(leakage_w, 4),
        "total_w_per_cube":   round(total, 4),
        "budget_w_per_cube":  budget_w,
        "vs_budget_pct":      round(100 * total / budget_w, 1) if budget_w > 0 else None,
        "raw":                raw or {},
        "error":              error,
    }


# ============================================================================
# Tool Wrappers (CACTI + Yosys)
# ============================================================================

def find_tool(name, hint=None):
    if hint and os.path.isfile(hint):
        return hint
    defaults = {
        "cacti": "/home/haotian/waferchip/cacti/cacti",
        "yosys": "/home/haotian/miniconda3/bin/yosys",
    }
    if name in defaults and os.path.isfile(defaults[name]):
        return defaults[name]
    from shutil import which
    return which(name)


def run_cacti(cacti_bin, cfg_path, out_path):
    """Run CACTI on one .cfg; return parsed metrics dict (or empty).
    CACTI reads tech_params/*.dat relative to CWD, so we chdir to its install dir."""
    cacti_dir = os.path.dirname(os.path.abspath(cacti_bin))
    try:
        r = subprocess.run([cacti_bin, "-infile", str(cfg_path.resolve())],
                           capture_output=True, text=True, timeout=120,
                           cwd=cacti_dir)
    except Exception as e:
        return {"_error": str(e)}
    out_path.write_text(r.stdout + (f"\n[STDERR]\n{r.stderr}\n" if r.returncode != 0 else ""))
    return parse_cacti(r.stdout)


def parse_cacti(text):
    """Extract Access time / Energy / Leakage / Area from CACTI 22nm output."""
    N = r"([-+]?[\d.]+(?:[eE][+-]?\d+)?)"
    patterns = {
        "access_time_ns":          rf"Access time \(ns\):\s+{N}",
        "cycle_time_ns":           rf"Cycle time \(ns\):\s+{N}",
        "dyn_read_energy_nj":      rf"Total dynamic read energy per access \(nJ\):\s+{N}",
        "dyn_write_energy_nj":     rf"Total dynamic write energy per access \(nJ\):\s+{N}",
        "leakage_mw":              rf"Total leakage power of a bank \(mW\):\s+{N}",
        "gate_leakage_mw":         rf"Total gate leakage power of a bank \(mW\):\s+{N}",
    }
    m = {}
    for k, p in patterns.items():
        x = re.search(p, text)
        if x:
            m[k] = float(x.group(1))
    am = re.search(rf"Cache height x width \(mm\):\s+{N}\s*x\s*{N}", text)
    if am:
        m["area_mm2"] = float(am.group(1)) * float(am.group(2))
    return m


def run_yosys(yosys_bin, rtl_path, top, out_path):
    """Synthesize one RTL file with ASAP7 (if libs available) else generic.
    Returns dict with cells_first, cells_last, and (if ASAP7 used) area_um2_7nm."""
    if USE_ASAP7:
        lib_args = " ".join(f"-liberty {lib}" for lib in ASAP7_LIBS)
        script = (
            f"read_verilog {rtl_path}; "
            f"synth -top {top}; "
            f"dfflibmap -liberty {ASAP7_SEQ_LIB}; "
            f"abc {lib_args}; "
            f"stat {lib_args}; "
            f"write_json {out_path}.json"
        )
    else:
        script = f"read_verilog {rtl_path}; synth -top {top}; stat; write_json {out_path}.json"

    try:
        r = subprocess.run([yosys_bin, "-Q", "-p", script],
                           capture_output=True, text=True, timeout=600)
    except Exception as e:
        return {"_error": str(e)}
    out_path.write_text(r.stdout + (f"\n[STDERR]\n{r.stderr}\n" if r.returncode != 0 else ""))
    if r.returncode != 0:
        return {"_error": f"yosys exit {r.returncode}"}
    return parse_yosys(r.stdout)


def parse_yosys(text):
    """Extract cells + (if ASAP7 used) chip area in µm² from Yosys 'stat'.

    - cells_first: pre-flatten count (small; for legacy dyn-power calibration)
    - cells_last:  post-flatten, post-ABC-mapping count (real complexity)
    - area_um2_7nm: real chip area at 7nm from ASAP7 `stat -liberty` (if available)
    - asap7_used: whether the chip-area line was found
    """
    matches = re.findall(r"Number of cells:\s+(\d+)", text)
    if not matches:
        return {}

    result = {
        "cells_first": int(matches[0]),
        "cells_last":  int(matches[-1]),
        "cells":       int(matches[0]),
    }

    # ASAP7 `stat -liberty` adds one line per module:
    #   Chip area for top module '\sa_16x16': 16400.517120     (multi-module designs)
    #   Chip area for module '\dma_engine': 642.234420         (single-module designs)
    # Take the LAST occurrence — top module always reported last after submodules.
    area_matches = re.findall(r"Chip area for (?:top )?module .*?:\s+([\d.]+)", text)
    if area_matches:
        result["area_um2_7nm"] = float(area_matches[-1])
        result["asap7_used"]   = True
    else:
        result["asap7_used"]   = False

    return result


# ============================================================================
# Per-Component Estimators (one function per tool type)
# Each estimator takes (code, defn) and returns a make_result() dict.
# Components are decoupled: each can be called/disabled independently.
# ============================================================================

def estimate_yosys(code, d):
    """Estimate one Yosys-modeled component (A, C, D, E, F)."""
    yosys = find_tool("yosys")
    if not yosys:
        return make_result(code, d["name"], "yosys", error="yosys binary not found")

    # Some components (E) have multiple RTL files
    rtl_files = d.get("rtl_files", [d.get("rtl_file")])
    tops      = d.get("top_modules", [d.get("top_module")])

    n_inst   = d["instances"]
    f_ghz    = d["clock_ghz"]
    util     = d["utilization"]
    # Effective switching activity α = full-throughput toggle rate × workload duty
    alpha    = d.get("switching_activity", TOGGLE_RATE_FULL * util)
    util_lay = d.get("layout_utilization", 0.65)   # P&R cell-area density

    # Per-module synth → gate-level physical area/power @ 7nm (per instance)
    cells_first_total = 0
    area_synth_um2 = 0.0     # per single instance, summed across this comp's RTL files
    area_layout_um2 = 0.0
    p_dyn_7nm = 0.0
    p_leak_7nm = 0.0
    any_asap7 = False
    cells_last_total = 0
    raw = {"per_module": {}}

    for rtl, top in zip(rtl_files, tops):
        rtl_path = YOSYS_INPUT / rtl
        if not rtl_path.exists():
            return make_result(code, d["name"], "yosys",
                               error=f"RTL not found: {rtl_path}")
        out_path = YOSYS_OUTPUT / f"{code}_{Path(rtl).stem}.txt"
        m = run_yosys(yosys, rtl_path, top, out_path)
        if "_error" in m:
            return make_result(code, d["name"], "yosys", error=m["_error"])
        cells_first_total += m.get("cells_first", 0)
        cells_last_total  += m.get("cells_last", 0)
        any_asap7 |= m.get("asap7_used", False)

        # Physical gate-level model from the stat output
        gb = gate.estimate_block(out_path.read_text(), activity=alpha,
                                 clock_ghz=f_ghz, utilization_layout=util_lay)
        area_synth_um2  += gb["area_synth_um2_7nm"]
        area_layout_um2 += gb["area_layout_mm2_7nm"] * 1e6
        p_dyn_7nm  += gb["p_dynamic_w_7nm"]
        p_leak_7nm += gb["p_leakage_w_7nm"]
        raw["per_module"][rtl] = {
            "cells": gb["total_cells"],
            "area_synth_um2_7nm": round(gb["area_synth_um2_7nm"], 2),
            "cap_total_ff": round(gb["cap_total_ff"], 1),
            "unmatched_types": len(gb["unmatched"]),
        }

    # Scale per-instance 7nm → per-cube 4nm
    area_layout_mm2_7nm = area_layout_um2 / 1e6 * n_inst          # synth ÷ util
    area_synth_4nm = area_layout_mm2_7nm * SCALE_7NM_TO_4NM["area"]   # ×0.62

    # Power-completeness factor: the skeleton-RTL gate model under-counts switching
    # cells the full design has (pipeline registers + full datapath). Pipeline FFs
    # toggle every cycle → real dynamic power. DFT/scan is EXCLUDED (idle in normal
    # operation). Default = pipeline × datapath part of rtl_completeness (no DFT).
    pwr_rtl = d.get("power_rtl_factor", 1.0)
    dynamic_w = p_dyn_7nm  * n_inst * SCALE_7NM_TO_4NM["power"] * pwr_rtl   # ×0.68
    leakage_w = (p_leak_7nm * n_inst * SCALE_7NM_TO_4NM["leakage"]
                 * LEAKAGE_TEMP_FACTOR * pwr_rtl)                  # ×0.80 ×temp

    if d.get("prefer_override") and d.get("area_override_mm2") is not None:
        area_mm2 = d["area_override_mm2"]
        raw["area_source"] = "first_principles_override (user-pinned)"
    elif any_asap7:
        # PHYSICAL area model (PLAN_ASAP7_REALISTIC §2/§3):
        #   ASAP7 synth (gate-limited) × RTL-completeness × physical-impl + wire
        rtl_c = d.get("rtl_completeness_factor", 1.0)   # skeleton RTL → full pipe+DFT
        phys  = d.get("physical_overhead_factor", 1.0)  # routing+clk+power-grid beyond util
        area_mm2 = area_synth_4nm * rtl_c * phys
        src = f"ASAP7 synth ÷util ×0.62 ×rtl({rtl_c}) ×phys({phys})"
        # Crossbar / interconnect: add analytical global-wire area (track-limited)
        if d.get("wire_area_model"):
            wa = gate.estimate_wire_area(**d["wire_area_model"])
            area_mm2 += wa["area_mm2"]
            raw["wire_area_mm2"] = round(wa["area_mm2"], 4)
            raw["wire_model"]    = wa
            src += f" + wire({wa['area_mm2']:.2f})"
        raw["area_source"]  = src
        raw["rtl_completeness_factor"] = rtl_c
        raw["physical_overhead_factor"] = phys
        raw["area_synth_4nm_mm2"] = round(area_synth_4nm, 6)
    else:
        area_mm2 = cells_last_total * n_inst / YOSYS_GE_DENSITY_PER_MM2
        raw["area_source"] = "yosys_cells generic (ASAP7 unavailable)"

    raw.update({
        "n_instances":            n_inst,
        "switching_activity":     alpha,
        "layout_utilization":     util_lay,
        "clock_ghz":              f_ghz,
        "vdd":                    gate.ASAP7_VDD,
        "leakage_temp_factor":    LEAKAGE_TEMP_FACTOR,
        "area_synth_mm2_7nm":     round(area_synth_um2 / 1e6 * n_inst, 6),
        "area_layout_mm2_7nm":    round(area_layout_mm2_7nm, 6),
        "area_4nm_mm2":           round(area_mm2, 6),
        "p_dyn_7nm_w":            round(p_dyn_7nm * n_inst, 6),
        "p_leak_7nm_w_25C":       round(p_leak_7nm * n_inst, 8),
        "scaling": dict(SCALE_7NM_TO_4NM),
    })
    return make_result(code, d["name"], "yosys",
                       area_mm2=area_mm2, dynamic_w=dynamic_w, leakage_w=leakage_w,
                       instances=n_inst, budget_w=d["budget_w"], raw=raw)


def estimate_cacti(code, d):
    """Estimate one CACTI-modeled component (B)."""
    cacti = find_tool("cacti")
    if not cacti:
        return make_result(code, d["name"], "cacti", error="cacti binary not found")

    f_ghz  = d["access_rate_ghz"]
    util   = d["utilization"]
    n_inst = d["instances"]

    area_22nm   = 0.0
    dyn_e_22nm  = 0.0    # nJ per access, summed read+write proxy
    leak_22nm   = 0.0    # mW
    raw = {"per_macro": {}}

    for cfg in d["cacti_cfgs"]:
        cfg_path = CACTI_INPUT / cfg
        if not cfg_path.exists():
            return make_result(code, d["name"], "cacti",
                               error=f"CACTI cfg not found: {cfg_path}")
        out_path = CACTI_OUTPUT / f"{code}_{Path(cfg).stem}.txt"
        m = run_cacti(cacti, cfg_path, out_path)
        if "_error" in m:
            return make_result(code, d["name"], "cacti", error=m["_error"])
        if "area_mm2" not in m:
            return make_result(code, d["name"], "cacti",
                               error=f"no metrics parsed from {cfg}; check CACTI output")

        # Aggregate
        area_22nm   += m["area_mm2"]
        # Average of read/write energy (or just read if write missing)
        e_rd = m.get("dyn_read_energy_nj", 0.0)
        e_wr = m.get("dyn_write_energy_nj", e_rd)
        dyn_e_22nm += (e_rd + e_wr) / 2.0
        leak_22nm  += m.get("leakage_mw", 0.0) + m.get("gate_leakage_mw", 0.0)
        raw["per_macro"][cfg] = m

    # 22nm → 4nm scaling
    area_4nm    = area_22nm  * SCALE_AREA
    dyn_e_4nm   = dyn_e_22nm * SCALE_ENERGY   # nJ per access at 4nm
    leak_4nm_mw = leak_22nm  * SCALE_LEAKAGE  # mW

    # Dynamic power: E[nJ] × f[GHz] = W directly (1 nJ × 1e9 /s = 1 W)
    dynamic_w   = dyn_e_4nm * f_ghz * util
    leakage_w   = leak_4nm_mw / 1000.0          # mW → W

    # Aggregate across instances (B already pre-aggregated to per-cube; instances=1)
    area_mm2    = area_4nm  * n_inst
    dynamic_w  *= n_inst
    leakage_w  *= n_inst

    # Apply optional area padding factor (e.g. periphery/decoder overhead in real ASIC)
    pad = d.get("area_padding_factor", 1.0)
    if pad != 1.0:
        raw["area_pre_padding_mm2"] = area_mm2
        raw["area_padding_factor"]  = pad
        area_mm2 *= pad

    # Physical area model: FF register-file (PLAN_ASAP7_REALISTIC §1).
    # Replaces CACTI macro area with ASAP7-DFF-based register-file area, because
    # tiny multiported double-buffered buffers are physically FF-RF. Keep CACTI POWER.
    if d.get("area_model") == "register_file":
        bits = int(d["rf_total_mb"] * 1024 * 1024 * 8)
        rf = gate.estimate_register_file(bits, overhead=d.get("rf_overhead", 2.0))
        raw["area_cacti_mm2"] = area_mm2
        area_mm2 = rf["area_mm2_7nm"] * SCALE_7NM_TO_4NM["area"]   # 7→4nm
        raw["area_source"]    = "FF register-file (ASAP7 DFF × bits × overhead) × 7→4nm"
        raw["register_file"]  = {**rf, "area_mm2_4nm": round(area_mm2, 4)}

    raw.update({
        "area_22nm_mm2":        area_22nm,
        "leakage_22nm_mw":      leak_22nm,
        "dyn_energy_22nm_nJ":   dyn_e_22nm,
        "access_rate_ghz":      f_ghz,
        "utilization":          util,
        "scaling": {"area": SCALE_AREA, "energy": SCALE_ENERGY, "leakage": SCALE_LEAKAGE},
    })
    return make_result(code, d["name"], "cacti",
                       area_mm2=area_mm2, dynamic_w=dynamic_w, leakage_w=leakage_w,
                       instances=n_inst, budget_w=d["budget_w"], raw=raw)


def estimate_reference(code, d):
    """Estimate one reference-value component (G, H)."""
    ref = d["reference"]
    return make_result(code, d["name"], "reference",
                       area_mm2=ref.get("area_mm2_per_cube", 0.0),
                       dynamic_w=ref["power_w_per_cube"],
                       leakage_w=0.0,
                       instances=1, budget_w=d["budget_w"], raw=ref)


def estimate_derived(code, d, prior_results):
    """Estimate one derived component (I, J) from prior A–F results."""
    method = d["method"]
    deps   = d["depends_on"]

    if method == "fraction_of_logic_dynamic":
        base = sum(prior_results[c]["dynamic_w_per_cube"]
                   for c in deps if c in prior_results)
        dyn = base * d["fraction"]
        return make_result(code, d["name"], "derived",
                           dynamic_w=dyn, instances=1, budget_w=d["budget_w"],
                           raw={"fraction": d["fraction"],
                                "base_dynamic_w": base, "deps": deps})

    if method == "sum_of_leakage":
        leak = sum(prior_results[c]["leakage_w_per_cube"]
                   for c in deps if c in prior_results)
        return make_result(code, d["name"], "derived",
                           leakage_w=leak, instances=1, budget_w=d["budget_w"],
                           raw={"summed_from": deps})

    if method == "fraction_of_logic_die_area":
        # L: area-only derived component (no power)
        base = sum(prior_results[c]["area_mm2_per_cube"]
                   for c in deps if c in prior_results)
        overhead = base * d["fraction"]
        return make_result(code, d["name"], "derived",
                           area_mm2=overhead, instances=1, budget_w=d["budget_w"],
                           raw={"fraction": d["fraction"], "base_area_mm2": base,
                                "deps": deps})

    return make_result(code, d["name"], "derived",
                       error=f"unknown derived method: {method}")


# ============================================================================
# Top-Level Driver
# ============================================================================

def estimate_one(code, defn, prior=None):
    tool = defn["tool"]
    if tool == "yosys":     return estimate_yosys(code, defn)
    if tool == "cacti":     return estimate_cacti(code, defn)
    if tool == "reference": return estimate_reference(code, defn)
    if tool == "derived":   return estimate_derived(code, defn, prior or {})
    return make_result(code, defn["name"], tool, error=f"unknown tool: {tool}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="Comma-separated component codes (e.g. A,B,C)")
    ap.add_argument("--no-derived", action="store_true",
                    help="Skip derived components (I, J)")
    ap.add_argument("--peak", action="store_true",
                    help="Peak/TDP mode: set all component utilizations to 1.0 "
                         "(full switching activity) instead of decode-average duty")
    args = ap.parse_args()

    if args.peak:
        # Force every component to full activity (α = TOGGLE_RATE_FULL × 1.0)
        for code, defn in COMPONENTS.items():
            if "utilization" in defn:
                defn["utilization"] = 1.0
        print("  [PEAK MODE] all utilizations forced to 1.0 (TDP estimate)\n")

    selected = set(args.only.split(",")) if args.only else set(COMPONENTS.keys())
    CACTI_OUTPUT.mkdir(parents=True, exist_ok=True)
    YOSYS_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("="*78)
    print("  AMMA HMP_REO Per-Cube Hardware Estimation (4nm)")
    print("="*78)

    results = {}
    # Pass 1: direct estimators (yosys/cacti/reference)
    for code, defn in COMPONENTS.items():
        if code not in selected: continue
        if defn["tool"] == "derived": continue
        print(f"\n[{code}] {defn['name']}  ({defn['tool']})")
        r = estimate_one(code, defn)
        results[code] = r
        if r["error"]:
            print(f"     ✗ ERROR: {r['error']}")
        else:
            print(f"     area={r['area_mm2_per_cube']:.4f} mm²  "
                  f"dyn={r['dynamic_w_per_cube']:.3f} W  "
                  f"lkg={r['leakage_w_per_cube']:.3f} W  "
                  f"total={r['total_w_per_cube']:.3f} W  "
                  f"(budget {defn['budget_w']:.2f} W → "
                  f"{r['vs_budget_pct'] or 0:.0f}%)")

    # Pass 2: derived (I, J, L) need prior results
    if not args.no_derived:
        for code in ("I", "J", "L"):
            if code not in selected: continue
            defn = COMPONENTS[code]
            print(f"\n[{code}] {defn['name']}  ({defn['tool']})")
            r = estimate_one(code, defn, prior=results)
            results[code] = r
            if code == "L":
                print(f"     area={r['area_mm2_per_cube']:.3f} mm²  "
                      f"({defn['fraction']*100:.0f}% of logic die)")
            else:
                print(f"     dyn={r['dynamic_w_per_cube']:.3f} W  "
                      f"lkg={r['leakage_w_per_cube']:.3f} W  "
                      f"total={r['total_w_per_cube']:.3f} W")

    # ---- Aggregate (POWER) ----
    # J is a VIEW of total leakage (sum of A-F lkg); excluded from sum to avoid double-count.
    cube_logic_w = sum(r["total_w_per_cube"] for c, r in results.items()
                       if c in "ABCDEFI" and not r.get("error"))
    cube_phy_w   = sum(r["total_w_per_cube"] for c, r in results.items()
                       if c in "GH" and not r.get("error"))
    cube_total_w = cube_logic_w + cube_phy_w

    # Static (leakage) vs Dynamic split across A–F + I (G/H are reference composites)
    static_w  = sum(r["leakage_w_per_cube"] for c, r in results.items()
                    if c in "ABCDEF" and not r.get("error"))
    dynamic_w = sum(r["dynamic_w_per_cube"] for c, r in results.items()
                    if c in "ABCDEFI" and not r.get("error"))
    ref_w     = sum(r["total_w_per_cube"] for c, r in results.items()
                    if c in "GH" and not r.get("error"))

    # ---- Aggregate (AREA) — split: logic-die vs HBM footprint ----
    # Logic die = A+B+C+D+E+F+G (AMMA compute) + K (HBM PHY+memctrl) + L (overhead).
    # H = 3D-stacked HBM stack; area is package footprint, NOT in logic-die total.
    logic_die_components = "ABCDEFGKL"
    logic_die_area = sum(r["area_mm2_per_cube"] for c, r in results.items()
                         if c in logic_die_components and not r.get("error"))
    hbm_footprint  = sum(r["area_mm2_per_cube"] for c, r in results.items()
                         if c == "H" and not r.get("error"))
    # Per-cube package footprint = max(logic die, HBM die) since they 3D-stack
    cube_footprint = max(logic_die_area, hbm_footprint)

    print("\n" + "="*78)
    print("  PER-CUBE SUMMARY")
    print("="*78)
    print(f"  -- Power --")
    print(f"  Logic-die (A+B+C+D+E+F+I+J): {cube_logic_w:>7.2f} W   "
          f"(paper target: 15 W → {cube_logic_w/15*100:.0f}%)")
    print(f"  Off-die    (G+H):            {cube_phy_w:>7.2f} W   "
          f"(paper target: 76.5 W)")
    print(f"  Total cube:                  {cube_total_w:>7.2f} W   "
          f"(paper target: 91.5 W → {cube_total_w/91.5*100:.0f}%)")
    print(f"  Static (leakage A-F):        {static_w:>7.3f} W   ({static_w/cube_total_w*100:.2f}% of cube)")
    print(f"  Dynamic (A-F+I):             {dynamic_w:>7.2f} W   ({dynamic_w/cube_total_w*100:.2f}% of cube)")
    print(f"  Reference G+H (mixed):       {ref_w:>7.2f} W   ({ref_w/cube_total_w*100:.2f}% of cube)")
    print(f"  -- Area --")
    print(f"  Logic-die (A+B+C+D+E+F+G+K+L): {logic_die_area:>7.2f} mm²  "
          f"({logic_die_area/121*100:.0f}% of 121 mm² base die)")
    def _a(code): return results.get(code, {}).get("area_mm2_per_cube", 0)
    compute_logic = _a("A") + _a("C") + _a("D") + _a("E") + _a("F")
    print(f"    A  MAC arrays:             {_a('A'):>7.2f} mm²")
    print(f"    B  SRAM 3MB:               {_a('B'):>7.2f} mm²")
    print(f"    C  Two-level xbar:         {_a('C'):>7.2f} mm²")
    print(f"    D  D2D MAC ctrl:           {_a('D'):>7.2f} mm²")
    print(f"    E  DMA + frontend:         {_a('E'):>7.2f} mm²")
    print(f"    F  Vector unit:            {_a('F'):>7.2f} mm²")
    print(f"    G  D2D PHY (UCIe):         {_a('G'):>7.2f} mm²")
    print(f"    K  HBM PHY + memctrl:      {_a('K'):>7.2f} mm²")
    print(f"    L  IO/decap/clock-phy:     {_a('L'):>7.2f} mm²")
    print(f"  HBM4 stack footprint (H):    {hbm_footprint:>7.2f} mm²   (3D-stacked, package area)")
    print(f"  Cube package footprint:      {cube_footprint:>7.2f} mm²   (= max of logic-die & HBM)")

    chip_w     = cube_total_w * N_CUBES
    chip_area  = cube_footprint * N_CUBES
    chip_logic_area = logic_die_area * N_CUBES
    chip_hbm_area   = hbm_footprint * N_CUBES

    print(f"\n  SYSTEM (×{N_CUBES} cubes):")
    print(f"  Total chip power:            {chip_w:>7.1f} W   "
          f"(paper TDP: 1440 W → {chip_w/1440*100:.0f}%)")
    print(f"  Total logic-die area:        {chip_logic_area:>7.2f} mm²  "
          f"(sum of 16 cube logic dies)")
    print(f"  Total HBM die area:          {chip_hbm_area:>7.1f} mm²  "
          f"(sum of 16 HBM stacks footprint)")
    print(f"  Total package area:          {chip_area:>7.1f} mm²  "
          f"(16 × cube footprint, side-by-side)")

    # Save
    output = {
        "_description":     "AMMA HMP_REO Per-Cube 4nm Estimation (A–J components)",
        "_methodology":     "PLAN.md §2 + §4",
        "_scaling_22nm_4nm": {"area": SCALE_AREA, "energy": SCALE_ENERGY,
                              "leakage": SCALE_LEAKAGE, "delay": SCALE_DELAY},
        "_yosys_4nm_constants": {
            "ge_density_per_mm2": YOSYS_GE_DENSITY_PER_MM2,
            "dyn_uW_per_GE_per_MHz": YOSYS_DYN_UW_PER_GE_PER_MHZ,
            "lkg_uW_per_GE": YOSYS_LKG_UW_PER_GE,
        },
        "components": results,
        "summary": {
            "power": {
                "per_cube_logic_w":  round(cube_logic_w, 3),
                "per_cube_phy_w":    round(cube_phy_w, 3),
                "per_cube_total_w":  round(cube_total_w, 3),
                "per_chip_total_w":  round(chip_w, 1),
            },
            "area_mm2": {
                "per_cube_logic_die":     round(logic_die_area, 4),
                "per_cube_hbm_footprint": round(hbm_footprint, 2),
                "per_cube_package":       round(cube_footprint, 2),
                "per_chip_logic_die_sum": round(chip_logic_area, 2),
                "per_chip_hbm_sum":       round(chip_hbm_area, 1),
                "per_chip_package":       round(chip_area, 1),
                "_note": "Logic-die area sums A+B+C+D+E+F+G. HBM (H) is 3D-stacked, "
                         "so per-cube footprint = max(logic die, HBM die). "
                         "Chip package area = 16 × cube footprint (side-by-side).",
            },
            "paper_targets": {
                "cube_logic_w": 15, "cube_total_w": 91.5,
                "chip_tdp_w": 1440, "chip_sram_mb": 48,
            },
        },
    }
    RESULT_FILE.write_text(json.dumps(output, indent=2))
    print(f"\n  Results saved: {RESULT_FILE}")


if __name__ == "__main__":
    main()
