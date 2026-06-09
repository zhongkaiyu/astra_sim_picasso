#!/usr/bin/env python3
"""
ASAP7 gate-level area + power model.

Replaces the "cells × magic constant" power hack with a physical estimate
derived from the actual synthesized cell histogram and the ASAP7 Liberty data:

  Area_synth = Σ (count[cell] × area[cell])              [µm² @ 7nm]
  Area_layout = Area_synth / utilization                 [adds P&R whitespace]
  C_total    = Σ (count[cell] × Σ input_pin_cap[cell])   [fF]
  P_dynamic  = α · C_total · Vdd² · f                     [W]   (α = activity)
  P_leakage  = Σ (count[cell] × avg_leakage[cell])        [W]   (state-averaged)

ASAP7 facts (from LIB/NLDM, TT corner):
  Vdd = 0.7 V,  cap unit = 1 fF,  leakage unit = 1 pW

Then 7nm → 4nm scaling is applied by the caller (run_estimation.py).
"""
import os
import re
import glob
from functools import lru_cache

ASAP7_LIB_DIR = "/home/haotian/waferchip/asap7/asap7sc7p5t_28/LIB/NLDM"
ASAP7_VDD     = 0.7        # V (TT corner, PVT_0P7V_25C)
CAP_UNIT_FF   = 1.0        # capacitive_load_unit (1, ff)
LEAK_UNIT_PW  = 1.0        # leakage_power_unit "1pW"


# ---------------------------------------------------------------------------
# Liberty parser — build cell → {area_um2, in_cap_ff, avg_leak_pw}
# ---------------------------------------------------------------------------

def _parse_one_lib(path):
    """Parse a single ASAP7 .lib; return {cell_name: {...}}."""
    txt = open(path).read()
    cells = {}

    # Split on "cell (NAME) {" — keep it simple with index scanning
    for m in re.finditer(r"\n  cell \((\w+)\)\s*\{", txt):
        name = m.group(1)
        start = m.end()
        # Find matching close brace by depth counting
        depth = 1
        i = start
        while i < len(txt) and depth > 0:
            ch = txt[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        block = txt[start:i]

        # area (semicolon optional — INVBUF lib omits it)
        am = re.search(r"\barea\s*:\s*([\d.]+)", block)
        area = float(am.group(1)) if am else 0.0

        # input pin capacitances: sum capacitance of pins with direction:input
        in_cap = 0.0
        for pm in re.finditer(r"\bpin \(\w+\)\s*\{(.*?)\n    \}", block, re.DOTALL):
            pblock = pm.group(1)
            dirm = re.search(r"direction\s*:\s*(\w+)", pblock)
            if dirm and dirm.group(1) == "input":
                cm = re.search(r"capacitance\s*:\s*([\d.]+)", pblock)
                if cm:
                    in_cap += float(cm.group(1))

        # leakage: average the non-zero VDD-related leakage_power values
        leaks = [float(x) for x in
                 re.findall(r"leakage_power \(\)\s*\{\s*value\s*:\s*([\d.]+)", block)]
        nz = [x for x in leaks if x > 0]
        avg_leak = (sum(nz) / len(nz)) if nz else 0.0

        cells[name] = {
            "area_um2":  area,
            "in_cap_ff": in_cap,
            "avg_leak_pw": avg_leak,
        }
    return cells


@lru_cache(maxsize=1)
def load_lib_db():
    """Load + merge all ASAP7 NLDM .lib files into one cell DB."""
    db = {}
    for path in sorted(glob.glob(os.path.join(ASAP7_LIB_DIR, "*.lib"))):
        db.update(_parse_one_lib(path))
    return db


# ---------------------------------------------------------------------------
# Parse Yosys `stat -liberty` flattened cell histogram
# ---------------------------------------------------------------------------

def parse_stat_histogram(stat_text):
    """Extract {cell_name: count} from the FINAL stat block (flattened).

    The last 'Number of cells:' block lists each ASAP7 cell with its count.
    """
    # Find the last "Number of cells:" and read the indented cell lines after it
    idxs = [m.start() for m in re.finditer(r"Number of cells:", stat_text)]
    if not idxs:
        return {}
    tail = stat_text[idxs[-1]:]
    hist = {}
    for m in re.finditer(r"^\s+(\w*ASAP7\w+)\s+(\d+)\s*$", tail, re.MULTILINE):
        hist[m.group(1)] = int(m.group(2))
    return hist


# ---------------------------------------------------------------------------
# Gate-level area + power
# ---------------------------------------------------------------------------

def estimate_block(stat_text, activity, clock_ghz, utilization_layout=0.65):
    """Return physical area + power for one synthesized module @ 7nm.

    Parameters
    ----------
    stat_text  : str   — Yosys stat -liberty output
    activity   : float — switching activity α (toggles/clk/2 × workload duty)
    clock_ghz  : float — clock frequency
    utilization_layout : float — P&R cell-area utilization (0.5-0.75)

    Returns dict (all @ 7nm; caller applies 7→4nm scaling).
    """
    db = load_lib_db()
    hist = parse_stat_histogram(stat_text)

    area_um2 = 0.0
    cap_ff   = 0.0
    leak_pw  = 0.0
    matched  = 0
    unmatched = {}
    total_cells = 0

    for cell, n in hist.items():
        total_cells += n
        info = db.get(cell)
        if info is None:
            unmatched[cell] = n
            continue
        matched += n
        area_um2 += n * info["area_um2"]
        cap_ff   += n * info["in_cap_ff"]
        leak_pw  += n * info["avg_leak_pw"]

    f_hz = clock_ghz * 1e9
    # Dynamic: P = α · C · V² · f   (C in F, so fF×1e-15)
    p_dyn_w  = activity * (cap_ff * 1e-15) * (ASAP7_VDD ** 2) * f_hz
    # Leakage: pW → W
    p_leak_w = leak_pw * 1e-12

    area_synth_mm2  = area_um2 / 1e6
    area_layout_mm2 = area_synth_mm2 / utilization_layout if utilization_layout > 0 else area_synth_mm2

    return {
        "total_cells":        total_cells,
        "matched_cells":      matched,
        "unmatched":          unmatched,
        "area_synth_um2_7nm": area_um2,
        "area_synth_mm2_7nm": area_synth_mm2,
        "area_layout_mm2_7nm": area_layout_mm2,
        "utilization_layout": utilization_layout,
        "cap_total_ff":       cap_ff,
        "p_dynamic_w_7nm":    p_dyn_w,
        "p_leakage_w_7nm":    p_leak_w,
        "activity":           activity,
        "clock_ghz":          clock_ghz,
        "vdd":                ASAP7_VDD,
    }


@lru_cache(maxsize=1)
def median_dff_area_um2():
    """Median ASAP7 DFF cell area (µm² @7nm) — for FF register-file storage."""
    db = load_lib_db()
    areas = sorted(v["area_um2"] for k, v in db.items()
                   if "DFF" in k and v["area_um2"] > 0)
    return areas[len(areas) // 2] if areas else 0.32


def estimate_register_file(bits, overhead=2.0):
    """FF-based register-file area @7nm (PLAN_ASAP7_REALISTIC §1).

    Tiny multiported double-buffered buffers are physically FF register files,
    not dense SRAM macros (this is why H100's blended SRAM density is ~4 mm²/MB).

      Area_7nm = bits × ASAP7_DFF_area × overhead
        overhead ≈ 1.8-2.0  (R/W mux + address decode + double-buffer select + routing)
    """
    dff = median_dff_area_um2()
    area_um2_7nm = bits * dff * overhead
    return {
        "bits":            bits,
        "dff_area_um2":    dff,
        "overhead":        overhead,
        "area_um2_7nm":    area_um2_7nm,
        "area_mm2_7nm":    area_um2_7nm / 1e6,
    }


def estimate_wire_area(n_bits, n_ports, span_mm, pitch_nm=130.0, fill=0.4):
    """Global-interconnect wire area @7nm-equiv (PLAN_ASAP7_REALISTIC §3).

    For a die-spanning crossbar the area is wire-track-limited, not gate-limited.
      A_wire = n_wires × span × pitch / fill
        n_wires = n_bits × n_ports × 2 (bidirectional)
        pitch   = metal pitch (~130 nm at 4nm-class M4-M6)
        fill    = fraction of wires needing dedicated tracks (rest route over cells)
    """
    n_wires = n_bits * n_ports * 2
    pitch_mm = pitch_nm * 1e-6
    area_mm2 = n_wires * span_mm * pitch_mm / max(fill, 1e-6) * fill  # track area × fill
    # = n_wires × span × pitch (fill cancels to model "fraction on dedicated tracks")
    area_mm2 = n_wires * span_mm * pitch_mm * fill
    return {
        "n_wires":     n_wires,
        "span_mm":     span_mm,
        "pitch_nm":    pitch_nm,
        "fill":        fill,
        "area_mm2":    area_mm2,
    }


if __name__ == "__main__":
    # Self-test on the A (MAC) stat output
    import sys
    db = load_lib_db()
    print(f"ASAP7 lib DB: {len(db)} cells loaded")
    print(f"  INVx1 example: {db.get('INVx1_ASAP7_75t_R')}")
    test = sys.argv[1] if len(sys.argv) > 1 else "yosys/output/A_sa_16x16.txt"
    if os.path.isfile(test):
        r = estimate_block(open(test).read(), activity=0.15, clock_ghz=2.0)
        for k, v in r.items():
            if k != "unmatched":
                print(f"  {k}: {v}")
        print(f"  unmatched types: {len(r['unmatched'])}")
