"""
Generates placeholder "lab_clean" and "lab_dirty" voltage-log CSVs that mimic
the real YT-Scope point-scan exports, so the student notebook can be built,
tested, and run BEFORE real panel data is collected.

INSTRUCTOR USE ONLY. Do not hand this script to students -- it hard-codes the
exact spatial soiling pattern used to fake the "dirty" panel, which is what
students are supposed to discover for themselves from the contour maps.

When real data is ready:
  1. Record clean + dirty scans with the point-scan rig (same 11x22, 3s/3s,
     top-right-start geometry).
  2. Drop the two real CSVs into the notebook's data/ folder, replacing (or
     alongside) these placeholder files.
  3. In the notebook's Configuration cell, point CSV_PATH_CLEAN / CSV_PATH_DIRTY
     at the real files. Nothing else needs to change.

Run:  python generate_placeholder_data.py
Output: data/lab_clean_r1_test_3s_interval.csv
        data/lab_dirty_r1_test_3s_interval.csv
"""

import os
import numpy as np

# ---- Scan geometry (must match the notebook's Configuration cell) ----
N_X = 11              # points across the width (left-right)
N_Y = 22               # points down the length (top-bottom)
START_TOP_RIGHT = True

DWELL_S = 3.0
MOVE_S = 3.0
SAMPLE_MS = 10          # matches the real logger's 10ms sample interval

SAMPLES_PER_DWELL = int(round(DWELL_S * 1000 / SAMPLE_MS))   # 300
SAMPLES_PER_MOVE = int(round(MOVE_S * 1000 / SAMPLE_MS))     # 300
CYCLE_LEN = SAMPLES_PER_DWELL + SAMPLES_PER_MOVE              # 600

SETTLE_TAU_S = 0.18     # how quickly the reading settles after the light stops moving
NOISE_SIGMA = 0.015     # per-sample sensor noise (volts), similar scale to the real logs

OUT_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUT_DIR, exist_ok=True)


def make_clean_grid(rng):
    """(N_Y x N_X) grid of 'true' clean-panel voltages. Smooth spatial
    variation (manufacturing + point-light-source non-uniformity) plus a
    little fixed per-cell variation. No soiling."""
    rows = np.arange(N_Y).reshape(-1, 1)
    cols = np.arange(N_X).reshape(1, -1)

    center_r, center_c = 6.0, 5.0
    dist2 = ((rows - center_r) / 10.0) ** 2 + ((cols - center_c) / 5.0) ** 2
    bump = 0.45 * np.exp(-dist2)

    cell_variation = rng.normal(0, 0.05, size=(N_Y, N_X))

    return 3.55 + bump + cell_variation


def make_loss_grid(rng):
    """Fractional power loss at each grid point for the 'dirty' scan:
    a broad bottom-heavy gradient (gravity-driven dust accumulation) plus
    one sharp localized blob (e.g. a leaf or bird dropping)."""
    rows = np.arange(N_Y).reshape(-1, 1)
    cols = np.arange(N_X).reshape(1, -1)

    baseline = 0.03
    gradient = 0.12 * (rows / (N_Y - 1))

    blob_r, blob_c, blob_amp, blob_radius = 16.0, 2.0, 0.35, 1.5
    blob = blob_amp * np.exp(-(((rows - blob_r) / blob_radius) ** 2 +
                                ((cols - blob_c) / blob_radius) ** 2))

    loss = baseline + gradient + blob
    return np.clip(loss, 0, 0.6)


def scan_order_targets(grid):
    """Flatten an (N_Y x N_X) grid [row 0 = top, col 0 = left] into the 1D
    sequence of target values in the order the light actually visits them
    (serpentine, starting top-right). Inverse of the notebook's
    reshape_to_grid()."""
    targets = []
    for row in range(N_Y):
        scanned_right_to_left = (row % 2 == 0) if START_TOP_RIGHT else (row % 2 == 1)
        col_order = range(N_X - 1, -1, -1) if scanned_right_to_left else range(N_X)
        for col in col_order:
            targets.append(grid[row, col])
    return np.array(targets)


def synthesize_trace(targets, rng):
    """Build one continuous noisy voltage trace: for each target point,
    MOVE_S seconds ramping in from the previous target, then DWELL_S seconds
    settling onto (and holding at) the target."""
    n_points = len(targets)
    total_samples = n_points * CYCLE_LEN
    trace = np.empty(total_samples)

    prev_target = targets[0]
    idx = 0
    for i, target in enumerate(targets):
        # --- move phase: linear ramp from prev_target to target ---
        t_move = np.arange(SAMPLES_PER_MOVE) * SAMPLE_MS / 1000.0
        move_vals = prev_target + (target - prev_target) * (t_move / MOVE_S)
        trace[idx:idx + SAMPLES_PER_MOVE] = move_vals
        idx += SAMPLES_PER_MOVE

        # --- dwell phase: exponential settle onto target, then hold ---
        t_dwell = np.arange(SAMPLES_PER_DWELL) * SAMPLE_MS / 1000.0
        dwell_vals = target + (prev_target - target) * np.exp(-t_dwell / SETTLE_TAU_S)
        trace[idx:idx + SAMPLES_PER_DWELL] = dwell_vals
        idx += SAMPLES_PER_DWELL

        prev_target = target

    trace += rng.normal(0, NOISE_SIGMA, size=trace.shape)
    return trace


def write_scope_csv(path, trace, sample_ms=SAMPLE_MS):
    """Write a trace out in the same layout as the real YT-Scope export:
    metadata header block, blank line, then 'time_ms,value' rows."""
    header = f"""Name,YT Scope Project,
File,{path},
Starttime of export,134280205304630000,Wednesday, July 8, 2026,14:42:10.463
Endtime of export,134280219834030000,Wednesday, July 8, 2026,15:06:23.403


Name,v
SymbolComment,
Data-Type,REAL32
SampleTime[ms],{sample_ms}
VariableSize,4
SymbolBased,True
IndexGroup,16448
IndexOffset,387028
SymbolName,MAIN.v
NetID,10.41.152.108.1.1
Port,851
Offset,0
ScaleFactor,1
BitMask,0xffffffffffffffff
Unit,(None)
Unit ScaleFactor,1
Unit Offset,0

"""
    with open(path, "w", newline="\r\n") as f:
        f.write(header)
        for i, v in enumerate(trace):
            f.write(f"{i * sample_ms},{v:.8f}\n")


def main():
    rng = np.random.default_rng(42)

    clean_grid = make_clean_grid(rng)
    loss_grid = make_loss_grid(rng)
    dirty_grid = clean_grid * (1 - loss_grid)

    clean_targets = scan_order_targets(clean_grid)
    dirty_targets = scan_order_targets(dirty_grid)

    clean_trace = synthesize_trace(clean_targets, np.random.default_rng(101))
    dirty_trace = synthesize_trace(dirty_targets, np.random.default_rng(202))

    clean_path = os.path.join(OUT_DIR, "lab_clean_r1_test_3s_interval.csv")
    dirty_path = os.path.join(OUT_DIR, "lab_dirty_r1_test_3s_interval.csv")

    write_scope_csv(clean_path, clean_trace)
    write_scope_csv(dirty_path, dirty_trace)

    print(f"Wrote {len(clean_trace):,} samples -> {clean_path}")
    print(f"Wrote {len(dirty_trace):,} samples -> {dirty_path}")


if __name__ == "__main__":
    main()
