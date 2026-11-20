"""Helper functions for the solar panel contour lab.

You do NOT need to read or change this file to complete the lab. It holds the
longer pieces of code so the notebook can focus on the data:

  parse_scope_csv       -- reads the raw instrument CSV (metadata header + samples)
  extract_point_values  -- slices the voltage trace into one average per scan point
  plot_panel_contour    -- draws a contour map of a panel grid

Keep this file in the same folder as the notebook.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom


def parse_scope_csv(path):
    """Parse a YT-Scope style export: metadata header, blank line, then
    'time_ms,value' rows. Returns (dataframe, sample_time_ms)."""
    with open(path, "r") as f:
        lines = f.readlines()

    sample_time_ms = None
    data_start = None

    for line in lines:
        if line.startswith("SampleTime[ms]"):
            sample_time_ms = float(line.strip().split(",")[1])

    for i in range(len(lines) - 1):
        if lines[i].strip() == "":
            parts = lines[i + 1].strip().split(",")
            if len(parts) == 2:
                try:
                    float(parts[0]); float(parts[1])
                    data_start = i + 1
                    break
                except ValueError:
                    continue

    if data_start is None:
        raise ValueError(f"Could not find start of numeric data in {path}")
    if sample_time_ms is None:
        raise ValueError(f"Could not find 'SampleTime[ms]' in the header of {path}")

    df = pd.read_csv(path, skiprows=data_start, header=None,
                      names=["time_ms", "volts"], engine="python",
                      on_bad_lines="skip")
    df = df[pd.to_numeric(df["time_ms"], errors="coerce").notna()].copy()
    df["time_ms"] = df["time_ms"].astype(float)
    df["volts"] = df["volts"].astype(float)
    df["time_s"] = df["time_ms"] / 1000.0

    return df, sample_time_ms


def extract_point_values(df, sample_time_ms, n_points, dwell_s=3.0, move_s=3.0, trim_s=0.5,
                          min_window_frac=0.25, label=""):
    """Slice a continuous voltage trace into `n_points` dwell windows and
    average each one (after trimming a settling transient). Returns a 1D
    array of length n_points, in scan order (point 0 = first point visited).

    If the recording ends partway through a dwell window (e.g. logging was
    stopped a moment early), that final window is averaged using whatever
    samples are actually available -- as long as it's at least
    `min_window_frac` of a full dwell window. A warning is printed either
    way so you always know if this happened. If MORE than one window is
    short, or a window has too few samples to trust, this raises an error
    instead of silently guessing."""
    samples_per_dwell = int(round(dwell_s * 1000 / sample_time_ms))
    samples_per_move = int(round(move_s * 1000 / sample_time_ms))
    cycle_len = samples_per_dwell + samples_per_move
    trim_samples = int(round(trim_s * 1000 / sample_time_ms))
    min_samples = int(round(samples_per_dwell * min_window_frac))

    volts = df["volts"].values
    values = np.empty(n_points)
    short_windows = []

    for i in range(n_points):
        start = i * cycle_len
        end = start + samples_per_dwell

        if end > len(volts):
            end = len(volts)
            missing_s = (start + samples_per_dwell - end) * sample_time_ms / 1000
            available = end - start
            if available < min_samples:
                raise ValueError(
                    f"[{label}] Point {i}/{n_points} only has {available} samples "
                    f"available (need at least {min_samples}). The recording likely "
                    f"stopped too early, or DWELL_S/MOVE_S/N_X/N_Y don't match how "
                    f"this file was recorded."
                )
            if i != n_points - 1:
                raise ValueError(
                    f"[{label}] Point {i}/{n_points} ran short of data, but it isn't "
                    f"the last point -- that's unexpected and likely means "
                    f"DWELL_S/MOVE_S/N_X/N_Y don't match this file's actual scan."
                )
            short_windows.append((i, missing_s))

        window_start = start + trim_samples if (end - start) > trim_samples else start
        window = volts[window_start:end]
        values[i] = window.mean()

    used_samples = min(n_points * cycle_len, len(volts))
    leftover_s = (len(volts) - used_samples) * sample_time_ms / 1000
    tag = f"[{label}] " if label else ""
    print(f"{tag}Extracted {n_points} points using {used_samples} of {len(volts)} "
          f"samples ({leftover_s:.2f}s left over at the end).")

    for i, missing_s in short_windows:
        print(f"  NOTE: point {i} (the last point) was short by {missing_s:.2f}s "
              f"-- recording likely stopped a bit early. Averaged using the "
              f"samples that were available.")

    return values


def plot_panel_contour(grid, panel_width_cm, panel_length_cm,
                        n_levels=15, cmap="viridis",
                        title="Panel Map", cbar_label="Volts",
                        smooth=True, upsample_factor=8,
                        show_points=False, filled=True, line_labels=False,
                        diverging=False, save_path=None):
    n_y, n_x = grid.shape

    if smooth:
        plot_grid = zoom(grid, upsample_factor, order=3)  # cubic spline upsample
    else:
        plot_grid = grid

    ny_p, nx_p = plot_grid.shape
    x = (np.arange(nx_p) + 0.5) * (panel_width_cm / nx_p)
    y = (np.arange(ny_p) + 0.5) * (panel_length_cm / ny_p)
    X, Y = np.meshgrid(x, y)

    norm = None
    if diverging:
        vmax = max(abs(plot_grid.min()), abs(plot_grid.max()))
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
        if cmap == "viridis":  # swap default to a diverging map if caller didn't override
            cmap = "RdYlGn_r"  # red = loss (positive %), green = gain (negative %)

    fig, ax = plt.subplots(figsize=(6, 10))

    if filled:
        cf = ax.contourf(X, Y, plot_grid, levels=n_levels, cmap=cmap, norm=norm)
        cbar = fig.colorbar(cf, ax=ax)
    else:
        cf = ax.contour(X, Y, plot_grid, levels=n_levels, cmap=cmap, norm=norm)
        cbar = fig.colorbar(cf, ax=ax)

    if line_labels:
        lines = ax.contour(X, Y, plot_grid, levels=n_levels, colors="white", linewidths=0.5)
        ax.clabel(lines, inline=True, fontsize=6, fmt="%.2f")

    if show_points:
        xp = (np.arange(n_x) + 0.5) * (panel_width_cm / n_x)
        yp = (np.arange(n_y) + 0.5) * (panel_length_cm / n_y)
        Xp, Yp = np.meshgrid(xp, yp)
        ax.scatter(Xp, Yp, s=8, c="white", edgecolors="black", linewidths=0.3, zorder=5)

    ax.invert_yaxis()  # so 0 (top edge) is at the top of the plot
    ax.set_xlabel("Distance from left edge (cm)")
    ax.set_ylabel("Distance from top edge (cm)")
    ax.set_title(title)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.show()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig, ax
