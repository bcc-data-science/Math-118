"""
Renders the two unlabeled "mystery" contour maps used in the notebook's
warm-up sensemaking activity (before any code is introduced). Panel A is the
clean scan, Panel B is the dirty scan -- deliberately unlabeled as such, with
no colorbar units, so students can compare shapes/patterns and predict
without being told the answer.

INSTRUCTOR USE ONLY -- do not hand this script to students, it identifies
which panel is which.

Run:  python generate_mystery_maps.py
Output: output/mystery_panel_A.png  (clean)
        output/mystery_panel_B.png  (dirty)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

from generate_placeholder_data import make_clean_grid, make_loss_grid, N_X, N_Y

OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)


def plot_mystery(grid, label, path):
    plot_grid = zoom(grid, 8, order=3)
    fig, ax = plt.subplots(figsize=(5, 8.5))
    cf = ax.contourf(plot_grid, levels=15, cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_ticks([])
    cbar.set_label("Relative output (unlabeled scale)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label, fontsize=16)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    rng = np.random.default_rng(42)
    clean_grid = make_clean_grid(rng)
    loss_grid = make_loss_grid(rng)
    dirty_grid = clean_grid * (1 - loss_grid)

    plot_mystery(clean_grid, "Panel A", os.path.join(OUT_DIR, "mystery_panel_A.png"))
    plot_mystery(dirty_grid, "Panel B", os.path.join(OUT_DIR, "mystery_panel_B.png"))


if __name__ == "__main__":
    main()
