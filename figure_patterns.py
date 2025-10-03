#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
corner_patterns_figure_v2.py

Creates a 2-row figure showing corner patterns from 3 experimental datasets:
- Row 1: Raw patterns from corners (3 per dataset, 9 total)
- Row 2: Masked + preprocessed patterns from same corners
- Vertical partition lines separate each dataset's group of 3
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import kikuchipy as kp

from utils import (
    ExperimentPatterns,
    get_radial_mask,
)

# ----------------------- Config -----------------------
DET_SHAPE = (60, 60)
EXP_DATASET_IDS = [1, 5, 10]
NOISE_LABELS = ["Low", "Medium", "High"]

# Corner positions (row, col) - excluding far corner (bottom-right)
# Scan is (149, 200)
CORNERS = [
    (0, 0),  # top-left
    (0, 199),  # top-right
    (148, 0),  # bottom-left
]

DPI = 300

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.linewidth": 0.5,
    }
)


# ----------------------- Helpers -----------------------
def _minmax01(a):
    """Normalize array to [0, 1], ignoring NaN values"""
    a = a.astype(np.float32, copy=True)
    valid = ~np.isnan(a)
    if not np.any(valid):
        return a
    mn, mx = float(np.nanmin(a)), float(np.nanmax(a))
    if mx > mn:
        a[valid] = (a[valid] - mn) / (mx - mn)
    else:
        a[valid] = 0.0
    return a


def load_exp_dataset(exp_id):
    """Load experimental dataset and return full 4D array (H, W, 60, 60)"""
    ds = kp.data.ni_gain(allow_download=True, number=exp_id)
    data = ds.data.astype(np.float32)
    return data


def preprocess_dataset(data):
    """Apply standard_clean preprocessing to entire dataset"""
    H, W = data.shape[:2]
    reshaped = data.reshape(H * W, DET_SHAPE[0], DET_SHAPE[1])

    pat_tensor = torch.from_numpy(reshaped)
    ep = ExperimentPatterns(pat_tensor.clone())
    ep.standard_clean()

    processed = ep.patterns.cpu().numpy()
    return processed.reshape(H, W, DET_SHAPE[0], DET_SHAPE[1])


def apply_mask_to_pattern(pattern, mask):
    """Apply circular mask to pattern, setting outside to NaN for transparency"""
    masked = pattern.copy()
    masked[~mask] = np.nan
    return masked


# ----------------------- Main -----------------------
def main():
    # Load mask
    mask = get_radial_mask(DET_SHAPE).bool().cpu().numpy()

    # Create figure with GridSpec for better control
    fig = plt.figure(figsize=(14, 3.5))

    # Create a 2x9 grid with extra spacing at columns 3 and 6 (between groups)
    gs = gridspec.GridSpec(
        2,
        11,
        figure=fig,
        left=0.02,
        right=0.98,
        top=0.90,
        bottom=0.08,
        hspace=0.12,
        wspace=0.03,
        width_ratios=[1, 1, 1, 0.15, 1, 1, 1, 0.15, 1, 1, 1],
    )

    # Map logical columns (0-8) to grid columns (accounting for gaps)
    col_mapping = {
        0: 0,
        1: 1,
        2: 2,  # Dataset 1
        3: 4,
        4: 5,
        5: 6,  # Dataset 2
        6: 8,
        7: 9,
        8: 10,  # Dataset 3
    }

    col_idx = 0

    # Process each dataset
    for dataset_idx, exp_id in enumerate(EXP_DATASET_IDS):
        print(f"Loading dataset {exp_id}...")
        data_raw = load_exp_dataset(exp_id)

        print(f"Preprocessing dataset {exp_id}...")
        data_proc = preprocess_dataset(data_raw)

        # Process each corner
        for corner_idx, (row, col) in enumerate(CORNERS):
            # Extract patterns
            raw_pattern = data_raw[row, col, :, :]
            proc_pattern = data_proc[row, col, :, :]
            masked_pattern = apply_mask_to_pattern(proc_pattern, mask)

            # Normalize
            raw_norm = _minmax01(raw_pattern)
            masked_norm = _minmax01(masked_pattern)

            # Get grid column
            grid_col = col_mapping[col_idx]

            # Plot raw pattern (row 0)
            ax_raw = fig.add_subplot(gs[0, grid_col])
            ax_raw.imshow(raw_norm, cmap="gray", vmin=0, vmax=1)
            ax_raw.axis("off")

            # Add title above middle column of each dataset
            if corner_idx == 1:
                ax_raw.set_title(
                    f"{NOISE_LABELS[dataset_idx]} Noise",
                    fontsize=16,
                    pad=10,
                    weight="bold",
                )

            # Plot preprocessed pattern (row 1)
            ax_proc = fig.add_subplot(gs[1, grid_col])
            ax_proc.imshow(masked_norm, cmap="gray", vmin=0, vmax=1)
            ax_proc.axis("off")

            # Add corner label for all patterns
            corner_label = f"({row},{col})"
            ax_proc.text(
                0.5,
                -0.12,
                corner_label,
                transform=ax_proc.transAxes,
                ha="center",
                va="top",
                fontsize=12,
            )

            col_idx += 1

    # Add row labels
    fig.text(
        0.008,
        0.70,
        "Raw",
        va="center",
        ha="center",
        fontsize=16,
        weight="bold",
        rotation=90,
    )
    fig.text(
        0.008,
        0.30,
        "Processed",
        va="center",
        ha="center",
        fontsize=16,
        weight="bold",
        rotation=90,
    )

    # Save figures
    plt.savefig("figure_patterns.png", dpi=DPI, bbox_inches="tight")
    plt.savefig("figure_patterns.pdf", bbox_inches="tight")
    print(f"✓ Figures saved: figure_patterns.png and figure_patterns.pdf")

    plt.close()


if __name__ == "__main__":
    main()
