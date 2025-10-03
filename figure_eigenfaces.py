"""
:Author: Zachary T. Varley
:Date: October 2025
:License: MIT License

Description: This script generates a publication-ready figure illustrating the
impact of dictionary size on the eigenimages derived from Electron Backscatter
Diffraction (EBSD) patterns. It uses Welford's online covariance update rule to
keep an updated estimate of the covariance matrix. Finally it computes
eigenimages via eigen decomposition of the covariance matrix. For 10 million
target samples, the full dataset would be approximately:

From dictionaries of varying sizes, specifically for a Nickel (Ni) master
pattern, and a 60x60 virtual detector with the same geometry as Hakon's dataset.
The figure showcases eigenimages corresponding to logarithmically spaced
eigenvalues across different dictionary sizes, highlighting how the eigenfaces
converge as the the number of samples in the fundamental zone tends to infinity.

"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import kikuchipy as kp
from utils import (
    MasterPattern,
    EBSDGeometry,
    OnlineCovMatrix,
    so3_cu_grid_laue,
    qu_apply,
    get_radial_mask,
    progressbar,
    get_radial_mask,
)

# Set publication-ready plotting style
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
    }
)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DETECTOR_SHAPE = (60, 60)
MAX_EIGENIMAGE = 512
CHUNK_SIZE = 4096 * 4

# Dictionary configurations
# DICTIONARY_SIZES = [1_000, 10_000, 100_000, 1_000_000] # Small (for testing plotting)
DICTIONARY_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
]  # Full (for publication)


def setup_geometry():
    """Initialize EBSD geometry and detector coordinates."""
    mask = get_radial_mask(DETECTOR_SHAPE).to(device).flatten()
    geometry = EBSDGeometry(
        detector_shape=DETECTOR_SHAPE,
        proj_center=(0.4221, 0.2179, 0.4954),
        tilts_degrees=(0, 70, 0),
    ).to(device)

    detector_coords = geometry.get_coords_sample_frame(binning=(1, 1))
    detector_coords /= detector_coords.norm(dim=-1, keepdim=True)
    return mask, detector_coords[mask]


def load_master_pattern():
    """Load and prepare Ni master pattern."""
    kp_mp = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    mp_data = torch.cat(
        [
            torch.from_numpy(kp_mp.data[0].astype(np.float32)),
            torch.from_numpy(kp_mp.data[1].astype(np.float32)),
        ],
        dim=-1,
    ).to(device)

    mp = MasterPattern(mp_data, laue_group=11)
    mp.normalize(norm_type="minmax")
    return mp


def process_dictionary_with_permutations(size, detector_coords, mp):
    """Process a single dictionary configuration with permutations for multiple runs."""
    print(f"Processing size: {size:,}")
    results = []

    oversampled_size = int(size * 24)  # Ni has 24 rotational symmetries
    edge_length = int(oversampled_size ** (1 / 3))
    ori = so3_cu_grid_laue(
        edge_length=edge_length,
        laue_id=11,
        device=torch.device("cpu"),
    )

    print(f"Target size: {size:,} | Sampled size: {len(ori):,}")

    # Permute the orientations for each run
    perm = torch.randperm(len(ori), device="cpu")
    ori_permuted = ori[perm]

    # cov = OnlineCovMatrix(detector_coords.shape[0]).to(device)
    pca = OnlineCovMatrix(
        detector_coords.shape[0],
        covmat_dtype=torch.float32,
        delta_dtype=torch.float32,
        correlation=False,
    ).to(device)
    for chunk in progressbar(
        torch.split(ori_permuted, CHUNK_SIZE),
        prefix=f"Size {len(ori):,}",
    ):
        rotated = qu_apply(chunk[:, None, :].to(device), detector_coords)
        pats = mp.interpolate(rotated).view(len(chunk), -1)
        pca(pats - pats.mean(dim=-1, keepdim=True))
        torch.cuda.synchronize()

    results.append(pca.get_eigenvectors().cpu().numpy())

    return results


def plot_eigenimage_grid(eigenimages_runs, mask, detector_shape):
    """Create publication-ready eigenimage grid visualization with reduced spacing."""
    # Select eigenimages to display (logarithmic scale)
    log_scale_indices = [2**i for i in range(int(np.log2(MAX_EIGENIMAGE)) + 1)]

    # Create figure with compact layout
    fig = plt.figure(
        figsize=(3.5 * len(log_scale_indices) / 4, 3.5 * len(DICTIONARY_SIZES) / 4),
        dpi=300,
    )

    # Create a gridspec with minimal spacing
    gs = fig.add_gridspec(
        nrows=len(DICTIONARY_SIZES),
        ncols=len(log_scale_indices),
        wspace=0.05,  # Reduced horizontal spacing between columns
        hspace=0.05,  # Reduced vertical spacing between rows
    )

    # Create axes from gridspec
    axs = np.array(
        [
            [fig.add_subplot(gs[i, j]) for j in range(len(log_scale_indices))]
            for i in range(len(DICTIONARY_SIZES))
        ]
    )

    # Set consistent grayscale normalization
    vmin, vmax = -0.1, 0.1

    # Process each dictionary size (rows)
    for row, size in enumerate(DICTIONARY_SIZES):
        vecs = eigenimages_runs[size][0]

        # Process each eigenimage (columns)
        for col, idx in enumerate(log_scale_indices):
            if idx >= vecs.shape[1]:
                break

            eigen_face_flat = vecs[:, -idx]
            # use the average cube to flip the sign if necessary
            # this ensures that the eigenfaces have a consistent parity
            # even a significantly small dictionary is used to compute them (10,000)
            avg_cube = (eigen_face_flat**3).mean()
            if avg_cube < 0:
                vecs[:, -idx] = -eigen_face_flat

            ax = axs[row, col]
            comp = np.zeros(np.prod(detector_shape))
            comp[mask.cpu().numpy()] = vecs[:, -idx]

            # fill the masked values with NaN
            comp[~mask.cpu().numpy()] = np.nan

            # Display eigenimage with consistent normalization
            im = ax.imshow(
                comp.reshape(detector_shape), cmap="gray", vmin=vmin, vmax=vmax
            )

            # Clean up axes
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

            # Add size labels on leftmost column
            if col == 0:
                if size < 1_000:
                    size_label = f"{size:,}"
                elif size >= 1_000_000:
                    size_label = f"{size//1_000_000}M"
                else:
                    size_label = f"{size//1_000}k"
                ax.text(
                    -0.3,
                    0.5,
                    size_label,
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                )

            # Add eigenvalue numbers on bottom row
            if row == len(DICTIONARY_SIZES) - 1:
                ax.text(
                    0.5,
                    -0.2,
                    f"{idx}",
                    transform=ax.transAxes,
                    va="top",
                    ha="center",
                    fontsize=10,
                )

    # Add row and column labels
    plt.figtext(
        0.5,
        0.01,
        "Eigenvector Number",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    plt.figtext(
        0.01,
        0.5,
        "Approximate Dictionary Size",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=12,
        fontweight="bold",
    )

    # Add a little extra margin to accommodate the labels
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.95)

    return fig


def main():
    # Setup
    mask, detector_coords = setup_geometry()
    mp = load_master_pattern()

    # Process dictionaries using separate sampling for each size
    eigenimages_runs = {
        n_entry: process_dictionary_with_permutations(
            n_entry,
            detector_coords,
            mp,
        )
        for n_entry in DICTIONARY_SIZES
    }

    # Create publication-ready plot
    grid_plot = plot_eigenimage_grid(eigenimages_runs, mask, DETECTOR_SHAPE)

    # Save as high-quality vector and raster formats
    plt.savefig("figure_eigenfaces.pdf", bbox_inches="tight")
    plt.savefig("figure_eigenfaces.png", bbox_inches="tight", dpi=300)

    print("Figures saved successfully.")
    plt.close()


if __name__ == "__main__":
    main()
