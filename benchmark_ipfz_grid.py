import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from orix.quaternion.symmetry import Oh
from orix.quaternion import Orientation, Quaternion
from orix.plot import IPFColorKeyTSL

# Import utilities for quaternion conversion
import torch
from ebsdtorch.s2_and_so3.orientations import qu2bu

# Publication-quality settings
DPI = 300
plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.linewidth": 0.5,
    }
)


def quaternions_to_ipf_rgb(quaternions, grid_shape=(149, 200), symmetry=Oh):
    """
    Convert quaternions to IPF-Z RGB image.

    Args:
        quaternions: (N, 4) array of quaternions
        grid_shape: (H, W) shape of the scan grid
        symmetry: orix symmetry object

    Returns:
        (H, W, 3) RGB array in [0, 1]
    """
    # Convert quaternions to Euler angles using ebsdtorch
    qu_tensor = torch.from_numpy(quaternions.astype(np.float32))
    euler = qu2bu(qu_tensor).cpu().numpy()  # Bunge Euler angles in radians

    # Create orix Orientation objects
    O = Orientation.from_euler(euler, symmetry=symmetry, degrees=False)

    # Get IPF-Z colors
    key = IPFColorKeyTSL(symmetry.laue)
    colors = key.orientation2color(O).astype(np.float32)

    # Reshape to grid
    H, W = grid_shape
    if colors.shape[0] != H * W:
        raise ValueError(f"Expected {H*W} orientations, got {colors.shape[0]}")

    ipf_map = colors.reshape(H, W, 3)
    return ipf_map


def create_ipf_grid(data_file, grid_shape=(149, 200), dtype_filter="FP32"):
    """Create 3x3 grid of IPF-Z maps for different noise levels and methods."""

    # Load benchmark data
    data = np.load(data_file, allow_pickle=True).item()

    # Get reference orientations (ground truth)
    ref_oris = data["reference_orientations"]

    # Filter for specific dtype
    mask = np.array(data["dtype"]) == dtype_filter

    methods = np.array(data["method"])[mask]
    resolutions = np.array(data["dict_resolution"])[mask]
    pca_comps = np.array(data["pca_components"])[mask]
    indexed_oris = np.array(data["indexed_orientations"], dtype=object)[mask]
    noise_ids = np.array(data["dataset_id"])[mask]

    # Use smallest resolution only
    smallest_resolution = min(np.unique(resolutions))
    res_mask = resolutions == smallest_resolution

    methods = methods[res_mask]
    pca_comps = pca_comps[res_mask]
    indexed_oris = indexed_oris[res_mask]
    noise_ids = noise_ids[res_mask]

    # Unique noise levels
    unique_noise_ids = sorted(np.unique(noise_ids))
    n_noise_levels = len(unique_noise_ids)

    # Noise labels
    noise_labels = {1: "Low", 5: "Medium", 10: "High"}

    # Method ordering
    method_order = ["DI", "PCA-512", "PCA-1024"]

    # Calculate figure size based on image aspect ratio (200/149 ≈ 1.34)
    img_aspect = grid_shape[1] / grid_shape[0]  # width / height
    subplot_height = 3.0
    subplot_width = subplot_height * img_aspect

    # Create figure with 4 columns (3 for data + 1 for legend)
    fig, axes = plt.subplots(
        n_noise_levels,
        4,
        figsize=(13, 8),
        squeeze=False,
    )

    # Populate grid
    for row_idx, noise_id in enumerate(unique_noise_ids):
        for col_idx, method_label in enumerate(method_order):
            ax = axes[row_idx][col_idx]

            # Find matching data
            if method_label == "DI":
                method_mask = (noise_ids == noise_id) & (methods == "DI")
            else:
                n_comp = int(method_label.split("-")[1])
                method_mask = (
                    (noise_ids == noise_id) & (methods == "PCA") & (pca_comps == n_comp)
                )

            if not np.any(method_mask):
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            # Get the orientations
            idx = np.where(method_mask)[0][0]
            oris = indexed_oris[idx]

            if len(oris) == 0:
                ax.text(
                    0.5, 0.5, "Empty", ha="center", va="center", transform=ax.transAxes
                )
                ax.axis("off")
                continue

            # Convert to IPF-Z map
            try:
                ipf_map = quaternions_to_ipf_rgb(oris, grid_shape=grid_shape)
                ax.imshow(ipf_map, interpolation="nearest")
            except Exception as e:
                print(f"Error processing {method_label} at noise_id {noise_id}: {e}")
                ax.text(
                    0.5, 0.5, f"Error", ha="center", va="center", transform=ax.transAxes
                )

            ax.axis("off")

            # Add column labels on top row
            if row_idx == 0:
                ax.set_title(method_label, fontsize=16, fontweight="bold", pad=10)

            # Add row labels on left column
            if col_idx == 0:
                noise_label = noise_labels.get(noise_id, f"ID {noise_id}")
                ax.text(
                    -0.05,
                    0.5,
                    f"{noise_label} Noise",
                    transform=ax.transAxes,
                    fontsize=16,
                    fontweight="bold",
                    rotation=90,
                    va="center",
                    ha="right",
                )

    # Handle column 4 (index 3): Reference scan and IPF key
    center_row = n_noise_levels // 2

    # Add reference scan (ground truth) to top row
    ax_ref = axes[0][3]
    try:
        ref_ipf_map = quaternions_to_ipf_rgb(ref_oris, grid_shape=grid_shape)
        ax_ref.imshow(ref_ipf_map, interpolation="nearest")
        ax_ref.axis("off")
        ax_ref.set_title("Ground Truth", fontsize=16, fontweight="bold", pad=10)
    except Exception as e:
        print(f"Error processing reference orientations: {e}")
        ax_ref.text(
            0.5, 0.5, "Error", ha="center", va="center", transform=ax_ref.transAxes
        )
        ax_ref.axis("off")

    # Add IPF color key in center row
    ax_legend = axes[center_row][3]
    ipf_key = IPFColorKeyTSL(Oh)
    ipf_fig = ipf_key.plot(return_figure=True)

    # Render the IPF figure to get its image data
    ipf_fig.canvas.draw()
    ipf_image = np.frombuffer(ipf_fig.canvas.tostring_argb(), dtype=np.uint8)
    ipf_image = ipf_image.reshape(ipf_fig.canvas.get_width_height()[::-1] + (4,))
    ipf_image = ipf_image[:, :, 1:4]  # Drop alpha channel

    # Close the temporary IPF figure
    plt.close(ipf_fig)

    # Display the IPF key image
    ax_legend.imshow(ipf_image)
    ax_legend.axis("off")

    # Turn off all other axes in column 4
    for row_idx in range(n_noise_levels):
        if row_idx != 0 and row_idx != center_row:
            axes[row_idx][3].axis("off")

    # Add overall title
    fig.suptitle(
        f"IPF-Z Maps",
        fontsize=18,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save figure
    output_dir = Path("benchmark_results")
    output_file = output_dir / "figure_ipf_grid.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"✓ IPF-Z grid saved to: {output_file}")
    print(f"  Using {smallest_resolution}° dictionary resolution")

    output_pdf = output_dir / "figure_ipf_grid.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    print(f"✓ IPF-Z grid PDF saved to: {output_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    data_file = Path("benchmark_results/benchmark_dictionary.npy")

    # Adjust grid_shape if your scan has different dimensions
    create_ipf_grid(
        data_file=data_file,
        grid_shape=(149, 200),  # Adjust based on your scan dimensions
        dtype_filter="FP32",
    )

    print("\n" + "=" * 70)
    print("IPF-Z grid generation complete!")
    print("=" * 70)
