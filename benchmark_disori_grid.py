import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

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


def create_disorientation_grid(data_file, grid_shape=(149, 200), dtype_filter="FP32"):
    """Create 3x3 grid of disorientation heatmaps for different noise levels and methods."""

    # Load benchmark data
    data = np.load(data_file, allow_pickle=True).item()

    # Filter for specific dtype
    mask = np.array(data["dtype"]) == dtype_filter

    methods = np.array(data["method"])[mask]
    resolutions = np.array(data["dict_resolution"])[mask]
    pca_comps = np.array(data["pca_components"])[mask]
    raw_disorientations = np.array(data["raw_disorientations"], dtype=object)[mask]
    noise_ids = np.array(data["dataset_id"])[mask]

    # Use smallest resolution only
    smallest_resolution = min(np.unique(resolutions))
    res_mask = resolutions == smallest_resolution

    methods = methods[res_mask]
    pca_comps = pca_comps[res_mask]
    raw_disorientations = raw_disorientations[res_mask]
    noise_ids = noise_ids[res_mask]

    # Unique noise levels
    unique_noise_ids = sorted(np.unique(noise_ids))
    n_noise_levels = len(unique_noise_ids)

    # Noise labels
    noise_labels = {1: "Low", 5: "Medium", 10: "High"}

    # Method ordering
    method_order = ["DI", "PCA-512", "PCA-1024"]

    # Create grayscale colormap for publication (dark gray = 0°, light gray = max error)
    # Inverted grayscale: dark for low error, light for high error
    cmap = plt.cm.gray_r  # Reversed grayscale (0=dark, 1=light)

    # Calculate figure size
    img_aspect = grid_shape[1] / grid_shape[0]  # width / height
    subplot_height = 3.0
    subplot_width = subplot_height * img_aspect

    # Create figure with 3 columns (no 4th column needed)
    fig, axes = plt.subplots(
        n_noise_levels,
        3,
        figsize=(3 * subplot_width + 1, n_noise_levels * subplot_height),
        squeeze=False,
    )

    # Set colorbar range to maximum cubic disorientation
    # Max cubic disorientation = 2*arccos((2 + sqrt(2)) / 4) ≈ 62.8°
    vmin = 0.0
    vmax = 2 * np.arccos((2 + np.sqrt(2)) / 4) * 180 / np.pi  # 62.8 degrees

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

            # Get the disorientations
            idx = np.where(method_mask)[0][0]
            disoris = raw_disorientations[idx]

            if len(disoris) == 0:
                ax.text(
                    0.5, 0.5, "Empty", ha="center", va="center", transform=ax.transAxes
                )
                ax.axis("off")
                continue

            # Reshape to grid
            H, W = grid_shape
            if len(disoris) != H * W:
                print(
                    f"Warning: Expected {H*W} values, got {len(disoris)} for {method_label} at noise_id {noise_id}"
                )
                ax.text(
                    0.5,
                    0.5,
                    f"Size Mismatch\n{len(disoris)} vs {H*W}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.axis("off")
                continue

            # Convert to heatmap
            try:
                disori_map = disoris.reshape(H, W)
                im = ax.imshow(
                    disori_map, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest"
                )

                # Add border around the image to distinguish from background
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor("black")
                    spine.set_linewidth(1.5)

            except Exception as e:
                print(f"Error processing {method_label} at noise_id {noise_id}: {e}")
                ax.text(
                    0.5, 0.5, f"Error", ha="center", va="center", transform=ax.transAxes
                )
                ax.axis("off")
                continue

            ax.set_xticks([])
            ax.set_yticks([])

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

    # Add colorbar to the right of the grid
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(
        "Indexing Disorientation (degrees)", rotation=270, labelpad=20, fontsize=14
    )

    # # Add overall title
    # fig.suptitle(
    #     f"Indexing Disorientation Angle Heatmaps",
    #     fontsize=18,
    #     fontweight="bold",
    #     y=0.995,
    # )

    plt.tight_layout(rect=[0, 0, 0.9, 1])

    # Save figure
    output_dir = Path("benchmark_results")
    output_file = output_dir / "figure_disori_grid.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"✓ Disorientation grid saved to: {output_file}")
    print(f"  Using {smallest_resolution}° dictionary resolution")
    print(f"  Colorbar range: {vmin:.2f}° to {vmax:.2f}°")

    output_pdf = output_dir / "figure_disori_grid.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    print(f"✓ Disorientation grid PDF saved to: {output_pdf}")

    plt.close(fig)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("DISORIENTATION STATISTICS")
    print("=" * 70)
    for row_idx, noise_id in enumerate(unique_noise_ids):
        noise_label = noise_labels.get(noise_id, f"ID {noise_id}")
        print(f"\n{noise_label} Noise (ID={noise_id}):")
        for method_label in method_order:
            # Find matching data
            if method_label == "DI":
                method_mask = (noise_ids == noise_id) & (methods == "DI")
            else:
                n_comp = int(method_label.split("-")[1])
                method_mask = (
                    (noise_ids == noise_id) & (methods == "PCA") & (pca_comps == n_comp)
                )

            if np.any(method_mask):
                idx = np.where(method_mask)[0][0]
                disoris = raw_disorientations[idx]
                if len(disoris) > 0:
                    mean_disori = np.mean(disoris)
                    median_disori = np.median(disoris)
                    max_disori = np.max(disoris)
                    frac_above_3 = (disoris > 3.0).mean() * 100
                    print(
                        f"  {method_label:12s}: mean={mean_disori:5.2f}°, median={median_disori:5.2f}°, max={max_disori:5.2f}°, >3°={frac_above_3:5.1f}%"
                    )


if __name__ == "__main__":
    data_file = Path("benchmark_results/benchmark_dictionary.npy")

    # Adjust grid_shape if your scan has different dimensions
    create_disorientation_grid(
        data_file=data_file,
        grid_shape=(149, 200),  # Adjust based on your scan dimensions
        dtype_filter="FP32",
    )

    print("\n" + "=" * 70)
    print("Disorientation grid generation complete!")
    print("=" * 70)
