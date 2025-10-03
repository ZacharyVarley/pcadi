import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from matplotlib.ticker import MultipleLocator

# Publication-quality settings
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

# Load benchmark data
data_file = Path("benchmark_results/benchmark_dictionary.npy")
data = np.load(data_file, allow_pickle=True).item()

# Filter for relevant data (FP32 dtype only)
dtype_mask = np.array(data["dtype"]) == "FP32"

methods = np.array(data["method"])[dtype_mask]
resolutions = np.array(data["dict_resolution"])[dtype_mask]
pca_comps = np.array(data["pca_components"])[dtype_mask]
raw_disoris = np.array(data["raw_disorientations"], dtype=object)[dtype_mask]
dataset_ids = np.array(data["dataset_id"])[dtype_mask]

# Get unique values
unique_resolutions = sorted(np.unique(resolutions))
unique_noise_ids = sorted(np.unique(dataset_ids))
n_resolutions = len(unique_resolutions)
n_noise_levels = len(unique_noise_ids)

# Map noise IDs to labels
noise_labels = {1: "Low", 5: "Medium", 10: "High"}

# Create figure with subplots (3x3 grid: rows=noise levels, cols=resolutions)
fig, axes = plt.subplots(
    n_noise_levels,
    n_resolutions,
    figsize=(3.5 * n_resolutions, 3.5 * n_noise_levels),
    sharex=True,
)
if n_resolutions == 1 and n_noise_levels == 1:
    axes = [[axes]]
elif n_resolutions == 1:
    axes = [[ax] for ax in axes]
elif n_noise_levels == 1:
    axes = [axes]

# Colors for each method
colors = {
    "DI": "#2E86AB",  # Blue
    "PCA-512": "#A23B72",  # Purple
    "PCA-1024": "#F18F01",  # Orange
}

# Line styles
line_styles = {
    "DI": "-",
    "PCA-512": "--",
    "PCA-1024": "-.",
}

# Plot PDFs for each noise level and resolution
x_range = np.linspace(0, 62.8, 500)

# First pass: compute all PDFs and find max y per row
row_max_y = {}
all_pdfs = {}

for row_idx, noise_id in enumerate(unique_noise_ids):
    row_max = 0
    for col_idx, resolution in enumerate(unique_resolutions):
        combo_mask = (dataset_ids == noise_id) & (resolutions == resolution)

        for method, pca_comp, disoris in zip(
            methods[combo_mask], pca_comps[combo_mask], raw_disoris[combo_mask]
        ):
            if len(disoris) == 0:
                continue

            if method == "DI":
                label = "DI"
            elif method == "PCA":
                label = f"PCA-{pca_comp}"
            else:
                continue

            if label not in colors:
                continue

            kde = gaussian_kde(disoris, bw_method="scott")
            pdf = kde(x_range)
            all_pdfs[(row_idx, col_idx, label)] = pdf
            row_max = max(row_max, pdf.max())

    row_max_y[row_idx] = row_max * 1.05  # Add 5% padding

# Second pass: plot with consistent y-scale per row
for row_idx, noise_id in enumerate(unique_noise_ids):
    for col_idx, resolution in enumerate(unique_resolutions):
        ax = axes[row_idx][col_idx]

        # Plot all stored PDFs for this subplot
        for label in ["DI", "PCA-512", "PCA-1024"]:
            if (row_idx, col_idx, label) in all_pdfs:
                pdf = all_pdfs[(row_idx, col_idx, label)]
                ax.plot(
                    x_range,
                    pdf,
                    color=colors[label],
                    linestyle=line_styles[label],
                    linewidth=1.5,
                    label=label,
                    alpha=0.9,
                )

        # Set y-limit for this row
        ax.set_ylim(0, row_max_y[row_idx])

        # Formatting
        if row_idx == n_noise_levels - 1:
            ax.set_xlabel("Disorientation (°)", fontsize=12)

        ax.set_xlim(0, 62.8)

        # Major ticks every 10 degrees, minor ticks every 5 degrees
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(5))

        # Thicker grid lines
        ax.grid(True, alpha=0.3, linewidth=0.8, which="major")
        ax.grid(True, alpha=0.15, linewidth=0.4, which="minor")

        # Title only on top row
        if row_idx == 0:
            ax.set_title(f"{resolution}° Dictionary", fontsize=12, fontweight="bold")

        # Y-label and legend only on first column
        if col_idx == 0:
            noise_label = noise_labels.get(noise_id, f"ID {noise_id}")
            ax.set_ylabel(
                f"{noise_label} Noise\nProbability Density",
                fontsize=14,
                fontweight="bold",
            )
            if row_idx == 0:
                ax.legend(
                    loc="lower right",
                    frameon=True,
                    framealpha=0.9,
                    edgecolor="gray",
                    fontsize=10,
                )

# Adjust layout
plt.tight_layout()

# # Save figure
# output_file = Path("benchmark_results/figure_disori_dists.png")
# plt.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
# print(f"✓ Figure saved to: {output_file}")

# # Also save as PDF for publications
# output_pdf = Path("benchmark_results/figure_disori_dists.pdf")
# plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
# print(f"✓ PDF version saved to: {output_pdf}")

plt.close(fig)

# Create second figure: single row with smallest resolution only
print("\n" + "=" * 70)
print("Creating single-row figure (smallest resolution only)...")
print("=" * 70)

smallest_res = min(unique_resolutions)
fig2, axes2 = plt.subplots(
    1, n_noise_levels, figsize=(3.5 * n_noise_levels, 3.5), sharex=True
)
if n_noise_levels == 1:
    axes2 = [axes2]

# Define inset x-limits for each noise level
inset_xlims = {1: 3, 5: 7, 10: 10}  # Low: 0-3, Medium: 0-7, High: 0-10

# Plot for smallest resolution only, independent y-axes
for col_idx, noise_id in enumerate(unique_noise_ids):
    ax = axes2[col_idx]

    combo_mask = (dataset_ids == noise_id) & (resolutions == smallest_res)

    # Store data for inset plotting
    plot_data = []

    for method, pca_comp, disoris in zip(
        methods[combo_mask], pca_comps[combo_mask], raw_disoris[combo_mask]
    ):
        if len(disoris) == 0:
            continue

        if method == "DI":
            label = "DI"
        elif method == "PCA":
            label = f"PCA-{pca_comp}"
        else:
            continue

        if label not in colors:
            continue

        kde = gaussian_kde(disoris, bw_method="scott")
        pdf = kde(x_range)

        ax.plot(
            x_range,
            pdf,
            color=colors[label],
            linestyle=line_styles[label],
            linewidth=1.5,
            label=label,
            alpha=0.9,
        )

        # Store for inset
        plot_data.append((label, kde))

    # Formatting
    ax.set_xlabel("Disorientation (°)", fontsize=12)
    ax.set_xlim(0, 62.8)

    # Major ticks every 10 degrees, minor ticks every 5 degrees
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))

    # Thicker grid lines
    ax.grid(True, alpha=0.3, linewidth=0.8, which="major")
    ax.grid(True, alpha=0.15, linewidth=0.4, which="minor")

    # Title with noise level
    noise_label = noise_labels.get(noise_id, f"ID {noise_id}")
    ax.set_title(
        f"{noise_label} Noise ({smallest_res}° Dictionary)",
        fontsize=12,
        fontweight="bold",
    )

    # Y-label only on first subplot
    if col_idx == 0:
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend(
            loc="lower right",
            frameon=True,
            framealpha=0.9,
            edgecolor="gray",
            fontsize=10,
        )

    # Create inset
    inset_xlim = inset_xlims.get(noise_id, 3)
    axins = ax.inset_axes(
        [0.5, 0.5, 0.45, 0.45]
    )  # [x, y, width, height] in axes coordinates

    # Create fine x range for inset
    x_inset = np.linspace(0, inset_xlim, 500)

    # Plot in inset
    for label, kde in plot_data:
        pdf_inset = kde(x_inset)
        axins.plot(
            x_inset,
            pdf_inset,
            color=colors[label],
            linestyle=line_styles[label],
            linewidth=1.2,
            alpha=0.9,
        )

    # Format inset
    axins.set_xlim(0, inset_xlim)
    axins.grid(True, alpha=0.2, linewidth=0.5)
    axins.tick_params(labelsize=8)

    # Add a box to indicate the zoomed region on the main plot
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5, linewidth=1)

# Adjust layout and save
plt.tight_layout()

output_file2 = Path("benchmark_results/figure_disori_dists.png")
plt.savefig(output_file2, dpi=DPI, bbox_inches="tight", facecolor="white")
print(f"✓ Single-resolution figure saved to: {output_file2}")

output_pdf2 = Path("benchmark_results/figure_disori_dists.pdf")
plt.savefig(output_pdf2, bbox_inches="tight", facecolor="white")
print(f"✓ Single-resolution PDF saved to: {output_pdf2}")

plt.close(fig2)

# Print statistics summary
print("\n" + "=" * 70)
print("DISORIENTATION STATISTICS SUMMARY")
print("=" * 70)

for noise_id in unique_noise_ids:
    noise_label = noise_labels.get(noise_id, f"ID {noise_id}")
    print(f"\n{noise_label} Noise (ID {noise_id})")
    print("=" * 70)

    for resolution in unique_resolutions:
        print(f"\nResolution: {resolution}°")
        print("-" * 50)

        combo_mask = (dataset_ids == noise_id) & (resolutions == resolution)

        for method, pca_comp, disoris in zip(
            methods[combo_mask], pca_comps[combo_mask], raw_disoris[combo_mask]
        ):
            if len(disoris) == 0:
                continue

            if method == "DI":
                label = "DI"
            elif method == "PCA":
                label = f"PCA-{pca_comp}"
            else:
                continue

            mean_disori = np.mean(disoris)
            median_disori = np.median(disoris)
            std_disori = np.std(disoris)
            frac_above_3 = np.mean(disoris > 3.0) * 100

            print(
                f"{label:12s} | Mean: {mean_disori:5.2f}° | "
                f"Median: {median_disori:5.2f}° | Std: {std_disori:5.2f}° | "
                f">3°: {frac_above_3:5.1f}%"
            )

print("\n" + "=" * 70)
