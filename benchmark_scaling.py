"""
Memory-efficient scaling analysis using upscaled PCA components.
Computes PCA on 60x60 images, then upscales components for larger sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import kikuchipy as kp
import time
import seaborn as sns
from pathlib import Path
import gc

# Import your utility functions
from utils import (
    MasterPattern,
    EBSDGeometry,
    sample_ori_fz_laue,
    qu_apply,
    get_radial_mask,
    progressbar,
    OnlineCovMatrix,
)

# Set publication-ready plotting style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
    }
)

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Realistic parameters
DICTIONARY_SIZE = 100_000  # Full realistic dictionary size
N_PCA_COMPONENTS = 512  # Standard number of components
N_TEST_PATTERNS = 500  # Fewer test patterns for faster benchmarking
DICT_CHUNK_SIZE = 2048  # Larger chunks since we're not storing full patterns
TEST_CHUNK_SIZE = 2048  # Process test patterns in chunks
N_TIMING_RUNS = 5  # Number of timing runs to average

# Base size for PCA computation, then larger sizes for scaling demo
BASE_SIZE = (60, 60)  # Compute PCA components here
IMAGE_SIZES = [
    (60, 60),  # Base size
    (128, 128),
    (256, 256),
    # (512, 512),  # Very slow, uncomment with smaller batch size if needed
]


def setup_geometry_for_size(detector_shape):
    """Initialize EBSD geometry for given detector shape."""
    mask = get_radial_mask(detector_shape).to(device).flatten()
    geometry = EBSDGeometry(
        detector_shape=detector_shape,
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


def compute_base_pca_components(detector_coords, mp, n_components):
    """Compute PCA components on the base size (60x60)."""
    print(
        f"Computing base PCA components ({n_components} components) for {DICTIONARY_SIZE:,} patterns..."
    )

    # Generate orientations
    ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=DICTIONARY_SIZE,
        device=torch.device("cpu"),
        permute=True,
    )

    # Use online covariance matrix (feasible for 60x60)
    cov = OnlineCovMatrix(detector_coords.shape[0]).to(device)

    # Process in chunks
    for chunk in progressbar(
        torch.split(ori, DICT_CHUNK_SIZE), prefix="Computing base PCA"
    ):
        chunk = chunk.to(device)
        rotated = qu_apply(chunk[:, None, :], detector_coords)
        pats = mp.interpolate(rotated).view(len(chunk), -1).to(torch.float16)
        # Center the patterns
        pats = pats - pats.mean(dim=1, keepdim=True)

        # Update covariance matrix (convert to float32 for numerical stability)
        cov(pats.to(torch.float32))

        # Clean up
        del rotated, pats, chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Get eigenvectors
    print("Computing eigenvectors...")
    eigenvectors = cov.get_eigenvectors()
    return eigenvectors[:, :n_components]


def upscale_pca_components(base_components, base_shape, target_shape):
    """Upscale PCA components from base size to target size."""
    print(f"Upscaling PCA components from {base_shape} to {target_shape}...")

    base_h, base_w = base_shape
    target_h, target_w = target_shape

    # Get base mask and target mask
    base_mask = get_radial_mask(base_shape).flatten()
    target_mask = get_radial_mask(target_shape).flatten()

    # Reshape base components to 2D images
    n_components = base_components.shape[1]
    upscaled_components = torch.zeros(target_mask.sum(), n_components, device=device)

    for i in range(n_components):
        # Reshape component to base image shape
        component_base = torch.zeros(base_h * base_w, device=device)
        component_base[base_mask] = base_components[:, i]
        component_2d = component_base.view(base_h, base_w)

        # Upscale using bilinear interpolation
        component_upscaled = F.interpolate(
            component_2d.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        # Apply target mask and store
        upscaled_components[:, i] = component_upscaled.flatten()[target_mask]

    return upscaled_components


def generate_test_patterns(n_patterns, detector_coords, mp):
    """Generate test patterns in batches."""
    print(f"Generating {n_patterns} test patterns...")

    # Generate orientations
    ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=n_patterns,
        device=torch.device("cpu"),
        permute=True,
    )

    # Generate patterns in chunks
    patterns = []
    for chunk in progressbar(
        torch.split(ori, TEST_CHUNK_SIZE), prefix="Generating test patterns"
    ):
        chunk = chunk.to(device)
        rotated = qu_apply(chunk[:, None, :], detector_coords)
        pats = mp.interpolate(rotated).view(len(chunk), -1)
        # Convert to float16 after interpolation and center the patterns
        pats = pats.to(torch.float16)
        pats = pats - pats.mean(dim=1, keepdim=True)
        patterns.append(pats.cpu())

        # Clean up GPU memory
        del rotated, pats, chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return torch.cat(patterns, dim=0)


def benchmark_conventional_di_batched(test_patterns, detector_coords, mp):
    """Benchmark conventional DI with batched dictionary generation."""
    print("Benchmarking conventional DI...")

    # Generate dictionary orientations
    dict_ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=DICTIONARY_SIZE,
        device=torch.device("cpu"),
        permute=True,
    )

    times = []

    for run in range(N_TIMING_RUNS):
        print(f"  Run {run+1}/{N_TIMING_RUNS}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Process test patterns in chunks
        all_best_matches = []

        for test_chunk in torch.split(test_patterns, TEST_CHUNK_SIZE):
            test_chunk = test_chunk.to(device)

            # For each test chunk, find best matches across entire dictionary
            chunk_best_scores = torch.full(
                (len(test_chunk),), -float("inf"), device=device, dtype=torch.float16
            )
            chunk_best_matches = torch.zeros(
                len(test_chunk), dtype=torch.long, device=device
            )

            # Process dictionary in chunks
            dict_start_idx = 0
            for dict_chunk in torch.split(dict_ori, DICT_CHUNK_SIZE):
                dict_chunk = dict_chunk.to(device)

                # Generate dictionary patterns
                rotated = qu_apply(dict_chunk[:, None, :], detector_coords)
                dict_pats = mp.interpolate(rotated).view(len(dict_chunk), -1)
                # Convert to float16 after interpolation and center the patterns
                dict_pats = dict_pats.to(torch.float16)
                dict_pats = dict_pats - dict_pats.mean(dim=1, keepdim=True)

                # Compute similarities for this chunk
                similarities = torch.mm(test_chunk, dict_pats.T)

                # Update best matches
                max_scores, max_indices = torch.max(similarities, dim=1)
                better_mask = max_scores > chunk_best_scores
                chunk_best_scores[better_mask] = max_scores[better_mask]
                chunk_best_matches[better_mask] = (
                    max_indices[better_mask] + dict_start_idx
                )

                dict_start_idx += len(dict_chunk)

                # Clean up
                del rotated, dict_pats, similarities, dict_chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            all_best_matches.append(chunk_best_matches.cpu())
            del test_chunk, chunk_best_scores, chunk_best_matches
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        times.append((end_time - start_time) / len(test_patterns))

    return np.mean(times), np.std(times)


def benchmark_pca_di_batched(test_patterns, detector_coords, mp, pca_components):
    """Benchmark PCA DI with batched processing."""
    print("Benchmarking PCA DI...")

    # Generate dictionary orientations
    dict_ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=DICTIONARY_SIZE,
        device=torch.device("cpu"),
        permute=True,
    )

    # Pre-project dictionary onto PCA space in chunks
    print("  Projecting dictionary onto PCA space...")
    dict_projected_chunks = []

    for dict_chunk in progressbar(
        torch.split(dict_ori, DICT_CHUNK_SIZE), prefix="  Projecting dictionary"
    ):
        dict_chunk = dict_chunk.to(device)

        # Generate dictionary patterns
        rotated = qu_apply(dict_chunk[:, None, :], detector_coords)
        dict_pats = mp.interpolate(rotated).view(len(dict_chunk), -1)
        # Convert to float16 after interpolation and center the patterns
        dict_pats = dict_pats.to(torch.float16)
        dict_pats = dict_pats - dict_pats.mean(dim=1, keepdim=True)

        # Project onto PCA space (convert to float32 for numerical stability)
        projected = torch.mm(dict_pats.to(torch.float32), pca_components)
        dict_projected_chunks.append(projected.cpu())

        # Clean up
        del rotated, dict_pats, projected, dict_chunk
        if device.type == "cuda":
            torch.cuda.empty_cache()

    times = []

    for run in range(N_TIMING_RUNS):
        print(f"  Run {run+1}/{N_TIMING_RUNS}")

        if device.type == "cuda":
            torch.cuda.synchronize()
        start_time = time.time()

        # Process test patterns in chunks
        all_best_matches = []

        for test_chunk in torch.split(test_patterns, TEST_CHUNK_SIZE):
            test_chunk = test_chunk.to(device)

            # Project test patterns onto PCA space (convert to float32 for numerical stability)
            test_projected = torch.mm(test_chunk.to(torch.float32), pca_components)

            # Find best matches across dictionary chunks
            chunk_best_scores = torch.full(
                (len(test_chunk),), -float("inf"), device=device
            )
            chunk_best_matches = torch.zeros(
                len(test_chunk), dtype=torch.long, device=device
            )

            dict_start_idx = 0
            for dict_projected_chunk in dict_projected_chunks:
                dict_projected_chunk = dict_projected_chunk.to(device)

                # Compute similarities in PCA space
                similarities = torch.mm(test_projected, dict_projected_chunk.T)

                # Update best matches
                max_scores, max_indices = torch.max(similarities, dim=1)
                better_mask = max_scores > chunk_best_scores
                chunk_best_scores[better_mask] = max_scores[better_mask]
                chunk_best_matches[better_mask] = (
                    max_indices[better_mask] + dict_start_idx
                )

                dict_start_idx += len(dict_projected_chunk)

                # Clean up
                del similarities, dict_projected_chunk
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            all_best_matches.append(chunk_best_matches.cpu())
            del test_chunk, test_projected, chunk_best_scores, chunk_best_matches
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        times.append((end_time - start_time) / len(test_patterns))

    return np.mean(times), np.std(times)


def main():
    """Main analysis function."""
    mp = load_master_pattern()

    # Compute base PCA components once
    print("Computing base PCA components on 60x60 images...")
    base_mask, base_detector_coords = setup_geometry_for_size(BASE_SIZE)
    base_pca_components = compute_base_pca_components(
        base_detector_coords, mp, N_PCA_COMPONENTS
    )

    results = {
        "image_size": [],
        "n_pixels": [],
        "conv_di_time": [],
        "conv_di_std": [],
        "pca_di_time": [],
        "pca_di_std": [],
        "speedup_factor": [],
        "dimensionality_reduction_factor": [],
    }

    print(f"\nAnalyzing scaling behavior with {DICTIONARY_SIZE:,} patterns...")
    print(f"Using {N_PCA_COMPONENTS} PCA components for all sizes")
    print(f"Test patterns: {N_TEST_PATTERNS}")

    for img_size in IMAGE_SIZES:
        print(f"\n{'='*20} Processing {img_size[0]}x{img_size[1]} images {'='*20}")

        # Setup geometry for this image size
        mask, detector_coords = setup_geometry_for_size(img_size)
        n_pixels = len(detector_coords)

        print(f"Effective pixels after masking: {n_pixels:,}")
        print(f"Dimensionality reduction factor: {n_pixels / N_PCA_COMPONENTS:.1f}x")

        # Get PCA components for this size
        if img_size == BASE_SIZE:
            pca_components = base_pca_components
        else:
            pca_components = upscale_pca_components(
                base_pca_components, BASE_SIZE, img_size
            )

        # Generate test patterns
        test_patterns = generate_test_patterns(N_TEST_PATTERNS, detector_coords, mp)

        # Benchmark conventional DI
        conv_time, conv_std = benchmark_conventional_di_batched(
            test_patterns, detector_coords, mp
        )

        # Benchmark PCA DI
        pca_time, pca_std = benchmark_pca_di_batched(
            test_patterns, detector_coords, mp, pca_components
        )

        # Calculate speedup
        speedup = conv_time / pca_time

        print(f"Results:")
        print(
            f"  Conventional DI: {conv_time*1000:.3f} ± {conv_std*1000:.3f} ms/pattern"
        )
        print(f"  PCA DI: {pca_time*1000:.3f} ± {pca_std*1000:.3f} ms/pattern")
        print(f"  Speedup: {speedup:.2f}x")

        # Store results
        results["image_size"].append(f"{img_size[0]}×{img_size[1]}")
        results["n_pixels"].append(n_pixels)
        results["conv_di_time"].append(conv_time)
        results["conv_di_std"].append(conv_std)
        results["pca_di_time"].append(pca_time)
        results["pca_di_std"].append(pca_std)
        results["speedup_factor"].append(speedup)
        results["dimensionality_reduction_factor"].append(n_pixels / N_PCA_COMPONENTS)

        # Clean up for next iteration
        del test_patterns, pca_components
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Create visualization
    create_scaling_plots(results)

    return results


def create_scaling_plots(results):
    """Create publication-quality plots showing scaling behavior."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    n_pixels = np.array(results["n_pixels"])
    speedup = np.array(results["speedup_factor"])
    conv_times = np.array(results["conv_di_time"]) * 1000  # Convert to ms
    pca_times = np.array(results["pca_di_time"]) * 1000
    dim_reduction = np.array(results["dimensionality_reduction_factor"])

    # Plot 1: Runtime vs Image Size (log-log)
    ax1.errorbar(
        n_pixels,
        conv_times,
        yerr=np.array(results["conv_di_std"]) * 1000,
        label="Conventional DI",
        marker="o",
        capsize=3,
        linewidth=2,
        markersize=6,
    )
    ax1.errorbar(
        n_pixels,
        pca_times,
        yerr=np.array(results["pca_di_std"]) * 1000,
        label=f"PCA DI ({N_PCA_COMPONENTS} components)",
        marker="s",
        capsize=3,
        linewidth=2,
        markersize=6,
    )
    ax1.set_xlabel("Number of Pixels")
    ax1.set_ylabel("Runtime (ms/pattern)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("A) Runtime Scaling with Image Size")

    # Add trend lines
    # Conventional DI should scale linearly with pixels
    if len(n_pixels) > 1:
        conv_slope = np.log(conv_times[-1] / conv_times[0]) / np.log(
            n_pixels[-1] / n_pixels[0]
        )
        pca_slope = np.log(pca_times[-1] / pca_times[0]) / np.log(
            n_pixels[-1] / n_pixels[0]
        )
        ax1.text(
            0.05,
            0.95,
            f"Conventional DI scaling: {conv_slope:.2f}\nPCA DI scaling: {pca_slope:.2f}",
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )

    # Plot 2: Speedup Factor vs Image Size
    ax2.semilogx(n_pixels, speedup, "ro-", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of Pixels")
    ax2.set_ylabel("Speedup Factor (×)")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("B) PCA-DI Speedup vs Image Size")

    # Add speedup annotations
    for i, (pixels, sp) in enumerate(zip(n_pixels, speedup)):
        ax2.annotate(
            f"{sp:.1f}×",
            (pixels, sp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
        )

    # Plot 3: Speedup vs Dimensionality Reduction Factor
    ax3.plot(dim_reduction, speedup, "go-", linewidth=2, markersize=8)

    # Add theoretical line (assuming some efficiency factor)
    efficiency = 0.3  # Empirical efficiency factor
    theoretical_speedup = dim_reduction * efficiency
    ax3.plot(
        dim_reduction,
        theoretical_speedup,
        "k--",
        alpha=0.7,
        label=f"Theoretical ({efficiency:.0%} efficiency)",
        linewidth=2,
    )

    ax3.set_xlabel("Dimensionality Reduction Factor")
    ax3.set_ylabel("Speedup Factor (×)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title("C) Speedup vs Dimensionality Reduction")

    # Plot 4: Scaling comparison table as text
    ax4.axis("off")
    ax4.set_title("D) Scaling Summary", pad=20)

    # Create table data
    table_data = []
    table_data.append(["Image Size", "Pixels", "Speedup", "Improvement"])
    for i, size in enumerate(results["image_size"]):
        improvement = speedup[i] / speedup[0] if i > 0 else 1.0
        table_data.append(
            [size, f"{n_pixels[i]:,}", f"{speedup[i]:.1f}×", f"{improvement:.1f}×"]
        )

    # Draw table
    table = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0.1, 0.3, 0.8, 0.6],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Add key insight text
    max_speedup = speedup[-1]
    max_improvement = max_speedup / speedup[0]
    insight_text = (
        f"Key Finding:\n"
        f"Speedup scales from {speedup[0]:.1f}× to {max_speedup:.1f}× \n"
        f"({max_improvement:.0f}× improvement with larger images)"
    )

    ax4.text(
        0.5,
        0.15,
        insight_text,
        transform=ax4.transAxes,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        fontsize=11,
        weight="bold",
    )

    plt.tight_layout()
    plt.savefig("figure_scaling.pdf", bbox_inches="tight")
    plt.savefig("figure_scaling.png", bbox_inches="tight", dpi=300)

    # Print summary
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS SUMMARY")
    print("=" * 80)
    print(
        f"{'Image Size':<15} {'Pixels':<10} {'DI (ms)':<12} {'PCA DI (ms)':<12} {'Speedup':<10} {'Improvement':<12}"
    )
    print("-" * 80)

    for i in range(len(results["image_size"])):
        improvement = speedup[i] / speedup[0] if i > 0 else 1.0
        print(
            f"{results['image_size'][i]:<15} "
            f"{results['n_pixels'][i]:<10,} "
            f"{results['conv_di_time'][i]*1000:<12.3f} "
            f"{results['pca_di_time'][i]*1000:<12.3f} "
            f"{results['speedup_factor'][i]:<10.1f} "
            f"{improvement:<12.2f}"
        )

    print("=" * 80)
    print(f"Dictionary size: {DICTIONARY_SIZE:,} patterns")
    print(f"PCA components: {N_PCA_COMPONENTS}")
    print(
        f"Speedup improvement from smallest to largest: {speedup[-1]/speedup[0]:.1f}×"
    )

    print(f"\nPlots saved with upscaled PCA components")


if __name__ == "__main__":
    results = main()
