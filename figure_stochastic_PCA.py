import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import kikuchipy as kp
from utils import (
    MasterPattern,
    EBSDGeometry,
    OnlineCovMatrix,
    sample_ori_fz_laue,
    qu_apply,
    get_radial_mask,
    progressbar,
    get_radial_mask,
    so3_fibonacci,
    OjaPCA,
    OjaPCAExp,
)
import seaborn as sns
import time

# Set publication-ready plotting style
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
    }
)

# Use seaborn to set colorblind friendly color palette
sns.set_palette("colorblind")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set seed for reproducibility
torch.manual_seed(42)
DETECTOR_SHAPE = (60, 60)
MAX_EIGENIMAGE = 2819
STEP_SIZE = 10
CHUNK_SIZE = 1024
N_TEST_SAMPLES = 1000  # Number of samples in the test set

# Fixed dictionary size of 100,000
DICTIONARY_SIZE = 100_000


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


def generate_test_patterns(detector_coords, mp):
    """Generate test patterns using super-fibonacci sampling."""
    test_ori = so3_fibonacci(N_TEST_SAMPLES, device)  # always returns double
    test_ori = test_ori.to(torch.float32)

    # Generate patterns for test orientations
    rotated = qu_apply(test_ori[:, None, :].to(device), detector_coords)
    test_patterns = mp.interpolate(rotated).view(N_TEST_SAMPLES, -1)

    # Center the patterns
    test_patterns = test_patterns - test_patterns.mean(dim=1, keepdim=True)

    return test_patterns


def generate_dictionary_patterns(size, detector_coords, mp):
    """Generate dictionary patterns for all methods to use."""
    print(f"Generating dictionary patterns, size: {size:,}")
    start_time = time.time()

    # Sample orientations
    ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=size,
        device=torch.device("cpu"),
        permute=True,
    )

    # permute the orientations for each run
    perm = torch.randperm(len(ori), device="cpu")
    ori = ori[perm]

    # Generate patterns in chunks
    patterns = []
    for chunk in progressbar(
        torch.split(ori, CHUNK_SIZE),
        prefix=f"Generating patterns {len(ori):,}",
    ):
        rotated = qu_apply(chunk[:, None, :].to(device), detector_coords)
        pats = mp.interpolate(rotated).view(len(chunk), -1)
        # Center the patterns
        pats = pats - pats.mean(dim=1, keepdim=True)
        patterns.append(pats)

    # Return list of chunks to avoid memory issues
    print(f"Dictionary generation time: {time.time() - start_time:.2f} seconds")
    return patterns


def train_pca_methods(patterns_chunks, methods, method_names):
    """Train all PCA methods using the same dictionary patterns."""
    results = {}
    times = {}

    for method_name, method in zip(method_names, methods):
        print(f"\nTraining {method_name}...")
        start_time = time.time()

        # Process each chunk
        for i, patterns in enumerate(patterns_chunks):
            if hasattr(method, "__call__"):
                # Online methods like OnlineCovMatrix, Oja
                method(patterns)
            else:
                # For SVD, we'll collect all patterns and do it at the end
                pass

        # Get eigenvectors
        if method_name == "SVD":
            # For SVD, concatenate all patterns and perform SVD
            print("Performing SVD...")
            all_patterns = torch.cat(patterns_chunks, dim=0)
            U, S, V = torch.svd(all_patterns, some=True)
            eigenvectors = V[:, :MAX_EIGENIMAGE].T
        else:
            # Online methods already have their eigenvectors
            if hasattr(method, "get_eigenvectors"):
                eigenvectors = method.get_eigenvectors()
            else:
                eigenvectors = method.get_components()

        results[method_name] = eigenvectors
        times[method_name] = time.time() - start_time
        print(f"{method_name} training time: {times[method_name]:.2f} seconds")

    return results, times


def evaluate_on_testset(
    eigenvectors_dict,
    test_patterns,
    max_components,
    step_size,
):
    """Evaluate how well eigenvectors from each method capture variance in test set.
    Optimized to avoid redundant matrix multiplications and keep computations on GPU."""
    # Keep test patterns on the device they're already on
    device = test_patterns.device

    # Compute total variance in test set
    test_variance = torch.var(test_patterns, dim=0, unbiased=False).sum().item()
    print(f"Total variance in test set: {test_variance:.4f}")

    results = {}

    for method_name, eigenvectors in eigenvectors_dict.items():
        print(f"\nEvaluating method: {method_name}")

        # Move eigenvectors to the same device as test patterns if needed
        eigenvectors = eigenvectors.to(device)

        # Limit components to evaluate
        n_components = min(max_components, eigenvectors.shape[0])

        # Component counts to evaluate
        # component_counts = [10] + list(range(step_size, n_components + 1, step_size))
        # go up from 10
        component_counts = list(range(10, n_components + 1, step_size))
        if n_components not in component_counts:
            component_counts.append(n_components)

        # Sort component counts to ensure we compute in ascending order
        component_counts = sorted(component_counts)

        # Calculate captured variance for each component count
        captured_variance = []
        x_values = []

        if method_name == "Krasulina":
            eigenvectors = eigenvectors.flip(0)

        # Pre-compute full projection once (for maximum components)
        if method_name == "OnlineCovMatrix":
            # Special case for OnlineCovMatrix
            full_eigenvectors = eigenvectors[:, -n_components:].mT
        else:
            full_eigenvectors = eigenvectors[:n_components]

        # Compute full projection once
        print(
            f"Computing full projection for {method_name} ({n_components} components)"
        )
        full_projected = torch.matmul(test_patterns, full_eigenvectors.T)

        # Now iterate through component counts and reuse the projection
        for k in progressbar(component_counts, prefix=f"Method {method_name}"):
            # Use only first k eigenvectors for reconstruction
            if method_name == "OnlineCovMatrix":
                k_projected = full_projected[:, -k:]
                k_eigenvectors = full_eigenvectors[-k:]
            else:
                k_projected = full_projected[:, :k]
                k_eigenvectors = full_eigenvectors[:k]

            # Reconstruct patterns
            reconstructed = torch.matmul(k_projected, k_eigenvectors)

            # Calculate reconstruction error
            reconstruction_error = torch.mean(
                torch.sum((test_patterns - reconstructed) ** 2, dim=1)
            ).item()

            # Calculate captured variance (1 - normalized error)
            captured_var = 1.0 - (reconstruction_error / test_variance)
            captured_variance.append(captured_var)
            x_values.append(k)

        # Only convert to numpy at the end for storing results
        results[method_name] = (np.array(x_values), np.array(captured_variance))

    return results


def plot_relative_variance_comparison(variance_results, method_names):
    """Plot variance relative to SVD for better visualization of differences."""
    # Create figure
    fig = plt.figure(figsize=(15, 5))  # Adjust size as needed
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

    # Create subplots
    ax1 = fig.add_subplot(gs[0, 0])  # Left plot
    ax2 = fig.add_subplot(gs[0, 1])  # Right plot

    # Line styles for different methods
    line_styles = ["-", "-", "-", "-", "-"]

    # Extract colors via seaborn color palette
    colors = sns.color_palette("colorblind", len(method_names))

    # Get SVD data for reference
    svd_x, svd_captured_var = variance_results["SVD"]

    # First plot: Absolute variance captured (original plot)
    for i, method_name in enumerate(method_names):
        # Get data
        x_values, captured_var = variance_results[method_name]

        # Plot variance captured
        ax1.plot(
            x_values,
            captured_var,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i],
            label=method_name,
        )

    # Set title and labels for absolute variance plot
    ax1.set_title(f"Absolute Variance Captured - Dictionary Size {DICTIONARY_SIZE:,}")
    ax1.set_xlabel("Number of Components")
    ax1.set_ylabel("Fraction of Variance Captured")

    # Set grid
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Set axis limits
    ax1.set_xlim(
        0, max(x_values[-1] for x_values, _ in variance_results.values()) * 1.05
    )
    ax1.set_ylim(0, 1.01)

    # Add legend to absolute plot
    ax1.legend(
        loc="lower right",
        title="PCA Method",
        ncol=2,
    )

    # Second plot: Relative to SVD
    for i, method_name in enumerate(method_names):
        if method_name == "SVD":
            continue  # Skip SVD as it would be a flat line at 1.0

        # Get data
        x_values, captured_var = variance_results[method_name]

        # no interpolation needed since SVD also had the same x_values
        _, svd_captured_var = variance_results["SVD"]

        # Calculate relative variance (method / SVD)
        relative_var = captured_var / svd_captured_var

        # Plot relative variance
        ax2.plot(
            x_values,
            relative_var,
            linestyle=line_styles[i % len(line_styles)],
            color=colors[i],
            label=method_name,
        )

    # Add horizontal line at y=1.0 for reference (SVD performance)
    ax2.axhline(y=1.0, color="gray", linestyle="-", alpha=0.5, label="SVD (reference)")

    # Set title and labels for relative variance plot
    ax2.set_title("Fraction of SVD's Optimal Variance Captured")
    ax2.set_xlabel("Number of Components")
    ax2.set_ylabel("Fraction of SVD's Captured Variance")

    # Set grid
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Set axis limits for relative variance
    ax2.set_xlim(
        0, max(x_values[-1] for x_values, _ in variance_results.values()) * 1.05
    )

    # focus on relative differences
    ax2.set_ylim(0.94, 1.005)

    # # Add legend to relative plot
    # ax2.legend(
    #     loc="lower right",
    #     title="PCA Method",
    #     ncol=2,
    # )

    # Add panel labels (A and B)
    ax1.text(
        -0.1,
        1.05,
        "A",
        transform=ax1.transAxes,
        fontweight="bold",
        va="top",
    )
    ax2.text(
        -0.1,
        1.05,
        "B",
        transform=ax2.transAxes,
        fontweight="bold",
        va="top",
    )

    plt.tight_layout()
    plt.savefig("figure_stoch_pca.pdf", bbox_inches="tight")
    plt.savefig("figure_stoch_pca.png", dpi=300, bbox_inches="tight")


def train_pca_methods_batched(
    detector_coords, mp, all_ori, methods, method_names, batch_size=128
):
    """Train all PCA methods using batched processing without storing all patterns."""
    results = {}
    times = {}

    for method_name, method in zip(method_names, methods):
        print(f"\nTraining {method_name}...")
        start_time = time.time()

        if method_name == "SVD":
            # SVD needs all patterns, but we'll generate them batch by batch and use online SVD
            # or collect minimal information for final computation
            pass

        # Process in batches, generate patterns on the fly
        for batch_idx in progressbar(
            range(0, len(all_ori), batch_size), prefix=f"Training {method_name}"
        ):
            # Get batch of orientations
            end_idx = min(batch_idx + batch_size, len(all_ori))
            ori_batch = all_ori[batch_idx:end_idx]

            # Generate patterns on the fly
            rotated = qu_apply(ori_batch[:, None, :].to(device), detector_coords)
            patterns_batch = mp.interpolate(rotated).view(len(ori_batch), -1)

            # Center the patterns
            patterns_batch = patterns_batch - patterns_batch.mean(dim=1, keepdim=True)

            # Pass to PCA method
            if method_name != "SVD":
                if hasattr(method, "__call__"):
                    method(patterns_batch)

            # Free memory
            del patterns_batch, rotated
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Extract eigenvectors
        if method_name == "SVD":
            # For SVD, we need to do a separate pass to compute the actual decomposition
            print("Computing SVD...")

            # Create a matrix to accumulate XX^T (covariance-like) information
            n_features = detector_coords.shape[0]
            svd_accumulator = torch.zeros(
                (n_features, n_features), dtype=torch.float32, device=device
            )

            # Second pass to compute covariance matrix for SVD
            for batch_idx in progressbar(
                range(0, len(all_ori), batch_size), prefix=f"SVD accumulation"
            ):
                # Get batch of orientations
                end_idx = min(batch_idx + batch_size, len(all_ori))
                ori_batch = all_ori[batch_idx:end_idx]

                # Generate patterns on the fly
                rotated = qu_apply(ori_batch[:, None, :].to(device), detector_coords)
                patterns_batch = mp.interpolate(rotated).view(len(ori_batch), -1)

                # Center the patterns
                patterns_batch = patterns_batch - patterns_batch.mean(
                    dim=1, keepdim=True
                )

                # Update accumulator
                svd_accumulator += torch.matmul(patterns_batch.T, patterns_batch)

                # Free memory
                del patterns_batch, rotated
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            # Compute eigendecomposition (which is equivalent to SVD for symmetric matrices)
            print("Computing eigendecomposition for SVD...")
            eigenvalues, eigenvectors = torch.linalg.eigh(svd_accumulator)

            # Sort in descending order (eigh returns in ascending)
            idx = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # Keep only the top MAX_EIGENIMAGE components
            eigenvectors = eigenvectors[:, :MAX_EIGENIMAGE].T

            results[method_name] = eigenvectors
        else:
            # Online methods already have their eigenvectors
            if hasattr(method, "get_eigenvectors"):
                results[method_name] = method.get_eigenvectors()
            else:
                results[method_name] = method.get_components()

        times[method_name] = time.time() - start_time
        print(f"{method_name} training time: {times[method_name]:.2f} seconds")

    return results, times


def main():
    start_total = time.time()

    # Setup
    mask, detector_coords = setup_geometry()
    mp = load_master_pattern()

    # Generate test set
    print("Generating test patterns...")
    test_patterns = generate_test_patterns(detector_coords, mp)
    print(f"Generated {test_patterns.shape[0]} test patterns.")

    # Generate orientations (not patterns) for dictionary
    print(f"Generating {DICTIONARY_SIZE:,} orientations...")
    all_ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=DICTIONARY_SIZE,
        device=torch.device("cpu"),
        permute=True,
    )

    # Permute the orientations for each run
    perm = torch.randperm(len(all_ori), device="cpu")
    all_ori = all_ori[perm].to(device)

    # Initialize PCA methods
    n_features = detector_coords.shape[0]
    n_components = min(MAX_EIGENIMAGE, n_features)

    methods = [
        # Online covariance matrix (from original script)
        OnlineCovMatrix(
            n_features,
            covmat_dtype=torch.float32,
            delta_dtype=torch.float32,
            correlation=False,
        ).to(device),
        # SVD (placeholder - will be handled separately)
        "SVD",
        # Oja with constant learning rate
        OjaPCA(
            n_features=n_features,
            n_components=n_components,
            eta=0.005,
        ).to(device),
        # Oja with exponential decay
        OjaPCAExp(
            n_features=n_features,
            n_components=n_components,
            total_steps=DICTIONARY_SIZE // CHUNK_SIZE,
            initial_eta=1.0,
            final_eta=1e-4,
        ).to(device),
    ]

    method_names = [
        "OnlineCovMatrix",
        "SVD",
        "Oja (fixed eta)",
        "Oja (exp decay)",
        # "Krasulina",
    ]

    # Train all methods with truly batched processing
    eigenvectors_dict, processing_times = train_pca_methods_batched(
        detector_coords, mp, all_ori, methods, method_names, batch_size=CHUNK_SIZE
    )

    # The rest of the function remains the same...
    # Evaluate on test set
    print("\nEvaluating methods on test set...")
    variance_results = evaluate_on_testset(
        eigenvectors_dict,
        test_patterns,
        max_components=MAX_EIGENIMAGE,
        step_size=STEP_SIZE,
    )

    # Create plot showing relative variance compared to SVD
    plot_relative_variance_comparison(variance_results, method_names)

    # Print timing summary
    print("\nProcessing Times Summary:")
    print("-" * 50)
    print(f"{'Method':<20} {'Time (s)':<10}")
    print("-" * 50)
    for method, proc_time in processing_times.items():
        print(f"{method:<20} {proc_time:<10.2f}")

    total_time = time.time() - start_total
    print(f"Total execution time: {total_time:.2f} seconds")


if __name__ == "__main__":
    main()
