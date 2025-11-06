import numpy as np
import matplotlib.pyplot as plt
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
)
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import seaborn as sns

# "font.family": "serif",
# "font.serif": ["Times New Roman", "DejaVu Serif"],
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

# use seaborn to set colorblind friendly color palette
sns.set_palette("colorblind")

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# set seed for reproducibility
torch.manual_seed(42)
DETECTOR_SHAPE = (60, 60)
MAX_EIGENIMAGE = 2048
CHUNK_SIZE = 4096 * 4
N_TEST_SAMPLES = 1000  # Number of samples in the test set

# Dictionary configurations
DICTIONARY_SIZES = [1_000, 10_000, 100_000]

# Noise configuration
NOISE_LEVELS = [0, 5, 1]  # 0 means no noise, higher value means LESS noise


def add_poisson_noise(tensor, lam):
    """
    Add Poisson noise to a tensor.

    Parameters:
    -----------
    tensor : torch.Tensor
        Input tensor to which noise will be added
    lam : float
        Lambda parameter for Poisson distribution (controls noise intensity)

    Returns:
    --------
    torch.Tensor: Tensor with Poisson noise added
    """
    if lam == 0:
        return tensor  # No noise

    # Scale tensor to positive range for Poisson distribution
    min_val = tensor.min()
    tensor_shifted = tensor - min_val

    # For Poisson noise, we need to first scale the intensity
    # Higher lambda means MORE noise (not less) in this implementation
    # We use lambda as a direct scaling factor for signal intensity
    signal_mean = tensor_shifted.mean()
    base_intensity = signal_mean

    # Apply Poisson noise - we are generating noise with mean = lambda * base_intensity
    # Then normalize by lambda to keep in same range
    noise = torch.poisson(torch.ones_like(tensor_shifted) * lam) / lam

    # Scale noise to match signal intensity
    scaled_noise = noise * base_intensity

    # Add noise to signal (not replacing with noise)
    noisy_tensor = tensor_shifted + (scaled_noise - base_intensity)

    # Shift back to original range
    noisy_tensor = noisy_tensor + min_val

    return noisy_tensor


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


def process_dictionary_with_permutations(size, detector_coords, mp):
    """Process a single dictionary configuration with permutations for multiple runs."""
    print(f"Processing size: {size:,}")
    results = []

    # Sample once for this dictionary size
    ori = sample_ori_fz_laue(
        laue_id=11,
        target_n_samples=size,
        device=torch.device("cpu"),
        permute=True,
    )

    # Permute the orientations for each run
    perm = torch.randperm(len(ori), device="cpu")
    ori_permuted = ori[perm]

    # Create PCA object
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

    # Get eigenvectors and eigenvalues
    eigenvectors = pca.get_eigenvectors().cpu()
    covmat = pca.get_covmat()
    eigenvalues, _ = torch.linalg.eigh(covmat)
    eigenvalues = eigenvalues.cpu()

    # Sort in descending order (eigh returns in ascending order)
    eigenvalues = eigenvalues.flip(0)
    eigenvectors = eigenvectors.flip(1)

    results.append((eigenvectors, eigenvalues))

    return results


def evaluate_on_testset(
    eigenvectors_dict, test_patterns, noise_level=0, max_components=1000, step_size=10
):
    """
    Evaluate how well eigenvectors from each dictionary capture variance in test set,
    using subsampling to efficiently handle many components.

    Parameters:
    -----------
    eigenvectors_dict : dict
        Dictionary with size as key and eigenvectors as value
    test_patterns : torch.Tensor
        Test patterns to evaluate on
    noise_level : float
        Poisson noise level to add to test patterns (0 = no noise)
    max_components : int
        Maximum number of components to evaluate
    step_size : int
        Interval between component counts to evaluate (e.g., 10 means test 1, 11, 21, etc.)

    Returns:
    --------
    dict: Dictionary with size as key and captured variance as value
    """
    # Add Poisson noise to test patterns if specified
    if noise_level > 0:
        test_patterns_noisy = add_poisson_noise(test_patterns, noise_level)
        print(f"Added Poisson noise with lambda={noise_level}")
    else:
        test_patterns_noisy = test_patterns

    # Move test patterns to CPU for consistency
    test_patterns_cpu = test_patterns_noisy.cpu()

    # Compute total variance in test set
    test_variance = torch.var(test_patterns_cpu, dim=0).sum().item()
    print(
        f"Total variance in test set with noise level {noise_level}: {test_variance:.4f}"
    )

    results = {}

    for size, (eigenvectors, _) in eigenvectors_dict.items():
        # Limit components to evaluate
        n_components = min(max_components, eigenvectors.shape[1])

        # Create subsampled list of component counts to evaluate
        # Always include 1 as the starting point
        component_counts = [1] + list(range(step_size, n_components + 1, step_size))
        # Add the maximum if it's not already included
        if n_components not in component_counts:
            component_counts.append(n_components)

        # Calculate captured variance for selected component counts
        captured_variance = []
        x_values = []

        for k in progressbar(component_counts, prefix=f"Size {size}"):
            # Get top-k eigenvectors
            top_k_eigenvectors = eigenvectors[:, :k]

            # Project test patterns onto eigenvectors
            projected = torch.matmul(test_patterns_cpu, top_k_eigenvectors)

            # Reconstruct patterns
            reconstructed = torch.matmul(projected, top_k_eigenvectors.T)

            # Calculate reconstruction error
            reconstruction_error = torch.mean(
                torch.sum((test_patterns_cpu - reconstructed) ** 2, dim=1)
            ).item()

            # Calculate captured variance (1 - normalized error)
            captured_var = 1.0 - (reconstruction_error / test_variance)
            captured_variance.append(captured_var)
            x_values.append(k)

        results[size] = (np.array(x_values), np.array(captured_variance))

    return results


def analyze_noise_effect(variance_results, dictionary_sizes, noise_levels):
    """
    Analyze the effect of noise on variance captured and determine optimal dictionary size.

    Parameters:
    -----------
    variance_results : dict
        Dictionary with noise level as key and dictionaries of variance captured results as value
    dictionary_sizes : list
        List of dictionary sizes
    noise_levels : list
        List of noise levels

    Returns:
    --------
    dict: Analysis results
    """
    # Component counts to analyze at
    component_counts = [50, 100, 500, 1000]

    analysis = {}

    for comp_count in component_counts:
        analysis[comp_count] = {}

        # For each noise level
        for noise in noise_levels:
            sizes_variance = []

            # For each dictionary size
            for size in dictionary_sizes:
                # Get variance captured data
                x_values, captured_var = variance_results[noise][size]

                # Find index of component count (or closest to it)
                idx = np.argmin(np.abs(x_values - comp_count))
                actual_comp = x_values[idx]
                variance = captured_var[idx]

                sizes_variance.append((size, variance))

            # Calculate diminishing returns
            diminishing_returns = []
            for i in range(1, len(sizes_variance)):
                prev_size, prev_var = sizes_variance[i - 1]
                curr_size, curr_var = sizes_variance[i]

                # Calculate improvement per log10 unit of dictionary size
                improvement = curr_var - prev_var
                log_diff = np.log10(curr_size) - np.log10(prev_size)
                improvement_rate = improvement / log_diff

                diminishing_returns.append(
                    (prev_size, curr_size, improvement, improvement_rate)
                )

            analysis[comp_count][noise] = {
                "captured_variance": sizes_variance,
                "improvements": diminishing_returns,
            }

    return analysis


def create_combined_figure(
    test_patterns, variance_results, dictionary_sizes, noise_levels
):
    """
    Create a combined figure with noisy patterns on the left and variance plot on the right.

    Parameters:
    -----------
    test_patterns : torch.Tensor
        Original test patterns
    variance_results : dict
        Dictionary with noise level as key and variance results as value
    dictionary_sizes : list
        List of dictionary sizes
    noise_levels : list
        List of noise levels
    """
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(12, 5), dpi=300)

    # Define grid spec to control subplot sizing
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2])

    # Left panel: Noisy patterns
    ax_left = fig.add_subplot(gs[0])

    # turn off the spines for the left panel
    for spine in ax_left.spines.values():
        spine.set_visible(False)
    ax_left.set_xticks([])
    ax_left.set_yticks([])

    # Select patterns to visualize
    pattern_indices = [0, 1, 2]
    num_patterns = len(pattern_indices)

    # Create a nested gridspec for the left panel
    gs_left = gs[0].subgridspec(num_patterns, len(noise_levels))

    # Plot patterns with different noise levels
    for i, idx in enumerate(pattern_indices):
        pattern = test_patterns[idx].clone().cpu()

        for j, noise in enumerate(noise_levels):
            # Add noise to pattern
            if noise > 0:
                pattern_vals = add_poisson_noise(pattern.unsqueeze(0), noise).squeeze(0)
            else:
                pattern_vals = pattern

            # Reshape pattern to 2D for visualization
            pattern_w_mask = torch.full(
                (DETECTOR_SHAPE[0], DETECTOR_SHAPE[1]), np.nan, dtype=torch.float32
            )
            mask = get_radial_mask(DETECTOR_SHAPE).to(device)

            # Normalize for better visualization
            min_val = pattern_vals.min()
            max_val = pattern_vals.max()
            pattern_vals = (
                (pattern_vals - min_val) / (max_val - min_val)
                if max_val > min_val
                else pattern_vals
            )
            pattern_w_mask[mask.cpu().numpy()] = pattern_vals.cpu()

            # Plot
            ax = fig.add_subplot(gs_left[i, j])
            im = ax.imshow(pattern_w_mask, cmap="gray", vmin=0, vmax=1)

            # Add labels
            if noise == 0:
                ax.set_title("Original" if i == 0 else "")
            else:
                ax.set_title(f"a={noise}" if i == 0 else "")

            ax.set_ylabel(f"Pattern {idx+1}" if j == 0 else "")
            ax.set_xticks([])
            ax.set_yticks([])

            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Right panel: Variance plot
    ax_right = fig.add_subplot(gs[1])

    # Line styles for noise levels
    line_styles = ["-", "--", "-.", ":"]

    # Extract colors via seaborn color palette
    # colors = sns.color_palette("colorblind", len(dictionary_sizes))
    colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]
    cmap = {
        1_000: colors[2],
        10_000: colors[1],
        100_000: colors[0],
    }

    # Plot for each dictionary size and noise level
    for i, size in enumerate(dictionary_sizes):
        for j, noise in enumerate(noise_levels):
            # Get data
            x_values, captured_var = variance_results[noise][size]

            # Label for the line
            if j == 0:  # Only include size in label for no noise
                label = f"Size: {size:,}"
            else:
                label = f"Size: {size:,}, a={noise}"

            # Plot variance captured
            ax_right.plot(
                x_values,
                captured_var,
                linestyle=line_styles[j % len(line_styles)],
                color=cmap[size],
                label=label,
                marker=None,
                markersize=2,
                markevery=len(x_values) // 10,  # Show fewer markers
            )

    # Set title and labels for right panel
    ax_right.set_title("Effect of Dictionary Size and Noise on Variance Captured")
    ax_right.set_xlabel("Number of Components")
    ax_right.set_ylabel("Fraction of Variance Captured")

    # Set grid for right panel
    ax_right.grid(True, linestyle="--", alpha=0.6)

    # Set axis limits for right panel
    ax_right.set_xlim(
        0, max(x_values[-1] for x_values, _ in variance_results[0].values()) * 1.05
    )
    ax_right.set_ylim(0, 1.05)

    # Add legend to right panel
    ax_right.legend(
        loc="lower right",
        title="Dictionary Size and Noise Level",
        ncol=3,
        handlelength=2,
        handletextpad=0.5,
        borderpad=0.5,
        labelspacing=0.5,
        columnspacing=1,
        borderaxespad=0.5,
    )

    # Add panel labels
    fig.text(0.01, 0.98, "A", fontsize=12, fontweight="bold")
    fig.text(0.38, 0.98, "B", fontsize=12, fontweight="bold")

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("figure_poisson.pdf", bbox_inches="tight")
    plt.savefig("figure_poisson.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # Setup
    mask, detector_coords = setup_geometry()
    mp = load_master_pattern()

    # Generate consistent test set
    print("Generating test patterns...")
    test_patterns = generate_test_patterns(detector_coords, mp)
    print(f"Generated {test_patterns.shape[0]} test patterns.")

    # Process dictionaries using separate sampling for each size
    eigenimages_runs = {
        n_entry: process_dictionary_with_permutations(
            n_entry,
            detector_coords,
            mp,
        )
        for n_entry in DICTIONARY_SIZES
    }

    # Prepare dictionary of eigenimages
    eigenimages_dict = {size: eigenimages_runs[size][0] for size in DICTIONARY_SIZES}

    # Evaluate on test set with different noise levels
    variance_results = {}
    for noise_level in NOISE_LEVELS:
        print(f"\nEvaluating with noise level a={noise_level}...")
        results = evaluate_on_testset(
            eigenimages_dict,
            test_patterns,
            noise_level=noise_level,
            max_components=2800,
            step_size=50,
        )
        variance_results[noise_level] = results

    print("Creating combined figure...")
    create_combined_figure(
        test_patterns, variance_results, DICTIONARY_SIZES, NOISE_LEVELS
    )


if __name__ == "__main__":
    main()
