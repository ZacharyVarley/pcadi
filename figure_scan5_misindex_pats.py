import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn.functional as F
import kikuchipy as kp

from orix.quaternion.symmetry import Oh
from orix.quaternion import Orientation
from orix.plot import IPFColorKeyTSL

from utils import (
    ExperimentPatterns,
    get_radial_mask,
    MasterPattern,
    EBSDGeometry,
    qu_apply,
)

# Import quaternion to Bunge conversion from benchmark_ipfz_grid.py
from torch import Tensor


@torch.jit.script
def qu2bu(qu: Tensor) -> Tensor:
    """
    Convert rotations given as quaternions to Bunge angles (ZXZ Euler angles).

    Args:
        qu (Tensor): shape (..., 4) quaternions in the format (w, x, y, z).

    Returns:
        torch.Tensor: shape (..., 3) Bunge angles in radians.
    """

    bu = torch.empty(qu.shape[:-1] + (3,), dtype=qu.dtype, device=qu.device)

    q03 = qu[..., 0] ** 2 + qu[..., 3] ** 2
    q12 = qu[..., 1] ** 2 + qu[..., 2] ** 2
    chi = torch.sqrt((q03 * q12))

    mask_chi_zero = chi == 0
    mA = (mask_chi_zero) & (q12 == 0)
    mB = (mask_chi_zero) & (q03 == 0)
    mC = ~mask_chi_zero

    bu[mA, 0] = torch.atan2(-2 * qu[mA, 0] * qu[mA, 3], qu[mA, 0] ** 2 - qu[mA, 3] ** 2)
    bu[mA, 1] = 0
    bu[mA, 2] = 0

    bu[mB, 0] = torch.atan2(2 * qu[mB, 1] * qu[mB, 2], qu[mB, 1] ** 2 - qu[mB, 2] ** 2)
    bu[mB, 1] = torch.pi
    bu[mB, 2] = 0

    bu[mC, 0] = torch.atan2(
        (qu[mC, 1] * qu[mC, 3] - qu[mC, 0] * qu[mC, 2]) / chi[mC],
        (-qu[mC, 0] * qu[mC, 1] - qu[mC, 2] * qu[mC, 3]) / chi[mC],
    )
    bu[mC, 1] = torch.atan2(2 * chi[mC], q03[mC] - q12[mC])
    bu[mC, 2] = torch.atan2(
        (qu[mC, 0] * qu[mC, 2] + qu[mC, 1] * qu[mC, 3]) / chi[mC],
        (qu[mC, 2] * qu[mC, 3] - qu[mC, 0] * qu[mC, 1]) / chi[mC],
    )

    # add 2pi to negative angles for first and last angles
    bu[..., 0] = torch.where(bu[..., 0] < 0, bu[..., 0] + 2 * torch.pi, bu[..., 0])
    bu[..., 2] = torch.where(bu[..., 2] < 0, bu[..., 2] + 2 * torch.pi, bu[..., 2])

    return bu


def quaternions_to_ipf_colors(quaternions, symmetry=Oh):
    """
    Convert quaternions to IPF-Z RGB colors.

    Args:
        quaternions: (N, 4) array of quaternions
        symmetry: orix symmetry object

    Returns:
        (N, 3) RGB array in [0, 1]
    """
    # Convert quaternions to Bunge Euler angles
    qu_tensor = torch.from_numpy(quaternions.astype(np.float32))
    euler = qu2bu(qu_tensor).cpu().numpy()  # Bunge Euler angles in radians

    # Create orix Orientation objects
    O = Orientation.from_euler(euler, symmetry=symmetry, degrees=False)

    # Get IPF-Z colors
    key = IPFColorKeyTSL(symmetry.laue)
    colors = key.orientation2color(O).astype(np.float32)

    return colors


# Publication-quality settings
DPI = 300
plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 10,
        "axes.linewidth": 0.5,
    }
)


def load_and_clean_patterns(
    pattern_id, size=(60, 60), grid_shape=(149, 200), device="cpu"
):
    """Load and preprocess experimental patterns (same as in benchmark.py)."""
    print(f"Loading experimental patterns (ID={pattern_id}, size={size})...", end=" ")

    exp_pats_all = kp.data.ni_gain(allow_download=True, number=pattern_id)

    # Original data shape is (scanH, scanW, patH, patW)
    original_shape = exp_pats_all.data.shape
    print(f"\n  Original shape: {original_shape}")

    # Convert to tensor and flatten to (scanH*scanW, patH, patW)
    exp_pats_data = torch.tensor(exp_pats_all.data, dtype=torch.float16, device=device)

    if exp_pats_data.shape[-2:] != size:
        exp_pats_data = F.interpolate(
            exp_pats_data.view(-1, 1, *exp_pats_data.shape[-2:]),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
    else:
        # Still need to flatten from (scanH, scanW, patH, patW) to (scanH*scanW, patH, patW)
        exp_pats_data = exp_pats_data.view(-1, *exp_pats_data.shape[-2:])

    # Apply the same preprocessing as in benchmark.py
    exp_pats = ExperimentPatterns(exp_pats_data)
    exp_pats.standard_clean()  # This does: subtract_overall_background, normalize, CLAHE, normalize

    # Apply radial mask for visualization (mask out corners)
    radial_mask = get_radial_mask(size).to(device)
    # Set masked regions to 0 (black background)
    exp_pats.patterns = exp_pats.patterns * radial_mask[None, :, :]

    # Convert to numpy for visualization
    patterns_np = exp_pats.patterns.cpu().numpy()

    print(
        f"  ✓ Loaded {patterns_np.shape[0]} patterns (flattened, preprocessed, masked)"
    )
    return patterns_np


def simulate_dictionary_patterns(orientations, mp, geom, mask, device):
    """
    Simulate dictionary patterns for given orientations (minimal preprocessing like benchmark.py).

    Args:
        orientations: (N, 4) array of quaternions
        mp: MasterPattern object
        geom: EBSDGeometry object
        mask: Radial mask tensor (2D boolean)
        device: torch device

    Returns:
        (N, H, W) numpy array of simulated patterns
    """
    ori_tensor = torch.from_numpy(orientations.astype(np.float32)).to(device)

    # Get detector coordinates
    detector_coords = geom.get_coords_sample_frame(binning=(1, 1))
    detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

    # Rotate detector coordinates by orientations
    batch_rot = qu_apply(ori_tensor[:, None, :], detector_coords[None, ...])

    # Interpolate from master pattern
    sim_patterns = (
        mp.interpolate(
            batch_rot,
            mode="bilinear",
            align_corners=False,
            normalize_coords=False,
            virtual_binning=1,
        )
        .squeeze()
        .view(len(ori_tensor), -1)
    )

    # Apply minimal preprocessing like benchmark.py: just mean subtraction per pattern
    sim_patterns -= torch.mean(sim_patterns, dim=-1, keepdim=True)

    # Reshape back to 2D patterns and normalize for display
    sim_patterns = sim_patterns.view(len(ori_tensor), *geom.detector_shape)

    # Normalize each pattern to [0, 1] for visualization
    sim_patterns_flat = sim_patterns.view(len(ori_tensor), -1)
    pat_mins = torch.min(sim_patterns_flat, dim=-1).values
    pat_maxs = torch.max(sim_patterns_flat, dim=-1).values
    sim_patterns_flat -= pat_mins[:, None]
    sim_patterns_flat /= 1e-4 + (pat_maxs[:, None] - pat_mins[:, None])
    sim_patterns = sim_patterns_flat.view(len(ori_tensor), *geom.detector_shape)

    # Apply radial mask (set masked regions to 0 for visualization)
    sim_patterns = sim_patterns * mask[None, :, :]

    return sim_patterns.cpu().numpy()


def create_misindexed_pattern_grid(
    data_file,
    grid_shape=(149, 200),
    disori_min=59.9,
    disori_max=60.1,
    n_examples=25,
    random_seed=42,
):
    """
    Create a 5x5 grid of patterns from Scan 1 that correspond to
    disorientation errors in a specific range in Scan 5.

    Args:
        data_file: Path to benchmark_dictionary.npy
        grid_shape: Shape of the EBSD scan
        disori_min: Minimum disorientation angle to select (degrees)
        disori_max: Maximum disorientation angle to select (degrees)
        n_examples: Number of examples to show (25 for 5x5 grid)
        random_seed: Random seed for reproducibility
    """

    # Load benchmark data
    print("Loading benchmark data...")
    data = np.load(data_file, allow_pickle=True).item()

    # Filter for Scan 5, DI method, FP32 dtype
    mask = (
        (np.array(data["dataset_id"]) == 5)
        & (np.array(data["method"]) == "DI")
        & (np.array(data["dtype"]) == "FP32")
    )

    if not np.any(mask):
        print("Error: No data found for Scan 5, DI method, FP32 dtype")
        return

    # Get the first matching entry (should be the smallest resolution)
    idx = np.where(mask)[0][0]
    disorientations = data["raw_disorientations"][idx]

    print(f"Found {len(disorientations)} disorientations for Scan 5")
    print(f"Min: {disorientations.min():.2f}°, Max: {disorientations.max():.2f}°")

    # Find indices where disorientation is in the specified range
    range_mask = (disorientations >= disori_min) & (disorientations <= disori_max)
    range_indices = np.where(range_mask)[0]
    range_values = disorientations[range_mask]

    print(
        f"Found {len(range_indices)} patterns with disorientation in [{disori_min}°, {disori_max}°]"
    )

    if len(range_indices) == 0:
        print("No patterns found in the specified range!")
        return

    # Randomly select n_examples
    np.random.seed(random_seed)
    if len(range_indices) > n_examples:
        selected_idx = np.random.choice(range_indices, size=n_examples, replace=False)
        selected_disoris = disorientations[selected_idx]
    else:
        selected_idx = range_indices
        selected_disoris = range_values
        print(
            f"Warning: Only {len(selected_idx)} patterns available, less than {n_examples}"
        )

    # Sort by disorientation value (highest first)
    sort_order = np.argsort(selected_disoris)[::-1]
    selected_idx = selected_idx[sort_order]
    selected_disoris = selected_disoris[sort_order]

    # Determine device for preprocessing
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load Scan 1 patterns (clean reference) with full preprocessing
    patterns_scan1 = load_and_clean_patterns(
        pattern_id=1, size=(60, 60), grid_shape=grid_shape, device=device
    )

    # Create 5x5 grid
    n_rows = 5
    n_cols = 5
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 12),
        squeeze=False,
    )

    print("\nCreating pattern grid...")
    for i, (idx_val, disori_val) in enumerate(zip(selected_idx, selected_disoris)):
        if i >= n_rows * n_cols:
            break

        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]

        # Get pattern from Scan 1
        pattern = patterns_scan1[idx_val]

        # Display pattern
        ax.imshow(pattern, cmap="gray", vmin=0, vmax=1, interpolation="nearest")

        # Convert linear index to 2D coordinates
        scan_row = idx_val // grid_shape[1]
        scan_col = idx_val % grid_shape[1]

        # Add title with disorientation value and position
        ax.set_title(f"{disori_val:.1f}° ({scan_row},{scan_col})", fontsize=9, pad=3)

        # Add border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)

        ax.set_xticks([])
        ax.set_yticks([])

    # Turn off any unused subplots
    for i in range(len(selected_idx), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row][col].axis("off")

    # Add overall title
    fig.suptitle(
        f"Scan 1 Patterns with Disorientation in Scan 5 [{disori_min}°-{disori_max}°]\n"
        f"Likely Σ3/Σ9 Twin Boundaries",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()

    # Save figure
    output_dir = Path("benchmark_results")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "figure_scan5_misindex_pats.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Pattern grid saved to: {output_file}")

    output_pdf = output_dir / "figure_scan5_misindex_pats.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    print(f"✓ Pattern grid PDF saved to: {output_pdf}")

    plt.close(fig)

    # Print statistics
    print("\n" + "=" * 70)
    print("PATTERN STATISTICS")
    print("=" * 70)
    print(f"Disorientation range filter: [{disori_min}°, {disori_max}°]")
    print(f"Patterns displayed: {min(len(selected_idx), n_rows * n_cols)}")
    print(
        f"Actual disorientation range: {selected_disoris.min():.2f}° - {selected_disoris.max():.2f}°"
    )
    print(f"Mean disorientation: {selected_disoris.mean():.2f}°")
    print(f"Median disorientation: {np.median(selected_disoris):.2f}°")
    print("=" * 70)

    # Return selected indices and disorientations for neighborhood visualization
    return selected_idx, selected_disoris, patterns_scan1


def create_neighborhood_windows(
    selected_idx,
    selected_disoris,
    patterns_scan1,
    patterns_scan5,
    disorientations_all,
    reference_orientations,
    mp,
    geom,
    mask,
    device,
    grid_shape=(149, 200),
    n_examples=10,
    window_size=5,
):
    """
    Create neighborhood windows (5x5) around the first n_examples high-error patterns.

    Args:
        selected_idx: Indices of selected high-error patterns
        selected_disoris: Disorientation values of selected patterns
        patterns_scan1: All patterns from Scan 1 (flattened)
        patterns_scan5: All patterns from Scan 5 (flattened)
        disorientations_all: All disorientation values
        reference_orientations: Reference orientations (quaternions) from Scan 1
        mp: MasterPattern object for simulating dictionary patterns
        geom: EBSDGeometry object
        mask: Radial mask tensor
        device: torch device
        grid_shape: Shape of the EBSD scan (H, W)
        n_examples: Number of neighborhood windows to create
        window_size: Size of the neighborhood window (5 = 5x5)
    """

    output_dir = Path("benchmark_results") / "misindexing_windows"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING NEIGHBORHOOD WINDOWS")
    print("=" * 70)

    # Convert all reference orientations to IPF colors
    print("  Computing IPF-Z colors for all orientations...")
    ipf_colors = quaternions_to_ipf_colors(reference_orientations, symmetry=Oh)
    print(f"  ✓ IPF colors computed ({ipf_colors.shape[0]} colors)")

    window_radius = window_size // 2  # 2 for a 5x5 window
    scanH, scanW = grid_shape

    valid_examples = []

    # Check which examples are far enough from the border
    for i in range(min(n_examples, len(selected_idx))):
        idx_val = selected_idx[i]
        disori_val = selected_disoris[i]

        # Convert linear index to 2D coordinates
        scan_row = idx_val // scanW
        scan_col = idx_val % scanW

        # Check if window fits within scan bounds
        if (
            scan_row >= window_radius
            and scan_row < scanH - window_radius
            and scan_col >= window_radius
            and scan_col < scanW - window_radius
        ):
            valid_examples.append((i, idx_val, disori_val, scan_row, scan_col))
        else:
            print(
                f"  Skipping pattern at ({scan_row},{scan_col}) - too close to border"
            )

    if len(valid_examples) == 0:
        print("No valid examples found (all too close to border)")
        return

    print(f"Creating {len(valid_examples)} neighborhood windows...")

    # Create a separate figure for each neighborhood
    for example_idx, (i, idx_val, disori_val, center_row, center_col) in enumerate(
        valid_examples, 1
    ):
        # Collect indices and orientations for this window
        window_linear_indices = []
        window_orientations = []

        for win_row in range(window_size):
            for win_col in range(window_size):
                actual_row = center_row - window_radius + win_row
                actual_col = center_col - window_radius + win_col
                linear_idx = actual_row * scanW + actual_col
                window_linear_indices.append(linear_idx)
                window_orientations.append(reference_orientations[linear_idx])

        window_orientations = np.array(window_orientations)

        # Simulate dictionary patterns for this window
        print(f"  Simulating dictionary patterns for window {example_idx}...", end=" ")
        dict_patterns = simulate_dictionary_patterns(
            window_orientations, mp, geom, mask, device
        )
        print("✓")

        # Create experimental pattern figures (Scan 1 and Scan 5)
        fig_exp1, axes_exp1 = plt.subplots(
            window_size,
            window_size,
            figsize=(10, 10),
            squeeze=False,
        )

        fig_exp5, axes_exp5 = plt.subplots(
            window_size,
            window_size,
            figsize=(10, 10),
            squeeze=False,
        )

        # Create dictionary pattern figure
        fig_dict, axes_dict = plt.subplots(
            window_size,
            window_size,
            figsize=(10, 10),
            squeeze=False,
        )

        # Fill all three figures
        for idx_in_window, (win_row, win_col) in enumerate(
            [(r, c) for r in range(window_size) for c in range(window_size)]
        ):
            linear_idx = window_linear_indices[idx_in_window]

            # Get patterns, disorientation, and IPF color
            exp_pattern1 = patterns_scan1[linear_idx]
            exp_pattern5 = patterns_scan5[linear_idx]
            dict_pattern = dict_patterns[idx_in_window]
            disori = disorientations_all[linear_idx]
            ipf_color = ipf_colors[linear_idx]  # RGB in [0, 1]

            # Check if this is the center pattern
            is_center = win_row == window_radius and win_col == window_radius

            # Plot experimental pattern from Scan 1
            ax_exp1 = axes_exp1[win_row][win_col]
            ax_exp1.imshow(
                exp_pattern1, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )

            for spine in ax_exp1.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ipf_color)
                spine.set_linewidth(3.5)

            ax_exp1.set_title(
                f"{disori:.1f}°",
                fontsize=8,
                pad=2,
                color="black",
                fontweight="bold" if is_center else "normal",
            )
            ax_exp1.set_xticks([])
            ax_exp1.set_yticks([])

            # Plot experimental pattern from Scan 5
            ax_exp5 = axes_exp5[win_row][win_col]
            ax_exp5.imshow(
                exp_pattern5, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )

            for spine in ax_exp5.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ipf_color)
                spine.set_linewidth(3.5)

            ax_exp5.set_title(
                f"{disori:.1f}°",
                fontsize=8,
                pad=2,
                color="black",
                fontweight="bold" if is_center else "normal",
            )
            ax_exp5.set_xticks([])
            ax_exp5.set_yticks([])

            # Plot dictionary pattern
            ax_dict = axes_dict[win_row][win_col]
            ax_dict.imshow(
                dict_pattern, cmap="gray", vmin=0, vmax=1, interpolation="nearest"
            )

            for spine in ax_dict.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor(ipf_color)
                spine.set_linewidth(3.5)

            ax_dict.set_title(
                f"{disori:.1f}°",
                fontsize=8,
                pad=2,
                color="black",
                fontweight="bold" if is_center else "normal",
            )
            ax_dict.set_xticks([])
            ax_dict.set_yticks([])

        # Add overall titles
        fig_exp1.suptitle(
            f"Window {example_idx} - Experimental Patterns from Scan 1 (Low Noise)\n"
            f"Center: ({center_row},{center_col}), {disori_val:.1f}°",
            fontsize=12,
            fontweight="bold",
        )

        fig_exp5.suptitle(
            f"Window {example_idx} - Experimental Patterns from Scan 5 (Medium Noise)\n"
            f"Center: ({center_row},{center_col}), {disori_val:.1f}°",
            fontsize=12,
            fontweight="bold",
        )

        fig_dict.suptitle(
            f"Window {example_idx} - Dictionary Patterns (Reference Orientations)\n"
            f"Center: ({center_row},{center_col}), {disori_val:.1f}°",
            fontsize=12,
            fontweight="bold",
        )

        plt.figure(fig_exp1.number)
        plt.tight_layout()
        plt.figure(fig_exp5.number)
        plt.tight_layout()
        plt.figure(fig_dict.number)
        plt.tight_layout()

        # Save experimental pattern figures
        output_file_exp1 = (
            output_dir / f"window_{example_idx:02d}_experimental_scan_1.png"
        )
        fig_exp1.savefig(
            output_file_exp1, dpi=DPI, bbox_inches="tight", facecolor="white"
        )

        output_pdf_exp1 = (
            output_dir / f"window_{example_idx:02d}_experimental_scan_1.pdf"
        )
        fig_exp1.savefig(output_pdf_exp1, bbox_inches="tight", facecolor="white")

        output_file_exp5 = (
            output_dir / f"window_{example_idx:02d}_experimental_scan_5.png"
        )
        fig_exp5.savefig(
            output_file_exp5, dpi=DPI, bbox_inches="tight", facecolor="white"
        )

        output_pdf_exp5 = (
            output_dir / f"window_{example_idx:02d}_experimental_scan_5.pdf"
        )
        fig_exp5.savefig(output_pdf_exp5, bbox_inches="tight", facecolor="white")

        # Save dictionary pattern figure
        output_file_dict = output_dir / f"window_{example_idx:02d}_dictionary.png"
        fig_dict.savefig(
            output_file_dict, dpi=DPI, bbox_inches="tight", facecolor="white"
        )

        output_pdf_dict = output_dir / f"window_{example_idx:02d}_dictionary.pdf"
        fig_dict.savefig(output_pdf_dict, bbox_inches="tight", facecolor="white")

        print(f"  ✓ Window {example_idx} saved (scan 1, scan 5, dictionary)")

        plt.close(fig_exp1)
        plt.close(fig_exp5)
        plt.close(fig_dict)

    print("=" * 70)


def create_comparison_grid(
    selected_idx,
    selected_disoris,
    patterns_scan1,
    patterns_scan5,
    reference_orientations,
    indexed_orientations_scan5,
    mp,
    geom,
    mask,
    device,
    grid_shape=(149, 200),
    n_examples=10,
):
    """
    Create a 4-row by 10-column comparison grid showing:
    - Row 1: Scan 1 experimental patterns (with IPF borders for reference orientation)
    - Row 2: Scan 5 experimental patterns (with IPF borders for indexed orientation)
    - Row 3: Dictionary patterns from Scan 1 reference orientation
    - Row 4: Dictionary patterns from Scan 5 indexed orientation

    Args:
        selected_idx: Selected pattern indices
        selected_disoris: Disorientations for selected patterns
        patterns_scan1: Scan 1 experimental patterns (H*W, Ph, Pw)
        patterns_scan5: Scan 5 experimental patterns (H*W, Ph, Pw)
        reference_orientations: Reference orientations from Scan 1 (H*W, 4)
        indexed_orientations_scan5: Indexed orientations from Scan 5 (H*W, 4)
        mp: Master pattern object
        geom: EBSD geometry
        mask: Radial mask
        device: torch device
        grid_shape: (H, W) shape of the scan grid
        n_examples: Number of examples to include (default 10)
    """
    print("\n" + "=" * 70)
    print("CREATING 4x10 COMPARISON GRID")
    print("=" * 70)

    # Take first n_examples from the selected indices
    n_to_show = min(n_examples, len(selected_idx))
    indices_to_show = selected_idx[:n_to_show]
    disoris_to_show = selected_disoris[:n_to_show]

    # Get orientations for these patterns
    ref_quats = reference_orientations[indices_to_show]  # (n, 4)
    idx_quats = indexed_orientations_scan5[indices_to_show]  # (n, 4)

    # Get IPF colors
    ref_colors = quaternions_to_ipf_colors(ref_quats)  # (n, 3)
    idx_colors = quaternions_to_ipf_colors(idx_quats)  # (n, 3)

    # Simulate dictionary patterns for both orientations
    print("Simulating dictionary patterns for reference orientations...", end=" ")
    dict_patterns_ref = simulate_dictionary_patterns(
        orientations=ref_quats, mp=mp, geom=geom, mask=mask, device=device
    )
    print("✓")

    print("Simulating dictionary patterns for indexed orientations...", end=" ")
    dict_patterns_idx = simulate_dictionary_patterns(
        orientations=idx_quats, mp=mp, geom=geom, mask=mask, device=device
    )
    print("✓")

    # Get experimental patterns
    exp_patterns_scan1 = patterns_scan1[indices_to_show]  # (n, Ph, Pw)
    exp_patterns_scan5 = patterns_scan5[indices_to_show]  # (n, Ph, Pw)

    # Create figure with 4 rows and 10 columns
    fig, axes = plt.subplots(4, n_to_show, figsize=(n_to_show * 1.2, 4 * 1.2))
    if n_to_show == 1:
        axes = axes.reshape(4, 1)

    # Plot patterns
    for col_idx in range(n_to_show):
        # Row 0: Scan 1 experimental with reference IPF border
        ax = axes[0, col_idx]
        ax.imshow(exp_patterns_scan1[col_idx], cmap="gray", vmin=0, vmax=1)
        border_color = ref_colors[col_idx]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel("Scan 1\nExp", fontsize=10, fontweight="bold")

        # Row 1: Scan 5 experimental with indexed IPF border
        ax = axes[1, col_idx]
        ax.imshow(exp_patterns_scan5[col_idx], cmap="gray", vmin=0, vmax=1)
        border_color = idx_colors[col_idx]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel("Scan 5\nExp", fontsize=10, fontweight="bold")

        # Row 2: Dictionary from reference orientation
        ax = axes[2, col_idx]
        ax.imshow(dict_patterns_ref[col_idx], cmap="gray", vmin=0, vmax=1)
        border_color = ref_colors[col_idx]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel("Dict\n(Ref)", fontsize=10, fontweight="bold")

        # Row 3: Dictionary from indexed orientation
        ax = axes[3, col_idx]
        ax.imshow(dict_patterns_idx[col_idx], cmap="gray", vmin=0, vmax=1)
        border_color = idx_colors[col_idx]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3.5)
        ax.set_xticks([])
        ax.set_yticks([])
        if col_idx == 0:
            ax.set_ylabel("Dict\n(Idx)", fontsize=10, fontweight="bold")

        # Add disorientation as column title
        disori_val = disoris_to_show[col_idx]
        axes[0, col_idx].set_title(f"{disori_val:.1f}°", fontsize=9)

    # Add main title
    fig.suptitle(
        "Pattern Comparison: Reference vs Indexed Orientations\n"
        "IPF-Z Border Colors Show Orientation",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    # Save figure
    output_dir = Path("benchmark_results/misindexing_windows")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "comparison_grid_4x10.png"
    fig.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")

    output_pdf = output_dir / "comparison_grid_4x10.pdf"
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")

    print(f"✓ Comparison grid saved to {output_file}")
    print(f"✓ Comparison grid saved to {output_pdf}")

    plt.close(fig)
    print("=" * 70)


if __name__ == "__main__":
    data_file = Path("benchmark_results/benchmark_dictionary.npy")

    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        print("Please run benchmark.py first to generate the data.")
    else:
        # Load benchmark data for neighborhood windows
        data = np.load(data_file, allow_pickle=True).item()

        # Filter for Scan 5, DI method, FP32 dtype
        mask = (
            (np.array(data["dataset_id"]) == 5)
            & (np.array(data["method"]) == "DI")
            & (np.array(data["dtype"]) == "FP32")
        )
        idx = np.where(mask)[0][0]
        disorientations_all = data["raw_disorientations"][idx]

        # Get reference orientations from Scan 1 (ground truth)
        reference_orientations = data["reference_orientations"]
        print(
            f"Loaded {len(reference_orientations)} reference orientations from Scan 1"
        )

        # Set up device and load master pattern for dictionary simulation
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("✓ Using CUDA for dictionary simulation")
        else:
            device = torch.device("cpu")
            print("✓ Using CPU for dictionary simulation")

        # Load master pattern
        print("Loading master pattern...", end=" ")
        kp_mp = kp.data.nickel_ebsd_master_pattern_small(
            projection="lambert", hemisphere="both"
        )
        mp_nh = torch.from_numpy(kp_mp.data[0].astype(np.float32)).to(device)
        mp_sh = torch.from_numpy(kp_mp.data[1].astype(np.float32)).to(device)
        master_pattern = torch.cat((mp_nh, mp_sh), dim=-1)
        mp = MasterPattern(master_pattern, laue_group=11)
        mp.normalize(norm_type="minmax")
        mp.apply_clahe()
        print("✓")

        # Set up geometry
        pattern_size = (60, 60)
        geom = EBSDGeometry(
            detector_shape=pattern_size, proj_center=(0.4221, 0.2179, 0.4954)
        ).to(device)
        radial_mask = get_radial_mask(pattern_size).to(device)
        print(
            f"✓ Geometry initialized for {pattern_size[0]}x{pattern_size[1]} patterns"
        )

        # Create main 5x5 grid of patterns with disorientations in the Σ3 twin range
        selected_idx, selected_disoris, patterns_scan1 = create_misindexed_pattern_grid(
            data_file=data_file,
            grid_shape=(149, 200),
            disori_min=59.9,
            disori_max=60.1,
            n_examples=25,
            random_seed=42,
        )

        # Load Scan 5 patterns (medium noise) for comparison
        print("\nLoading Scan 5 patterns for neighborhood windows...")
        patterns_scan5 = load_and_clean_patterns(
            pattern_id=5, size=(60, 60), grid_shape=(149, 200), device=device
        )

        # Create neighborhood windows for examples with IPF-colored borders
        create_neighborhood_windows(
            selected_idx=selected_idx,
            selected_disoris=selected_disoris,
            patterns_scan1=patterns_scan1,
            patterns_scan5=patterns_scan5,
            disorientations_all=disorientations_all,
            reference_orientations=reference_orientations,
            mp=mp,
            geom=geom,
            mask=radial_mask,
            device=device,
            grid_shape=(149, 200),
            n_examples=2,
            window_size=5,
        )

        # Get indexed orientations from Scan 5 for comparison grid
        indexed_orientations_scan5 = data["indexed_orientations"][idx]
        print(
            f"\nLoaded {len(indexed_orientations_scan5)} indexed orientations from Scan 5"
        )

        # Create 4x10 comparison grid
        create_comparison_grid(
            selected_idx=selected_idx,
            selected_disoris=selected_disoris,
            patterns_scan1=patterns_scan1,
            patterns_scan5=patterns_scan5,
            reference_orientations=reference_orientations,
            indexed_orientations_scan5=indexed_orientations_scan5,
            mp=mp,
            geom=geom,
            mask=radial_mask,
            device=device,
            grid_shape=(149, 200),
            n_examples=10,
        )

        print("\n" + "=" * 70)
        print("Pattern grid generation complete!")
        print("=" * 70)
