import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

from orix.quaternion.symmetry import Oh
from orix.quaternion import Orientation, Misorientation

from utils import disori_angle_laue, disorientation, ori_to_fz_laue

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


def get_sigma3_quaternions_torch():
    """
    Get quaternions for 60° rotations around all 8 <111> axes as torch tensor.

    Returns:
        (8, 4) torch tensor of quaternions (w, x, y, z)
    """
    axes = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                axis = torch.tensor([i, j, k], dtype=torch.float32)
                axis = axis / torch.norm(axis)
                axes.append(axis)

    axes = torch.stack(axes)  # (8, 3)
    angle = 60.0 * np.pi / 180.0  # 60 degrees in radians

    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    # Convert axis-angle to quaternions
    quats = torch.zeros(8, 4, dtype=torch.float32)
    quats[:, 0] = c  # w component
    quats[:, 1:] = axes * s  # x, y, z components

    # double check normed and in the fundamental zone
    quats /= torch.norm(quats, dim=1, keepdim=True)
    quats = ori_to_fz_laue(quats, laue_id=11)

    return quats


def get_sigma9_quaternions_torch():
    """
    Get quaternions for 38.9° rotations around all 12 <110> axes as torch tensor.

    Returns:
        (12, 4) torch tensor of quaternions (w, x, y, z)
    """
    axes = []
    # Generate all 12 <110> directions
    for i in [-1, 1]:
        for j in [-1, 1]:
            axes.append(torch.tensor([i, j, 0], dtype=torch.float32))
            axes.append(torch.tensor([i, 0, j], dtype=torch.float32))
            axes.append(torch.tensor([0, i, j], dtype=torch.float32))

    # Normalize axes
    axes = torch.stack(axes)  # (12, 3)
    axes = axes / torch.norm(axes, dim=1, keepdim=True)

    angle = 38.9 * np.pi / 180.0  # 38.9 degrees in radians

    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    # Convert axis-angle to quaternions
    quats = torch.zeros(12, 4, dtype=torch.float32)
    quats[:, 0] = c  # w component
    quats[:, 1:] = axes * s  # x, y, z components

    # Normalize and move to fundamental zone
    quats /= torch.norm(quats, dim=1, keepdim=True)
    quats = ori_to_fz_laue(quats, laue_id=11)

    return quats


def get_sigma27a_quaternions_torch():
    """
    Get quaternions for 31.59° rotations around all 12 <110> axes as torch tensor.

    Returns:
        (12, 4) torch tensor of quaternions (w, x, y, z)
    """
    axes = []
    # Generate all 12 <110> directions
    for i in [-1, 1]:
        for j in [-1, 1]:
            axes.append(torch.tensor([i, j, 0], dtype=torch.float32))
            axes.append(torch.tensor([i, 0, j], dtype=torch.float32))
            axes.append(torch.tensor([0, i, j], dtype=torch.float32))

    # Normalize axes
    axes = torch.stack(axes)  # (12, 3)
    axes = axes / torch.norm(axes, dim=1, keepdim=True)

    angle = 31.59 * np.pi / 180.0  # 31.59 degrees in radians

    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    # Convert axis-angle to quaternions
    quats = torch.zeros(12, 4, dtype=torch.float32)
    quats[:, 0] = c  # w component
    quats[:, 1:] = axes * s  # x, y, z components

    # Normalize and move to fundamental zone
    quats /= torch.norm(quats, dim=1, keepdim=True)
    quats = ori_to_fz_laue(quats, laue_id=11)

    return quats


def get_sigma27b_quaternions_torch():
    """
    Get quaternions for 35.43° rotations around all 24 <210> axes as torch tensor.

    Returns:
        (24, 4) torch tensor of quaternions (w, x, y, z)
    """
    axes = []
    # Generate all 24 <210> directions
    for i in [-2, 2]:
        for j in [-1, 1]:
            axes.append(torch.tensor([i, j, 0], dtype=torch.float32))
            axes.append(torch.tensor([i, 0, j], dtype=torch.float32))
            axes.append(torch.tensor([j, i, 0], dtype=torch.float32))
            axes.append(torch.tensor([0, i, j], dtype=torch.float32))
            axes.append(torch.tensor([j, 0, i], dtype=torch.float32))
            axes.append(torch.tensor([0, j, i], dtype=torch.float32))

    # Normalize axes
    axes = torch.stack(axes)  # (24, 3)
    axes = axes / torch.norm(axes, dim=1, keepdim=True)

    angle = 35.43 * np.pi / 180.0  # 35.43 degrees in radians

    half_angle = angle / 2.0
    s = np.sin(half_angle)
    c = np.cos(half_angle)

    # Convert axis-angle to quaternions
    quats = torch.zeros(24, 4, dtype=torch.float32)
    quats[:, 0] = c  # w component
    quats[:, 1:] = axes * s  # x, y, z components

    # Normalize and move to fundamental zone
    quats /= torch.norm(quats, dim=1, keepdim=True)
    quats = ori_to_fz_laue(quats, laue_id=11)

    return quats


def compute_sigma3_deviation(quats1, quats2, sigma3_quats, laue_id=11):
    """
    Compute minimum deviation from Σ3 twin relationship for pairs of orientations.

    Steps:
    1. Compute disorientation quaternions between quats1 and quats2 using cubic symmetry
    2. Compare each disorientation quaternion against all 8 Σ3 twin quaternions
    3. Return the minimum angle deviation for each pair

    Args:
        quats1: (N, 4) quaternions for first set of orientations
        quats2: (N, 4) quaternions for second set of orientations
        sigma3_quats: (8, 4) Σ3 twin quaternions (60° rotations about <111>)
        laue_id: Laue group ID (default 11 for cubic Oh)

    Returns:
        (N,) minimum deviation angles in radians from any Σ3 twin relationship
    """
    N = quats1.shape[0]

    # Step 1: Compute disorientation quaternions between neighboring pairs
    # This gives us the symmetry-reduced misorientation
    disori_quats = disorientation(quats1, quats2, laue_id, laue_id)  # (N, 4)

    # Step 2: Reshape for broadcasting with Σ3 quaternions
    # disori_quats: (N, 1, 4) and sigma3: (1, 8, 4) -> automatic broadcast to (N, 8, 4)
    disori_expanded = disori_quats.unsqueeze(1)  # (N, 1, 4)
    sigma3_expanded = sigma3_quats.unsqueeze(0)  # (1, 8, 4)

    # Step 3: Compute angles between each disorientation and all 8 Σ3 twins
    # Broadcasting handles (N, 1, 4) and (1, 8, 4) -> (N, 8) automatically
    deviation_angles = disori_angle_laue(
        disori_expanded,
        sigma3_expanded,
        laue_id,
        laue_id,
    )  # (N, 8)

    # Step 4: Take minimum deviation across all 8 Σ3 possibilities
    return torch.min(deviation_angles, dim=1).values  # (N,)


def compute_sigma9_deviation(quats1, quats2, sigma9_quats, laue_id=11):
    """
    Compute minimum deviation from Σ9 twin relationship for pairs of orientations.

    Steps:
    1. Compute disorientation quaternions between quats1 and quats2 using cubic symmetry
    2. Compare each disorientation quaternion against all 12 Σ9 twin quaternions
    3. Return the minimum angle deviation for each pair

    Args:
        quats1: (N, 4) quaternions for first set of orientations
        quats2: (N, 4) quaternions for second set of orientations
        sigma9_quats: (12, 4) Σ9 twin quaternions (38.9° rotations about <110>)
        laue_id: Laue group ID (default 11 for cubic Oh)

    Returns:
        (N,) minimum deviation angles in radians from any Σ9 twin relationship
    """
    N = quats1.shape[0]

    # Step 1: Compute disorientation quaternions between neighboring pairs
    disori_quats = disorientation(quats1, quats2, laue_id, laue_id)  # (N, 4)

    # Step 2: Reshape for broadcasting with Σ9 quaternions
    # disori_quats: (N, 1, 4) and sigma9: (1, 12, 4) -> automatic broadcast to (N, 12, 4)
    disori_expanded = disori_quats.unsqueeze(1)  # (N, 1, 4)
    sigma9_expanded = sigma9_quats.unsqueeze(0)  # (1, 12, 4)

    # Step 3: Compute angles between each disorientation and all 12 Σ9 twins
    deviation_angles = disori_angle_laue(
        disori_expanded,
        sigma9_expanded,
        laue_id,
        laue_id,
    )  # (N, 12)

    # Step 4: Take minimum deviation across all 12 Σ9 possibilities
    return torch.min(deviation_angles, dim=1).values  # (N,)


def compute_sigma27a_deviation(quats1, quats2, sigma27a_quats, laue_id=11):
    """
    Compute minimum deviation from Σ27a twin relationship for pairs of orientations.

    Steps:
    1. Compute disorientation quaternions between quats1 and quats2 using cubic symmetry
    2. Compare each disorientation quaternion against all 12 Σ27a twin quaternions
    3. Return the minimum angle deviation for each pair

    Args:
        quats1: (N, 4) quaternions for first set of orientations
        quats2: (N, 4) quaternions for second set of orientations
        sigma27a_quats: (12, 4) Σ27a twin quaternions (31.59° rotations about <110>)
        laue_id: Laue group ID (default 11 for cubic Oh)

    Returns:
        (N,) minimum deviation angles in radians from any Σ27a twin relationship
    """
    N = quats1.shape[0]

    # Step 1: Compute disorientation quaternions between neighboring pairs
    disori_quats = disorientation(quats1, quats2, laue_id, laue_id)  # (N, 4)

    # Step 2: Reshape for broadcasting with Σ27a quaternions
    disori_expanded = disori_quats.unsqueeze(1)  # (N, 1, 4)
    sigma27a_expanded = sigma27a_quats.unsqueeze(0)  # (1, 12, 4)

    # Step 3: Compute angles between each disorientation and all 12 Σ27a twins
    deviation_angles = disori_angle_laue(
        disori_expanded,
        sigma27a_expanded,
        laue_id,
        laue_id,
    )  # (N, 12)

    # Step 4: Take minimum deviation across all 12 Σ27a possibilities
    return torch.min(deviation_angles, dim=1).values  # (N,)


def compute_sigma27b_deviation(quats1, quats2, sigma27b_quats, laue_id=11):
    """
    Compute minimum deviation from Σ27b twin relationship for pairs of orientations.

    Steps:
    1. Compute disorientation quaternions between quats1 and quats2 using cubic symmetry
    2. Compare each disorientation quaternion against all 24 Σ27b twin quaternions
    3. Return the minimum angle deviation for each pair

    Args:
        quats1: (N, 4) quaternions for first set of orientations
        quats2: (N, 4) quaternions for second set of orientations
        sigma27b_quats: (24, 4) Σ27b twin quaternions (35.43° rotations about <210>)
        laue_id: Laue group ID (default 11 for cubic Oh)

    Returns:
        (N,) minimum deviation angles in radians from any Σ27b twin relationship
    """
    N = quats1.shape[0]

    # Step 1: Compute disorientation quaternions between neighboring pairs
    disori_quats = disorientation(quats1, quats2, laue_id, laue_id)  # (N, 4)

    # Step 2: Reshape for broadcasting with Σ27b quaternions
    disori_expanded = disori_quats.unsqueeze(1)  # (N, 1, 4)
    sigma27b_expanded = sigma27b_quats.unsqueeze(0)  # (1, 24, 4)

    # Step 3: Compute angles between each disorientation and all 24 Σ27b twins
    deviation_angles = disori_angle_laue(
        disori_expanded,
        sigma27b_expanded,
        laue_id,
        laue_id,
    )  # (N, 24)

    # Step 4: Take minimum deviation across all 24 Σ27b possibilities
    return torch.min(deviation_angles, dim=1).values  # (N,)


def create_boundary_map(
    reference_orientations,
    grid_shape=(149, 200),
    device="cpu",
    check_diagonals=False,
):
    """
    Create a boundary map highlighting Σ3, Σ9, Σ27a, and Σ27b twin boundaries using vectorized operations.

    Uses disori_angle_laue to compute the minimum disorientation angle between
    neighboring orientations and the CSL twin relationships:
    - Σ3: 60° rotations about <111> (8 axes), Brandon: 15°/√3 ≈ 8.66°
    - Σ9: 38.9° rotations about <110> (12 axes), Brandon: 15°/√9 = 5°
    - Σ27a: 31.59° rotations about <110> (12 axes), Brandon: 15°/√27 ≈ 2.89°
    - Σ27b: 35.43° rotations about <210> (24 axes), Brandon: 15°/√27 ≈ 2.89°

    Args:
        reference_orientations: (N, 4) array of quaternions
        grid_shape: (H, W) shape of the scan
        device: torch device to use for computation
        check_diagonals: if True, also check diagonal neighbors (default: False)

    Returns:
        tuple of boundary map arrays
    """
    H, W = grid_shape

    print(f"Checking Σ3, Σ9, Σ27a, and Σ27b boundaries...")
    print(f"  Σ3 Brandon criterion: {15.0/np.sqrt(3.0):.2f}°")
    print(f"  Σ9 Brandon criterion: {15.0/np.sqrt(9.0):.2f}°")
    print(f"  Σ27 Brandon criterion: {15.0/np.sqrt(27.0):.2f}°")
    print(f"Processing {H}x{W} = {H*W} pixels...")

    # Convert to torch tensor
    ori_tensor = torch.from_numpy(reference_orientations).to(device)
    ori_grid = ori_tensor.reshape(H, W, 4)

    # Get CSL quaternions
    sigma3_quats = get_sigma3_quaternions_torch().to(device)  # (8, 4)
    sigma9_quats = get_sigma9_quaternions_torch().to(device)  # (12, 4)
    sigma27a_quats = get_sigma27a_quaternions_torch().to(device)  # (12, 4)
    sigma27b_quats = get_sigma27b_quaternions_torch().to(device)  # (24, 4)

    # Brandon thresholds in radians
    brandon_rad_sigma3 = (15.0 / np.sqrt(3.0)) * np.pi / 180.0
    brandon_rad_sigma9 = (15.0 / np.sqrt(9.0)) * np.pi / 180.0
    brandon_rad_sigma27 = (15.0 / np.sqrt(27.0)) * np.pi / 180.0

    # Initialize boundary maps (separate for each CSL type and general boundaries)
    boundary_map_sigma3 = np.zeros((H, W), dtype=np.uint8)
    boundary_map_sigma9 = np.zeros((H, W), dtype=np.uint8)
    boundary_map_sigma27a = np.zeros((H, W), dtype=np.uint8)
    boundary_map_sigma27b = np.zeros((H, W), dtype=np.uint8)
    boundary_map_general = np.zeros((H, W), dtype=np.uint8)

    # General boundary threshold: 3 degrees in radians
    general_boundary_threshold = 3.0 * np.pi / 180.0

    # Check horizontal boundaries (i, j) <-> (i, j+1)
    print("  Checking horizontal boundaries...", end=" ")
    quats_left = ori_grid[:, :-1, :].reshape(-1, 4)  # (H*(W-1), 4)
    quats_right = ori_grid[:, 1:, :].reshape(-1, 4)  # (H*(W-1), 4)

    # Compute disorientation angles (reuse for all checks)
    disori_angles_horiz = disori_angle_laue(quats_left, quats_right, 11, 11)

    # Compute minimum deviation from Σ3, Σ9, Σ27a, and Σ27b twins
    min_dev_sigma3 = compute_sigma3_deviation(quats_left, quats_right, sigma3_quats)
    min_dev_sigma9 = compute_sigma9_deviation(quats_left, quats_right, sigma9_quats)
    min_dev_sigma27a = compute_sigma27a_deviation(
        quats_left, quats_right, sigma27a_quats
    )
    min_dev_sigma27b = compute_sigma27b_deviation(
        quats_left, quats_right, sigma27b_quats
    )

    # Check Brandon criteria and general boundary threshold
    is_sigma3_horiz = (
        (min_dev_sigma3 < brandon_rad_sigma3).cpu().numpy().reshape(H, W - 1)
    )
    is_sigma9_horiz = (
        (min_dev_sigma9 < brandon_rad_sigma9).cpu().numpy().reshape(H, W - 1)
    )
    is_sigma27a_horiz = (
        (min_dev_sigma27a < brandon_rad_sigma27).cpu().numpy().reshape(H, W - 1)
    )
    is_sigma27b_horiz = (
        (min_dev_sigma27b < brandon_rad_sigma27).cpu().numpy().reshape(H, W - 1)
    )
    is_general_horiz = (
        (disori_angles_horiz > general_boundary_threshold)
        .cpu()
        .numpy()
        .reshape(H, W - 1)
    )

    # Vectorized update: mark both pixels on each side of horizontal boundaries
    boundary_map_sigma3[:, :-1] |= is_sigma3_horiz
    boundary_map_sigma3[:, 1:] |= is_sigma3_horiz
    boundary_map_sigma9[:, :-1] |= is_sigma9_horiz
    boundary_map_sigma9[:, 1:] |= is_sigma9_horiz
    boundary_map_sigma27a[:, :-1] |= is_sigma27a_horiz
    boundary_map_sigma27a[:, 1:] |= is_sigma27a_horiz
    boundary_map_sigma27b[:, :-1] |= is_sigma27b_horiz
    boundary_map_sigma27b[:, 1:] |= is_sigma27b_horiz
    boundary_map_general[:, :-1] |= is_general_horiz
    boundary_map_general[:, 1:] |= is_general_horiz
    print("✓")

    # Check vertical boundaries (i, j) <-> (i+1, j)
    print("  Checking vertical boundaries...", end=" ")
    quats_top = ori_grid[:-1, :, :].reshape(-1, 4)  # ((H-1)*W, 4)
    quats_bottom = ori_grid[1:, :, :].reshape(-1, 4)  # ((H-1)*W, 4)

    # Compute disorientation angles (reuse for all checks)
    disori_angles_vert = disori_angle_laue(quats_top, quats_bottom, 11, 11)

    # Compute minimum deviation from Σ3, Σ9, Σ27a, and Σ27b twins
    min_dev_sigma3 = compute_sigma3_deviation(quats_top, quats_bottom, sigma3_quats)
    min_dev_sigma9 = compute_sigma9_deviation(quats_top, quats_bottom, sigma9_quats)
    min_dev_sigma27a = compute_sigma27a_deviation(
        quats_top, quats_bottom, sigma27a_quats
    )
    min_dev_sigma27b = compute_sigma27b_deviation(
        quats_top, quats_bottom, sigma27b_quats
    )

    # Check Brandon criteria and general boundary threshold
    is_sigma3_vert = (
        (min_dev_sigma3 < brandon_rad_sigma3).cpu().numpy().reshape(H - 1, W)
    )
    is_sigma9_vert = (
        (min_dev_sigma9 < brandon_rad_sigma9).cpu().numpy().reshape(H - 1, W)
    )
    is_sigma27a_vert = (
        (min_dev_sigma27a < brandon_rad_sigma27).cpu().numpy().reshape(H - 1, W)
    )
    is_sigma27b_vert = (
        (min_dev_sigma27b < brandon_rad_sigma27).cpu().numpy().reshape(H - 1, W)
    )
    is_general_vert = (
        (disori_angles_vert > general_boundary_threshold)
        .cpu()
        .numpy()
        .reshape(H - 1, W)
    )

    # Vectorized update: mark both pixels on each side of vertical boundaries
    boundary_map_sigma3[:-1, :] |= is_sigma3_vert
    boundary_map_sigma3[1:, :] |= is_sigma3_vert
    boundary_map_sigma9[:-1, :] |= is_sigma9_vert
    boundary_map_sigma9[1:, :] |= is_sigma9_vert
    boundary_map_sigma27a[:-1, :] |= is_sigma27a_vert
    boundary_map_sigma27a[1:, :] |= is_sigma27a_vert
    boundary_map_sigma27b[:-1, :] |= is_sigma27b_vert
    boundary_map_sigma27b[1:, :] |= is_sigma27b_vert
    boundary_map_general[:-1, :] |= is_general_vert
    boundary_map_general[1:, :] |= is_general_vert
    print("✓")

    if check_diagonals:
        # Check diagonal boundaries (top-left to bottom-right): (i, j) <-> (i+1, j+1)
        print("  Checking diagonal (↘) boundaries...", end=" ")
        quats_topleft = ori_grid[:-1, :-1, :].reshape(-1, 4)  # ((H-1)*(W-1), 4)
        quats_bottomright = ori_grid[1:, 1:, :].reshape(-1, 4)  # ((H-1)*(W-1), 4)

        # Compute disorientation angles (reuse for all checks)
        disori_angles_diag1 = disori_angle_laue(
            quats_topleft, quats_bottomright, 11, 11
        )

        # Compute minimum deviation from Σ3, Σ9, Σ27a, and Σ27b twins
        min_dev_sigma3 = compute_sigma3_deviation(
            quats_topleft, quats_bottomright, sigma3_quats
        )
        min_dev_sigma9 = compute_sigma9_deviation(
            quats_topleft, quats_bottomright, sigma9_quats
        )
        min_dev_sigma27a = compute_sigma27a_deviation(
            quats_topleft, quats_bottomright, sigma27a_quats
        )
        min_dev_sigma27b = compute_sigma27b_deviation(
            quats_topleft, quats_bottomright, sigma27b_quats
        )

        # Check Brandon criteria and general boundary threshold
        is_sigma3_diag1 = (
            (min_dev_sigma3 < brandon_rad_sigma3).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_sigma9_diag1 = (
            (min_dev_sigma9 < brandon_rad_sigma9).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_sigma27a_diag1 = (
            (min_dev_sigma27a < brandon_rad_sigma27).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_sigma27b_diag1 = (
            (min_dev_sigma27b < brandon_rad_sigma27).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_general_diag1 = (
            (disori_angles_diag1 > general_boundary_threshold)
            .cpu()
            .numpy()
            .reshape(H - 1, W - 1)
        )

        # Vectorized update: mark both pixels on each side of diagonal boundaries
        boundary_map_sigma3[:-1, :-1] |= is_sigma3_diag1
        boundary_map_sigma3[1:, 1:] |= is_sigma3_diag1
        boundary_map_sigma9[:-1, :-1] |= is_sigma9_diag1
        boundary_map_sigma9[1:, 1:] |= is_sigma9_diag1
        boundary_map_sigma27a[:-1, :-1] |= is_sigma27a_diag1
        boundary_map_sigma27a[1:, 1:] |= is_sigma27a_diag1
        boundary_map_sigma27b[:-1, :-1] |= is_sigma27b_diag1
        boundary_map_sigma27b[1:, 1:] |= is_sigma27b_diag1
        boundary_map_general[:-1, :-1] |= is_general_diag1
        boundary_map_general[1:, 1:] |= is_general_diag1
        print("✓")

        # Check diagonal boundaries (top-right to bottom-left): (i, j+1) <-> (i+1, j)
        print("  Checking diagonal (↙) boundaries...", end=" ")
        quats_topright = ori_grid[:-1, 1:, :].reshape(-1, 4)  # ((H-1)*(W-1), 4)
        quats_bottomleft = ori_grid[1:, :-1, :].reshape(-1, 4)  # ((H-1)*(W-1), 4)

        # Compute disorientation angles (reuse for all checks)
        disori_angles_diag2 = disori_angle_laue(
            quats_topright, quats_bottomleft, 11, 11
        )

        # Compute minimum deviation from Σ3, Σ9, Σ27a, and Σ27b twins
        min_dev_sigma3 = compute_sigma3_deviation(
            quats_topright, quats_bottomleft, sigma3_quats
        )
        min_dev_sigma9 = compute_sigma9_deviation(
            quats_topright, quats_bottomleft, sigma9_quats
        )
        min_dev_sigma27a = compute_sigma27a_deviation(
            quats_topright, quats_bottomleft, sigma27a_quats
        )
        min_dev_sigma27b = compute_sigma27b_deviation(
            quats_topright, quats_bottomleft, sigma27b_quats
        )

        # Check Brandon criteria and general boundary threshold
        is_sigma3_diag2 = (
            (min_dev_sigma3 < brandon_rad_sigma3).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_sigma9_diag2 = (
            (min_dev_sigma9 < brandon_rad_sigma9).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_sigma27a_diag2 = (
            (min_dev_sigma27a < brandon_rad_sigma27).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_sigma27b_diag2 = (
            (min_dev_sigma27b < brandon_rad_sigma27).cpu().numpy().reshape(H - 1, W - 1)
        )
        is_general_diag2 = (
            (disori_angles_diag2 > general_boundary_threshold)
            .cpu()
            .numpy()
            .reshape(H - 1, W - 1)
        )

        # Vectorized update: mark both pixels on each side of diagonal boundaries
        boundary_map_sigma3[:-1, 1:] |= is_sigma3_diag2
        boundary_map_sigma3[1:, :-1] |= is_sigma3_diag2
        boundary_map_sigma9[:-1, 1:] |= is_sigma9_diag2
        boundary_map_sigma9[1:, :-1] |= is_sigma9_diag2
        boundary_map_sigma27a[:-1, 1:] |= is_sigma27a_diag2
        boundary_map_sigma27a[1:, :-1] |= is_sigma27a_diag2
        boundary_map_sigma27b[:-1, 1:] |= is_sigma27b_diag2
        boundary_map_sigma27b[1:, :-1] |= is_sigma27b_diag2
        boundary_map_general[:-1, 1:] |= is_general_diag2
        boundary_map_general[1:, :-1] |= is_general_diag2
        print("✓")

    # Combine into single map: encoding CSL boundary types
    # We'll create a combined Σ27 map (either 27a or 27b)
    boundary_map_sigma27 = (boundary_map_sigma27a > 0) | (boundary_map_sigma27b > 0)

    boundary_map = np.zeros((H, W), dtype=np.uint8)
    boundary_map[boundary_map_sigma3 > 0] = 1
    boundary_map[boundary_map_sigma9 > 0] = 2
    boundary_map[boundary_map_sigma27 > 0] = 4  # Σ27 (either a or b)
    # Note: combinations will result in additive values

    n_sigma3 = np.sum(boundary_map_sigma3)
    n_sigma9 = np.sum(boundary_map_sigma9)
    n_sigma27a = np.sum(boundary_map_sigma27a)
    n_sigma27b = np.sum(boundary_map_sigma27b)
    n_sigma27_combined = np.sum(boundary_map_sigma27)

    print(f"  Found {n_sigma3} Σ3 pixels ({100*n_sigma3/(H*W):.2f}%)")
    print(f"  Found {n_sigma9} Σ9 pixels ({100*n_sigma9/(H*W):.2f}%)")
    print(f"  Found {n_sigma27a} Σ27a pixels ({100*n_sigma27a/(H*W):.2f}%)")
    print(f"  Found {n_sigma27b} Σ27b pixels ({100*n_sigma27b/(H*W):.2f}%)")
    print(
        f"  Found {n_sigma27_combined} Σ27 (a or b) pixels ({100*n_sigma27_combined/(H*W):.2f}%)"
    )

    # Analyze fraction of general boundaries (>3°) that are CSL boundaries
    print(f"\nGeneral boundary analysis (>3° disorientation):")
    n_general = np.sum(boundary_map_general)
    print(f"  Total general boundary pixels: {n_general} ({100*n_general/(H*W):.2f}%)")

    if n_general > 0:
        # Find which general boundary pixels are also CSL boundaries
        general_and_sigma3 = np.sum(
            (boundary_map_general > 0) & (boundary_map_sigma3 > 0)
        )
        general_and_sigma9 = np.sum(
            (boundary_map_general > 0) & (boundary_map_sigma9 > 0)
        )
        general_and_sigma27 = np.sum(
            (boundary_map_general > 0) & (boundary_map_sigma27 > 0)
        )

        frac_sigma3 = 100 * general_and_sigma3 / n_general
        frac_sigma9 = 100 * general_and_sigma9 / n_general
        frac_sigma27 = 100 * general_and_sigma27 / n_general

        print(f"  Fraction that are Σ3: {frac_sigma3:.2f}%")
        print(f"  Fraction that are Σ9: {frac_sigma9:.2f}%")
        print(f"  Fraction that are Σ27: {frac_sigma27:.2f}%")

    return (
        boundary_map,
        boundary_map_general,
        boundary_map_sigma3,
        boundary_map_sigma9,
        boundary_map_sigma27a,
        boundary_map_sigma27b,
    )


def visualize_boundary_map(
    boundary_map,
    boundary_map_general,
    boundary_map_sigma27a,
    boundary_map_sigma27b,
    output_dir,
):
    """
    Visualize and save the boundary map in publication-ready format.

    Args:
        boundary_map: (H, W) array with CSL boundary encoding
        boundary_map_general: (H, W) array with general boundaries (>3° disorientation)
        boundary_map_sigma27a: (H, W) array with Σ27a boundaries
        boundary_map_sigma27b: (H, W) array with Σ27b boundaries
        output_dir: Directory to save figures
    """
    H, W = boundary_map.shape

    # Create RGB image: white background
    rgb_map = np.ones((H, W, 3), dtype=np.float32)

    # Black for general boundaries (>3°) - applied first so CSL colors can overwrite
    rgb_map[boundary_map_general > 0] = [0.0, 0.0, 0.0]

    # Red for Σ3 boundaries (1, 0, 0)
    rgb_map[boundary_map == 1] = [1.0, 0.0, 0.0]
    rgb_map[(boundary_map & 1) > 0] = [1.0, 0.0, 0.0]

    # Blue for Σ9 boundaries (0, 0, 1)
    rgb_map[boundary_map == 2] = [0.0, 0.0, 1.0]
    rgb_map[(boundary_map & 2) > 0] = [0.0, 0.0, 1.0]

    # Dark yellow for Σ27 boundaries (combined Σ27a or Σ27b) - RGB: (0.7, 0.7, 0.0)
    # Apply before other CSL types so they can overwrite if overlapping
    boundary_map_sigma27_combined = (boundary_map_sigma27a > 0) | (
        boundary_map_sigma27b > 0
    )
    rgb_map[boundary_map_sigma27_combined] = [0.7, 0.7, 0.0]

    # # Purple for both Σ3 and Σ9 [(1, 0, 1)]
    # rgb_map[((boundary_map & 1) > 0) and (boundary_map & 2) > 0] = [1.0, 0.0, 1.0]

    # Create figure with white background
    fig, ax = plt.subplots(figsize=(10, 7.5), facecolor="white")
    ax.set_facecolor("white")

    # Display image with thin black border
    im = ax.imshow(rgb_map, interpolation="nearest")

    # Clean title
    ax.set_title("Σ3, Σ9, and Σ27 Boundary Map", fontsize=16, pad=10)
    ax.axis("off")

    # Add thin black border around image
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("black")
        spine.set_linewidth(0.5)

    plt.tight_layout()

    # Save figure
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "figure_sigma_boundary_map.png"
    plt.savefig(output_file, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"✓ Boundary map saved to: {output_file}")

    output_pdf = output_dir / "figure_sigma_boundary_map.pdf"
    plt.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    print(f"✓ Boundary map PDF saved to: {output_pdf}")

    plt.close(fig)


if __name__ == "__main__":
    data_file = Path("benchmark_results/benchmark_dictionary.npy")

    if not data_file.exists():
        print(f"Error: {data_file} not found!")
        print("Please run benchmark.py first to generate the data.")
    else:
        print("\n" + "=" * 70)
        print("CSL BOUNDARY MAP GENERATION")
        print("=" * 70)

        # Load benchmark data
        print("\nLoading reference orientations...")
        data = np.load(data_file, allow_pickle=True).item()
        reference_orientations = data["reference_orientations"]
        print(f"✓ Loaded {len(reference_orientations)} orientations")

        # Create boundary map
        print("\nGenerating boundary map...")
        (
            boundary_map,
            boundary_map_general,
            boundary_map_sigma3,
            boundary_map_sigma9,
            boundary_map_sigma27a,
            boundary_map_sigma27b,
        ) = create_boundary_map(
            reference_orientations=reference_orientations,
            grid_shape=(149, 200),
        )

        # Visualize and save
        print("\nSaving boundary map...")
        output_dir = Path("benchmark_results")
        visualize_boundary_map(
            boundary_map,
            boundary_map_general,
            boundary_map_sigma27a,
            boundary_map_sigma27b,
            output_dir,
        )

        # Analyze Scan 5 indexing errors at boundaries
        print("\n" + "=" * 70)
        print("SCAN 5 INDEXING ERROR ANALYSIS")
        print("=" * 70)

        # Filter for Scan 5, DI method, FP32 dtype (same as figure_scan5_misindex_pats.py)
        mask = (
            (np.array(data["dataset_id"]) == 5)
            & (np.array(data["method"]) == "DI")
            & (np.array(data["dtype"]) == "FP32")
        )

        if np.any(mask):
            idx = np.where(mask)[0][0]
            disorientations_scan5 = data["raw_disorientations"][idx]

            print(f"Loaded {len(disorientations_scan5)} disorientations for Scan 5")

            # Reshape to grid
            H, W = 149, 200
            disori_grid = disorientations_scan5.reshape(H, W)

            # Find pixels with indexing errors > 3°
            error_threshold = 3.0
            indexing_errors = disori_grid > error_threshold
            n_errors = np.sum(indexing_errors)

            print(
                f"\nPixels with indexing error > {error_threshold}°: {n_errors} ({100*n_errors/(H*W):.2f}%)"
            )

            if n_errors > 0:
                # Analyze where these errors occurred
                errors_at_sigma3 = np.sum(indexing_errors & (boundary_map_sigma3 > 0))
                errors_at_sigma9 = np.sum(indexing_errors & (boundary_map_sigma9 > 0))
                errors_at_sigma27a = np.sum(
                    indexing_errors & (boundary_map_sigma27a > 0)
                )
                errors_at_sigma27b = np.sum(
                    indexing_errors & (boundary_map_sigma27b > 0)
                )
                errors_at_general = np.sum(indexing_errors & (boundary_map_general > 0))

                # Errors at CSL boundaries
                boundary_map_sigma27_combined = (boundary_map_sigma27a > 0) | (
                    boundary_map_sigma27b > 0
                )
                errors_at_csl = np.sum(
                    indexing_errors
                    & (
                        (boundary_map_sigma3 > 0)
                        | (boundary_map_sigma9 > 0)
                        | (boundary_map_sigma27_combined > 0)
                    )
                )

                # Errors at non-CSL general boundaries (general but not CSL)
                errors_at_noncsl_general = np.sum(
                    indexing_errors
                    & (boundary_map_general > 0)
                    & (boundary_map_sigma3 == 0)
                    & (boundary_map_sigma9 == 0)
                    & (boundary_map_sigma27_combined == 0)
                )

                # Errors not at any boundary
                errors_not_at_boundary = np.sum(
                    indexing_errors & (boundary_map_general == 0)
                )

                frac_at_sigma3 = 100 * errors_at_sigma3 / n_errors
                frac_at_sigma9 = 100 * errors_at_sigma9 / n_errors
                frac_at_sigma27a = 100 * errors_at_sigma27a / n_errors
                frac_at_sigma27b = 100 * errors_at_sigma27b / n_errors
                frac_at_csl = 100 * errors_at_csl / n_errors
                frac_at_noncsl_general = 100 * errors_at_noncsl_general / n_errors
                frac_not_at_boundary = 100 * errors_not_at_boundary / n_errors

                print(f"\nWhere indexing errors occurred:")
                print(f"  At Σ3 boundaries: {errors_at_sigma3} ({frac_at_sigma3:.2f}%)")
                print(f"  At Σ9 boundaries: {errors_at_sigma9} ({frac_at_sigma9:.2f}%)")
                print(
                    f"  At Σ27a boundaries: {errors_at_sigma27a} ({frac_at_sigma27a:.2f}%)"
                )
                print(
                    f"  At Σ27b boundaries: {errors_at_sigma27b} ({frac_at_sigma27b:.2f}%)"
                )
                print(
                    f"  At CSL boundaries (Σ3, Σ9, or Σ27): {errors_at_csl} ({frac_at_csl:.2f}%)"
                )
                print(
                    f"  At general boundaries (non-CSL): {errors_at_noncsl_general} ({frac_at_noncsl_general:.2f}%)"
                )
                print(
                    f"  Not at boundaries: {errors_not_at_boundary} ({frac_not_at_boundary:.2f}%)"
                )

                # Sanity check
                total_accounted = (
                    errors_at_csl + errors_at_noncsl_general + errors_not_at_boundary
                )
                print(
                    f"\n  Total accounted: {total_accounted}/{n_errors} (sanity check)"
                )
        else:
            print("Warning: No Scan 5 data found for analysis")

        print("\n" + "=" * 70)
        print("BOUNDARY MAP GENERATION COMPLETE!")
        print("=" * 70)
