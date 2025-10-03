import os
import csv
import time
import platform
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import kikuchipy as kp
from tqdm import tqdm
from torch.fft import irfftn

from utils import (
    ExperimentPatterns,
    MasterPattern,
    EBSDGeometry,
    dictionary_index_orientations,
    get_radial_mask,
    disori_angle_laue,
    compute_pca_components_covmat,
    orientation_grid_refinement,
    sample_ori_fz_laue_angle,
    qu_apply,
)
from utils_knn import ChunkedKNN


# Constants
OUTDIR = Path("benchmark_results")
TIMING_CSV = OUTDIR / "timing_breakdown.csv"
REFERENCE_RESOLUTION = 1.0
DICT_CHUNK = 4096 * 3
EXP_CHUNK = 4096 * 3


def get_device():
    """Get the best available device (CUDA, MPS, XPU, or CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ Using CUDA device: {gpu_name} ({total_mem:.1f} GB)")
        return device
    if torch.backends.mps.is_available():
        print("✓ Using MPS device (Apple Silicon)")
        return torch.device("mps")
    if torch.backends.xpu.is_available():
        print("✓ Using XPU device (Intel)")
        return torch.device("xpu")
    print("⚠ Using CPU device (no GPU acceleration)")
    return torch.device("cpu")


def get_sync_function(device):
    """Return appropriate synchronization function for device."""
    sync_map = {
        "cuda": torch.cuda.synchronize,
        "mps": torch.mps.synchronize,
        "xpu": torch.xpu.synchronize,
    }
    return sync_map.get(device.type, lambda: None)


def setup_output_directory():
    """Create output directory and initialize timing CSV."""
    OUTDIR.mkdir(exist_ok=True)
    print(f"✓ Output directory: {OUTDIR.absolute()}")

    if not TIMING_CSV.exists():
        with open(TIMING_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "os",
                    "python",
                    "torch",
                    "device",
                    "method",
                    "pattern_size",
                    "dict_resolution",
                    "dict_size",
                    "pca_components",
                    "dtype",
                    "n_dict_patterns",
                    "n_exp_patterns",
                    "dict_sim_pps",
                    "pca_pps",
                    "dict_proj_pps",
                    "knn_pps",
                    "notes",
                ]
            )
        print(f"✓ Created timing CSV: {TIMING_CSV.name}")
    else:
        print(f"✓ Appending to existing timing CSV: {TIMING_CSV.name}")


def write_timing_row(**kwargs):
    """Append a timing result row to the CSV file."""
    with open(TIMING_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                platform.platform(),
                platform.python_version(),
                torch.__version__,
                kwargs.get("device", "cpu"),
                kwargs.get("method", ""),
                kwargs.get("pattern_size", ""),
                kwargs.get("dict_resolution", ""),
                kwargs.get("dict_size", 0),
                kwargs.get("pca_components", ""),
                kwargs.get("dtype", ""),
                kwargs.get("n_dict_patterns", 0),
                kwargs.get("n_exp_patterns", 0),
                f"{kwargs.get('dict_sim_pps', 0):.2f}",
                f"{kwargs.get('pca_pps', 0):.2f}",
                f"{kwargs.get('dict_proj_pps', 0):.2f}",
                f"{kwargs.get('knn_pps', 0):.2f}",
                kwargs.get("notes", ""),
            ]
        )


def load_exp_patterns(pattern_id, device, size=(60, 60)):
    """Load and preprocess experimental patterns."""
    print(f"  Loading experimental patterns (ID={pattern_id}, size={size})...", end=" ")
    exp_pats_all = kp.data.ni_gain(allow_download=True, number=pattern_id)
    exp_pats_data = torch.tensor(exp_pats_all.data, dtype=torch.float16, device=device)

    if exp_pats_data.shape[-2:] != size:
        exp_pats_data = F.interpolate(
            exp_pats_data.view(-1, 1, *exp_pats_data.shape[-2:]),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

    exp_pats = ExperimentPatterns(exp_pats_data)
    exp_pats.standard_clean()
    print(f"✓ Loaded {exp_pats.n_patterns} patterns")
    return exp_pats


def load_master_pattern(device):
    """Load and preprocess master pattern."""
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
    print(f"✓ Master pattern loaded (shape: {master_pattern.shape})")
    return mp


def simulate_dictionary(mp, geom, ori_tensor, mask, device, sync):
    """Simulate dictionary patterns for given orientations."""
    dict_size = ori_tensor.shape[0]
    n_masked_pixels = int((mask == 1).sum().item())

    print(
        f"  Simulating dictionary: {dict_size} orientations, {n_masked_pixels} pixels..."
    )

    sim_patterns = torch.empty(
        (dict_size, n_masked_pixels), dtype=torch.float16, device=torch.device("cpu")
    )

    detector_coords = geom.get_coords_sample_frame(binning=(1, 1))
    detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

    n_chunks = (dict_size + DICT_CHUNK - 1) // DICT_CHUNK
    for chunk_idx, start_idx in enumerate(range(0, dict_size, DICT_CHUNK), 1):
        ori_batch = ori_tensor[start_idx : start_idx + DICT_CHUNK]
        batch_rot = qu_apply(ori_batch[:, None, :], detector_coords[None, ...])

        sim_batch = (
            mp.interpolate(
                batch_rot,
                mode="bilinear",
                align_corners=False,
                normalize_coords=False,
                virtual_binning=1,
            )
            .squeeze()
            .view(len(ori_batch), -1)
        )

        sim_batch = sim_batch[:, mask]
        sim_batch -= torch.mean(sim_batch, dim=-1, keepdim=True)
        sim_patterns[start_idx : start_idx + len(ori_batch)] = sim_batch.to(
            torch.device("cpu")
        )

        if chunk_idx % 10 == 0 or chunk_idx == n_chunks:
            print(
                f"    Progress: {chunk_idx}/{n_chunks} chunks ({100*chunk_idx/n_chunks:.1f}%)"
            )

    sync()
    print(f"  ✓ Dictionary simulation complete")
    return sim_patterns


def time_dictionary_simulation(mp, geom, ori_tensor, mask, device, sync, n_trials=10):
    """Time dictionary simulation with multiple trials for robust measurement."""
    print(f"  Timing dictionary simulation ({n_trials} trials)...", end=" ")
    n_dict_patterns = ori_tensor.shape[0]

    detector_coords = geom.get_coords_sample_frame(binning=(1, 1))
    detector_coords = detector_coords / detector_coords.norm(dim=-1, keepdim=True)

    # Warmup
    for start_idx in range(0, n_dict_patterns, DICT_CHUNK):
        ori_batch = ori_tensor[start_idx : start_idx + DICT_CHUNK]
        batch_rot = qu_apply(ori_batch[:, None, :], detector_coords[None, ...])
        sim_batch = (
            mp.interpolate(
                batch_rot,
                mode="bilinear",
                align_corners=False,
                normalize_coords=False,
                virtual_binning=1,
            )
            .squeeze()
            .view(len(ori_batch), -1)
        )
        sim_batch = sim_batch[:, mask]
        sim_batch -= torch.mean(sim_batch, dim=-1, keepdim=True)
    sync()

    # Timed trials
    trial_times = []
    for trial in range(n_trials):
        t_start = time.time()
        for start_idx in range(0, n_dict_patterns, DICT_CHUNK):
            ori_batch = ori_tensor[start_idx : start_idx + DICT_CHUNK]
            batch_rot = qu_apply(ori_batch[:, None, :], detector_coords[None, ...])
            sim_batch = (
                mp.interpolate(
                    batch_rot,
                    mode="bilinear",
                    align_corners=False,
                    normalize_coords=False,
                    virtual_binning=1,
                )
                .squeeze()
                .view(len(ori_batch), -1)
            )
            sim_batch = sim_batch[:, mask]
            sim_batch -= torch.mean(sim_batch, dim=-1, keepdim=True)
        sync()
        trial_times.append(time.time() - t_start)

    avg_time = np.mean(trial_times)
    dict_sim_pps = n_dict_patterns / avg_time
    print(f"✓ {dict_sim_pps:.1f} patterns/s (avg over {n_trials} trials)")
    return dict_sim_pps


def time_pca_computation(
    mp, geom, mask, n_components, resolution, device, sync, n_trials=10
):
    """Time PCA computation with multiple trials for robust measurement."""
    print(f"  Timing PCA computation ({n_trials} trials)...", end=" ")

    # Warmup
    pca_warmup = compute_pca_components_covmat(
        master_patterns=mp,
        geometry=geom,
        n_pca_components=n_components,
        signal_mask=mask,
        dictionary_resolution_learn_deg=resolution,
        dictionary_chunk_size=DICT_CHUNK,
        virtual_binning=1,
    )
    sync()

    # Estimate number of patterns used in PCA
    ori_tensor_pca = sample_ori_fz_laue_angle(
        laue_id=mp.laue_group, angular_resolution_deg=resolution, device=device
    )
    n_pca_patterns = ori_tensor_pca.shape[0]

    # Timed trials
    trial_times = []
    for trial in range(n_trials):
        t_start = time.time()
        pca_trial = compute_pca_components_covmat(
            master_patterns=mp,
            geometry=geom,
            n_pca_components=n_components,
            signal_mask=mask,
            dictionary_resolution_learn_deg=resolution,
            dictionary_chunk_size=DICT_CHUNK,
            virtual_binning=1,
        )
        sync()
        trial_times.append(time.time() - t_start)

    avg_time = np.mean(trial_times)
    pca_pps = n_pca_patterns / avg_time
    print(f"✓ {pca_pps:.1f} patterns/s (avg over {n_trials} trials)")
    return pca_pps


def time_dictionary_projection(
    sim_patterns, pca_matrix, n_components, device, sync, n_trials=10
):
    """Time dictionary projection with multiple trials for robust measurement."""
    print(
        f"    Timing dict projection for {n_components} components ({n_trials} trials)...",
        end=" ",
    )
    n_dict_patterns = sim_patterns.shape[0]
    pca_subset = pca_matrix[:, -n_components:]

    # Warmup
    for sim_batch in torch.split(sim_patterns, DICT_CHUNK):
        proj = torch.matmul(sim_batch.to(device).to(pca_subset.dtype), pca_subset)
        sync()

    # Timed trials
    trial_times = []
    for trial in range(n_trials):
        t_start = time.time()
        for sim_batch in torch.split(sim_patterns, DICT_CHUNK):
            proj = torch.matmul(sim_batch.to(device).to(pca_subset.dtype), pca_subset)
            sync()
        trial_times.append(time.time() - t_start)

    avg_time = np.mean(trial_times)
    dict_proj_pps = n_dict_patterns / avg_time
    print(f"✓ {dict_proj_pps:.1f} patterns/s")
    return dict_proj_pps


def project_dictionary(sim_patterns, pca_matrix, device, sync):
    """Project dictionary patterns onto PCA basis."""
    projected = []
    n_chunks = len(list(torch.split(sim_patterns, DICT_CHUNK)))

    for chunk_idx, sim_batch in enumerate(torch.split(sim_patterns, DICT_CHUNK), 1):
        projected.append(
            torch.matmul(sim_batch.to(device).to(pca_matrix.dtype), pca_matrix)
        )
        sync()
        if chunk_idx % 10 == 0 or chunk_idx == n_chunks:
            print(
                f"      Projecting dictionary: {chunk_idx}/{n_chunks} chunks", end="\r"
            )

    if n_chunks > 1:
        print()  # New line after progress

    return torch.cat(projected, dim=0)


def run_knn_indexing(data_patterns, query_patterns, dtype_str, dtype, device, sync):
    """Run KNN indexing with specified dtype."""
    knn = ChunkedKNN(
        query_size=query_patterns.shape[0],
        topk=1,
        distance_metric="dotprod",
        match_device=torch.device("cpu") if dtype_str == "INT8" else device,
        match_dtype=dtype,
        quantized=(dtype_str == "INT8"),
        transpose_data=(dtype_str != "INT8"),
    )

    for data_batch in torch.split(data_patterns, DICT_CHUNK):
        knn.set_data_chunk(data_batch.to(device))
        for query_start in range(0, query_patterns.shape[0], EXP_CHUNK):
            knn.query(
                query_patterns[query_start : query_start + EXP_CHUNK],
                start_idx=query_start,
            )
        sync()
    sync()

    return knn.retrieve_topk()


def compute_accuracy_metrics(indexed_oris, ref_oris, laue_group):
    """Calculate disorientation metrics."""
    disori = disori_angle_laue(
        indexed_oris, ref_oris, laue_id_1=laue_group, laue_id_2=laue_group
    ) * (180.0 / np.pi)

    return {
        "mean_disorientation": torch.mean(disori).item(),
        "fraction_above_3deg": (disori > 3.0).float().mean().item(),
        "raw_disorientations": disori.cpu().numpy(),
    }


def benchmark_di_method(
    sim_patterns,
    exp_flat,
    ori_tensor,
    ref_oris,
    laue_group,
    dtype_str,
    dtype,
    device,
    sync,
    dict_sim_pps,
    pca_pps,
    n_dict_patterns,
):
    """Benchmark Dictionary Indexing (DI) method."""
    print(f"    Running DI with {dtype_str}...", end=" ")

    t_knn_start = time.time()
    indices, _ = run_knn_indexing(
        sim_patterns, exp_flat, dtype_str, dtype, device, sync
    )
    t_knn = time.time() - t_knn_start
    knn_pps = exp_flat.shape[0] / t_knn

    indexed_oris = ori_tensor[indices].squeeze()
    metrics = compute_accuracy_metrics(indexed_oris, ref_oris, laue_group)

    print(
        f"✓ {knn_pps:.1f} patterns/s, {metrics['mean_disorientation']:.2f}° mean error"
    )

    return {
        "indexed_orientations": indexed_oris.cpu().numpy(),
        "timings": {
            "dict_sim_pps": dict_sim_pps,
            "pca_pps": pca_pps,
            "dict_proj_pps": 0.0,
            "knn_pps": knn_pps,
        },
        "n_dict_patterns": n_dict_patterns,
        **metrics,
    }


def benchmark_pca_method(
    sim_patterns,
    exp_flat,
    ori_tensor,
    ref_oris,
    laue_group,
    pca_matrix,
    n_components,
    dtype_str,
    dtype,
    device,
    sync,
    dict_sim_pps,
    pca_pps,
    dict_proj_pps,
    n_dict_patterns,
):
    """Benchmark PCA-based Dictionary Indexing (PCA-DI) method."""
    print(f"    Running PCA-DI with {n_components} components, {dtype_str}...", end=" ")

    # Project dictionary
    proj_dict = project_dictionary(
        sim_patterns, pca_matrix[:, -n_components:], device, sync
    )

    # Project experimental patterns
    proj_exp = torch.matmul(
        exp_flat.to(pca_matrix.dtype), pca_matrix[:, -n_components:]
    )
    sync()

    # Run KNN
    t_knn_start = time.time()
    indices, _ = run_knn_indexing(proj_dict, proj_exp, dtype_str, dtype, device, sync)
    t_knn = time.time() - t_knn_start
    knn_pps = exp_flat.shape[0] / t_knn

    indexed_oris = ori_tensor[indices].squeeze()
    metrics = compute_accuracy_metrics(indexed_oris, ref_oris, laue_group)

    print(
        f"✓ {knn_pps:.1f} patterns/s, {metrics['mean_disorientation']:.2f}° mean error"
    )

    return {
        "indexed_orientations": indexed_oris.cpu().numpy(),
        "timings": {
            "dict_sim_pps": dict_sim_pps,
            "pca_pps": pca_pps,
            "dict_proj_pps": dict_proj_pps,
            "knn_pps": knn_pps,
        },
        "n_dict_patterns": n_dict_patterns,
        **metrics,
    }


def compute_reference_orientations(mp, geom, mask, device):
    """Compute ground-truth reference orientations."""
    print("  Computing reference orientations...")
    exp_ref = load_exp_patterns(1, device, size=geom.detector_shape)

    print("    Running coarse dictionary indexing...", end=" ")
    dictionary_index_orientations(
        mp,
        geom,
        exp_ref,
        dictionary_resolution_degrees=REFERENCE_RESOLUTION,
        dictionary_chunk_size=DICT_CHUNK,
        signal_mask=mask,
        virtual_binning=1,
        experiment_chunk_size=EXP_CHUNK,
        match_dtype=torch.float32,
        match_device=device,
    )
    print("✓")

    print("    Running orientation grid refinement...", end=" ")
    orientation_grid_refinement(
        master_patterns=mp,
        geometry=geom,
        experiment_patterns=exp_ref,
        batch_size=256,
        virtual_binning=1,
        n_iter=7,
        grid_semi_edge_in_degrees=1.0,
        kernel_radius_in_steps=1,
        axial_grid_dimension=3,
        average_pattern_center=True,
        match_dtype=torch.float32,
    )
    print("✓")

    ref_oris = exp_ref.get_orientations().squeeze()
    print(f"  ✓ Reference orientations computed ({len(ref_oris)} orientations)")
    return ref_oris


def benchmark_spherical_fft(
    L, device, sync, n_iterations=100, batch_size=8, z_sym=4, warmup_iterations=50
):
    """Benchmark inverse r3DFFT for spherical harmonics indexing."""
    print(f"    Benchmarking Spherical FFT with L={L}...", end=" ")
    m_dim_size = (2 * L - 1) // z_sym  # Using 4-fold symmetry for Nickel
    mock_coeffs = torch.rand(
        batch_size,
        m_dim_size,
        2 * L - 1,
        L + 1,
        device=device,
        dtype=torch.complex64,
    )

    # Extended warmup for stable timings
    for _ in range(warmup_iterations):
        result = irfftn(mock_coeffs, dim=(-3, -2, -1), norm="forward")
        sync()

    start_time = time.time()
    for _ in range(n_iterations):
        result = irfftn(mock_coeffs, dim=(-3, -2, -1), norm="forward")
        sync()
    end_time = time.time()

    cross_corr_rate = (n_iterations * batch_size) / (end_time - start_time)
    print(f"✓ {cross_corr_rate:.1f} patterns/s (theoretical)")

    return cross_corr_rate


def run_benchmarks(config):
    """Main benchmark execution."""
    start_time = time.time()
    print("\n" + "=" * 70)
    print("EBSD DICTIONARY INDEXING BENCHMARK")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    device = get_device()
    sync = get_sync_function(device)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("✓ TF32 enabled for matrix operations")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✓ GPU memory cleared")

    setup_output_directory()

    print("\n" + "-" * 70)
    print("CONFIGURATION")
    print("-" * 70)
    print(f"Pattern sizes: {config['pattern_sizes']}")
    print(f"Dictionary resolutions: {config['resolutions']}°")
    print(f"PCA components: {config['pca_components']}")
    print(f"Data types: {config['dtypes']}")
    print(f"Noise levels (dataset IDs): {config['noise_ids']}")
    if config.get("spherical_L_values"):
        print(f"Spherical L values: {config['spherical_L_values']}")
    print("-" * 70)

    mp = load_master_pattern(device)
    results = None

    total_patterns = len(config["pattern_sizes"])
    for pattern_idx, pattern_size in enumerate(config["pattern_sizes"], 1):
        print(f"\n{'='*70}")
        print(
            f"PATTERN SIZE {pattern_idx}/{total_patterns}: {pattern_size[0]}x{pattern_size[1]}"
        )
        print(f"{'='*70}")

        geom = EBSDGeometry(
            detector_shape=pattern_size, proj_center=(0.4221, 0.2179, 0.4954)
        ).to(device)
        mask = get_radial_mask(pattern_size).to(device).flatten()
        print(f"✓ Geometry initialized ({int((mask == 1).sum().item())} active pixels)")

        # Compute reference orientations
        ref_oris = compute_reference_orientations(mp, geom, mask, device)

        # Compute PCA basis
        print(f"  Computing PCA basis (10000 components)...", end=" ")
        pca_matrix = compute_pca_components_covmat(
            master_patterns=mp,
            geometry=geom,
            n_pca_components=10000,
            signal_mask=mask,
            dictionary_resolution_learn_deg=REFERENCE_RESOLUTION,
            dictionary_chunk_size=DICT_CHUNK,
            virtual_binning=1,
        ).to(torch.float32)
        print(f"✓ PCA basis ready (shape: {pca_matrix.shape})")

        # Initialize results dictionary
        if results is None:
            results = {
                "dataset_id": [],
                "method": [],
                "pattern_size": [],
                "dict_resolution": [],
                "dict_size": [],
                "pca_components": [],
                "dtype": [],
                "dict_sim_pps": [],
                "pca_pps": [],
                "dict_proj_pps": [],
                "knn_pps": [],
                "mean_disorientation": [],
                "fraction_above_3deg": [],
                "raw_disorientations": [],
                "indexed_orientations": [],
                "reference_orientations": ref_oris.cpu().numpy(),
                "L_value": [],
                "spherical_fft_pps": [],
            }

        # Benchmark each resolution
        total_resolutions = len(config["resolutions"])
        for res_idx, resolution in enumerate(config["resolutions"], 1):
            print(f"\n{'-'*70}")
            print(f"RESOLUTION {res_idx}/{total_resolutions}: {resolution}°")
            print(f"{'-'*70}")

            # Generate orientations and simulate dictionary
            print(f"  Generating orientation grid...", end=" ")
            ori_tensor = sample_ori_fz_laue_angle(
                laue_id=mp.laue_group, angular_resolution_deg=resolution, device=device
            )
            n_dict_patterns = ori_tensor.shape[0]
            print(f"✓ {n_dict_patterns} orientations generated")

            sim_patterns = simulate_dictionary(mp, geom, ori_tensor, mask, device, sync)

            # Time dictionary simulation separately
            dict_sim_pps = time_dictionary_simulation(
                mp, geom, ori_tensor, mask, device, sync, n_trials=10
            )

            # Compute PCA for this resolution
            print(f"  Computing PCA basis for this resolution...", end=" ")
            pca_res = compute_pca_components_covmat(
                master_patterns=mp,
                geometry=geom,
                n_pca_components=pca_matrix.shape[1],
                signal_mask=mask,
                dictionary_resolution_learn_deg=resolution,
                dictionary_chunk_size=DICT_CHUNK,
                virtual_binning=1,
            ).to(torch.float32)
            print(f"✓")

            # Time PCA computation separately
            pca_pps = time_pca_computation(
                mp,
                geom,
                mask,
                pca_matrix.shape[1],
                resolution,
                device,
                sync,
                n_trials=10,
            )

            # Time dictionary projection for each PCA component count
            print(f"  Timing dictionary projections for PCA components:")
            dict_proj_timings = {}
            for n_components in config["pca_components"]:
                dict_proj_pps = time_dictionary_projection(
                    sim_patterns, pca_res, n_components, device, sync, n_trials=10
                )
                dict_proj_timings[n_components] = dict_proj_pps

            # Load experimental patterns
            print(f"  Loading experimental pattern sets...")
            exp_sets = {}
            for idx, noise_id in enumerate(config["noise_ids"], 1):
                exp_sets[noise_id] = load_exp_patterns(
                    noise_id, device, size=pattern_size
                )

            # Benchmark all combinations
            for noise_id, exp_pats in exp_sets.items():
                print(f"\n  Noise Level: ID={noise_id}")
                exp_flat = exp_pats.patterns.view(
                    -1, pattern_size[0] * pattern_size[1]
                )[:, mask]
                exp_flat -= torch.mean(exp_flat, dim=-1, keepdim=True)
                n_exp_patterns = exp_flat.shape[0]
                print(f"    Prepared {n_exp_patterns} experimental patterns")

                # Benchmark DI methods
                print(f"    Dictionary Indexing (DI):")
                for dtype_str in config["dtypes"]:
                    dtype = config["dtype_mapping"][dtype_str]

                    di_result = benchmark_di_method(
                        sim_patterns,
                        exp_flat,
                        ori_tensor,
                        ref_oris,
                        mp.laue_group,
                        dtype_str,
                        dtype,
                        device,
                        sync,
                        dict_sim_pps,
                        pca_pps,
                        n_dict_patterns,
                    )

                    store_results(
                        results,
                        noise_id,
                        "DI",
                        pattern_size,
                        resolution,
                        0,
                        dtype_str,
                        n_dict_patterns,
                        n_exp_patterns,
                        di_result,
                        device,
                    )

                # Benchmark PCA-DI methods
                print(f"    PCA-based Dictionary Indexing (PCA-DI):")
                for dtype_str in config["dtypes"]:
                    dtype = config["dtype_mapping"][dtype_str]

                    for n_components in config["pca_components"]:
                        pca_result = benchmark_pca_method(
                            sim_patterns,
                            exp_flat,
                            ori_tensor,
                            ref_oris,
                            mp.laue_group,
                            pca_res,
                            n_components,
                            dtype_str,
                            dtype,
                            device,
                            sync,
                            dict_sim_pps,
                            pca_pps,
                            dict_proj_timings[n_components],
                            n_dict_patterns,
                        )

                        store_results(
                            results,
                            noise_id,
                            "PCA",
                            pattern_size,
                            resolution,
                            n_components,
                            dtype_str,
                            n_dict_patterns,
                            n_exp_patterns,
                            pca_result,
                            device,
                        )

            # Benchmark Spherical FFT (theoretical timings)
            if config.get("spherical_L_values"):
                print(f"\n  Theoretical Spherical Harmonics Indexing:")
                for L in config["spherical_L_values"]:
                    spherical_pps = benchmark_spherical_fft(
                        L,
                        device,
                        sync,
                        n_iterations=100,
                        batch_size=8,
                        warmup_iterations=50,
                    )

                    # Store spherical results for each noise level
                    for noise_id in config["noise_ids"]:
                        results["dataset_id"].append(noise_id)
                        results["method"].append("Spherical")
                        results["pattern_size"].append(
                            f"{pattern_size[0]}x{pattern_size[1]}"
                        )
                        results["dict_resolution"].append(resolution)
                        results["dict_size"].append(n_dict_patterns)
                        results["pca_components"].append(0)
                        results["dtype"].append("FP32")
                        results["dict_sim_pps"].append(0.0)
                        results["pca_pps"].append(0.0)
                        results["dict_proj_pps"].append(0.0)
                        results["knn_pps"].append(0.0)
                        results["mean_disorientation"].append(0.0)
                        results["fraction_above_3deg"].append(0.0)
                        results["raw_disorientations"].append(np.array([]))
                        results["indexed_orientations"].append(np.array([]))
                        results["L_value"].append(L)
                        results["spherical_fft_pps"].append(spherical_pps)

    # Save results
    print(f"\n{'='*70}")
    print("SAVING RESULTS")
    print(f"{'='*70}")

    output_file = OUTDIR / "benchmark_dictionary.npy"
    np.save(
        output_file,
        {
            k: np.array(
                v,
                dtype=(
                    object
                    if k in ["raw_disorientations", "indexed_orientations"]
                    else None
                ),
            )
            for k, v in results.items()
        },
    )

    total_time = time.time() - start_time
    total_results = len(results["dataset_id"])

    print(f"✓ Timing breakdown saved to: {TIMING_CSV}")
    print(f"✓ Benchmark metrics saved to: {output_file}")
    print(f"✓ Total results collected: {total_results}")
    print(
        f"✓ Total execution time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)"
    )
    print(f"\n{'='*70}")
    print("BENCHMARK COMPLETE!")
    print(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")


def store_results(
    results_dict,
    noise_id,
    method,
    pattern_size,
    resolution,
    n_components,
    dtype,
    n_dict_patterns,
    n_exp_patterns,
    benchmark_result,
    device,
):
    """Store benchmark results and write timing row."""
    results_dict["dataset_id"].append(noise_id)
    results_dict["method"].append(method)
    results_dict["pattern_size"].append(f"{pattern_size[0]}x{pattern_size[1]}")
    results_dict["dict_resolution"].append(resolution)
    results_dict["dict_size"].append(n_dict_patterns)
    results_dict["pca_components"].append(n_components)
    results_dict["dtype"].append(dtype)
    results_dict["dict_sim_pps"].append(benchmark_result["timings"]["dict_sim_pps"])
    results_dict["pca_pps"].append(benchmark_result["timings"]["pca_pps"])
    results_dict["dict_proj_pps"].append(benchmark_result["timings"]["dict_proj_pps"])
    results_dict["knn_pps"].append(benchmark_result["timings"]["knn_pps"])
    results_dict["mean_disorientation"].append(benchmark_result["mean_disorientation"])
    results_dict["fraction_above_3deg"].append(benchmark_result["fraction_above_3deg"])
    results_dict["raw_disorientations"].append(benchmark_result["raw_disorientations"])
    results_dict["indexed_orientations"].append(
        benchmark_result["indexed_orientations"]
    )
    results_dict["L_value"].append(0)
    results_dict["spherical_fft_pps"].append(0.0)

    write_timing_row(
        device=str(device),
        method=method if method == "DI" else "PCA-DI",
        pattern_size=f"{pattern_size[0]}x{pattern_size[1]}",
        dict_resolution=resolution,
        dict_size=n_dict_patterns,
        pca_components=n_components,
        dtype=dtype,
        n_dict_patterns=n_dict_patterns,
        n_exp_patterns=n_exp_patterns,
        notes=f"noise_id={noise_id}",
        **benchmark_result["timings"],
    )


if __name__ == "__main__":
    config = {
        "noise_ids": [1, 5, 10],
        "resolutions": [1.4, 2.0, 3.0],
        # "resolutions": [3.0, 4.0, 5.0], # for quickly testing plotting scripts
        "pca_components": [512, 1024],
        "dtypes": ["FP32", "FP16", "INT8"],
        "pattern_sizes": [(60, 60)],
        "dtype_mapping": {
            "FP32": torch.float32,
            "FP16": torch.float16,
            "INT8": torch.float32,
        },
        "spherical_L_values": [63, 88, 123, 158],
    }

    run_benchmarks(config)
