# refinement_basin_3deg.py
# Figure to justify the ~3° refinement basin using REGULAR DI (no PCA).
# Outputs: figure_3deg_basin.png, figure_3deg_basin.pdf
#
# Depends on utils.py and kikuchipy.

import numpy as np
import torch
import matplotlib.pyplot as plt
import kikuchipy as kp
from pathlib import Path

from utils import (
    ExperimentPatterns,
    MasterPattern,
    EBSDGeometry,
    get_radial_mask,
    disori_angle_laue,
    dictionary_index_orientations,
    orientation_grid_refinement,
)

# -----------------------------
# Config (mirrors your bench)
# -----------------------------
DET_SHAPE = (60, 60)
PROJ_CENTER = (0.4221, 0.2179, 0.4954)
LAUE_ID = 11

REF_INDEX_RES = 2.0  # deg; for Scan 1 reference
DICT_RES_START = 2.0  # deg; initial DI for Scans 5 & 10
REFINE_ITERS = 7
GRID_SEMI_EDGE_DEG = 2.0

MAX_START = 6.0  # focus on basin
BIN_WIDTH = 0.1  # deg

OUT_FIG = Path("figure_3deg_basin")


# -----------------------------
# Helpers
# -----------------------------
def dev():
    d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Using: {d} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})"
    )
    return d


def load_mp_geom(device):
    geom = EBSDGeometry(detector_shape=DET_SHAPE, proj_center=PROJ_CENTER).to(device)
    mp_ = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    nh = torch.from_numpy(mp_.data[0].astype(np.float32)).to(device)
    sh = torch.from_numpy(mp_.data[1].astype(np.float32)).to(device)
    master = torch.concat((nh, sh), dim=-1)
    mp = MasterPattern(master, laue_group=LAUE_ID).to(device)
    mp.normalize(norm_type="minmax")
    mp.apply_clahe()
    return mp, geom


def load_exp(scan_id):
    arr = kp.data.ni_gain(allow_download=True, number=scan_id).data
    t = torch.tensor(arr).to(torch.float16)
    pats = ExperimentPatterns(t)
    pats.standard_clean()
    return pats


def build_reference(mp, geom, mask, device):
    exp1 = load_exp(1)
    dictionary_index_orientations(
        mp,
        geom,
        exp1,
        dictionary_resolution_degrees=REF_INDEX_RES,
        dictionary_chunk_size=8192,
        signal_mask=mask,
        virtual_binning=1,
        experiment_chunk_size=16384,
        match_dtype=torch.float16,
        match_device=device,
    )
    orientation_grid_refinement(
        master_patterns=mp,
        geometry=geom,
        experiment_patterns=exp1,
        batch_size=256,
        virtual_binning=1,
        n_iter=REFINE_ITERS,
        grid_semi_edge_in_degrees=GRID_SEMI_EDGE_DEG,
        kernel_radius_in_steps=1,
        axial_grid_dimension=1,
        average_pattern_center=True,
        match_dtype=torch.float16,
    )
    return exp1.get_orientations()


def start_finish(mp, geom, scan_id, mask, device, start_res_deg):
    exp = load_exp(scan_id)
    dictionary_index_orientations(
        mp,
        geom,
        exp,
        dictionary_resolution_degrees=start_res_deg,
        dictionary_chunk_size=8192,
        signal_mask=mask,
        virtual_binning=1,
        experiment_chunk_size=29800,
        match_dtype=torch.float16,
        match_device=device,
    )
    ori_start = exp.get_orientations().clone()
    orientation_grid_refinement(
        master_patterns=mp,
        geometry=geom,
        experiment_patterns=exp,
        batch_size=256,
        virtual_binning=1,
        n_iter=REFINE_ITERS,
        grid_semi_edge_in_degrees=GRID_SEMI_EDGE_DEG,
        kernel_radius_in_steps=1,
        axial_grid_dimension=1,
        average_pattern_center=True,
        match_dtype=torch.float16,
    )
    ori_finish = exp.get_orientations().clone()
    return ori_start, ori_finish


def disori_deg(ref, a, b, laue):
    start = (disori_angle_laue(ref, a, laue, laue) * 180.0 / np.pi).cpu().numpy()
    finish = (disori_angle_laue(ref, b, laue, laue) * 180.0 / np.pi).cpu().numpy()
    return start, finish


def binned_stats(start_deg, finish_deg, max_x=MAX_START, bin_width=BIN_WIDTH):
    start = np.asarray(start_deg)
    finish = np.asarray(finish_deg)
    keep = (start >= 0) & (start <= max_x)
    start = start[keep]
    finish = finish[keep]
    improve = start - finish  # positive = moved toward 0°
    bins = np.arange(0, max_x + bin_width, bin_width)
    centers = 0.5 * (bins[:-1] + bins[1:])

    def agg(vec):
        return (np.median(vec), np.percentile(vec, 25), np.percentile(vec, 75))

    mF = np.full_like(centers, np.nan, dtype=float)
    lF = np.full_like(centers, np.nan, dtype=float)
    uF = np.full_like(centers, np.nan, dtype=float)
    mI = np.full_like(centers, np.nan, dtype=float)
    lI = np.full_like(centers, np.nan, dtype=float)
    uI = np.full_like(centers, np.nan, dtype=float)
    for i in range(len(centers)):
        idx = (start >= bins[i]) & (start < bins[i + 1])
        if np.any(idx):
            mF[i], lF[i], uF[i] = agg(finish[idx])
            mI[i], lI[i], uI[i] = agg(improve[idx])
    return centers, (mF, lF, uF), (mI, lI, uI)


# -----------------------------
# Main
# -----------------------------
def main():
    device = dev()
    mp, geom = load_mp_geom(device)
    mask = get_radial_mask(DET_SHAPE).to(device)

    print("Building Scan 1 reference…")
    ref = build_reference(mp, geom, mask, device)

    print("Scan 5: DI, refine…")
    s5_start_o, s5_finish_o = start_finish(mp, geom, 5, mask, device, DICT_RES_START)
    s5_start, s5_finish = disori_deg(ref, s5_start_o, s5_finish_o, LAUE_ID)

    print("Scan 10: DI, refine…")
    s10_start_o, s10_finish_o = start_finish(mp, geom, 10, mask, device, DICT_RES_START)
    s10_start, s10_finish = disori_deg(ref, s10_start_o, s10_finish_o, LAUE_ID)

    # Binned curves (median + IQR)
    x5, (mF5, lF5, uF5), (mI5, lI5, uI5) = binned_stats(s5_start, s5_finish)
    x10, (mF10, lF10, uF10), (mI10, lI10, uI10) = binned_stats(s10_start, s10_finish)

    # Figure: 2 panels (finishing vs starting; improvement vs starting)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: finishing vs starting; y = x baseline; vertical line at 3°
    ax1.plot(
        [0, MAX_START], [0, MAX_START], lw=1.2, color="0.3", label="y = x (no change)"
    )
    ax1.fill_between(x5, lF5, uF5, alpha=0.18, label="Scan 5 IQR")
    ax1.plot(x5, mF5, lw=2.0, label="Scan 5 median")
    ax1.fill_between(x10, lF10, uF10, alpha=0.18, label="Scan 10 IQR")
    ax1.plot(x10, mF10, lw=2.0, label="Scan 10 median")
    ax1.axvline(3.0, ls="--", lw=1.2, color="k")
    ax1.set_xlim(0, MAX_START)
    ax1.set_ylim(0, MAX_START)
    ax1.set_xlabel("Starting disorientation (deg)")
    ax1.set_ylabel("Finishing disorientation (deg)")
    ax1.set_title("Finishing vs. starting (lower = closer to 0°)")
    ax1.legend(loc="upper left", frameon=True)

    # Right: improvement vs starting (start − finish); vertical line at 3°
    ax2.axhline(0.0, ls="--", lw=1.2, color="k")
    ax2.fill_between(x5, lI5, uI5, alpha=0.18, label="Scan 5 IQR")
    ax2.plot(x5, mI5, lw=2.0, label="Scan 5 median")
    ax2.fill_between(x10, lI10, uI10, alpha=0.18, label="Scan 10 IQR")
    ax2.plot(x10, mI10, lw=2.0, label="Scan 10 median")
    ax2.axvline(3.0, ls="--", lw=1.2, color="k")
    ax2.set_xlim(0, MAX_START)
    ax2.set_xlabel("Starting disorientation (deg)")
    ax2.set_ylabel("Improvement (start − finish, deg)")
    ax2.set_title("Refinement moves **toward 0°** when start < ~3°")
    ax2.legend(loc="upper left", frameon=True)

    fig.suptitle("Refinement basin around ~3° (regular DI; reference = Scan 1 refined)")
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=300)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
