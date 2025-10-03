#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_pcadi_assets_compact_v7.py

Changes from v6 per feedback:
  • Use SAME component counts for dict & experimental loadings -> LOAD_C_LIST = [...]
  • Rename loadings files to use C{components} (avoid K confusion; K reserved for k-NN=3)
  • Grayscale everywhere ('gray' cmap) for loadings, dot-product matrix, bars
  • Halve white padding between projection rows (LOAD_VPAD = 4)
  • On the upscaled dot-product matrix, outline top-3 dictionary matches per experimental

Outputs (PNG only):
  stack_raw_border_noise{ID}_N{n}.png
  stack_dict_border_N{n}.png
  stack_masked_noise{ID}_N{n}.png         (RGBA; alpha outside; smooth thin ring)
  stack_masked_dict_N{n}.png              (RGBA; alpha outside; smooth thin ring)
  components_masked_N{n}.png              (RGBA; alpha outside; smooth thin ring)

  loadings_dict_C{C}.png
  loadings_noise{ID}_C{C}.png

  dotprod_matrix_noise{ID}.png            (grayscale; upscaled)
  dotprod_colorbar.png                    (grayscale bar)
  topk3_noise{ID}.png                     (grayscale; truncated/sorted per experimental)

Preproc:
  • Master pattern on S²: minmax + CLAHE (once) → simulate dictionary
  • Experimental: ep.standard_clean()  # static + CLAHE
  • Zero-mean per masked pattern BEFORE projection
"""

from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import kikuchipy as kp

from utils import (
    ExperimentPatterns,
    MasterPattern,
    EBSDGeometry,
    get_radial_mask,
    sample_ori_fz_laue_angle,
    compute_pca_components_covmat,
    qu_apply,
)

# ----------------------- Config -----------------------
OUTDIR = Path("figure_main_summary")
ASSETDIR = OUTDIR / "assets"
ASSETDIR.mkdir(parents=True, exist_ok=True)

DET_SHAPE = (60, 60)
PROJ_CENTER = (0.4221, 0.2179, 0.4954)
LAUE_NI = 11

EXP_DATASET_IDS = [1, 5, 10]
EXP_STACK_NS = [4]
DICT_STACK_NS = [10]
PC_STACK_NS = [10]

PCA_K = 64  # learned PCs
DICT_RES_DEG = 3.0
TOPK = 3  # k-NN top-k

# Upscaling & borders
UPSCALE = 2
RING_THICK = 2 * UPSCALE
RECT_BORDER = 1
STACK_OFFSET_PX = 10 * UPSCALE
DPI = 300

# Loadings (same options for dict & experimental)
LOAD_C_LIST = [8, 16]  # number of components to display in loadings strips
LOAD_CELL = 14  # px per coeff tile
LOAD_BORDER = 1  # black border thickness per tile
LOAD_VPAD = 4  # white gap between rows (halved from 8)

# Dot-product rendering
DOT_CELL = 32  # px per cell in matrix/TopK images

# Colormaps (grayscale)
CMAP_GRAY = "gray"

SEED = 42

plt.rcParams.update(
    {
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
    }
)


# ----------------------- Helpers -----------------------
def _ensure_nhw(x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    if x.ndim == 4:
        a, b, h, w = x.shape
        return x.view(a * b, h, w)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected (N,H,W) or (A,B,H,W), got {tuple(x.shape)}")


def _to_np_hw(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.ndim != 2:
        raise ValueError(f"Expected (H,W), got {x.shape}")
    return x


def _minmax01(a):
    a = a.astype(np.float32, copy=False)
    mn, mx = float(a.min()), float(a.max())
    return (a - mn) / (mx - mn) if mx > mn else np.zeros_like(a, dtype=np.float32)


def _upsample2x(img01):
    t = torch.from_numpy(img01[None, None, :, :]).float()
    u = F.interpolate(t, scale_factor=UPSCALE, mode="bilinear", align_corners=False)
    return u[0, 0].numpy()


def _stack_gray_bordered(imgs01, offset, border=RECT_BORDER):
    H, W = imgs01[0].shape
    n = len(imgs01)
    pad = 6
    th = H + (n - 1) * offset + 2 * pad + 2 * border
    tw = W + (n - 1) * offset + 2 * pad + 2 * border
    canvas = np.ones((th, tw), dtype=np.float32)  # white
    for i in range(n - 1, -1, -1):
        y = pad + i * offset
        x = pad + i * offset
        by, bx = y - border, x - border
        canvas[by : by + H + 2 * border, bx : bx + W + 2 * border] = 0.0  # black border
        canvas[y : y + H, x : x + W] = imgs01[i]
    return canvas


def _alpha_composite(dst_rgba, src_rgba, y, x):
    h, w = src_rgba.shape[:2]
    dst = dst_rgba[y : y + h, x : x + w, :]
    a_s = src_rgba[..., 3:4]
    a_d = dst[..., 3:4]
    a_out = a_s + a_d * (1.0 - a_s)
    safe = (a_out > 1e-8).astype(np.float32)
    rgb_out = src_rgba[..., :3] * a_s + dst[..., :3] * a_d * (1.0 - a_s)
    rgb_out = np.where(safe, rgb_out / np.clip(a_out, 1e-8, 1.0), dst[..., :3])
    dst_rgba[y : y + h, x : x + w, :3] = rgb_out
    dst_rgba[y : y + h, x : x + w, 3:4] = a_out


def _stack_rgba_masked_upscaled(imgs01, mask_lores_bool, ring_thick, offset):
    H0, W0 = imgs01[0].shape
    mask_hi = get_radial_mask((H0 * UPSCALE, W0 * UPSCALE)).cpu().numpy().astype(bool)

    H, W = mask_hi.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    r_est = np.sqrt(float(mask_hi.sum()) / np.pi)
    d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ring_mask = (d >= r_est - ring_thick) & (d <= r_est + 1.0)

    n = len(imgs01)
    pad = 6
    th = H + (n - 1) * offset + 2 * pad
    tw = W + (n - 1) * offset + 2 * pad
    canvas = np.zeros((th, tw, 4), dtype=np.float32)

    for i in range(n - 1, -1, -1):
        y = pad + i * offset
        x = pad + i * offset
        up = _upsample2x(imgs01[i])
        im = up.copy()
        im[~mask_hi] = im[mask_hi].mean()
        rgb = np.stack([_minmax01(im)] * 3, axis=-1)
        a = mask_hi.astype(np.float32)
        rgb[ring_mask] = 0.0
        a[ring_mask] = 1.0
        tile = np.concatenate([rgb, a[..., None]], axis=-1)
        _alpha_composite(canvas, tile, y, x)

    return canvas  # RGBA


def _pca_strip_gray(scores, C_show, cell=LOAD_CELL, bw=LOAD_BORDER, vpad=LOAD_VPAD):
    """
    Build PCA loadings strip (grayscale):
      • Each coefficient -> grayscale square with thin black border
      • Left→right in component order (no sorting)
      • White rows between patterns
    """
    if cell <= 2 * bw:
        raise ValueError(f"`cell` ({cell}) must be > 2*bw ({2*bw})")
    C_show = int(min(C_show, scores.shape[1]))
    S = np.asarray(scores[:, :C_show], dtype=np.float32)
    S -= S.mean(axis=1, keepdims=True)
    vmax = np.maximum(np.abs(S).max(axis=1, keepdims=True), 1e-8)
    S = S / vmax  # in [-1,1]

    # map [-1,1] -> [0,1] for grayscale
    S01 = (S + 1.0) * 0.5

    rows = []
    row_w = C_show * cell
    for i in range(S01.shape[0]):
        row = np.ones((cell, row_w), dtype=np.float32)  # start white
        for j in range(C_show):
            val = float(S01[i, j])
            inner = np.full((cell - 2 * bw, cell - 2 * bw), val, dtype=np.float32)
            tile = np.zeros((cell, cell), dtype=np.float32)  # black border
            tile[bw:-bw, bw:-bw] = inner
            x0 = j * cell
            row[:, x0 : x0 + cell] = tile
        rows.append(row)
        if i < S01.shape[0] - 1:
            rows.append(np.ones((vpad, row_w), dtype=np.float32))  # white gap

    strip = np.vstack(rows)
    return np.repeat(strip[..., None], 3, axis=-1)  # RGB


def _draw_rect_outline(img_rgb, top, left, h, w, color=(0, 0, 0), thick=2):
    """Draw rectangle outline (in-place) on an RGB image with given thickness."""
    H, W, _ = img_rgb.shape
    t0, l0 = int(top), int(left)
    t1, l1 = int(top + h - 1), int(left + w - 1)
    t0 = max(0, min(H - 1, t0))
    t1 = max(0, min(H - 1, t1))
    l0 = max(0, min(W - 1, l0))
    l1 = max(0, min(W - 1, l1))
    for k in range(thick):
        r0, r1 = max(0, t0 - k), min(H - 1, t0 + k)
        rb0, rb1 = max(0, t1 - k), min(H - 1, t1 + k)
        img_rgb[r0 : r1 + 1, l0 : l1 + 1, :] = color
        img_rgb[rb0 : rb1 + 1, l0 : l1 + 1, :] = color
        c0, c1 = max(0, l0 - k), min(W - 1, l0 + k)
        cb0, cb1 = max(0, l1 - k), min(W - 1, l1 + k)
        img_rgb[t0 : t1 + 1, c0 : c1 + 1, :] = color
        img_rgb[t0 : t1 + 1, cb0 : cb1 + 1, :] = color


def _save_gray(path, img01):
    plt.imsave(path, np.clip(img01, 0, 1), cmap=CMAP_GRAY, dpi=DPI)


def _save_rgb(path, img_rgb01):
    plt.imsave(path, np.clip(img_rgb01, 0, 1), dpi=DPI)


def _save_rgba(path, img_rgba01):
    plt.imsave(path, np.clip(img_rgba01, 0, 1), dpi=DPI)


# ----------------------- Core steps -----------------------
def load_mp_geom(device):
    geom = EBSDGeometry(detector_shape=DET_SHAPE, proj_center=PROJ_CENTER).to(device)
    ds = kp.data.nickel_ebsd_master_pattern_small(
        projection="lambert", hemisphere="both"
    )
    mp_nh = torch.from_numpy(ds.data[0].astype(np.float32)).to(device)
    mp_sh = torch.from_numpy(ds.data[1].astype(np.float32)).to(device)
    master_pattern = torch.concat((mp_nh, mp_sh), dim=-1)
    mp = MasterPattern(master_pattern, laue_group=LAUE_NI).to(device)
    mp.normalize(norm_type="minmax")
    mp.apply_clahe()
    return mp, geom


def simulate_dict(mp, geom, n, res_deg, device):
    torch.manual_seed(SEED)
    ori = sample_ori_fz_laue_angle(LAUE_NI, res_deg, device=device, permute=True)[:n]
    rays = geom.get_coords_sample_frame(binning=(1, 1)).view(-1, 3)
    rays = (rays / rays.norm(dim=-1, keepdim=True))[None, :, :]
    sims = mp.interpolate(
        qu_apply(ori[:, None, :], rays),
        mode="bilinear",
        align_corners=False,
        normalize_coords=True,
        virtual_binning=1,
    ).view(n, *DET_SHAPE)
    return sims


def learn_pca(mp, geom, mask_bool, k):
    try:
        P = compute_pca_components_covmat(
            mp,
            geom,
            n_pca_components=k,
            signal_mask=mask_bool.to(mp.master_pattern.device),
            dictionary_resolution_learn_deg=1.5,
            dictionary_chunk_size=8192,
            virtual_binning=1,
        )
    except TypeError:
        P = compute_pca_components_covmat(
            mp,
            geom,
            k,
            mask_bool.to(mp.master_pattern.device),
            dictionary_resolution_learn_deg=1.5,
            dictionary_chunk_size=8192,
            virtual_binning=1,
        )
    return P.detach().cpu().numpy() if isinstance(P, torch.Tensor) else P


def project_scores(pats, mask_bool, Pmk):
    if isinstance(pats, np.ndarray):
        pats = torch.from_numpy(pats)
    X = pats.detach().cpu().to(torch.float32)

    m = mask_bool
    if isinstance(m, torch.Tensor):
        m = m.detach().cpu().numpy()
    m = m.astype(bool)
    if m.ndim == 2:
        m = m.reshape(-1)
    HW = m.size

    shp = tuple(X.shape)
    if shp[-1] == HW:
        X2 = X.reshape(-1, HW)
    elif len(shp) >= 2 and shp[-2] * shp[-1] == HW:
        X2 = X.reshape(-1, shp[-2] * shp[-1])
    else:
        raise ValueError(
            f"project_scores: cannot reconcile shapes. patterns={shp}, mask HW={HW}"
        )

    Xm = X2.numpy()[:, m]  # masked pixels
    Xm -= Xm.mean(axis=1, keepdims=True)  # zero-mean per masked pattern
    return Xm @ Pmk


def load_exp_set(exp_id, nshow):
    ds = kp.data.ni_gain(allow_download=True, number=exp_id)
    raw = _ensure_nhw(torch.from_numpy(ds.data.astype(np.float32)))
    ep = ExperimentPatterns(raw.clone())
    ep.standard_clean()  # static + CLAHE
    proc = _ensure_nhw(ep.patterns)
    return raw[:nshow], proc[:nshow]


# ----------------------- Main -----------------------
def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mp, geom = load_mp_geom(device)
    mask = get_radial_mask(DET_SHAPE).bool().cpu().numpy()

    # Dictionary + PCA
    N_DICT_TOTAL = max(DICT_STACK_NS)
    dict_all = simulate_dict(mp, geom, N_DICT_TOTAL, DICT_RES_DEG, device)
    Pmk = learn_pca(mp, geom, torch.from_numpy(mask), PCA_K)
    dict_scores = project_scores(dict_all, torch.from_numpy(mask), Pmk)

    # Experimental sets & scores
    N_EXP_TOTAL = max(EXP_STACK_NS)
    exp_raw, exp_proc, exp_scores = {}, {}, {}
    for eid in EXP_DATASET_IDS:
        r, p = load_exp_set(eid, N_EXP_TOTAL)
        exp_raw[eid], exp_proc[eid] = r, p
        exp_scores[eid] = project_scores(p, torch.from_numpy(mask), Pmk)

    # -------- RAW stacks (bordered) --------
    for n in EXP_STACK_NS:
        for eid in EXP_DATASET_IDS:
            imgs = [_minmax01(_to_np_hw(exp_raw[eid][i])) for i in range(n)]
            up = [_upsample2x(im) for im in imgs]
            _save_gray(
                ASSETDIR / f"stack_raw_border_noise{eid}_N{n}.png",
                _stack_gray_bordered(up, offset=STACK_OFFSET_PX, border=RECT_BORDER),
            )

    for n in DICT_STACK_NS:
        imgs = [_minmax01(_to_np_hw(dict_all[i])) for i in range(n)]
        up = [_upsample2x(im) for im in imgs]
        _save_gray(
            ASSETDIR / f"stack_dict_border_N{n}.png",
            _stack_gray_bordered(up, offset=STACK_OFFSET_PX, border=RECT_BORDER),
        )

    # -------- MASKED stacks (bordered ring; RGBA) --------
    for n in EXP_STACK_NS:
        for eid in EXP_DATASET_IDS:
            imgs = [_minmax01(_to_np_hw(exp_proc[eid][i])) for i in range(n)]
            rgba = _stack_rgba_masked_upscaled(
                imgs, mask, ring_thick=RING_THICK, offset=STACK_OFFSET_PX
            )
            _save_rgba(ASSETDIR / f"stack_masked_noise{eid}_N{n}.png", rgba)

    for n in DICT_STACK_NS:
        imgs = [_minmax01(_to_np_hw(dict_all[i])) for i in range(n)]
        rgba = _stack_rgba_masked_upscaled(
            imgs, mask, ring_thick=RING_THICK, offset=STACK_OFFSET_PX
        )
        _save_rgba(ASSETDIR / f"stack_masked_dict_N{n}.png", rgba)

    # -------- PCA components: masked stacks (RGBA) --------
    H, W = DET_SHAPE
    mflat = mask.flatten().astype(bool)
    for n in PC_STACK_NS:
        pc_imgs = []
        for k in range(n):
            tile = np.zeros(H * W, dtype=np.float32)
            tile[mflat] = Pmk[:, k]
            pc_imgs.append(_minmax01(tile.reshape(H, W)))
        rgba = _stack_rgba_masked_upscaled(
            pc_imgs, mask, ring_thick=RING_THICK, offset=STACK_OFFSET_PX
        )
        _save_rgba(ASSETDIR / f"components_masked_N{n}.png", rgba)

    # -------- Loadings strips (same C options for dict & exp) --------
    for C_show in LOAD_C_LIST:
        strip_d = _pca_strip_gray(dict_scores, C_show)
        _save_rgb(ASSETDIR / f"loadings_dict_C{C_show}.png", strip_d)
        for eid in EXP_DATASET_IDS:
            strip_e = _pca_strip_gray(exp_scores[eid], C_show)
            _save_rgb(ASSETDIR / f"loadings_noise{eid}_C{C_show}.png", strip_e)

    # -------- Dot-product matrix (grayscale, upscaled) + single red circle per row --------
    eid = EXP_DATASET_IDS[-1]  # e.g., show the highest-noise set
    E, D = exp_scores[eid], dict_scores
    M = E @ D.T  # (N_exp x N_dict), pure dot product

    # normalize to [0,1] for grayscale
    vmin, vmax = float(M.min()), float(M.max())
    M01 = ((M - vmin) / (vmax - vmin + 1e-8)).astype(np.float32)

    # grayscale RGB, then upscale each cell to DOT_CELL x DOT_CELL
    M_rgb = np.repeat(M01[:, :, None], 3, axis=2)  # (rows=N_exp, cols=N_dict, 3)
    M_big = np.repeat(np.repeat(M_rgb, DOT_CELL, axis=0), DOT_CELL, axis=1)

    def _draw_circle_outline(
        img_rgb, row_idx, col_idx, cell=DOT_CELL, color=(1.0, 0.0, 0.0), thick=3
    ):
        """
        Draw a red circular outline centered on the (row_idx, col_idx) cell.
        img_rgb: upscaled RGB image
        """
        H, W, _ = img_rgb.shape
        cy = int(row_idx * cell + cell / 2.0)
        cx = int(col_idx * cell + cell / 2.0)
        r = cell * 0.45  # radius; slightly inset from cell edges

        yy, xx = np.ogrid[:H, :W]
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        inner2 = (r - thick) ** 2
        outer2 = (r) ** 2
        ring = (dist2 >= inner2) & (dist2 <= outer2)
        img_rgb[ring] = color  # red outline

    # one highlight per row (reduce along dictionary dimension)
    best_cols = np.argmax(M, axis=1)  # shape: (N_exp,)
    for r, c in enumerate(best_cols):
        _draw_circle_outline(M_big, r, c, cell=DOT_CELL, color=(1.0, 0.0, 0.0), thick=3)

    _save_rgb(ASSETDIR / f"dotprod_matrix_noise{eid}.png", M_big)

    # grayscale colorbar (unchanged)
    grad = np.linspace(0, 1, 512)[None, :]
    bar = np.repeat(grad, 32, axis=0)
    bar_rgb = np.repeat(bar[:, :, None], 3, axis=2).astype(np.float32)
    _save_rgb(ASSETDIR / "dotprod_colorbar.png", bar_rgb)

    # (Optional) keep the separate top-3 truncated panel as-is if you like;
    # it’s independent from the single-circle overlay above.

    # top-3 truncated (sorted desc) per experimental, grayscale, upscaled
    top_sorted = np.sort(M, axis=1)[:, -TOPK:][:, ::-1]  # (N_exp x 3)
    T01 = (top_sorted - vmin) / max(vmax - vmin, 1e-8)
    T_rgb = np.repeat(T01[:, :, None], 3, axis=2).astype(np.float32)
    T_big = np.repeat(np.repeat(T_rgb, DOT_CELL, axis=0), DOT_CELL, axis=1)
    _save_rgb(ASSETDIR / f"topk{TOPK}_noise{eid}.png", T_big)

    # provenance
    meta = dict(
        det_shape=DET_SHAPE,
        proj_center=PROJ_CENTER,
        laue_group=LAUE_NI,
        exp_dataset_ids=EXP_DATASET_IDS,
        exp_stack_ns=EXP_STACK_NS,
        dict_stack_ns=DICT_STACK_NS,
        pc_stack_ns=PC_STACK_NS,
        pca_k=PCA_K,
        dict_res_deg=DICT_RES_DEG,
        topk=TOPK,
        upscale=UPSCALE,
        ring_thick=RING_THICK,
        stack_offset_px=STACK_OFFSET_PX,
        rect_border=RECT_BORDER,
        load_cell=LOAD_CELL,
        load_border=LOAD_BORDER,
        load_vpad=LOAD_VPAD,
        load_c_list=LOAD_C_LIST,
        dot_cell=DOT_CELL,
        seed=SEED,
    )
    (ASSETDIR / "params.json").write_text(json.dumps(meta, indent=2))
    print(f"✓ Assets written to: {ASSETDIR.resolve()}")


if __name__ == "__main__":
    main()
