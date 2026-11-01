"""Image-sim m-bias with the response weighted like the shear mean (#395).

For each branch of one grid, rerun the calibration of
calibrate_comprehensive_cat.py in memory (same config, same code), then
recompute the response with three weightings and write one recalibrated
copy of the cut catalogue per weighting:

  iv    : w_iv on every branch. Check: must reproduce the catalogue's R, RJ.
  des_c : w_des = W(snr, size_ratio) from the no-shear quantities, the same
          weight on every branch (no weight response).
  des_d : w_des table W evaluated on each branch's own snr and size_ratio
          (weight response included). The table itself is fixed: built once
          from the no-shear sample, as in get_w_des.
  iv_d  : w_iv = 1/(2 sigma_eps^2 + C), C = G1_ERR + G2_ERR, re-evaluated on
          each branch (weight response included). The catalogue keeps C only
          for no-shear, so C_b = C_ns * exp(f(snr_b, size_b) - f(snr_ns,
          size_ns)), with f a degree-2 polynomial fit of ln C_ns in
          (ln snr, ln size). Each object keeps its own residual, so iv_d is
          iv at no shear. Objects without a valid no-shear snr or size
          (T_ns <= 0) take C_b = exp(f(snr_b, size_b)). A proxy: misses any
          change of C at fixed (snr, size).
  iv_box: as iv_d, but the change of C comes from the w_des boxes instead
          of the fit: C_b = C_ns * Cbox(snr_b, size_b) / Cbox(snr_ns,
          size_ns), Cbox = mean C of the no-shear selection per box. Values
          outside the range go to the nearest edge box; a galaxy landing in
          an empty box keeps C_ns.

Each weighting gives R = R_shear + R_sel (split) and R_joint. Leave-one-
tile-out responses go in the R_JK HDU, as in the calibration script.

Usage: python wdes_response.py <grid dir> <out dir>
"""

import os
import sys

import numpy as np
from astropy.io import fits

from sp_validation import calibration
from sp_validation import catalog_builders as sp_joint
from sp_validation.calibration import metacal

BRANCHES = {"ns": "NOSHEAR", "p1": "1P", "m1": "1M", "p2": "2P", "m2": "2M"}
N_BINS = 20
DEG_C = 2


def poly_terms(x, y, deg=DEG_C):
    return np.array([x**i * y**j for i in range(deg + 1) for j in range(deg + 1 - i)]).T


def fit_lnC(snr, size, C):
    """Coefficients of the polynomial fit of ln C in (ln snr, ln size)."""
    x, y = np.log(snr), np.log(size)
    coef, *_ = np.linalg.lstsq(poly_terms(x, y), np.log(C), rcond=None)
    resid = np.log(C) - poly_terms(x, y) @ coef
    return coef, resid


def eval_lnC(snr, size, coef):
    return poly_terms(np.log(snr), np.log(size)) @ coef


def w_table(snr, size_ratio, e1, e2, R11, R22, cm):
    """Bin edges and cell values of w_des, as in calibration.get_w_des."""
    edges_s = calibration.create_bins(
        snr, N_BINS, x_min=cm["gal_snr_min"], x_max=cm["gal_snr_max"]
    )
    edges_r = calibration.create_bins(
        size_ratio, N_BINS, x_min=cm["gal_rel_size_min"], x_max=cm["gal_rel_size_max"]
    )
    i_s, i_r = cell_index(snr, size_ratio, edges_s, edges_r, like_pandas=True)
    table = np.full((N_BINS, N_BINS), np.nan)
    for i in range(N_BINS):
        for j in range(N_BINS):
            sel = (i_s == i) & (i_r == j)
            if sel.sum() == 0:
                continue
            noise = 0.5 * (np.mean(e1[sel] ** 2) + np.mean(e2[sel] ** 2))
            resp = 0.5 * (np.mean(R11[sel]) + np.mean(R22[sel]))
            table[i, j] = resp**2 / noise
    return edges_s, edges_r, table


def cell_index(snr, size_ratio, edges_s, edges_r, like_pandas=False):
    """Cell indices. like_pandas: as pd.cut(right=True) plus the code's rule
    that values outside the range go to the last bin. Otherwise values
    outside the range go to the nearest edge cell."""
    out = []
    for x, e in ((snr, edges_s), (size_ratio, edges_r)):
        idx = np.searchsorted(e, x, side="left") - 1
        if like_pandas:
            idx[(x <= e[0]) | (x > e[-1]) | ~np.isfinite(x)] = N_BINS - 1
        else:
            idx = np.clip(idx, 0, N_BINS - 1)
        out.append(idx)
    return out


def eval_table(snr, size_ratio, edges_s, edges_r, table):
    i_s, i_r = cell_index(snr, size_ratio, edges_s, edges_r)
    w = table[i_s, i_r]
    n_bad = int(np.sum(~np.isfinite(w)))
    w[~np.isfinite(w)] = 0.0
    return w, n_bad


def responses(gm, wts, keep=None):
    """Split R = R_shear + R_sel and joint R, weights per branch.

    wts[b] is a weight per object of the metacal input (gm.ns index), used
    on branch b's selection. keep: bool per object, False drops it.
    """
    h2 = 2 * gm._step
    branch = {"p1": gm.p1, "m1": gm.m1, "p2": gm.p2, "m2": gm.m2}
    sel = {}
    for b, idx in gm.mask_dict.items():
        sel[b] = idx if keep is None else idx[keep[idx]]

    def mean(data, b, comp):
        return np.average(data[comp][sel[b]], weights=wts[b][sel[b]])

    ma = gm.mask_dict["ns"]
    k_ns = np.ones(len(ma), bool) if keep is None else keep[ma]
    R_shear = np.average(gm.R_shear[:, :, k_ns], axis=2, weights=wts["ns"][ma][k_ns])

    R_sel = np.zeros((2, 2))
    R_joint = np.zeros((2, 2))
    for a, comp in enumerate(("g1", "g2")):
        for b, (p, m) in enumerate((("p1", "m1"), ("p2", "m2"))):
            R_sel[a, b] = (mean(gm.ns, p, comp) - mean(gm.ns, m, comp)) / h2
            R_joint[a, b] = (mean(branch[p], p, comp) - mean(branch[m], m, comp)) / h2
    return R_shear + R_sel, R_joint, R_shear, R_sel


def snr_size(dat_m, key):
    snr = dat_m[f"NGMIX_FLUX_{key}"] / dat_m[f"NGMIX_FLUX_ERR_{key}"]
    size = dat_m[f"NGMIX_T_{key}"] / dat_m[f"NGMIX_T_PSF_RECONV_{key}"]
    return np.asarray(snr, float), np.asarray(size, float)


def run_branch(sim_dir, out_root):
    os.chdir(sim_dir)
    name = os.path.basename(sim_dir.rstrip("/"))
    print(f"=== {name}", flush=True)

    obj = sp_joint.CalibrateCat()
    config = obj.read_config_set_params("config_mask.yaml")
    dat, dat_ext = obj.read_cat(load_into_memory=False)
    masks, labels = sp_joint.get_masks_from_config(config, dat, dat_ext, verbose=False)
    mask_combined = sp_joint.Mask.from_list(masks, label="combined", verbose=False)
    cm = config["metacal"]
    gm = metacal(
        dat,
        mask_combined._mask,
        snr_min=cm["gal_snr_min"],
        snr_max=cm["gal_snr_max"],
        rel_size_min=cm["gal_rel_size_min"],
        rel_size_max=cm["gal_rel_size_max"],
        size_corr_ell=cm["gal_size_corr_ell"],
        sigma_eps=cm["sigma_eps_prior"],
        global_R_weight=cm["global_R_weight"],
        verbose=False,
    )
    _, g_uncorr, _, mask_metacal, _, _ = calibration.get_calibrated_m_c(
        gm, additive_correction=cm.get("additive_correction", True)
    )
    cat_gal = {}
    sp_joint.compute_weights_gatti(
        cat_gal,
        g_uncorr,
        gm,
        dat,
        mask_combined,
        mask_metacal,
        num_bins=N_BINS,
        snr_min=cm["gal_snr_min"],
        snr_max=cm["gal_snr_max"],
        size_ratio_min=cm["gal_rel_size_min"],
        size_ratio_max=cm["gal_rel_size_max"],
    )

    dat_m = dat[mask_combined._mask]
    tiles = np.char.strip(np.asarray(dat["TILE_ID"]).astype(str)[mask_combined._mask])

    # w_des table from the no-shear sample, checked against the catalogue's
    edges_s, edges_r, table = w_table(
        cat_gal["snr"],
        cat_gal["size_ratio"],
        g_uncorr[0],
        g_uncorr[1],
        cat_gal["R_g11"],
        cat_gal["R_g22"],
        cm,
    )
    snr_ns, size_ns = snr_size(dat_m, "NOSHEAR")
    w_c, bad_c = eval_table(snr_ns, size_ns, edges_s, edges_r, table)
    ma = gm.mask_dict["ns"]
    dev = np.max(np.abs(w_c[ma] / cat_gal["w_des"] - 1))
    print(f"  w_des table check: max rel dev on S_ns = {dev:.2e}", flush=True)
    if dev > 1e-6:
        raise ValueError(f"{name}: w_des table does not reproduce w_des")

    w_d = {"ns": w_c}
    for b, key in BRANCHES.items():
        if b == "ns":
            continue
        s, r = snr_size(dat_m, key)
        w_d[b], n_bad = eval_table(s, r, edges_s, edges_r, table)
        print(f"  {b}: {n_bad} objects in empty w_des cells", flush=True)
    print(f"  c: {bad_c} objects in empty w_des cells", flush=True)

    # w_iv per branch: C from a fit on the no-shear selection
    sig2 = 2 * cm["sigma_eps_prior"] ** 2
    C_ns = np.asarray(
        dat_m["NGMIX_G1_ERR_NOSHEAR"] + dat_m["NGMIX_G2_ERR_NOSHEAR"], float
    )
    w_iv_ns = 1 / (sig2 + C_ns)
    dev = np.max(np.abs(w_iv_ns / gm.ns["w"] - 1))
    print(f"  w_iv check: max rel dev = {dev:.2e}", flush=True)
    if dev > 1e-12:
        raise ValueError(f"{name}: w_iv from G_ERR does not reproduce gm.ns['w']")
    coef, resid = fit_lnC(snr_ns[ma], size_ns[ma], C_ns[ma])
    print(
        f"  ln C fit (deg {DEG_C}) on S_ns: rms resid {resid.std():.4f},"
        f" coef {np.array2string(coef, precision=4)}",
        flush=True,
    )
    with np.errstate(invalid="ignore"):
        f_ns = eval_lnC(snr_ns, size_ns, coef)
    no_ns = ~np.isfinite(f_ns)
    w_iv_d = {"ns": gm.ns["w"]}
    for b, key in BRANCHES.items():
        if b == "ns":
            continue
        s, r = snr_size(dat_m, key)
        with np.errstate(invalid="ignore"):
            f_b = eval_lnC(s, r, coef)
        C_b = np.where(no_ns, np.exp(f_b), C_ns * np.exp(f_b - f_ns))
        w_iv_d[b] = 1 / (sig2 + C_b)
        sel = gm.mask_dict[b]
        print(
            f"  iv_d {b}: {int(no_ns[sel].sum())} objects on S_{b} without"
            " valid no-shear snr/size, fit value used",
            flush=True,
        )
        if not np.all(np.isfinite(w_iv_d[b][sel])):
            raise ValueError(f"{name}: non-finite iv_d weight on S_{b}")
        ratio = w_iv_d[b][sel] / w_iv_ns[sel] - 1
        comp = "g1" if b in ("p1", "m1") else "g2"
        rho = np.corrcoef(ratio, gm.ns[comp][sel])[0, 1]
        print(
            f"  iv_d {b}: w/w_ns - 1 median {np.median(ratio):+.2e}"
            f" rms {ratio.std():.2e}, corr with {comp}_ns {rho:+.3f}",
            flush=True,
        )

    # w_iv per branch: change of C from the mean C per w_des box
    i_s, i_r = cell_index(snr_ns[ma], size_ns[ma], edges_s, edges_r, like_pandas=True)
    Cbox = np.full((N_BINS, N_BINS), np.nan)
    for i in range(N_BINS):
        for j in range(N_BINS):
            t = (i_s == i) & (i_r == j)
            if t.any():
                Cbox[i, j] = C_ns[ma][t].mean()
    j_s, j_r = cell_index(snr_ns, size_ns, edges_s, edges_r)
    Cbox_ns = Cbox[j_s, j_r]
    w_iv_box = {"ns": gm.ns["w"]}
    for b, key in BRANCHES.items():
        if b == "ns":
            continue
        s, r = snr_size(dat_m, key)
        k_s, k_r = cell_index(s, r, edges_s, edges_r)
        ratio_C = Cbox[k_s, k_r] / Cbox_ns
        sel = gm.mask_dict[b]
        n_empty = int(np.sum(~np.isfinite(ratio_C[sel])))
        n_moved = int(np.sum((k_s != j_s)[sel] | (k_r != j_r)[sel]))
        ratio_C[~np.isfinite(ratio_C)] = 1.0
        w_iv_box[b] = 1 / (sig2 + C_ns * ratio_C)
        ratio = w_iv_box[b][sel] / w_iv_ns[sel] - 1
        comp = "g1" if b in ("p1", "m1") else "g2"
        rho = np.corrcoef(ratio, gm.ns[comp][sel])[0, 1]
        print(
            f"  iv_box {b}: {n_moved} of {len(sel)} change box, {n_empty} in empty"
            f" box; w/w_ns - 1 rms {ratio.std():.2e}, corr with {comp}_ns {rho:+.3f}",
            flush=True,
        )

    schemes = {
        "iv": {b: gm.ns["w"] for b in BRANCHES},
        "iv_d": w_iv_d,
        "iv_box": w_iv_box,
        "des_c": {b: w_c for b in BRANCHES},
        "des_d": w_d,
    }

    # Check: iv reproduces the response of the catalogue on disk
    cut_path = os.path.join(sim_dir, "shape_catalog_cut_ngmix.fits")
    with fits.open(cut_path) as hdul:
        hdr = hdul[0].header
        R_disk = np.array([[hdr[f"R_{i}{j}"] for j in (1, 2)] for i in (1, 2)])
        RJ_disk = np.array([[hdr[f"RJ_{i}{j}"] for j in (1, 2)] for i in (1, 2)])
        e_uncal_disk = np.array([hdul[1].data["e1_uncal"], hdul[1].data["e2_uncal"]])
    R_iv, RJ_iv, _, _ = responses(gm, schemes["iv"])
    print(
        f"  iv check: |R - R_disk| = {np.max(np.abs(R_iv - R_disk)):.2e},"
        f" |RJ - RJ_disk| = {np.max(np.abs(RJ_iv - RJ_disk)):.2e}",
        flush=True,
    )
    if np.max(np.abs(R_iv - R_disk)) > 1e-8 or np.max(np.abs(RJ_iv - RJ_disk)) > 1e-8:
        raise ValueError(f"{name}: iv response does not reproduce the catalogue")
    if np.max(np.abs(e_uncal_disk - g_uncorr)) > 1e-10:
        raise ValueError(f"{name}: object order differs from the catalogue")

    labels_jk = np.unique(tiles)
    for scheme, wts in schemes.items():
        R, RJ, R_sh, R_se = responses(gm, wts)
        n_eff = {
            b: float(
                wts[b][gm.mask_dict[b]].sum() ** 2
                / (wts[b][gm.mask_dict[b]] ** 2).sum()
            )
            for b in BRANCHES
        }
        print(
            f"  {scheme}: R_shear diag {R_sh[0, 0]:.4f} {R_sh[1, 1]:.4f}"
            f"  R_sel diag {R_se[0, 0]:+.4f} {R_se[1, 1]:+.4f}"
            f"  RJ diag {RJ[0, 0]:.4f} {RJ[1, 1]:.4f}"
            f"  N_eff(ns) {n_eff['ns']:.0f} of {len(ma)}",
            flush=True,
        )
        R_jk = np.empty((len(labels_jk), 2, 2))
        RJ_jk = np.empty((len(labels_jk), 2, 2))
        for k, t in enumerate(labels_jk):
            R_jk[k], RJ_jk[k], _, _ = responses(gm, wts, keep=tiles != t)

        out_dir = os.path.join(out_root, scheme, name)
        os.makedirs(out_dir, exist_ok=True)
        with fits.open(cut_path) as hdul:
            g = np.linalg.inv(R) @ g_uncorr
            hdul[1].data["e1"] = g[0]
            hdul[1].data["e2"] = g[1]
            h0 = hdul[0].header
            for i in (1, 2):
                for j in (1, 2):
                    h0[f"R_{i}{j}"] = R[i - 1, j - 1]
                    h0[f"RJ_{i}{j}"] = RJ[i - 1, j - 1]
                    h0[f"R_G{i}{j}"] = R_sh[i - 1, j - 1]
                    h0[f"R_S{i}{j}"] = R_se[i - 1, j - 1]
            h0["RWEIGHT"] = (scheme, "weighting of the response (wdes_response.py)")
            cols = [
                fits.Column(
                    name="TILE_ID",
                    array=labels_jk,
                    format=f"{max(len(t) for t in labels_jk)}A",
                )
            ]
            for key, resp in (("R", R_jk), ("RJ", RJ_jk)):
                for i in (1, 2):
                    for j in (1, 2):
                        cols.append(
                            fits.Column(
                                name=f"{key}_{i}{j}",
                                array=resp[:, i - 1, j - 1],
                                format="D",
                            )
                        )
            jk = fits.BinTableHDU.from_columns(cols, name="R_JK")
            hdus = fits.HDUList([hdul[0], hdul[1], jk])
            hdus.writeto(
                os.path.join(out_dir, "shape_catalog_cut_ngmix.fits"), overwrite=True
            )


def main():
    grid_dir, out_root = sys.argv[1], sys.argv[2]
    num = grid_dir.rstrip("/").split("_")[-1]
    for b in ("1z2z", "1p2z", "1m2z", "1z2p", "1z2m"):
        run_branch(os.path.join(grid_dir, f"{b}_grid_{num}"), out_root)


if __name__ == "__main__":
    main()
