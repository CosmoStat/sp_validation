"""Catalogue-level products behind the dashboard sky maps and weight figure.

Two cached products, each computed once from the calibrated catalogue and
read by the renderers in :mod:`sp_validation.dashboard`:

``field_maps_<version>.npz``
    Per-pixel sums on a nested HEALPix grid: galaxy counts and weight sums,
    weighted ellipticity and ellipticity-squared sums, and the PSF model
    (ellipticity, FWHM) at the galaxy positions; plus star counts and the
    star-minus-PSF-model residual sums. Means, dispersions and effective
    densities are ratios of these sums, so a renderer can coarsen the grid
    by summing children before dividing. Only occupied pixels are stored.

``weight_cells_<version>.npz``
    The galaxy weight on the SNR x size-ratio grid it is built on
    (:func:`sp_validation.calibration.get_w_des`): per-cell counts and the
    weight range within each cell, the grid edges, and the distinct weight
    values with their multiplicities. Also contains response histograms,
    equal-count leakage-cell regression sums and binned additive-bias sums.
    Quantile boundaries use a temporary float32 buffer of the seven leakage
    inputs; only sufficient statistics are saved.

``star_residuals_<version>.npz``
    Histograms and full-sample moments of valid-HSM star residuals.

Usage (inside the container)::

    python -m sp_validation.dashboard.field_maps \\
        --galaxies shape_catalog_cut_ngmix.fits --stars stars.hdf5 \\
        --version SP_v2.0 --out-dir cosmo_val/
"""

import argparse
import json
from pathlib import Path

import fitsio
import healpy as hp
import numpy as np

from sp_validation.io import read_catalogue

GALAXY_COLUMNS = [
    "RA",
    "Dec",
    "w_des",
    "e1",
    "e2",
    "e1_uncal",
    "e2_uncal",
    "R_g11",
    "R_g12",
    "R_g21",
    "R_g22",
    "e1_PSF",
    "e2_PSF",
    "fwhm_PSF",
    "mag",
    "snr",
    "NGMIX_T_NOSHEAR",
    "NGMIX_T_PSF_RECONV_NOSHEAR",
]
BIAS_EDGES = {
    "PSF e1": np.linspace(-0.2, 0.2, 21),
    "PSF e2": np.linspace(-0.2, 0.2, 21),
    "PSF FWHM": np.linspace(0.3, 2.0, 21),
    "r magnitude": np.linspace(18, 27, 21),
    "SNR": np.geomspace(10, 500, 21),
    "size ratio": np.linspace(0.707, 3, 21),
    "galaxy size": np.linspace(0, 5, 21),
    "RA": np.linspace(0, 360, 21),
    "Dec": np.linspace(-30, 90, 21),
    "weight": np.linspace(0, 50, 21),
}

STAR_COLUMNS = [
    "RA",
    "DEC",
    "HSM_G1_PSF",
    "HSM_G2_PSF",
    "HSM_T_PSF",
    "HSM_G1_STAR",
    "HSM_G2_STAR",
    "HSM_T_STAR",
    "HSM_M4_1_STAR",
    "HSM_M4_2_STAR",
    "HSM_M4_1_PSF",
    "HSM_M4_2_PSF",
    "HSM_FLAG_PSF",
    "HSM_FLAG_STAR",
]
GALAXY_SUMS = [
    "n_gal",
    "sum_w",
    "sum_w2",
    "sum_w_e1",
    "sum_w_e2",
    "sum_w2_ee",
    "sum_w_e1_psf",
    "sum_w_e2_psf",
    "sum_w_fwhm_psf",
    "sum_mag",
]
STAR_SUMS = [
    "n_star",
    "sum_de1",
    "sum_de2",
    "sum_dT_over_T_psf",
    "sum_T_psf",
]


def weight_grid_edges(n_bins, snr_range, size_ratio_range):
    """Log-spaced SNR and size-ratio edges of the weight grid."""
    snr_edges = np.logspace(*np.log10(snr_range), n_bins + 1)
    size_edges = np.logspace(*np.log10(size_ratio_range), n_bins + 1)
    return snr_edges, size_edges


def weight_cell_index(values, edges):
    """Cell index of each value, matching ``calibration.cut_to_bins``.

    Bins are right-closed, ``(edge[i], edge[i+1]]``; a value outside the
    edges belongs to the last cell, as ``cut_to_bins`` assigns it.
    """
    n_bins = len(edges) - 1
    index = np.searchsorted(edges, values, side="left") - 1
    outside = (index < 0) | (index >= n_bins) | ~np.isfinite(values)
    index[outside] = n_bins - 1
    return index


def _accumulate(total, pixels, weights, npix):
    total += np.bincount(pixels, weights=weights, minlength=npix)


def leakage_cells(data, n_bins=8):
    """Equal-count resolution bins, then equal-count SNR within each bin."""
    valid = np.all(np.isfinite(data), axis=1) & (data[:, 2] > 0)
    data = data[valid]
    resolution_edges = np.quantile(data[:, 1], np.linspace(0, 1, n_bins + 1))
    if np.any(np.diff(resolution_edges) <= 0):
        raise ValueError("resolution quantiles are not distinct")
    rbin = np.clip(
        np.searchsorted(resolution_edges, data[:, 1], side="right") - 1, 0, n_bins - 1
    )
    snr_edges = np.zeros((n_bins, n_bins + 1))
    sums = np.zeros((2, 7, n_bins, n_bins))
    for r in range(n_bins):
        d = data[rbin == r].astype(np.float64)
        snr_edges[r] = np.quantile(d[:, 0], np.linspace(0, 1, n_bins + 1))
        if np.any(np.diff(snr_edges[r]) <= 0):
            raise ValueError("SNR quantiles are not distinct")
        idx = np.clip(
            np.searchsorted(snr_edges[r], d[:, 0], side="right") - 1, 0, n_bins - 1
        )
        w = d[:, 2]
        for component in range(2):
            x, y = d[:, 3 + 2 * component], d[:, 4 + 2 * component]
            for k, v in enumerate(
                (w, w * x, w * y, w * x * x, w * x * y, w * y * y, np.ones_like(w))
            ):
                sums[component, k, r] = np.bincount(idx, weights=v, minlength=n_bins)
    return resolution_edges, snr_edges, sums


def galaxy_products(path, nside, n_bins, snr_range, size_ratio_range, chunk):
    """Stream the galaxy catalogue once into map sums and weight-cell tallies."""
    npix = hp.nside2npix(nside)
    sums = {key: np.zeros(npix) for key in GALAXY_SUMS}
    snr_edges, size_edges = weight_grid_edges(n_bins, snr_range, size_ratio_range)
    cell_count = np.zeros((n_bins, n_bins), dtype=np.int64)
    cell_w_min = np.full((n_bins, n_bins), np.inf)
    cell_w_max = np.full((n_bins, n_bins), -np.inf)
    out_of_range = {"snr_low": 0, "snr_high": 0, "size_low": 0, "size_high": 0}
    weights = []
    cell_response = np.zeros((n_bins, n_bins))
    cell_ee = np.zeros((n_bins, n_bins))
    response_edges = np.linspace(-5, 5, 501)
    response_hist = np.zeros((4, 500), dtype=np.int64)
    response_outside = np.zeros(4, dtype=np.int64)
    bias_sums = np.zeros((len(BIAS_EDGES), 2, 10, 20))

    with fitsio.FITS(path) as fits:
        table = fits[1]
        n_rows = table.get_nrows()
        leakage_data = np.empty((n_rows, 7), dtype=np.float32)
        for start in range(0, n_rows, chunk):
            rows = table[GALAXY_COLUMNS][start : start + chunk]
            pix = hp.ang2pix(nside, rows["RA"], rows["Dec"], lonlat=True, nest=True)
            w = rows["w_des"].astype(np.float64)
            e1, e2 = rows["e1"], rows["e2"]
            for key, value in (
                ("n_gal", None),
                ("sum_w", w),
                ("sum_w2", w**2),
                ("sum_w_e1", w * e1),
                ("sum_w_e2", w * e2),
                ("sum_w2_ee", w**2 * (e1**2 + e2**2)),
                ("sum_w_e1_psf", w * rows["e1_PSF"]),
                ("sum_w_e2_psf", w * rows["e2_PSF"]),
                ("sum_w_fwhm_psf", w * rows["fwhm_PSF"]),
                ("sum_mag", rows["mag"]),
            ):
                _accumulate(sums[key], pix, value, npix)

            snr = rows["snr"]
            size_ratio = rows["NGMIX_T_NOSHEAR"] / rows["NGMIX_T_PSF_RECONV_NOSHEAR"]
            out_of_range["snr_low"] += int(np.sum(snr <= snr_range[0]))
            out_of_range["snr_high"] += int(np.sum(snr > snr_range[1]))
            out_of_range["size_low"] += int(np.sum(size_ratio <= size_ratio_range[0]))
            out_of_range["size_high"] += int(np.sum(size_ratio > size_ratio_range[1]))
            i = weight_cell_index(snr, snr_edges)
            j = weight_cell_index(size_ratio, size_edges)
            flat = i * n_bins + j
            cell_count += np.bincount(flat, minlength=n_bins**2).reshape(n_bins, -1)
            response = 0.5 * (rows["R_g11"] + rows["R_g22"])
            ee = 0.5 * (rows["e1_uncal"] ** 2 + rows["e2_uncal"] ** 2)
            cell_response += np.bincount(
                flat, weights=response, minlength=n_bins**2
            ).reshape(n_bins, -1)
            cell_ee += np.bincount(flat, weights=ee, minlength=n_bins**2).reshape(
                n_bins, -1
            )
            for k, name in enumerate(("R_g11", "R_g22", "R_g12", "R_g21")):
                hist, _ = np.histogram(rows[name], bins=response_edges)
                response_hist[k] += hist
                response_outside[k] += len(rows) - hist.sum()
            with np.errstate(invalid="ignore", divide="ignore"):
                resolution = 1 / (1 + np.sqrt(size_ratio))
            for column, data in enumerate(
                (snr, resolution, w, rows["e1_PSF"], e1, rows["e2_PSF"], e2)
            ):
                leakage_data[start : start + len(rows), column] = data
            props = (
                rows["e1_PSF"],
                rows["e2_PSF"],
                rows["fwhm_PSF"],
                rows["mag"],
                snr,
                size_ratio,
                np.sqrt(np.maximum(rows["NGMIX_T_NOSHEAR"], 0)),
                rows["RA"],
                rows["Dec"],
                w,
            )
            for p, (x, edges) in enumerate(zip(props, BIAS_EDGES.values())):
                idx = np.searchsorted(edges, x, side="right") - 1
                ok = (idx >= 0) & (idx < 20) & np.isfinite(x)
                b, xx, ww = idx[ok], x[ok].astype(float), w[ok]
                for component in range(2):
                    y = rows[f"e{component + 1}"][ok].astype(float)
                    for k, v in enumerate(
                        (
                            ww,
                            ww * xx,
                            ww * y,
                            ww * xx * xx,
                            ww * xx * y,
                            ww * y * y,
                            ww * ww,
                            np.ones_like(ww),
                            ww * ww * y,
                            ww * ww * y * y,
                        )
                    ):
                        bias_sums[p, component, k] += np.bincount(
                            b, weights=v, minlength=20
                        )
            np.minimum.at(cell_w_min.ravel(), flat, w)
            np.maximum.at(cell_w_max.ravel(), flat, w)
            weights.append(w)

    resolution_edges, leakage_snr_edges, leakage_sums = leakage_cells(leakage_data)
    del leakage_data
    values, multiplicity = np.unique(np.concatenate(weights), return_counts=True)
    cells = {
        "snr_edges": snr_edges,
        "size_ratio_edges": size_edges,
        "count": cell_count,
        "sum_response": cell_response,
        "sum_uncal_ee": cell_ee,
        "response_edges": response_edges,
        "response_hist": response_hist,
        "response_outside": response_outside,
        "resolution_edges": resolution_edges,
        "leakage_snr_edges": leakage_snr_edges,
        "leakage_sums": leakage_sums,
        "bias_names": np.array(list(BIAS_EDGES)),
        "bias_edges": np.array(list(BIAS_EDGES.values())),
        "bias_sums": bias_sums,
        "w_min": np.where(cell_count > 0, cell_w_min, np.nan),
        "w_max": np.where(cell_count > 0, cell_w_max, np.nan),
        "w_values": values,
        "w_multiplicity": multiplicity,
        "n_rows": n_rows,
        "out_of_range": json.dumps(out_of_range),
    }
    return sums, cells


def star_sums(path, nside):
    """Star counts and star-minus-PSF-model residual sums per pixel."""
    npix = hp.nside2npix(nside)
    stars = read_catalogue(path, columns=STAR_COLUMNS)
    keep = (stars["HSM_FLAG_PSF"] == 0) & (stars["HSM_FLAG_STAR"] == 0)
    stars = stars[keep]
    pix = hp.ang2pix(nside, stars["RA"], stars["DEC"], lonlat=True, nest=True)
    t_star, t_psf = stars["HSM_T_STAR"], stars["HSM_T_PSF"]
    sums = {key: np.zeros(npix) for key in STAR_SUMS}
    for key, value in (
        ("n_star", None),
        ("sum_de1", stars["HSM_G1_STAR"] - stars["HSM_G1_PSF"]),
        ("sum_de2", stars["HSM_G2_STAR"] - stars["HSM_G2_PSF"]),
        ("sum_dT_over_T_psf", (t_star - t_psf) / t_psf),
        ("sum_T_psf", t_psf),
    ):
        _accumulate(sums[key], pix, value, npix)
    residuals = [
        stars["HSM_G1_STAR"] - stars["HSM_G1_PSF"],
        stars["HSM_G2_STAR"] - stars["HSM_G2_PSF"],
        stars["HSM_M4_1_STAR"] - stars["HSM_M4_1_PSF"],
        stars["HSM_M4_2_STAR"] - stars["HSM_M4_2_PSF"],
        np.sqrt(np.where((t_star > 0) & (t_psf > 0), t_star / t_psf, np.nan)) - 1,
    ]
    edges = (
        [np.linspace(-0.5, 0.5, 401)] * 2
        + [np.linspace(-0.5, 0.5, 401)] * 2
        + [np.linspace(-1, 1, 401)]
    )
    histograms = {}
    for i, (values, bins) in enumerate(zip(residuals, edges)):
        values = values[np.isfinite(values)]
        histograms[f"edges_{i}"] = bins
        histograms[f"hist_{i}"] = np.histogram(values, bins=bins)[0]
        histograms[f"mean_{i}"] = values.mean()
        histograms[f"std_{i}"] = values.std()
        histograms[f"outside_{i}"] = len(values) - histograms[f"hist_{i}"].sum()
    return sums, int(keep.sum()), histograms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--galaxies", required=True, help="calibrated galaxy FITS")
    parser.add_argument("--stars", required=True, help="star/PSF catalogue")
    parser.add_argument("--version", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--weight-bins", type=int, default=20)
    parser.add_argument("--snr-range", type=float, nargs=2, default=(10.0, 500.0))
    parser.add_argument("--size-ratio-range", type=float, nargs=2, default=(0.707, 3.0))
    parser.add_argument("--chunk", type=int, default=4_000_000)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    gal, cells = galaxy_products(
        args.galaxies,
        args.nside,
        args.weight_bins,
        args.snr_range,
        args.size_ratio_range,
        args.chunk,
    )
    stars, n_stars, star_hist = star_sums(args.stars, args.nside)
    occupied = np.flatnonzero((gal["n_gal"] > 0) | (stars["n_star"] > 0))
    np.savez_compressed(
        out_dir / f"field_maps_{args.version}.npz",
        version=args.version,
        selection="calibrated cut catalogue; stars with both HSM flags zero",
        nside=args.nside,
        nest=True,
        pixels=occupied,
        n_stars_used=n_stars,
        **{key: value[occupied] for key, value in {**gal, **stars}.items()},
    )
    np.savez_compressed(
        out_dir / f"star_residuals_{args.version}.npz",
        version=args.version,
        selection="PSF validation stars with both HSM flags zero",
        units="dimensionless HSM ellipticity, fourth moments and radius ratio",
        **star_hist,
    )
    np.savez_compressed(
        out_dir / f"weight_cells_{args.version}.npz",
        version=args.version,
        selection="calibrated cut catalogue",
        units="SNR dimensionless; sizes are ngmix T ratios",
        **cells,
    )


if __name__ == "__main__":
    main()
