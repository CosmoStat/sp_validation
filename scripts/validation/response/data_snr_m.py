"""Compare SNR-resolved image-sim m to a data response-weighted histogram."""

import argparse
import json
from pathlib import Path

import inputs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from sp_validation import sim_m_snr


def from_cells(path, weight_path, target_edges):
    """Rebin data cell sums, assuming uniform density in log SNR within cells."""
    with np.load(path) as f, fits.open(weight_path) as hdus:
        edges = f["snr_edges"].copy()
        np.testing.assert_allclose(edges, hdus["W_EDGES"].data[0])
        np.testing.assert_allclose(f["size_edges"], hdus["W_EDGES"].data[1])
        weights = hdus["W_DES"].data
        count = f["count"].copy()
        np.testing.assert_array_equal(count, hdus["W_COUNT"].data)
        r = np.stack([f["sum_R_g11"], f["sum_R_g22"]], axis=-1)
        cell_response = np.sum(r * weights[:, :, None], axis=1)
        cell_weight = np.sum(count * weights, axis=1)
        cell_count = np.sum(count, axis=1)
    a, b = np.log(edges[:-1]), np.log(edges[1:])
    x, y = np.log(target_edges[:-1]), np.log(target_edges[1:])
    frac = np.maximum(0, np.minimum(b[:, None], y) - np.maximum(a[:, None], x))
    frac /= (b - a)[:, None]
    return dict(
        count=frac.T @ cell_count,
        sum_w=frac.T @ cell_weight,
        sum_w_r=frac.T @ cell_response,
    )


def run(product_path, cells, full_dir, output):
    """Save an effective-m summary and a provisional data histogram plot."""
    product = sim_m_snr.read_product(product_path)
    hist = from_cells(cells, inputs.PRODUCT["path"], product["edges"])
    # Pool simulation selection/shear response ratios over external branches
    # and grids with their no-shear weight sums. This is a stand-in for the
    # data selection response, not a measurement of that response.
    shear = np.zeros_like(hist["sum_w_r"])
    selection = np.zeros_like(shear)
    for grid in range(1, 5):
        with np.load(output / f"review_default_{grid}.npz") as f:
            for c in range(2):
                shear[:, c] += (f["sum_w"] * f["shear"][:, :, c, c]).sum(axis=0)
                selection[:, c] += (f["sum_w"] * f["selection"][:, :, c, c]).sum(axis=0)
    if np.any(shear == 0) or np.any(hist["sum_w"] <= 0):
        raise ValueError("Cannot form response ratios for empty bins")
    ratio = selection / shear
    total = np.zeros((len(hist["sum_w"]), 2, 2))
    for c in range(2):
        total[:, c, c] = hist["sum_w_r"][:, c] / hist["sum_w"] * (1 + ratio[:, c])
    value, cov, share = sim_m_snr.effective_m(product, hist["sum_w"], total)
    full = []
    for i in range(1, 5):
        full.append(json.loads((full_dir / f"production_{i}.json").read_text()))
    fs = np.array([[r["m1"], r["m2"]] for r in full])
    fe = np.array([[r["m1_err"], r["m2_err"]] for r in full])
    fw = 1 / fe**2
    mean = (fs * fw).sum(axis=0) / fw.sum(axis=0)
    err = 1 / np.sqrt(fw.sum(axis=0))
    grid_products = []
    for i in range(1, 5):
        with np.load(output / f"snr_grid_{i}.npz") as f:
            grid_products.append({k: f[k].copy() for k in f.files})
    grid_weights = sim_m_snr.combine_grids(grid_products)["grid_weights"]
    difference_cov = np.zeros((2, 2))
    fw /= fw.sum(axis=0)
    for i, (grid, full_result) in enumerate(zip(grid_products, full, strict=True)):
        if not np.array_equal(
            grid["labels"].astype(str), full_result["jackknife_tiles"]
        ):
            raise ValueError("Full and SNR products must have identical tile deletions")
        draws = np.asarray(grid["deleted"])
        full_draws = np.asarray(full_result["mc_jackknife"])
        delta = np.stack(
            [
                np.sum(
                    draws[:, :, 2 * c] * share[:, c] * grid_weights[i, 2 * c :: 4],
                    axis=1,
                )
                - fw[i, c] * full_draws[:, 2 * c]
                for c in range(2)
            ],
            axis=1,
        )
        delta -= delta.mean(axis=0)
        difference_cov += (len(delta) - 1) / len(delta) * delta.T @ delta
    summary = dict(
        method="Provisional: log-uniform interpolation of data shear response in DES cells; simulation bin selection/shear response ratio stands in for data selection response. Fixed data distribution and response ratio.",
        sim_selection_over_shear=ratio.tolist(),
        data_total_response=total.tolist(),
        count=hist["count"].tolist(),
        sum_w=hist["sum_w"].tolist(),
        sum_w_r=hist["sum_w_r"].tolist(),
        response_share=share.tolist(),
        effective_m=value.tolist(),
        covariance=cov.tolist(),
        full_m=mean.tolist(),
        full_error=err.tolist(),
        effective_minus_full=(value - mean).tolist(),
        difference_covariance=difference_cov.tolist(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_snr_provisional.json").write_text(json.dumps(summary, indent=2))
    edges = product["edges"]
    center = np.sqrt(edges[:-1] * edges[1:])
    errors = np.sqrt(np.diag(product["covariance"])).reshape(-1, 4)
    fig, (a, b) = plt.subplots(
        2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    for c, color in enumerate(("C0", "C1")):
        a.errorbar(
            center,
            product["value"][:, 2 * c] * 1000,
            yerr=errors[:, 2 * c] * 1000,
            fmt="o-",
            capsize=3,
            color=color,
            label=f"m{c + 1} in SNR bins",
        )
        a.axhspan(
            (mean[c] - err[c]) * 1000,
            (mean[c] + err[c]) * 1000,
            color=color,
            alpha=0.12,
            label=f"m{c + 1} full population ±1σ",
        )
    a.axhline(0, color="0.4", lw=0.8)
    a.set_ylabel("m × 10⁻³")
    a.legend(fontsize=8)
    b.stairs(share.mean(axis=1), edges, fill=True, alpha=0.4)
    b.set_ylabel("Data response\nshare per bin")
    b.set_xlabel("No-shear SNR")
    b.set_xscale("log")
    fig.suptitle("Image-sim m and provisional v2 response shares")
    fig.tight_layout()
    fig.savefig(output / "snr_m.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(p)
    p.add_argument("--product-snr", type=Path, required=True)
    p.add_argument("--cells", type=Path, required=True)
    p.add_argument("--production", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs.configure(args)
    run(args.product_snr, args.cells, args.production, args.output)
