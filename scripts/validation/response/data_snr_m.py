"""Compare SNR-resolved image-sim m to a data response-weighted histogram."""

import argparse
import json
from pathlib import Path

import inputs
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sp_validation import sim_m_snr


def load_histogram(path, target_edges):
    """Load halo-clear data bin responses and validate their provenance and sums."""
    data = json.loads(path.read_text())
    if not np.array_equal(data["edges"], target_edges):
        raise ValueError("Data and simulation SNR edges differ")
    if not data["ensemble_response"]["available"]:
        raise ValueError("Data ensemble response is unavailable")
    if not data["weight_verification"]["w_des_is_exact_production_weight"]:
        raise ValueError("Data weights do not match the production table")
    bins = data["per_bin"]
    weights = np.asarray(bins["sum_w"], dtype=float)
    total = np.asarray(bins["R_total"], dtype=float)
    shear = np.asarray(bins["R_shear"], dtype=float)
    selection = np.asarray(bins["R_sel"], dtype=float)
    count = np.asarray(bins["N"], dtype=int)
    np.testing.assert_allclose(total, shear + selection, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(weights.sum(), data["totals"]["sum_w"], rtol=1e-12)
    if count.sum() != data["totals"]["selected"]:
        raise ValueError("Data bin counts do not sum to the selected population")
    return dict(
        count=count,
        sum_w=weights,
        total=total,
        full_response=np.asarray(data["totals"]["R_total"]),
        convention=data["convention"],
    )


def run(product_path, histogram, full_dir, output):
    """Save an effective-m summary and the data response-share plot."""
    product = sim_m_snr.read_product(product_path)
    hist = load_histogram(histogram, product["edges"])
    value, cov, share = sim_m_snr.effective_m(product, hist["sum_w"], hist["total"])
    weighted_response = np.average(hist["total"], axis=0, weights=hist["sum_w"])
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
        method="Halo-clear v2 data, fixed no-shear bin weights and measured total responses; data histogram and response covariance neglected.",
        data_product=str(histogram),
        data_bin_convention=hist["convention"],
        simulation_bin_convention="(lower, upper] intersected with production cuts",
        data_total_response=hist["total"].tolist(),
        weighted_bin_response=weighted_response.tolist(),
        direct_full_data_response=hist["full_response"].tolist(),
        weighted_minus_full_response=(
            weighted_response - hist["full_response"]
        ).tolist(),
        count=hist["count"].tolist(),
        sum_w=hist["sum_w"].tolist(),
        response_share=share.tolist(),
        effective_m=value.tolist(),
        covariance=cov.tolist(),
        full_m=mean.tolist(),
        full_error=err.tolist(),
        effective_minus_full=(value - mean).tolist(),
        difference_covariance=difference_cov.tolist(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "data_snr_exact.json").write_text(json.dumps(summary, indent=2))
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
    fig.suptitle("Image-sim m and halo-clear v2 data response shares")
    fig.tight_layout()
    fig.savefig(output / "snr_m.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(p)
    p.add_argument("--product-snr", type=Path, required=True)
    p.add_argument("--histogram", type=Path, required=True)
    p.add_argument("--production", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs.configure(args)
    run(args.product_snr, args.histogram, args.production, args.output)
