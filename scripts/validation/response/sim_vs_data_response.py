"""Direct ns response and occupancy comparison; run on an allocated node only.

Use the shared exact-error join, selection and independent-table evaluation. Density matching uses the ns cell's
p_data/p_sim as a fixed parent-object multiplier on every artificial branch:
it changes the population without adding an artificial density-weight derivative.
DES W and IV errors still vary with each artificial branch as in production.
"""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import inputs
import matplotlib.pyplot as plt
import numpy as np
from inputs import load, table_weights

from sp_validation import calibration, response


def pack(gm, weights):
    r = response.responses(gm.branches, gm.mask_dict, weights, gm.R_shear, gm._step)
    return {
        k: dict(
            matrix=r[k].tolist(),
            diagonal=np.diag(r[k]).tolist(),
            mean_diagonal=float(np.trace(r[k]) / 2),
        )
        for k in ("shear", "selection", "total", "joint")
    }


def ratios(results):
    def r(a, b, kind):
        return results[a][kind]["mean_diagonal"] / results[b][kind]["mean_diagonal"]

    return {
        f"{kind}_{des}_over_{base}": r(des, base, kind)
        for kind in ("shear", "total")
        for des in ("des_v2", "des_self")
        for base in ("unweighted", "iv_squared")
    }


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    table, nmin = calibration.load_des_weight_product(inputs.PRODUCT)
    es, er, wd, data_count = table
    pd = data_count / data_count.sum()
    result = dict(
        product=inputs.PRODUCT,
        data_count=data_count.tolist(),
        edges_snr=es.tolist(),
        edges_size=er.tolist(),
        density_matching="Fixed ns-cell p_data/p_sim on parent objects across all artificial branches; DES table fixed, not refitted after matching.",
        grids={},
    )
    maps, cells, fractions = [], [], []
    for grid in range(1, 5):
        # Capture legacy unsquared weights on the same joined input arrays.
        old = {}
        original = calibration.metacal.get_variance_ivweights
        names = dict(NOSHEAR="ns", **{"1P": "p1", "1M": "m1", "2P": "p2", "2M": "m2"})

        def capture(data, sigma_eps, mask=None, legacy=False, branch="NOSHEAR"):
            old[names[branch]] = original(data, sigma_eps, mask, True, branch)[2]
            return original(data, sigma_eps, mask, legacy, branch)

        with patch.object(
            calibration.metacal, "get_variance_ivweights", staticmethod(capture)
        ):
            gm, tiles = load(grid, "1z2z", "iv", exact=True)
        own, _ = load(grid, "1z2z", "des", min_count=3)
        for b in response.BRANCHES:
            np.testing.assert_array_equal(gm.mask_dict[b], own.mask_dict[b])
        np.testing.assert_allclose(gm.R_shear, own.R_shear)
        np.testing.assert_array_equal(es, own.W[0])
        np.testing.assert_array_equal(er, own.W[1])
        policies = dict(
            unweighted={b: np.ones(len(gm.ns["g1"])) for b in response.BRANCHES},
            iv_squared=gm.branch_weights,
            iv_unsquared=old,
            des_v2=table_weights(gm, table),
            des_self=table_weights(gm, own.W),
        )
        ns = gm.mask_dict["ns"]
        snr = gm.ns["flux"] / gm.ns["flux_err"]
        size = gm.ns["T"] / gm.ns["Tpsf"]
        index = response.cell_index(snr, es) * len(wd[0]) + response.cell_index(
            size, er
        )
        shape = wd.shape
        count = np.bincount(index[ns], minlength=wd.size).reshape(shape)
        assert np.all(count > 0), "Data occupancy cannot be matched in empty sim cells"
        ps = count / count.sum()
        factor = pd / ps
        density = factor.ravel()[index]
        before = {k: pack(gm, w) for k, w in policies.items()}
        after = {
            k: pack(gm, {b: w[b] * density for b in response.BRANCHES})
            for k, w in policies.items()
        }
        rg = (gm.R_shear[0, 0] + gm.R_shear[1, 1]) / 2
        rsum = np.bincount(index[ns], weights=rg, minlength=wd.size).reshape(shape)
        rcell = rsum / count
        outside = (
            (snr[ns] < es[0])
            | (snr[ns] > es[-1])
            | (size[ns] < er[0])
            | (size[ns] > er[-1])
        )
        assert not outside.any()
        # Check exact occupancy match and direct cell-sum shear identity.
        matched = np.bincount(
            index[ns], weights=density[ns], minlength=wd.size
        ).reshape(shape)
        np.testing.assert_allclose(matched / matched.sum(), pd, atol=1e-13)
        np.testing.assert_allclose(
            np.sum(pd * rcell), after["unweighted"]["shear"]["mean_diagonal"]
        )
        summaries = dict(
            median_snr=float(np.median(snr[ns])),
            median_size=float(np.median(size[ns])),
            fraction_snr_below_30=float(np.mean(snr[ns] < 30)),
            occupancy_total_variation=float(np.abs(ps - pd).sum() / 2),
            density_factor_min=float(factor.min()),
            density_factor_max=float(factor.max()),
            density_effective_n=float(
                density[ns].sum() ** 2 / np.sum(density[ns] ** 2)
            ),
            per_cell_R_min=float(rcell.min()),
            per_cell_R_max=float(rcell.max()),
        )
        result["grids"][str(grid)] = dict(
            selected=len(ns),
            tiles=len(np.unique(tiles)),
            sigma_eps=gm._sigma_eps,
            responses=before,
            ratios=ratios(before),
            occupancy_matched_responses=after,
            occupancy_matched_ratios=ratios(after),
            count=count.tolist(),
            sim_fraction=ps.tolist(),
            sim_over_data_fraction=(ps / pd).tolist(),
            mean_R_cell=rcell.tolist(),
            mean_R_snr=(rsum.sum(1) / count.sum(1)).tolist(),
            mean_R_size=(rsum.sum(0) / count.sum(0)).tolist(),
            summaries=summaries,
        )
        maps.append(ps / pd)
        cells.append(rcell)
        fractions.append(ps)
        print(
            "GRID", grid, json.dumps(result["grids"][str(grid)]["ratios"]), flush=True
        )
        print("MATCHED", grid, json.dumps(ratios(after)), flush=True)
        del gm, own
    (output / "results.json").write_text(json.dumps(result, indent=2))

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for i, ratio in enumerate(maps):
        ax = axes[0, i]
        im = ax.pcolormesh(
            es, er, np.log10(ratio).T, cmap="RdBu_r", vmin=-1.5, vmax=1.5
        )
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="SNR",
            ylabel="T / T_PSF,reconv",
            title=f"Grid {i + 1}: selected sim / data fraction",
        )
    fig.colorbar(im, ax=axes[0], label="log10 occupancy fraction ratio")
    cs, cr = np.sqrt(es[:-1] * es[1:]), np.sqrt(er[:-1] * er[1:])
    for j in (0, 1):
        ax = axes[1, j]
        coords = cs if j == 0 else cr
        axis = 1 if j == 0 else 0
        ax.plot(coords, pd.sum(axis), color="black", lw=2, label="v2 W_COUNT")
        for i, ps in enumerate(fractions):
            ax.plot(coords, ps.sum(axis), label=f"Grid {i + 1}")
        ax.set(
            xscale="log",
            xlabel="SNR" if j == 0 else "T / T_PSF,reconv",
            ylabel="Fraction per log bin",
        )
        ax.legend(fontsize=8)
        ax = axes[1, j + 2]
        for i, ps in enumerate(fractions):
            ax.plot(coords, ps.sum(axis) / pd.sum(axis), label=f"Grid {i + 1}")
        ax.axhline(1, color="black", ls="--")
        ax.set(
            xscale="log",
            xlabel="SNR" if j == 0 else "T / T_PSF,reconv",
            ylabel="Sim / data marginal fraction",
        )
    fig.savefig(output / "occupancy.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for i, rc in enumerate(cells):
        ax = axes[0, i]
        im = ax.pcolormesh(es, er, rc.T, cmap="viridis", vmin=0.2, vmax=1.1)
        ax.set(
            xscale="log",
            yscale="log",
            xlabel="SNR",
            ylabel="T / T_PSF,reconv",
            title=f"Grid {i + 1}: unweighted cell mean R_g",
        )
        for j, (coords, key) in enumerate(((cs, "mean_R_snr"), (cr, "mean_R_size"))):
            axes[1, j].plot(
                coords, result["grids"][str(i + 1)][key], label=f"Grid {i + 1}"
            )
    fig.colorbar(im, ax=axes[0], label="Mean (R_g11 + R_g22) / 2")
    for j in (0, 1):
        axes[1, j].set(
            xscale="log",
            xlabel="SNR" if j == 0 else "T / T_PSF,reconv",
            ylabel="Unweighted mean R_g",
        )
        axes[1, j].legend()
    for j, base in enumerate(("unweighted", "iv_squared")):
        ax = axes[1, j + 2]
        kind = "shear" if j == 0 else "total"
        key = f"{kind}_des_v2_over_{base}"
        for name, field, marker in (
            ("Original", "ratios", "o"),
            ("Data occupancy", "occupancy_matched_ratios", "s"),
        ):
            ax.plot(
                range(1, 5),
                [result["grids"][str(i)][field][key] for i in range(1, 5)],
                marker=marker,
                label=name,
            )
        ax.axhline(
            0.734 / 0.602 if j == 0 else (0.734 - 0.022) / (0.610 - 0.002),
            color="black",
            ls="--",
            label="v2 data",
        )
        ax.set(
            xlabel="Grid",
            ylabel="R_shear DES / unw" if j == 0 else "R_total DES / IV squared",
            xticks=range(1, 5),
        )
        ax.legend(fontsize=8)
    fig.savefig(output / "per_cell_response.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(p)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs.configure(args)
    run(args.output)
