"""Measure the full-matrix shear/selection calibration split on unblinded grids.

Run inside the sp_validation container on a Slurm compute node. All writes go
under --output. Inputs follow the sibling response validation scripts.
The numerator is held identical between the shear-only and total calibration.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import inputs
import matplotlib.pyplot as plt
from inputs import EXTERNAL

from sp_validation import calibration, catalog_builders, image_sim_config, io, response
from sp_validation.image_sims import ImageSimMBias

CONFIGS = ("selection_des", "selection_unweighted", "no_selection_unweighted")
METRICS = (
    "R_shear",
    "R_sel",
    "m_shear_only",
    "m",
    "delta_m_sel",
    "N_slope",
    "scalar_m_shear_only",
    "scalar_m",
    "scalar_delta_m_sel",
)
ORDER = [f"{metric}_{i}" for metric in METRICS for i in (1, 2)]


def covariance(draws):
    delta = draws - draws.mean(axis=0)
    return (len(draws) - 1) / len(draws) * delta.T @ delta


def package(value, draws, labels):
    cov = covariance(draws)
    return dict(
        value=np.asarray(value).tolist(),
        error=np.sqrt(np.maximum(0, np.diag(cov))).tolist(),
        covariance=cov.tolist(),
        deleted=draws.tolist(),
        labels=labels.tolist(),
        order=ORDER,
    )


def estimator(config, cats, matrix):
    """Use ImageSimMBias.run, supplying the same in-memory modern catalogue fields
    its FITS loader supplies; swap only the applied response for the split.
    """
    est = ImageSimMBias({**config, "w_cols": ["none"], "pair_match": False})
    est.cats = {}
    for name, cat in cats.items():
        r = cat["full"][matrix]
        jr = cat["jk"][matrix]
        calibrated = np.linalg.solve(r, cat["raw"])
        est.cats[name] = dict(
            weight_scheme="des",
            W_policy="fixed-external",
            e1=calibrated[0],
            e2=calibrated[1],
            raw=cat["raw"],
            w={"none": cat["weights"]},
            tiles=cat["tiles"],
            jk_labels=cat["labels"],
            jk_R=jr,
        )
    return est.run(False)


def run_grid(grid, output, production=None):
    cfg_m = yaml.safe_load(
        (inputs.BASE / f"grid_{grid}/results/m_bias_config.yaml").read_text()
    )
    all_cats = {key: {} for key in CONFIGS}
    reference_labels = None
    parent_configs = {}
    # Only four externally sheared branches enter m; no-external-shear catalogue
    # is not used by ImageSimMBias.run for the modern full-population estimator.
    for name in EXTERNAL:
        directory = inputs.BASE / f"grid_{grid}/{name}_grid_{grid}"
        original = yaml.safe_load((directory / "config_mask.yaml").read_text())
        parent_configs[name] = original
        data = io.read_catalogue(directory / "shape_catalog_comprehensive_ngmix.fits")
        masks, _ = catalog_builders.get_masks_from_config(original, data, verbose=False)
        pre = catalog_builders.Mask.from_list(
            masks, label="combined", verbose=False
        )._mask
        tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
        for key in CONFIGS:
            cfg = image_sim_config.calibration_config(
                original, ["w_des"], source="external", product=inputs.PRODUCT
            )
            if key == "no_selection_unweighted":
                cm = cfg["metacal"]
                cm.update(
                    gal_snr_min=-np.inf,
                    gal_snr_max=np.inf,
                    gal_rel_size_min=-np.inf,
                    gal_rel_size_max=np.inf,
                )
            gm = calibration.metacal_from_config(data, pre, cfg["metacal"])
            if key != "selection_des":
                # Retain selection and per-object responses; use unit weights.
                gm.branch_weights = {b: np.ones(len(tiles)) for b in response.BRANCHES}
            full = gm.response_estimates()
            labels, jk, _ = gm.response_jackknife(tiles, refit_w=False)
            if reference_labels is None:
                reference_labels = labels
            np.testing.assert_array_equal(labels, reference_labels)
            idx = gm.mask_dict["ns"]
            raw = np.array(gm.branches["ns"])[:, idx]
            all_cats[key][name] = dict(
                full=full,
                jk=jk,
                labels=labels,
                raw=raw,
                weights=gm.branch_weights["ns"][idx],
                tiles=tiles[idx],
                counts={b: len(gm.mask_dict[b]) for b in response.BRANCHES},
                parent_count=len(tiles),
                nonfinite_coordinates={
                    b: int(
                        np.sum(
                            ~np.isfinite(
                                getattr(gm, b)["flux"] / getattr(gm, b)["flux_err"]
                            )
                            | ~np.isfinite(getattr(gm, b)["T"] / getattr(gm, b)["Tpsf"])
                        )
                    )
                    for b in response.BRANCHES
                },
            )
            print(grid, name, key, all_cats[key][name]["counts"], flush=True)
            del gm
        del data
    result = dict(grid=grid, parent_configs=parent_configs, configurations={})
    for key, cats in all_cats.items():
        total = estimator(cfg_m, cats, "total")
        shear = estimator(cfg_m, cats, "shear")
        value = np.empty((len(METRICS), 2))
        draws = np.empty((len(reference_labels), len(METRICS), 2))
        for i, (p, m) in enumerate((EXTERNAL[:2], EXTERNAL[2:])):
            for dst, field in ((0, "shear"), (1, "selection")):
                value[dst, i] = (
                    cats[p]["full"][field][i, i] + cats[m]["full"][field][i, i]
                ) / 2
                draws[:, dst, i] = (
                    cats[p]["jk"][field][:, i, i] + cats[m]["jk"][field][:, i, i]
                ) / 2
            value[2, i], value[3, i] = shear[f"m{i + 1}"], total[f"m{i + 1}"]
            draws[:, 2, i] = np.array(shear["mc_jackknife"])[:, 2 * i]
            draws[:, 3, i] = np.array(total["mc_jackknife"])[:, 2 * i]
            value[4, i] = value[3, i] - value[2, i]
            draws[:, 4, i] = draws[:, 3, i] - draws[:, 2, i]
            gin = cfg_m["shear_amplitude"]
            value[5, i] = (cats[p]["full"]["c_e"][i] - cats[m]["full"]["c_e"][i]) / (
                2 * gin
            )
            draws[:, 5, i] = (
                cats[p]["jk"]["c_e"][:, i] - cats[m]["jk"]["c_e"][:, i]
            ) / (2 * gin)
            for dst, field in ((6, "shear"), (7, "total")):
                value[dst, i] = (
                    cats[p]["full"]["c_e"][i] / cats[p]["full"][field][i, i]
                    - cats[m]["full"]["c_e"][i] / cats[m]["full"][field][i, i]
                ) / (2 * gin) - 1
                draws[:, dst, i] = (
                    cats[p]["jk"]["c_e"][:, i] / cats[p]["jk"][field][:, i, i]
                    - cats[m]["jk"]["c_e"][:, i] / cats[m]["jk"][field][:, i, i]
                ) / (2 * gin) - 1
            value[8, i] = value[7, i] - value[6, i]
            draws[:, 8, i] = draws[:, 7, i] - draws[:, 6, i]
        entry = package(
            value.ravel(), draws.reshape(len(reference_labels), -1), reference_labels
        )
        entry["production_estimator"] = total
        entry["branch_responses"] = {
            name: dict(
                full={k: v.tolist() for k, v in c["full"].items()},
                deleted={k: v.tolist() for k, v in c["jk"].items()},
                counts=c["counts"],
                parent_count=c["parent_count"],
                nonfinite_coordinates=c["nonfinite_coordinates"],
            )
            for name, c in cats.items()
        }
        result["configurations"][key] = entry
    if production is not None:
        reference = json.loads((production / f"production_{grid}.json").read_text())
        total = result["configurations"]["selection_des"]["production_estimator"]
        for name in ("m1", "m2", "m1_err", "m2_err", "mc_covariance", "mc_jackknife"):
            np.testing.assert_allclose(total[name], reference[name], rtol=0, atol=1e-12)
        result["production_reproduction_max_abs"] = max(
            abs(total[k] - reference[k]) for k in ("m1", "m2", "m1_err", "m2_err")
        )
    (output / f"grid_{grid}.json").write_text(json.dumps(result, indent=2))
    return result


def combine(grids, key, coherent=False):
    entries = [g["configurations"][key] for g in grids]
    values = np.array([e["value"] for e in entries])
    covs = np.array([e["covariance"] for e in entries])
    variances = np.diagonal(covs, axis1=1, axis2=2)
    if coherent:
        variances = np.tile(variances[:, 6:8], (1, len(METRICS)))
    # Exact-zero selection response has zero covariance: use equal grid weights.
    w = np.divide(1, variances, out=np.zeros_like(variances), where=variances > 0)
    w[:, w.sum(axis=0) == 0] = 1
    w /= w.sum(axis=0)
    cov = np.sum(covs * w[:, :, None] * w[:, None, :], axis=0)
    return dict(
        value=np.sum(w * values, axis=0).tolist(),
        error=np.sqrt(np.maximum(0, np.diag(cov))).tolist(),
        covariance=cov.tolist(),
        grid_weights=w.tolist(),
        order=ORDER,
    )


def plot(results, output):
    labels = [
        "No SNR/size cuts\nUnweighted",
        "SNR/size cuts\nUnweighted",
        "SNR/size cuts\nDES (v2 table)",
    ]
    keys = CONFIGS[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for i, ax in enumerate(axes):
        for offset, metric, color, marker, label in (
            (-0.20, 2, "#0072B2", "o", "Shear-response only"),
            (0, 4, "#D55E00", "s", "Selection contribution"),
            (0.20, 3, "#222222", "D", "Total"),
        ):
            vals = [
                results["combined_common_m_weights"][k]["value"][2 * metric + i] * 1000
                for k in keys
            ]
            errs = [
                results["combined_common_m_weights"][k]["error"][2 * metric + i] * 1000
                for k in keys
            ]
            ax.errorbar(
                np.arange(3) + offset,
                vals,
                yerr=errs,
                fmt=marker,
                color=color,
                capsize=3,
                label=label,
                ms=5,
            )
        ax.axhline(0, color=".6", lw=0.8)
        ax.set_xticks(range(3), labels)
        ax.set_title(f"Component {i + 1}")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(r"Multiplicative bias $\times 10^3$")
    axes[0].legend(fontsize=8)
    fig.suptitle("Shear-only + selection contribution = total", fontsize=12)
    fig.text(
        0.5,
        0.01,
        "Four-grid means; common inverse-variance total-m weights preserve the split. Shared-tile jackknife errors.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(output / "m_split.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--production",
        type=Path,
        help="Check against existing per-grid production JSON",
    )
    parser.add_argument(
        "--reuse", action="store_true", help="Combine already measured grid files only"
    )
    args = parser.parse_args()
    inputs.configure(args)
    args.output.mkdir(parents=True, exist_ok=True)
    start_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    grids = [
        json.loads((args.output / f"grid_{i}.json").read_text())
        if args.reuse
        else run_grid(i, args.output, args.production)
        for i in range(1, 5)
    ]
    results = dict(
        metadata=dict(
            checkout_head_start=start_head,
            checkout_head_end=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            product=inputs.PRODUCT,
            product_sha256=hashlib.sha256(
                Path(inputs.PRODUCT["path"]).read_bytes()
            ).hexdigest(),
            assumption="Grids treated as independent, conditional on the named fixed external table; no external-table uncertainty or cross-grid tile covariance included.",
            combined="Each reported quantity has its own inverse-jackknife-variance grid weights. These marginal means need not sum. combined_common_m_weights uses total-m weights for all quantities, preserving m = m_shear_only + delta_m_sel and used in the figure.",
            response_summary="For component i, R_shear/R_sel are the average diagonal i,i across its positive and negative external branches. Every branch matrix and deletion is also saved; numerator calibration uses each branch matrix separately.",
            decomposition="Exact production full-matrix counterfactual: same raw weighted population, numerator and deletions; replace R_total by R_shear. Scalar diagonal-only branchwise counterparts are also saved. There is no unique physical numerator split; changing selection also changes N, so delta_m_sel isolates only the calibration-denominator effect within a configuration.",
            no_selection="Existing parent config dat cuts retained (FLAGS<=2, overlap=True, N_EPOCH>=2, mag 15..30, MCAL_TYPES_FAIL=0, no-shear PSF g1/g2 != -10); each artificial branch also requires NGMIX_FLAGS_branch=0. SNR/size bounds replaced by +/-infinity: NaN/infinite coordinates can still fail comparisons. Per-branch counts and nonfinite coordinate counts saved. Unit weights on all branches.",
        ),
        grids=grids,
        combined={k: combine(grids, k) for k in CONFIGS},
        combined_common_m_weights={k: combine(grids, k, True) for k in CONFIGS},
    )
    # Keep covariance between configurations too: all use identical tile labels.
    cross_order = [f"{key}:{name}" for key in CONFIGS for name in ORDER]
    cross_covariances = []
    for grid in grids:
        entries = grid["configurations"]
        joint_draws = np.concatenate(
            [np.asarray(entries[key]["deleted"]) for key in CONFIGS], axis=1
        )
        cross_cov = covariance(joint_draws)
        grid["cross_configuration_covariance"] = dict(
            order=cross_order, covariance=cross_cov.tolist()
        )
        cross_covariances.append(cross_cov)
        for key in CONFIGS:
            entry = entries[key]
            values = np.asarray(entry["value"])
            draws = np.asarray(entry["deleted"])
            np.testing.assert_allclose(
                values[8:10], values[6:8] - values[4:6], atol=1e-14
            )
            np.testing.assert_allclose(
                draws[:, 8:10], draws[:, 6:8] - draws[:, 4:6], atol=1e-14
            )
        # Confirm no hidden finite-coordinate or fit-flag rejection remains.
        for branch in entries["no_selection_unweighted"]["branch_responses"].values():
            assert all(n == branch["parent_count"] for n in branch["counts"].values())
            assert not any(branch["nonfinite_coordinates"].values())
    for method in ("combined", "combined_common_m_weights"):
        weights = np.concatenate(
            [np.asarray(results[method][key]["grid_weights"]) for key in CONFIGS],
            axis=1,
        )
        cov = np.sum(
            np.asarray(cross_covariances) * weights[:, :, None] * weights[:, None, :],
            axis=0,
        )
        results[f"{method}_cross_configuration_covariance"] = dict(
            order=cross_order, covariance=cov.tolist()
        )
    results["metadata"]["N_slope"] = (
        "Raw weighted external-shear slope: (A_plus(e_ns)-A_minus(e_ns))/(2*g_in). This is N/g in the question's notation."
    )
    results["metadata"]["no_selection_verified"] = (
        "Every retained parent row passes all five artificial branch masks; nonfinite SNR/size counts are zero on all four grids. R_sel and its deletion covariance are exactly zero."
    )
    v = np.array(results["combined"]["selection_des"]["value"])[6:8]
    e = np.array(results["combined"]["selection_des"]["error"])[6:8]
    print("Production combined m:", v, "+/-", e, flush=True)
    (args.output / "results.json").write_text(json.dumps(results, indent=2))
    plot(results, args.output)
    for key, entry in results["combined"].items():
        print(key, dict(zip(ORDER, zip(entry["value"], entry["error"]))), flush=True)


if __name__ == "__main__":
    main()
