"""Validate DES case d on Martin's unblinded grids (compute node only).

python scripts/validation/validate_response.py --grid 1 --output /path/to/results
The input shear comes from the grid manifest-derived m_bias_config.yaml.
No blinded catalogue is accepted by this script.
"""

import argparse
import json
import resource
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from astropy.io import fits

from sp_validation import calibration, catalog_builders, io
from sp_validation.catalog import match_catalogs_radec
from sp_validation.response_diagnostics import pair_products, plot_response


def run(grid, output):
    start = time.monotonic()
    base = (
        Path(
            "/automnt/n17data/mkilbing/astro/Runs/shapepipe/CFIS/v2.0/image_sims/new_workflow"
        )
        / f"grid_{grid}"
    )
    reference = Path("/n23data1/mkilbing/imsim_wdes_response") / f"grid_{grid}/des_d"
    output.mkdir(parents=True, exist_ok=True)
    with open(base / "results/m_bias_config.yaml") as fh:
        cfg = yaml.safe_load(fh)
    g_in = cfg["shear_amplitude"]
    products = {}
    summary = {
        "grid": grid,
        "injected_shear": g_in,
        "branches": {},
        "sp_validation_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "weight_scheme": "des",
        "W_policy": "fixed",
        "metacal_step": 0.01,
        "grid_bins": 20,
        "mean_subtraction": False,
        "paired_match_radius_deg": cfg["match_radius_deg"],
    }
    for name in ("1z2z", "1p2z", "1m2z", "1z2p", "1z2m"):
        branch_start = time.monotonic()
        label = f"{name}_grid_{grid}"
        directory = base / label
        with open(directory / "config_mask.yaml") as fh:
            mask_config = yaml.safe_load(fh)
        data = io.read_catalogue(directory / "shape_catalog_comprehensive_ngmix.fits")
        masks, _ = catalog_builders.get_masks_from_config(
            mask_config, data, verbose=False
        )
        pre = catalog_builders.Mask.from_list(
            masks, label="combined", verbose=False
        )._mask
        cm = mask_config["metacal"].copy()
        cm.pop("global_R_weight", None)
        cm.update(weight_scheme="des", w_col="w_des", des_num_bins=20)
        gm = calibration.metacal_from_config(data, pre, cm)
        shear, raw, w, selected, _, _ = calibration.get_calibrated_m_c(gm, False)
        tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
        labels, jk, _ = gm.response_jackknife(tiles)
        cut = data[pre][selected]
        groups = tiles[selected]
        outdir = output / label
        outdir.mkdir(exist_ok=True)
        path = outdir / "shape_catalog_cut_ngmix.fits"
        refpath = reference / label / path.name
        with fits.open(refpath) as hdus:
            np.testing.assert_allclose(raw[0], hdus[1].data["e1_uncal"], atol=1e-12)
            np.testing.assert_allclose(raw[1], hdus[1].data["e2_uncal"], atol=1e-12)
            refR = np.array(
                [[hdus[0].header[f"R_{i}{j}"] for j in (1, 2)] for i in (1, 2)]
            )
            refRJ = np.array(
                [[hdus[0].header[f"RJ_{i}{j}"] for j in (1, 2)] for i in (1, 2)]
            )
            w_diff = np.max(abs(w / hdus[1].data["w_des"] - 1))
            hdus[1].data["e1"] = shear[0]
            hdus[1].data["e2"] = shear[1]
            hdus[1].data["w_iv"] = gm.ns["w_iv"][selected]
            # Keep only the table/header; write the new uncertainty product below.
            fits.HDUList([hdus[0].copy(), hdus[1].copy()]).writeto(path, overwrite=True)
        with fits.open(path, mode="update") as hdus:
            for i in range(2):
                for j in range(2):
                    for prefix, val in [
                        ("R", gm.R),
                        ("R_G", gm.R_shear_global),
                        ("R_S", gm.R_selection),
                    ]:
                        hdus[0].header[
                            f"{prefix}{'_' if prefix == 'R' else ''}{i + 1}{j + 1}"
                        ] = val[i, j]
        calibration.save_response_product(path, gm, tiles)
        means = np.array(
            [
                np.average(raw[:, groups != t], axis=1, weights=w[groups != t])
                for t in labels
            ]
        )
        g_jk = np.linalg.solve(jk["total"], means[..., None])[..., 0]
        products[name] = dict(
            g=shear,
            raw=raw,
            w=w,
            groups=groups,
            labels=labels,
            g_jk=g_jk,
            ra=cut["RA"],
            dec=cut["Dec"],
        )
        summary["branches"][name] = dict(
            n=len(w),
            R=gm.R.tolist(),
            RJ=gm.R_joint.tolist(),
            max_abs_R_difference=float(np.max(abs(gm.R - refR))),
            max_abs_RJ_difference=float(np.max(abs(gm.R_joint - refRJ))),
            max_relative_W_difference=float(w_diff),
            branch_counts=gm.branch_counts,
            branch_migration=gm.branch_migration,
            branch_weight_sums={k: float(v) for k, v in gm.branch_weight_sums.items()},
            seconds=time.monotonic() - branch_start,
        )
        if name == "1z2z":
            plot_response(gm, tiles, cut["RA"], cut["Dec"], outdir)
            pair = pair_products(
                cut["RA"], cut["Dec"], w, gm.R_shear, min_sep=1, max_sep=120, nbins=8
            )
            np.savez(outdir / "pair_products.npz", **pair)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for k, comp in enumerate((0, 3)):
                ratio = (
                    pair["products"][:, comp, comp]
                    / pair["global_products"][comp, comp]
                    - 1
                )
                axes[0].plot(pair["theta"], ratio, "o-", label=("11", "22")[k])
            axes[0].axhline(0, color="k", lw=0.5)
            axes[0].set(
                xscale="log",
                xlabel="Separation [arcmin]",
                ylabel="Pair Rg product / global product − 1",
            )
            axes[0].legend()
            for i, j in ((0, 1), (0, 2), (3, 1), (3, 2)):
                axes[1].plot(
                    pair["theta"],
                    pair["products"][:, i, j] - pair["global_products"][i, j],
                    "o-",
                    label=f"{i},{j}",
                )
            axes[1].set(
                xscale="log",
                xlabel="Separation [arcmin]",
                ylabel="Off-diagonal pair − global product",
            )
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(output / "pair_response.png", dpi=160)
            plt.close(fig)
            summary["stage1"] = dict(
                note="Shear-only; selection/weight derivatives omitted. No total-response adequacy verdict.",
                max_diagonal_fraction=float(
                    np.nanmax(
                        abs(
                            pair["products"][:, [0, 3], [0, 3]]
                            / pair["global_products"][[0, 3], [0, 3]]
                            - 1
                        )
                    )
                ),
            )
    with open(reference / "results/m_bias_results.yaml") as fh:
        martin = yaml.safe_load(fh)["weights"]["w_des"]
    summary["m"] = {}
    for comp, (p, m) in enumerate((("1p2z", "1m2z"), ("1z2p", "1z2m"))):
        plus, minus = products[p], products[m]
        means = [np.average(x["g"][comp], weights=x["w"]) for x in (plus, minus)]
        unpaired = (means[0] - means[1]) / (2 * g_in) - 1
        ip, im = match_catalogs_radec(
            plus["ra"],
            plus["dec"],
            minus["ra"],
            minus["dec"],
            thresh_deg=cfg["match_radius_deg"],
        )
        pairw = 0.5 * (plus["w"][ip] + minus["w"][im])
        paired = np.average(
            (plus["g"][comp, ip] - minus["g"][comp, im]) / (2 * g_in) - 1, weights=pairw
        )
        np.testing.assert_array_equal(plus["labels"], minus["labels"])
        m_jk = (plus["g_jk"][:, comp] - minus["g_jk"][:, comp]) / (2 * g_in) - 1
        error = np.sqrt((len(m_jk) - 1) / len(m_jk) * np.sum((m_jk - m_jk.mean()) ** 2))
        summary["m"][f"m{comp + 1}"] = dict(
            martin_paired=martin[f"m{comp + 1}"],
            new_paired=float(paired),
            new_unpaired=float(unpaired),
            unpaired_fixed_W_tile_jk_error=float(error),
        )
    summary["seconds"] = time.monotonic() - start
    summary["max_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    with open(output / "validation.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.grid, args.output)
