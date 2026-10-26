"""Exact branch-IV / DES and fixed/refit-W grid measurements (unblinded).

Run only in the container on an allocated compute node.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from numpy.lib import recfunctions

from sp_validation import calibration, catalog_builders, io, response

BASE = Path(
    "/automnt/n17data/mkilbing/astro/Runs/shapepipe/CFIS/v2.0/image_sims/new_workflow"
)
EXTERNAL = ("1p2z", "1m2z", "1z2p", "1z2m")
SUFFIX = dict(ns="NOSHEAR", p1="1P", m1="1M", p2="2P", m2="2M")


def load(grid, name, scheme, exact=True):
    directory = BASE / f"grid_{grid}/{name}_grid_{grid}"
    with open(directory / "config_mask.yaml") as f:
        cfg = yaml.safe_load(f)
    data = io.read_catalogue(directory / "shape_catalog_comprehensive_ngmix.fits")
    if scheme == "iv" and exact:
        # The HDF5 merge and extraction dropped errors, but tile finals kept them.
        missing = [
            f"NGMIX_G{i}_ERR_{s}" for s in ("1P", "1M", "2P", "2M") for i in (1, 2)
        ]
        errors = {k: np.full(len(data), np.nan) for k in missing}
        tile = np.char.strip(np.asarray(data["TILE_ID"]).astype(str))
        for label in np.unique(tile):
            idx = np.flatnonzero(tile == label)
            disk_label = f"{float(label):.3f}"
            paths = list(
                (directory / "product/tiles").glob(f"*/{disk_label}/final_cat-*.fits")
            )
            if not paths:
                raise FileNotFoundError(f"{directory}: tile {label!r}")
            path = paths[0]
            with fits.open(path) as hdus:
                raw = hdus[1].data
                order = np.argsort(raw["NUMBER"])
                pos = order[np.searchsorted(raw["NUMBER"][order], data["NUMBER"][idx])]
                np.testing.assert_array_equal(raw["NUMBER"][pos], data["NUMBER"][idx])
                for k in ("NGMIX_G1_NOSHEAR", "NGMIX_G2_ERR_NOSHEAR"):
                    np.testing.assert_allclose(
                        raw[k][pos], data[k][idx], equal_nan=True
                    )
                for k in missing:
                    errors[k][idx] = raw[k][pos]
        data = recfunctions.append_fields(
            np.asarray(data), list(errors), list(errors.values()), usemask=False
        )
    masks, _ = catalog_builders.get_masks_from_config(cfg, data, verbose=False)
    pre = catalog_builders.Mask.from_list(masks, label="combined", verbose=False)._mask
    cm = cfg["metacal"].copy()
    cm.pop("global_R_weight", None)
    cm.update(weight_scheme=scheme, w_col=f"w_{scheme}", des_num_bins=20)
    gm = calibration.metacal_from_config(data, pre, cm)
    tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
    return gm, tiles


def calibrated_mean(gm, weights, keep=None):
    result = response.responses(
        gm.branches, gm.mask_dict, weights, gm.R_shear, gm._step, keep
    )
    return result["c_g"]


def covariance(x):
    delta = x - x.mean(axis=0)
    return (len(x) - 1) / len(x) * delta.T @ delta


def measure(grid, output):
    with open(BASE / f"grid_{grid}/results/m_bias_config.yaml") as f:
        gin = yaml.safe_load(f)["shear_amplitude"]
    measurements = {}
    common_labels = None
    for scheme in ("des", "iv"):
        means, deletes = {}, {}
        for name in EXTERNAL:
            gm, tiles = load(grid, name, scheme)
            labels = np.unique(tiles)
            if common_labels is None:
                common_labels = labels
            np.testing.assert_array_equal(labels, common_labels)
            policies = (
                {"des_fixed": gm.branch_weights}
                if scheme == "des"
                else {
                    "iv_exact": gm.branch_weights,
                    "iv_frozen": {
                        b: gm.branch_weights["ns"] for b in response.BRANCHES
                    },
                }
            )
            for policy, weights in policies.items():
                means.setdefault(policy, {})[name] = calibrated_mean(gm, weights)
                _, jk, _ = response.jackknife(
                    gm.branches, gm.mask_dict, weights, gm.R_shear, gm._step, tiles
                )
                deletes.setdefault(policy, {})[name] = jk["c_g"]
            if scheme == "des" and hasattr(gm, "refit_weights"):
                means.setdefault("des_refit", {})[name] = gm.c_g
                _, jk, _ = gm.response_jackknife(tiles, refit_w=True)
                deletes.setdefault("des_refit", {})[name] = jk["c_g"]
            print(grid, name, scheme, len(gm.ns["g1"]), flush=True)
        for policy in means:

            def mc(values):
                out = []
                for i, (p, m) in enumerate((("1p2z", "1m2z"), ("1z2p", "1z2m"))):
                    out.extend(
                        [
                            (values[p][..., i] - values[m][..., i]) / (2 * gin) - 1,
                            (values[p][..., i] + values[m][..., i]) / 2,
                        ]
                    )
                return np.array(out).T

            value, jk = mc(means[policy]), mc(deletes[policy])
            cov = covariance(jk)
            measurements[policy] = dict(
                value=value.tolist(),
                error=np.sqrt(np.diag(cov)).tolist(),
                covariance=cov.tolist(),
                deleted=jk.tolist(),
                labels=labels.tolist(),
            )
    diff = np.array(measurements["iv_exact"]["deleted"]) - np.array(
        measurements["iv_frozen"]["deleted"]
    )
    measurements["iv_exact_minus_frozen"] = dict(
        value=(
            np.array(measurements["iv_exact"]["value"])
            - np.array(measurements["iv_frozen"]["value"])
        ).tolist(),
        error=np.sqrt(np.diag(covariance(diff))).tolist(),
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / f"weights_grid_{grid}.json").write_text(
        json.dumps(measurements, indent=2)
    )


def check_production(grid, output):
    """Exercise FITS R_JK refit-W means through the production m estimator."""
    from sp_validation.image_sims import ImageSimMBias

    directory = output / f"production_grid_{grid}"
    with open(BASE / f"grid_{grid}/results/m_bias_config.yaml") as f:
        cfg = yaml.safe_load(f)
    for name in ("1z2z",) + EXTERNAL:
        gm, tiles = load(grid, name, "des")
        label = f"{name}_grid_{grid}"
        path = directory / label / "shape_catalog_cut_ngmix.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        source = (
            Path("/n23data1/mkilbing/imsim_wdes_response")
            / f"grid_{grid}/des_d/{label}/{path.name}"
        )
        with fits.open(source) as hdus:
            fits.HDUList([hdus[0].copy(), hdus[1].copy()]).writeto(path, overwrite=True)
        calibration.get_calibrated_m_c(gm, False)
        calibration.save_response_product(path, gm, tiles)
    estimator = ImageSimMBias(
        {**cfg, "grids_dir": str(directory), "w_cols": ["w_des"], "pair_match": False}
    )
    estimator.load_catalogs(False)
    result = estimator.run(False)
    direct = json.loads((output / f"weights_grid_{grid}.json").read_text())["des_refit"]
    for j, key in enumerate(("m1", "c1", "m2", "c2")):
        np.testing.assert_allclose(result[key], direct["value"][j], atol=1e-13)
        np.testing.assert_allclose(result[key + "_err"], direct["error"][j], atol=1e-13)
    np.testing.assert_allclose(
        result["mc_covariance"], direct["covariance"], atol=1e-13
    )
    (output / f"production_grid_{grid}.json").write_text(json.dumps(result, indent=2))
    print("production matches direct refit-W:", grid, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--production-check", action="store_true")
    a = p.parse_args()
    if a.production_check:
        check_production(a.grid, a.output)
    else:
        measure(a.grid, a.output)
