"""Shared unblinded grid inputs, branch-error join and response helpers."""

from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from numpy.lib import recfunctions

from sp_validation import calibration, catalog_builders, io, response

BASE = Path(
    "/automnt/n17data/mkilbing/astro/Runs/shapepipe/CFIS/v2.0/image_sims/new_workflow"
)
REFERENCE = Path("/n23data1/mkilbing/imsim_wdes_response")
PRODUCT = dict(
    name="v2-d533a078",
    path=(
        "/n17data/cdaley/unions/validation/2026-10-26_response-425/"
        "products/des_weight_product-d533a078.fits"
    ),
)
EXTERNAL = ("1p2z", "1m2z", "1z2p", "1z2m")


def add_arguments(parser):
    parser.add_argument("--base", type=Path, default=BASE)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--product", type=Path, default=Path(PRODUCT["path"]))
    parser.add_argument("--product-name", default=PRODUCT["name"])


def configure(args):
    global BASE, REFERENCE, PRODUCT
    BASE, REFERENCE = args.base, args.reference
    PRODUCT = dict(name=args.product_name, path=str(args.product))


def load(grid, name, scheme, exact=True, min_count=None):
    directory = BASE / f"grid_{grid}/{name}_grid_{grid}"
    with open(directory / "config_mask.yaml") as f:
        cfg = yaml.safe_load(f)
    data = io.read_catalogue(directory / "shape_catalog_comprehensive_ngmix.fits")
    if scheme == "iv" and exact:
        # Tile finals supply branch errors absent from the comprehensive inputs.
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
            with fits.open(paths[0]) as hdus:
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
    if min_count is not None:
        cm["des_min_count"] = min_count
    gm = calibration.metacal_from_config(data, pre, cm)
    tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
    return gm, tiles


def covariance(x):
    delta = x - x.mean(axis=0)
    return (len(x) - 1) / len(x) * delta.T @ delta


def table_weights(gm, table):
    return {
        b: response.evaluate_w(
            table,
            getattr(gm, b)["flux"] / getattr(gm, b)["flux_err"],
            getattr(gm, b)["T"] / getattr(gm, b)["Tpsf"],
        )
        for b in response.BRANCHES
    }
