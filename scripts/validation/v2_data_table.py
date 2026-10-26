"""Measure production residual m with the named v2 data table on allocated nodes."""

import json
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from round2_weights import BASE, EXTERNAL

from sp_validation import calibration, catalog_builders, image_sim_config, io
from sp_validation.image_sims import ImageSimMBias

PRODUCT = dict(
    name="v2-d533a078",
    path="/n17data/cdaley/scratch/response-pr/v2-table/des_weight_product-d533a078.fits",
)
OUT = Path("/n17data/cdaley/scratch/response-pr/v2-production")


def run(grid):
    directory = OUT / f"grid_{grid}"
    clipping = {}
    for name in ("1z2z",) + EXTERNAL:
        label = f"{name}_grid_{grid}"
        source = BASE / f"grid_{grid}" / label
        cfg = yaml.safe_load((source / "config_mask.yaml").read_text())
        cfg = image_sim_config.calibration_config(
            cfg, ["w_des"], source="external", product=PRODUCT
        )
        data = io.read_catalogue(source / "shape_catalog_comprehensive_ngmix.fits")
        masks, _ = catalog_builders.get_masks_from_config(cfg, data, verbose=False)
        pre = catalog_builders.Mask.from_list(
            masks, label="combined", verbose=False
        )._mask
        gm = calibration.metacal_from_config(data, pre, cfg["metacal"])
        tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
        clipping[name] = {}
        for branch in gm.branches:
            d = getattr(gm, branch)
            sel = gm.mask_dict[branch]
            snr, size = (
                d["flux"][sel] / d["flux_err"][sel],
                d["T"][sel] / d["Tpsf"][sel],
            )
            es, er = gm.W[:2]
            outside = (snr < es[0]) | (snr > es[-1]) | (size < er[0]) | (size > er[-1])
            parent_snr = d["flux"] / d["flux_err"]
            parent_size = d["T"] / d["Tpsf"]
            parent_out = (
                (parent_snr < es[0])
                | (parent_snr > es[-1])
                | (parent_size < er[0])
                | (parent_size > er[-1])
            )
            clipping[name][branch] = dict(
                selected=len(snr),
                outside=int(outside.sum()),
                fraction=float(outside.mean()),
                parent=len(parent_snr),
                parent_outside=int(parent_out.sum()),
            )
        path = directory / label / "shape_catalog_cut_ngmix.fits"
        path.parent.mkdir(parents=True, exist_ok=True)
        template = (
            Path("/n23data1/mkilbing/imsim_wdes_response")
            / f"grid_{grid}/des_d"
            / label
            / path.name
        )
        calibrated, raw, weights, *_ = calibration.get_calibrated_m_c(gm, False)
        with fits.open(template) as hdus:
            product = fits.HDUList([hdus[0].copy(), hdus[1].copy()])
            for i in range(2):
                np.testing.assert_allclose(product[1].data[f"e{i + 1}_uncal"], raw[i])
                product[1].data[f"e{i + 1}"] = calibrated[i]
            product[1].data["w_des"] = weights
            product.writeto(path, overwrite=True)
        calibration.save_response_product(path, gm, tiles)
        print(grid, name, cfg, flush=True)
    cfg = yaml.safe_load((BASE / f"grid_{grid}/results/m_bias_config.yaml").read_text())
    est = ImageSimMBias(
        {**cfg, "grids_dir": str(directory), "w_cols": ["w_des"], "pair_match": False}
    )
    est.load_catalogs(False)
    prod = est.run(False)
    print(json.dumps(prod, indent=2), flush=True)
    (OUT / f"production_{grid}.json").write_text(json.dumps(prod, indent=2))
    (OUT / f"clipping_{grid}.json").write_text(json.dumps(clipping, indent=2))


if __name__ == "__main__":
    import sys

    OUT.mkdir(parents=True, exist_ok=True)
    run(int(sys.argv[1]))
