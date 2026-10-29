"""Reproduce production ImageSimMBias with and without the FLAGS parent cut."""

import argparse
import json
from pathlib import Path

import inputs
import numpy as np
import yaml
from astropy.io import fits

from sp_validation import calibration, catalog_builders, image_sim_config, io
from sp_validation.image_sims import ImageSimMBias


def run(grid, output):
    """Save calibrated products and shared-tile m estimates for one grid."""
    for variant in ("base", "noflags"):
        directory = output / variant / f"grid_{grid}"
        counts = {}
        for name in inputs.EXTERNAL:
            label = f"{name}_grid_{grid}"
            source = inputs.BASE / f"grid_{grid}" / label
            cfg = yaml.safe_load((source / "config_mask.yaml").read_text())
            if variant == "noflags":
                cfg["dat"] = [c for c in cfg["dat"] if c["col_name"] != "FLAGS"]
            cfg = image_sim_config.calibration_config(
                cfg, ["w_des"], source="external", product=inputs.PRODUCT
            )
            data = io.read_catalogue(source / "shape_catalog_comprehensive_ngmix.fits")
            masks, _ = catalog_builders.get_masks_from_config(cfg, data, verbose=False)
            pre = catalog_builders.Mask.from_list(
                masks, label="combined", verbose=False
            )._mask
            gm = calibration.metacal_from_config(data, pre, cfg["metacal"])
            tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
            calibrated, raw, weights, sel, *_ = calibration.get_calibrated_m_c(
                gm, False
            )
            selected = np.flatnonzero(pre)[sel]
            counts[name] = dict(parent=int(pre.sum()), selected=len(selected))
            cols = [
                fits.Column(name=k, format="D", array=v)
                for k, v in (
                    ("RA", data["RA"][selected]),
                    ("Dec", data["Dec"][selected]),
                    ("e1", calibrated[0]),
                    ("e2", calibrated[1]),
                    ("e1_uncal", raw[0]),
                    ("e2_uncal", raw[1]),
                    ("w_des", weights),
                )
            ]
            labels = np.asarray(data["TILE_ID"][selected]).astype(str)
            cols.append(
                fits.Column(
                    name="TILE_ID", format=f"{max(map(len, labels))}A", array=labels
                )
            )
            path = directory / label / "shape_catalog_cut_ngmix.fits"
            path.parent.mkdir(parents=True, exist_ok=True)
            fits.HDUList(
                [fits.PrimaryHDU(), fits.BinTableHDU.from_columns(cols)]
            ).writeto(path, overwrite=True)
            calibration.save_response_product(path, gm, tiles)
            print(grid, variant, name, counts[name], flush=True)
        cfg = yaml.safe_load(
            (inputs.BASE / f"grid_{grid}/results/m_bias_config.yaml").read_text()
        )
        cfg.update(
            grids_dir=str(directory),
            w_cols=["w_des"],
            pair_match=False,
            branches=list(inputs.EXTERNAL),
        )
        estimator = ImageSimMBias(cfg)
        estimator.load_catalogs(False)
        result = estimator.run(False)
        (output / f"{variant}_{grid}.json").write_text(
            json.dumps({"result": result, "counts": counts}, indent=2)
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(p)
    p.add_argument("--grid", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    inputs.configure(args)
    args.output.mkdir(parents=True, exist_ok=True)
    run(args.grid, args.output)
