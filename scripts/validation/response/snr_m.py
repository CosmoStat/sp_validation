"""Measure fixed-DES, branch-selected SNR-bin image-simulation m per grid."""

import argparse
import json
from pathlib import Path

import inputs
import numpy as np
import yaml

from sp_validation import calibration, catalog_builders, image_sim_config, io, sim_m_snr


def run(grid, output, edges):
    """Measure one grid and save the full shared-tile deletions."""
    cats = {}
    for name in inputs.EXTERNAL:
        directory = inputs.BASE / f"grid_{grid}" / f"{name}_grid_{grid}"
        cfg = yaml.safe_load((directory / "config_mask.yaml").read_text())
        cfg = image_sim_config.calibration_config(
            cfg, ["w_des"], source="external", product=inputs.PRODUCT
        )
        data = io.read_catalogue(directory / "shape_catalog_comprehensive_ngmix.fits")
        masks, _ = catalog_builders.get_masks_from_config(cfg, data, verbose=False)
        pre = catalog_builders.Mask.from_list(
            masks, label="combined", verbose=False
        )._mask
        gm = calibration.metacal_from_config(data, pre, cfg["metacal"])
        tiles = np.char.strip(np.asarray(data["TILE_ID"][pre]).astype(str))
        cats[name] = (gm, tiles)
        print(grid, name, len(tiles), flush=True)
    cfg = yaml.safe_load(
        (inputs.BASE / f"grid_{grid}/results/m_bias_config.yaml").read_text()
    )
    result = sim_m_snr.grid_product(cats, cfg["shear_amplitude"], edges)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / f"snr_grid_{grid}.npz",
        **{k: result[k] for k in ("edges", "value", "covariance", "deleted", "labels")},
    )
    (output / f"snr_counts_{grid}.json").write_text(
        json.dumps(result["counts"], indent=2)
    )
    print(grid, result["value"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    inputs.add_arguments(parser)
    parser.add_argument("--grid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--edges", type=float, nargs="+", default=sim_m_snr.DEFAULT_SNR_EDGES
    )
    args = parser.parse_args()
    inputs.configure(args)
    run(args.grid, args.output, args.edges)
