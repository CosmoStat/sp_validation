"""Run an n(z) recipe: SOM direct calibration of a shear catalogue's bins.

``prepare`` builds the two inputs (:mod:`sp_validation.nz_inputs`); ``run``
trains the SOM on the calibrators, groups its cells, assigns every source,
selects gold groups on the sources themselves, and writes the products:

- ``nz_tomo.txt``: z and one density per bin, integrating to the bin's share of
  the retained shear weight (cosmo_val's ``shear.redshift_path``);
- ``nz_nontomo.txt``: their sum, the n(z) of every ``TOM_BIN_ID > 0`` row;
- ``tom_bin.parquet``: ``TOM_BIN_ID`` and ``ZBAND`` row-aligned with the
  catalogue, with its ``TILE_ID`` and ``NUMBER``;
- ``realisations.npz``: draws of every bin at once (:mod:`nz_ensemble`), from
  ``ensemble_inputs.npz`` (per-group source tables, their response to
  magnitude offsets, and the calibrators' BPZ gradients);
- ``calibration_arrays.npz``, ``som_model.npz``, ``summary.json`` and the recipe.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import nz_calibration as nz
from .data_roots import load_yaml


def prepare(config, workers=1):
    """Write the calibrator and source files the recipe names."""
    from . import nz_inputs

    inputs = config["inputs"]
    cal = inputs["calibrators"]
    np.savez(
        cal["output"],
        **nz_inputs.calibrators(
            cal["specz_dir"],
            cal["compilation"],
            cal["surveys"],
            cal["z_min"],
            config["z2_offset"],
        ),
    )
    src = inputs["sources"]
    np.savez(
        src["output"],
        **nz_inputs.sources(
            src["catalogue"], src["shapes"], src["photometry"], workers=workers
        ),
    )


def output_grid(config):
    grid = config["grid"]
    return np.linspace(
        grid["min"], grid["max"], round((grid["max"] - grid["min"]) / grid["step"]) + 1
    )


def run(config, output, *, workers=1):
    """Calibrate the recipe's catalogue and write its n(z) products to ``output``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cal = np.load(config["inputs"]["calibrators"]["output"])
    src = np.load(config["inputs"]["sources"]["output"])
    edges = np.asarray(config["edges"], float)
    grid = output_grid(config)
    n_groups = config["groups"]

    model = nz.train(cal["mag"], workers=workers, **config["som"])
    model["group_map"] = nz.group_cells(model, n_groups)
    model["edges"] = edges
    _, cal_cell = nz.assign(cal["mag"], model, workers=workers)
    cal_group = model["group_map"][cal_cell]

    weight, photoz = src["weight"], src["photoz"]
    prelim, cell, channel = nz.source_groups(
        src["mag"],
        photoz,
        weight,
        model,
        offset=config["z2_offset"],
        workers=workers,
    )
    group = np.where(cell >= 0, model["group_map"][np.maximum(cell, 0)], -1)
    threshold = nz.gold_threshold(cal["z"], cal["photoz"], **config["gold_qc"])
    est = nz.calibrate(
        prelim,
        group,
        weight,
        photoz,
        cal["z"],
        cal["photoz"],
        cal_group,
        edges,
        n_groups,
        grid,
        threshold=threshold,
    )
    tom = nz.tom_bin_ids(prelim, group, est["gold"])
    total_weight = float(weight.sum())
    for b in est["bins"]:
        b["weight_fraction_of_all"] = b["weight_sum"] / total_weight

    fractions = [b["weight_sum"] for b in est["bins"]]
    nz.write_nz(output / "nz_tomo.txt", grid, est["bin_hists"], fractions)
    mixture = (np.asarray(fractions) / np.sum(fractions)) @ est["bin_hists"]
    nz.write_nz(output / "nz_nontomo.txt", grid, mixture[None], [1.0])
    pq.write_table(
        pa.table(
            dict(
                TILE_ID=src["tile"],
                NUMBER=src["number"],
                TOM_BIN_ID=tom,
                ZBAND=channel,
            )
        ),
        output / "tom_bin.parquet",
    )
    np.savez(output / "som_model.npz", **model)
    eligible = cell >= 0
    n_cells = len(model["code"])
    np.savez_compressed(
        output / "calibration_arrays.npz",
        z=cal["z"],
        cal_photoz=cal["photoz"],
        cal_cell=cal_cell,
        cal_group=cal_group,
        grid=grid,
        edges=edges,
        gold=est["gold"],
        cal_weights=est["cal_weights"],
        bin_hists=est["bin_hists"],
        source_weight_cell=np.bincount(
            cell[eligible], weights=weight[eligible], minlength=n_cells
        ),
        source_photoz_cell=np.bincount(
            cell[eligible],
            weights=weight[eligible] * photoz[eligible],
            minlength=n_cells,
        ),
    )
    complete = nz.detected(src["mag"][:, :4]).all(1) & (channel > 0) & (weight > 0)
    summary = dict(
        name=config["name"],
        limitation=config["limitation"],
        bins=est["bins"],
        n_rows=len(weight),
        n_calibrators=len(cal["z"]),
        total_weight=total_weight,
        retention=dict(
            complete_photometry=float(weight[complete].sum() / total_weight),
            bpz_bounds=float(weight[prelim >= 0].sum() / total_weight),
            gold=float(np.sum(fractions) / total_weight),
        ),
        tom_bin_counts=np.bincount(tom, minlength=len(edges)).tolist(),
        gold_qc_threshold=threshold,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "recipe.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    if "ensemble" in config:
        from . import nz_ensemble

        np.savez_compressed(
            output / "ensemble_inputs.npz",
            cal_photoz_gradient=nz_ensemble.photoz_gradient(
                cal["mag"], cal["photoz"], model
            ),
            **nz_ensemble.ensemble_inputs(
                src["mag"],
                photoz,
                weight,
                prelim,
                channel,
                model,
                offset=config["z2_offset"],
                workers=workers,
            ),
        )
        nz_ensemble.run(output, output / "realisations.npz", config["ensemble"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("--prepare", action="store_true", help="build the inputs first")
    parser.add_argument("--output", help="override the recipe's output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--figures", metavar="DIRECTORY")
    args = parser.parse_args(argv)
    config = load_yaml(args.recipe)
    if args.prepare:
        prepare(config, workers=args.workers)
    output = Path(args.output or config["output"])
    run(config, output, workers=args.workers)
    if args.figures:
        from .nz_diagnostics import render

        render(
            dict(
                config.get("diagnostics", {}),
                calibration_arrays=output / "calibration_arrays.npz",
                model=output / "som_model.npz",
                summary=output / "summary.json",
                side=config["som"]["side"],
            ),
            args.figures,
        )


if __name__ == "__main__":
    main()
