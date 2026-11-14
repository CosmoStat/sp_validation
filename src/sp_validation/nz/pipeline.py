"""Run an n(z) recipe: SOM direct calibration of a shear catalogue's bins.

A recipe names its inputs, every choice of the calibration and its output
directory. ``run`` reads the calibrators and sources (:mod:`.inputs`), and
``calibrate`` trains the SOM on the calibrators, groups its cells, places the
bin edges, assigns every source, selects gold groups on the sources
themselves, and writes:

- ``nz_tomo.txt``: z and one density per bin, integrating to the bin's share of
  the retained shear weight (cosmo_val's ``shear.redshift_path``);
- ``nz_nontomo.txt``: their sum, the n(z) of every ``TOM_BIN_ID > 0`` row;
- ``tom_bin.parquet``: ``OBJECT_ID``, ``TOM_BIN_ID`` and ``ZBAND``, row-aligned
  with the catalogue;
- ``realisations.npz``: draws of every bin at once (:mod:`.ensemble`), from
  ``ensemble_inputs.npz`` (per-group source tables, their response to
  magnitude offsets, and the calibrators' BPZ gradients);
- ``summary.json``: bin edges, means and weights, and the photometry match
  (:func:`.inputs.match_report`);
- ``calibrators.npz``, ``som_model.npz`` and ``calibration_arrays.npz``;
- ``recipe.yaml``: the recipe file, byte for byte.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..data_roots import load_yaml
from ..object_id import OBJECT_ID
from . import calibration as nz
from . import ensemble, inputs


def output_grid(config):
    grid = config["grid"]
    return np.linspace(
        grid["min"], grid["max"], round((grid["max"] - grid["min"]) / grid["step"]) + 1
    )


def run(recipe, output=None, *, workers=1):
    """Calibrate the catalogue of the recipe file ``recipe``.

    Products go to the recipe's ``output`` directory, or to ``output``.
    """
    config = load_yaml(recipe)
    output = Path(output or config["output"])
    cal = inputs.calibrators(
        **config["inputs"]["calibrators"], offset=config["z2_offset"]
    )
    src = inputs.sources(**config["inputs"]["sources"], workers=workers)
    summary = calibrate(config, cal, src, output, workers=workers)
    shutil.copyfile(recipe, output / "recipe.yaml")
    return summary


def calibrate(config, cal, src, output, *, workers=1):
    """Calibrate sources ``src`` on calibrators ``cal`` and write the products.

    ``cal`` and ``src`` are the arrays of :func:`.inputs.calibrators` and
    :func:`.inputs.sources`; ``config`` is a loaded recipe.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "calibrators.npz", **cal)
    edges = nz.quantile_edges(
        src["mag"],
        src["photoz"],
        src["weight"],
        offset=config["z2_offset"],
        **config["bins"],
    )
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
                **{OBJECT_ID: pa.array(src["object_id"], mask=src["object_id"] < 0)},
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
        gold_qc=json.dumps(config["gold_qc"]),
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
    summary = dict(
        name=config["name"],
        limitation=config["limitation"],
        bins=est["bins"],
        n_rows=len(weight),
        n_calibrators=len(cal["z"]),
        total_weight=total_weight,
        match=inputs.match_report(src["status"], src["mag"], weight),
        retention=dict(
            bpz_bounds=float(weight[prelim >= 0].sum() / total_weight),
            gold=float(np.sum(fractions) / total_weight),
        ),
        tom_bin_counts=np.bincount(tom, minlength=len(edges)).tolist(),
        edges=edges.tolist(),
        gold_qc_threshold=threshold,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if "ensemble" in config:
        np.savez_compressed(
            output / "ensemble_inputs.npz",
            cal_photoz_gradient=ensemble.photoz_gradient(
                cal["mag"], cal["photoz"], model
            ),
            **ensemble.ensemble_inputs(
                src["mag"],
                photoz,
                weight,
                prelim,
                channel,
                model,
                est["gold"],
                offset=config["z2_offset"],
                workers=workers,
            ),
        )
        ensemble.run(output, output / "realisations.npz", config["ensemble"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("--output", help="write here, not to the recipe's release")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--figures", metavar="DIRECTORY")
    args = parser.parse_args(argv)
    config = load_yaml(args.recipe)
    output = Path(args.output or config["output"])
    run(args.recipe, output, workers=args.workers)
    if args.figures:
        from .diagnostics import render

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
