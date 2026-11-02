"""Run the catalogue's saved n(z) recipe independently of cosmo_val."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import nz_calibration as nz


def rebuild_model(config, workers):
    """Train/group the map and derive gold on the recipe's reference sample."""
    cal = np.load(config["inputs"]["calibrators"])
    settings = config["som"]
    model = nz.train(cal["mag"], workers=workers, **settings)
    model["group_map"] = nz.group_cells(model, config["groups"])
    model["edges"] = np.array(config["edges"])
    source = np.load(config["inputs"]["gold_reference"])
    prelim, group, _ = nz.source_groups(
        source["mag"],
        source["photoz"],
        source["weight"],
        model,
        offset=config["z2_offset"],
        use_z2=False,
        workers=workers,
    )
    _, cal_cell = nz.assign(cal["mag"], model, workers=workers)
    est = nz.calibrate(
        prelim,
        group,
        source["weight"],
        source["photoz"],
        cal["z"],
        cal["photoz"],
        model["group_map"][cal_cell],
        model["edges"],
        config["groups"],
        output_grid(config),
        threshold=nz.gold_threshold(cal["z"], cal["photoz"], **config["gold_qc"]),
    )
    model["gold"] = est["gold"]
    return model


def output_grid(config):
    grid = config["grid"]
    return np.linspace(
        grid["min"], grid["max"], round((grid["max"] - grid["min"]) / grid["step"]) + 1
    )


def run(config, output, *, workers=1, rebuild=False):
    """Recalibrate full-area weights with the frozen SOM, edges and gold masks."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = (
        rebuild_model(config, workers)
        if rebuild
        else dict(np.load(config["inputs"]["model"]))
    )
    edges = np.asarray(config["edges"])
    # The recipe and the frozen definition describe the same catalogue selection.
    np.testing.assert_array_equal(model["edges"], edges)
    np.savez(output / "som_assignment_model.npz", **model)
    cal = np.load(config["inputs"]["calibrators"])
    _, cal_cell = nz.assign(cal["mag"], model, workers=workers)
    cal_group = model["group_map"][cal_cell]
    data = np.load(config["photometry"]["output"])
    mag, photoz, weight = data["mag"], data["zb"], data["weight"]
    grid = output_grid(config)
    total_weight = weight.sum()
    prelim, group, channel = nz.source_groups(
        mag,
        photoz,
        weight,
        model,
        offset=config["z2_offset"],
        use_z2=True,
        workers=workers,
    )
    print(
        f"Assigned: {np.count_nonzero(prelim >= 0)} eligible rows",
        flush=True,
    )
    est = nz.calibrate(
        prelim,
        group,
        weight,
        photoz,
        cal["z"],
        cal["photoz"],
        cal_group,
        edges,
        config["groups"],
        grid,
        gold=model["gold"],
    )
    tom = nz.tom_bin_ids(prelim, group, est["gold"])
    retained = sum(b["weight_sum"] for b in est["bins"])
    complete = (
        nz.detected(mag[:, :4]).all(1)
        & (channel > 0)
        & np.isfinite(weight)
        & (weight > 0)
    )
    retention = dict(
        all=1.0,
        complete_photometry=float(weight[complete].sum() / total_weight),
        bpz_bounds=float(weight[prelim >= 0].sum() / total_weight),
        gold=float(retained / total_weight),
    )
    for b in est["bins"]:
        b["weight_fraction_of_all"] = b["weight_sum"] / float(total_weight)
    nz.write_nz(
        output / "nz_tomo.txt",
        grid,
        est["bin_hists"],
        [b["weight_sum"] for b in est["bins"]],
    )
    tile = pa.DictionaryArray.from_arrays(
        pa.array(data["tile"]), pa.array(data["tile_names"].astype(str))
    ).cast(pa.string())
    pq.write_table(
        pa.table(
            dict(
                TILE_ID=tile,
                NUMBER=data["number"],
                TOM_BIN_ID=tom,
                ZBAND=channel,
            )
        ),
        output / "tom_bin.parquet",
    )
    print("TOM_BIN_ID counts:", np.bincount(tom, minlength=len(edges)), flush=True)
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
    )
    summary = dict(
        bins=est["bins"],
        retention=retention,
        total_weight=float(total_weight),
        n_rows=len(weight),
        retained_rows=sum(b["n_rows"] for b in est["bins"]),
        retained_weight_fraction=retention["gold"],
        gold_qc_threshold=nz.gold_threshold(
            cal["z"], cal["photoz"], **config["gold_qc"]
        ),
        method=config["name"],
        limitation=config["limitation"],
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "recipe.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("--output", help="override recipe output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--rebuild-model", action="store_true")
    parser.add_argument("--figures", metavar="DIRECTORY")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.recipe).read_text())
    output = Path(args.output or config["output"])
    run(config, output, workers=args.workers, rebuild=args.rebuild_model)
    if args.figures:
        from .nz_diagnostics import render

        render(
            dict(
                config["diagnostics"],
                calibration_arrays=output / "calibration_arrays.npz",
                model=output / "som_assignment_model.npz",
                summary=output / "summary.json",
                source_catalogue=config["inputs"]["gold_reference"],
                side=config["som"]["side"],
                workers=args.workers,
            ),
            args.figures,
        )


if __name__ == "__main__":
    main()
