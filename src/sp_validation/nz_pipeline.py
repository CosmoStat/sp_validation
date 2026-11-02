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
    estimates, retention = {}, {}
    total_weight = weight.sum()
    for variant, use_z2 in [("both", True), ("zonly", False)]:
        prelim, group, channel = nz.source_groups(
            mag,
            photoz,
            weight,
            model,
            offset=config["z2_offset"],
            use_z2=use_z2,
            workers=workers,
        )
        print(
            f"Assigned {variant}: {np.count_nonzero(prelim >= 0)} eligible rows",
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
        estimates[variant] = est
        tom = nz.tom_bin_ids(prelim, group, est["gold"])
        retained = sum(b["weight_sum"] for b in est["bins"])
        complete = (
            nz.detected(mag[:, :4]).all(1)
            & (channel > 0)
            & np.isfinite(weight)
            & (weight > 0)
        )
        retention[variant] = dict(
            all=1.0,
            complete_photometry=float(weight[complete].sum() / total_weight),
            bpz_bounds=float(weight[prelim >= 0].sum() / total_weight),
            gold=float(retained / total_weight),
        )
        for b in est["bins"]:
            b["weight_fraction_of_all"] = b["weight_sum"] / float(total_weight)
        stem = "nz_tomo.txt" if use_z2 else "nz_zonly_tomo.txt"
        nz.write_nz(
            output / stem,
            grid,
            est["bin_hists"],
            [b["weight_sum"] for b in est["bins"]],
        )
        if use_z2:
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
            print(
                "TOM_BIN_ID counts:", np.bincount(tom, minlength=len(edges)), flush=True
            )
    primary = estimates["both"]
    np.savez_compressed(
        output / "calibration_arrays.npz",
        z=cal["z"],
        cal_photoz=cal["photoz"],
        cal_cell=cal_cell,
        cal_group=cal_group,
        grid=grid,
        edges=edges,
        gold=primary["gold"],
        cal_weights=primary["cal_weights"],
        bin_hists=primary["bin_hists"],
        cal_weights_zonly=estimates["zonly"]["cal_weights"],
    )
    summary = dict(
        bins=primary["bins"],
        retention=retention,
        bins_zonly=estimates["zonly"]["bins"],
        total_weight=float(total_weight),
        n_rows=len(weight),
        retained_rows=sum(b["n_rows"] for b in primary["bins"]),
        retained_weight_fraction=retention["both"]["gold"],
        gold_qc_threshold=nz.gold_threshold(
            cal["z"], cal["photoz"], **config["gold_qc"]
        ),
        method=config["name"],
        limitation="EGS-only, provisional calibration for validation and null tests; "
        "no independent-field variance or spectroscopic-selection model.",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "recipe.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return summary


def verify(config, output):
    """Gate delivered means/IDs and round-trip the actual cosmo_val/SACC readers."""
    import sacc

    from .cosmo_val import CosmologyValidation
    from .sacc_io import new_sacc

    output = Path(output)
    reference = config["reference"]
    actual = np.loadtxt(output / "nz_tomo.txt")
    expected = np.loadtxt(reference["nz"])
    z = actual[:, 0]
    means = np.trapezoid(z[:, None] * actual[:, 1:], z, axis=0)
    means /= np.trapezoid(actual[:, 1:], z, axis=0)
    ref_mean = np.trapezoid(expected[:, :1] * expected[:, 1:], expected[:, 0], axis=0)
    ref_mean /= np.trapezoid(expected[:, 1:], expected[:, 0], axis=0)
    error = abs(means - ref_mean)
    assert np.all(error < 1e-4), error
    mismatches, count = 0, 0
    # Both sidecars preserve the selected-catalogue row order.
    a = pq.ParquetFile(output / "tom_bin.parquet")
    b = pq.ParquetFile(reference["tom_bins"])
    assert a.metadata.num_rows == b.metadata.num_rows
    for aa, bb in zip(
        a.iter_batches(columns=["TILE_ID", "NUMBER", "TOM_BIN_ID"]),
        b.iter_batches(columns=["TILE_ID", "NUMBER", "TOM_BIN_ID"]),
    ):
        assert aa.column(0).equals(bb.column(0))
        assert aa.column(1).equals(bb.column(1))
        mismatches += np.count_nonzero(
            aa.column(2).to_numpy() != bb.column(2).to_numpy()
        )
        count += len(aa)
    assert count == a.metadata.num_rows, (count, a.metadata.num_rows)
    assert mismatches == 0, mismatches
    reader = object.__new__(CosmologyValidation)
    reader.cc = {"nz": {"shear": {"redshift_path": str(output / "nz_tomo.txt")}}}
    rz, densities = reader.read_redshift_distribution("nz", is_tomography=True)
    np.testing.assert_array_equal(rz, z)
    np.testing.assert_array_equal(densities, actual[:, 1:])
    _, summed = reader.read_redshift_distribution("nz", is_tomography=False)
    np.testing.assert_array_equal(summed, actual[:, 1:].sum(1))
    tracers = new_sacc([(rz, d) for d in densities.T])
    tracers.save_fits(str(output / "nz_tracers.fits"), overwrite=True)
    loaded = sacc.Sacc.load_fits(str(output / "nz_tracers.fits"))
    for k in range(densities.shape[1]):
        tracer = loaded.get_tracer(f"source_{k}")
        np.testing.assert_array_equal(tracer.z, rz)
        np.testing.assert_array_equal(tracer.nz, densities[:, k])
    result = dict(
        means=means.tolist(),
        reference_means=ref_mean.tolist(),
        absolute_mean_errors=error.tolist(),
        rows=count,
        tom_bin_mismatches=int(mismatches),
        max_density_difference=float(abs(actual - expected).max()),
        cosmo_val_reader="passed",
        sacc_nz_roundtrip="passed",
    )
    (output / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe")
    parser.add_argument("--output", help="override recipe output directory")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--prepare", action="store_true", help="match DR6 before calibration"
    )
    parser.add_argument("--rebuild-model", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--figures", metavar="DIRECTORY")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.recipe).read_text())
    output = Path(args.output or config["output"])
    if args.prepare:
        from .nz_photometry import prepare_nz_photometry

        p = config["photometry"]
        prepare_nz_photometry(p["selection"], p["dr6_root"], p["output"])
    run(config, output, workers=args.workers, rebuild=args.rebuild_model)
    if args.verify:
        verify(config, output)
    if args.figures:
        from .nz_diagnostics import render

        render(
            dict(
                config["diagnostics"],
                calibration_arrays=output / "calibration_arrays.npz",
                model=output / "som_assignment_model.npz",
                summary=output / "summary.json",
                source_catalogue=config["inputs"]["gold_reference"],
                workers=args.workers,
            ),
            args.figures,
        )


if __name__ == "__main__":
    main()
