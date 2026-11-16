"""The response of each tomographic bin of a calibrated sample.

The sample's response (:mod:`.pipeline`) is that of all its galaxies. A
tomographic bin has its own: the same estimator on the bin's rows, with the
sample's cuts, cells and weight table held fixed
(``docs/source/response_calibration.md``). The bins are the n(z) stage's
(:mod:`sp_validation.nz`), which gives ``TOM_BIN_ID`` to the galaxies the
sample selects. The selection response also needs the galaxies a sheared
metacal branch selects and the no-shear branch does not, which have no
``TOM_BIN_ID``. So :func:`run` gives every preselected row its bin from its
photometry with the n(z) stage's saved model (:func:`tomographic_bins`), the
bin held fixed across the branches, and reduces the rows to sums per
(tile, bin) and cell in one pass over the shapes' HEALPix pixels.

Two checks make the bins' sums the release's, and the run stops unless both
hold:

- every galaxy of ``nz/tom_bin.parquet`` with an object ID is selected here
  with its ``TOM_BIN_ID``, and the galaxies selected here it does not name
  are as many as its rows without an ID (:func:`check_bins`);
- the sums of the bins, bin 0 (no bin) included, are those of the whole
  sample, ``catalogs/response_sums.npz``.

Each bin's response, additive bias and delete-one-tile jackknife covariance
(``response.record``) go into ``catalogs/response.json`` under ``bins``,
``{"column": "TOM_BIN_ID", "records": {"0": ..., "1": ...}}``, bin 0 being
the galaxies in no bin; the rest of the file, the whole sample's record, is
kept as it is. ``provenance/bin_response_checks.json`` records what the checks
compared.

The run then rewrites the catalogue (:func:`recalibrate`): every row gets its
``TOM_BIN_ID`` from ``nz/tom_bin.parquet`` and the shear of its own bin,
``e = R_k⁻¹ (g − c_e,k)``, where the sample build wrote ``R⁻¹ (g − c_e)`` with
the whole sample's. A record with ``bins`` therefore says the catalogue's
``e1``, ``e2`` carry each row's bin's ``R`` and ``c_e``; one without, the top
level's. The rewrite is staged whole and then installed with its record
(:func:`install`), so the record always describes the catalogue beside it.
The whole sample's statistics in ``catalogs/survey.json`` (σ_e) are then
taken again from the rewritten shear.

The run reads, under the release directory, the sample's
``catalogs/calibrated``, ``catalogs/response.json``, ``des_weights.fits`` and
``response_sums.npz``, and the n(z) stage's ``nz/tom_bin.parquet`` and
``provenance/nz_som_model.npz``, ``nz_calibration_arrays.npz`` and
``nz_recipe.yaml``. It is rerun whenever the bins, the cuts or the weights
change.
"""

import argparse
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from hats.io import write_parquet_metadata

from .. import io, response
from ..assembly.object_id import OBJECT_ID, object_id
from ..data_roots import load_yaml
from ..nz import calibration as nz
from ..nz.inputs import MAGS
from ..survey import effective_survey_stats
from . import pipeline, snapshot

COLUMN = "TOM_BIN_ID"
NZ = dict(
    bins="nz/tom_bin.parquet",
    model="provenance/nz_som_model.npz",
    arrays="provenance/nz_calibration_arrays.npz",
    recipe="provenance/nz_recipe.yaml",
)


def tomographic_bins(data, model, gold, offset):
    """Return the ``TOM_BIN_ID`` of every row of ``data`` from its photometry.

    The n(z) stage's assignment (:func:`.nz.calibration.source_groups`, the
    map's groups, :func:`.nz.calibration.tom_bin_ids`) with its saved
    ``model``, ``gold`` mask and z2 ``offset``. A row without a DR6 object ID
    has no photometry, as in :func:`.nz.inputs.sources`, and is in no bin.
    Every row counts as weighted, so a row's bin does not depend on the
    metacal branch.
    """
    mag = np.column_stack([data[name] for name in MAGS]).astype(np.float32)
    photoz = np.array(data["Z_B"], dtype=np.float64)
    missing = object_id(data["TILE_ID"], data["NUMBER"]).is_null()
    missing = missing.to_numpy(zero_copy_only=False)
    mag[missing], photoz[missing] = np.nan, np.nan
    prelim, cell, _ = nz.source_groups(
        mag, photoz, np.ones(len(photoz)), model, offset=offset
    )
    group = np.where(cell >= 0, model["group_map"][np.maximum(cell, 0)], -1)
    return nz.tom_bin_ids(prelim, group, gold)


def split(sums, width):
    """Return the sums of bins 0 to ``width - 1``, each grouped by tile, from
    ``sums`` grouped by ``tile key * width + bin``."""
    return [
        replace(
            sums,
            arrays={k: v[..., b::width, :] for k, v in sums.arrays.items()},
            labels=sums.labels[b::width] // width,
        )
        for b in range(width)
    ]


_state = {}


def _start(catalogue, selection, tiles, edges, model, gold, offset):
    _state.update(
        catalogue=io.Catalogue(catalogue),
        selection=selection,
        tiles=tiles,
        edges=edges,
        model=model,
        gold=gold,
        offset=offset,
        columns=pipeline.source_columns(selection) + ["Z_B", *MAGS],
    )


def measure(pixel):
    """One pixel: its sums per (tile, bin), and the object IDs (-1 for none)
    and bins of the galaxies the sample selects."""
    data = _state["catalogue"].chunks[pixel].to_structured(_state["columns"])
    keep = pipeline.preselect(data, _state["selection"], _state["tiles"])[1][-1]
    metacal = _state["selection"]["metacal"]
    cuts = response.Cuts.from_config(metacal)
    tom = tomographic_bins(data, _state["model"], _state["gold"], _state["offset"])
    width = len(_state["gold"]) + 1
    sums = response.accumulate(
        data,
        cuts,
        _state["edges"],
        preselection=keep,
        groups=pipeline.tile_key(data["TILE_ID"]) * width + tom,
        step=metacal.get("step", 0.01),
    )
    selected = keep & response.select(response.branch_columns(data, "ns"), cuts)
    ids = object_id(data["TILE_ID"][selected], data["NUMBER"][selected])
    return sums, np.asarray(ids.fill_null(-1)), tom[selected]


def check_bins(ids, bins, release, width):
    """Fail unless every galaxy of the release's ``tom_bin.parquet`` that has
    an object ID is one of ``ids`` (-1 for none) with the same bin in ``bins``
    (0 to ``width - 1``), and the galaxies here it does not name by ID are as
    many as its rows without one. Returns what was compared, with the bins
    here of the galaxies it does not name (``unnamed_per_bin``).

    A row the release holds without an ID is not compared: the row has no
    other key, and the n(z) stage, which joins photometry by ID, gave it bin 0.
    """
    table = pq.read_table(release / NZ["bins"], columns=[OBJECT_ID, COLUMN])
    theirs = np.asarray(table[OBJECT_ID].fill_null(-1))
    released = np.asarray(table[COLUMN])
    order = np.argsort(ids)
    at = order[np.minimum(np.searchsorted(ids, theirs, sorter=order), len(ids) - 1)]
    named = theirs >= 0
    found = named & (ids[at] == theirs)
    here = np.ones(len(ids), dtype=bool)
    here[at[found]] = False
    moved = np.zeros((width, width), np.int64)
    np.add.at(moved, (released[found], bins[at[found]]), 1)
    differ = moved.sum() - np.trace(moved)
    absent = np.count_nonzero(named & ~found)
    if differ or absent or np.count_nonzero(here) != np.count_nonzero(~named):
        pairs = ", ".join(
            f"{n} from {a} to {b}"
            for (a, b), n in np.ndenumerate(moved)
            if n and a != b
        )
        raise ValueError(
            f"{NZ['bins']} does not hold the sample's bins: {differ} galaxies have "
            f"another bin here ({pairs or 'none'}), {absent} of its galaxies with "
            f"an object ID are not selected here, and {np.count_nonzero(here)} "
            f"selected here are not among its {np.count_nonzero(named)} with an "
            f"ID, while it has {np.count_nonzero(~named)} without one"
        )
    return dict(
        matched_by_object_id=int(np.count_nonzero(found)),
        released_without_object_id=int(np.count_nonzero(~named)),
        unnamed_per_bin=np.bincount(bins[here], minlength=width),
    )


def check_sums(per_bin, release):
    """Fail unless the sums of the bins add to the release's."""
    saved = response.Sums.load(release / "catalogs/response_sums.npz")
    whole = sum(per_bin)
    if not (
        np.array_equal(whole.labels, saved.labels)
        and all(map(np.array_equal, whole.edges, saved.edges))
        and all(
            np.allclose(whole.arrays[k], v, rtol=1e-12, atol=1e-9)
            for k, v in saved.arrays.items()
        )
    ):
        raise ValueError("the bins' sums do not add to catalogs/response_sums.npz")


def applied(record, width):
    """Return the response (``width``, 2, 2) and additive bias (``width``, 2)
    the catalogue of ``record`` carries for a row of each bin: each bin's own
    when the record holds ``bins``, else the whole sample's."""
    if "bins" in record:
        records = [record["bins"]["records"][str(b)] for b in range(width)]
    else:
        records = [record] * width
    return (
        np.array([r["R"] for r in records]),
        np.array([r["c_e"] for r in records]),
    )


def _start_recalibrate(ids, bins, before, after):
    _state.update(ids=np.load(ids, mmap_mode="r"), bins=np.load(bins, mmap_mode="r"))
    _state.update(before=before, after=after)


def recalibrate(job):
    """Rewrite one pixel file of the catalogue to ``staged``: each row's
    ``TOM_BIN_ID`` (0 for one ``nz/tom_bin.parquet`` does not name by object
    ID), and ``e1``, ``e2`` calibrated
    with its bin's response and additive bias in place of those the file
    carries.

    @sc [decision:dr6_catalogue.sample_calibration] each-row-its-bins-response
    ``e = R_k⁻¹ (g − c_e,k)`` for a row of bin k, with ``g = R e + c_e``
    recovered from the shear and the calibration the file carries.
    """
    path, staged = job
    table = pq.read_table(path)
    ids = object_id(
        table["TILE_ID"].to_numpy(zero_copy_only=False), table["NUMBER"].to_numpy()
    )
    ids = np.asarray(ids.fill_null(-1))
    at = np.minimum(np.searchsorted(_state["ids"], ids), len(_state["ids"]) - 1)
    named = _state["ids"][at] == ids
    tom = np.where(named, _state["bins"][at], 0).astype(_state["bins"].dtype)
    old = np.asarray(table[COLUMN]) if COLUMN in table.column_names else tom
    (R_old, c_old), (R_new, c_new) = _state["before"], _state["after"]
    e = np.array([table["e1"].to_numpy(), table["e2"].to_numpy()])
    g = np.einsum("nij,jn->in", R_old[old], e) + c_old[old].T
    e = np.einsum("nij,jn->in", np.linalg.inv(R_new)[tom], g - c_new[tom].T)
    for name, values in (("e1", e[0]), ("e2", e[1]), (COLUMN, tom)):
        if name in table.column_names:
            table = table.drop_columns(name)
        table = table.append_column(name, pa.array(values))
    staged.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table, staged, compression="zstd", compression_level=3, row_group_size=100_000
    )
    return np.bincount(tom, minlength=len(R_new))


def install(release):
    """Install a staged rewrite of the catalogue of ``release``: every staged
    file over the catalogue's, then the staged record over
    ``catalogs/response.json``. The record is staged after every file, so a
    staging directory holding one is a whole rewrite, and installing it again
    after an interruption completes it."""
    target = release / "catalogs/calibrated"
    staging = release / "catalogs/calibrated.staging"
    for staged in sorted((staging / "dataset").rglob("*.parquet")):
        os.replace(staged, target / staged.relative_to(staging))
    write_parquet_metadata(target)
    os.replace(staging / "response.json", release / "catalogs/response.json")
    shutil.rmtree(staging)


def recalibrate_catalogue(release, record, workers):
    """Rewrite the catalogue of ``release`` from the calibration its
    ``catalogs/response.json`` records to that of ``record``
    (:func:`applied`), with the bins of its ``nz/tom_bin.parquet``, and make
    ``record`` its record (:func:`install`). Returns the rows per bin."""
    target = release / "catalogs/calibrated"
    staging = release / "catalogs/calibrated.staging"
    staging.mkdir()
    width = len(record["bins"]["records"])
    before = json.loads((release / "catalogs/response.json").read_text())
    table = pq.read_table(release / NZ["bins"], columns=[OBJECT_ID, COLUMN])
    table = table.filter(table[OBJECT_ID].is_valid())
    ids = np.asarray(table[OBJECT_ID])
    order = np.argsort(ids)
    np.save(staging / "ids.npy", ids[order])
    np.save(staging / "bins.npy", np.asarray(table[COLUMN])[order])
    files = sorted((target / "dataset").rglob("*.parquet"))
    jobs = [(path, staging / path.relative_to(target)) for path in files]
    start = (
        staging / "ids.npy",
        staging / "bins.npy",
        applied(before, width),
        applied(record, width),
    )
    with ProcessPoolExecutor(
        workers, initializer=_start_recalibrate, initargs=start
    ) as pool:
        rows = sum(pool.map(recalibrate, jobs))
    pipeline._dump(staging / "response.json", record)
    install(release)
    return rows


def run(recipe, output=None, *, workers=1):
    """Measure the response of each bin of the sample of the recipe file
    ``recipe``, built under its ``output`` directory or under ``output``,
    write it into the sample's ``catalogs/response.json`` and calibrate each
    row of the catalogue with its bin's. Returns the ``bins`` entry.

    @sc [decision:dr6_catalogue.sample_wide_calibration] bins-on-the-sample-table
    Each bin's sums are those of the sample's rows in the bin, and its
    response, additive bias and jackknife are taken with the sample's weight
    table held fixed, the weights the catalogue carries.
    """
    config = load_yaml(recipe)
    release = Path(output or config["output"])
    staging = release / "catalogs/calibrated.staging"
    if (staging / "response.json").exists():
        install(release)  # a rewrite an earlier run staged and did not finish
    shutil.rmtree(staging, ignore_errors=True)
    selection = yaml.safe_load((pipeline.ROOT / config["selection"]).read_text())
    tiles = sorted(snapshot.read(config["snapshot"])["objects"])
    # The collection, so the shapes' rows come with their PhotoPipe extension.
    source = config["catalogue"]
    table = response.read_table(release / "catalogs/des_weights.fits")
    model = dict(np.load(release / NZ["model"]))
    gold = np.load(release / NZ["arrays"])["gold"]
    offset = yaml.safe_load((release / NZ["recipe"]).read_text())["z2_offset"]
    width = len(gold) + 1
    labels = (pipeline.tile_key(tiles)[:, None] * width + np.arange(width)).ravel()

    with io.Catalogue(source) as catalogue:
        pixels = list(catalogue.chunks)
    ids, bins = [], []
    start = (source, selection, tiles, table[:2], model, gold, offset)
    with ProcessPoolExecutor(workers, initializer=_start, initargs=start) as pool:

        def parts():
            for done, (sums, i, b) in enumerate(pool.map(measure, pixels), 1):
                ids.append(i)
                bins.append(b)
                if done % 100 == 0:
                    print(f"measured {done}/{len(pixels)} pixels", flush=True)
                yield sums

        per_bin = split(response.Sums.gather(parts(), labels), width)
    ids, bins = np.concatenate(ids), np.concatenate(bins)
    compared = check_bins(ids, bins, release, width)
    check_sums(per_bin, release)

    path = release / "catalogs/response.json"
    whole = json.loads(path.read_text())
    records = {}
    for b in range(width):
        _, _, covariance = response.jackknife(per_bin[b], table)
        records[str(b)] = response.record(
            response.estimate(per_bin[b], table), whole["weights"], covariance
        )
    entry = dict(column=COLUMN, records=records)
    written = recalibrate_catalogue(release, whole | dict(bins=entry), workers)
    survey = json.loads((release / "catalogs/survey.json").read_text())
    data = io.read_catalogue(
        release / "catalogs/calibrated", columns=["e1", "e2", "w_des"]
    )
    stats = effective_survey_stats(
        data["e1"], data["e2"], data["w_des"], survey["area_deg2"]
    )
    survey["samples"]["all"] = {name: float(value) for name, value in stats.items()}
    pipeline._dump(release / "catalogs/survey.json", survey)
    pipeline._dump(
        release / "provenance/bin_response_checks.json",
        dict(
            selected_rows=len(ids),
            without_object_id=int(np.count_nonzero(ids < 0)),
            selected_per_bin=np.bincount(bins, minlength=width),
            catalogue_rows_per_bin=written,
            bins_match=NZ["bins"],
            **compared,
            sums_add_to="catalogs/response_sums.npz",
        ),
    )
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recipe", help="sample recipe (config/calibration/sample/)")
    parser.add_argument(
        "--output", help="the sample's directory (default: the recipe's output)"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    entry = run(args.recipe, args.output, workers=args.workers)
    for b, record in entry["records"].items():
        print(f"{COLUMN} {b}: R diagonal {np.diag(record['R']).tolist()}")
