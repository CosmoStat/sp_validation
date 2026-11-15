"""The response of each tomographic bin of a calibrated sample.

A synthetic sample (``test_sample``) is given photometry, binned by the n(z)
stage's own code, and measured per bin by ``sample.bins``. The bins' responses
must equal the estimator on each bin's rows in memory, and a run whose bins or
sums are not the release's must fail.
"""

import json
import shutil

import numpy as np
import numpy.testing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from numpy.lib.recfunctions import append_fields
from test_sample import SELECTION, TILES, build, shapes

from sp_validation import masks, response
from sp_validation.assembly.object_id import OBJECT_ID, object_id
from sp_validation.galaxy import classification_galaxy_overlap_ra_dec
from sp_validation.nz import calibration as nz
from sp_validation.nz import inputs
from sp_validation.nz import pipeline as nz_pipeline
from sp_validation.sample import bins, pipeline

OFFSET = -0.11
NZ_CONFIG = dict(
    name="toy",
    limitation="toy",
    z2_offset=OFFSET,
    bins=dict(count=4, range=[0.1, 1.2]),
    som=dict(side=6, epochs=5, seed=1),
    groups=12,
    gold_qc=dict(floor=0.4, nmad_factor=5),
    grid=dict(min=0, max=5, step=0.05),
)


def photometry(n, rng):
    """GAaP magnitudes (z missing for a third, z2 then filled) and BPZ whose
    colours follow a redshift."""
    z = rng.uniform(0, 1.4, n)
    r = rng.uniform(20, 24, n)
    colours = np.array([0.6 + 0.5 * z, 0.3 + 0.9 * z, 0.1 + 0.6 * z, 0.05 + 0.4 * z])
    colours += rng.normal(0, 0.05, colours.shape)
    g = r + colours[1]
    u = g + colours[0]
    i = r - colours[2]
    zband = i - colours[3]
    mag = np.column_stack([u, g, r, i, zband, zband - OFFSET])
    mag[rng.uniform(size=n) < 0.3, 4] = 99
    mag[rng.uniform(size=n) < 0.05, 0] = 99
    return mag, z, np.clip(z + rng.normal(0, 0.05, n), 0, None)


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    """The sample of shapes with photometry, a few without object IDs, and the
    n(z) stage's products installed in its release directory."""
    root = tmp_path_factory.mktemp("bins")
    rng = np.random.default_rng(7)
    data = shapes()
    data["NUMBER"][rng.choice(len(data), 60, replace=False)] = -1
    mag, _, photoz = photometry(len(data), rng)
    ids = np.asarray(object_id(data["TILE_ID"], data["NUMBER"]).fill_null(-1))
    data = append_fields(
        data,
        [*inputs.MAGS, "Z_B", "PHOTOPIPE_STATUS", OBJECT_ID],
        [*mag.T.astype(np.float32), photoz, np.zeros(len(data), np.int16), ids],
        usemask=False,
    )
    built = build(root, data)

    cal_mag, cal_z, cal_photoz = photometry(800, rng)
    cal_mag = nz.z_magnitudes(cal_mag, OFFSET)[0]
    complete = nz.detected(cal_mag).all(1)
    cal = dict(mag=cal_mag[complete], z=cal_z[complete], photoz=cal_photoz[complete])
    src = inputs.sources(root / "release/catalogs/calibrated", root / "shapes")
    nz_pipeline.calibrate(NZ_CONFIG, cal, src, root / "nz_run")
    out = root / "release"
    (out / "nz").mkdir()
    shutil.copyfile(root / "nz_run/tom_bin.parquet", out / bins.NZ["bins"])
    shutil.copyfile(root / "nz_run/som_model.npz", out / bins.NZ["model"])
    shutil.copyfile(root / "nz_run/calibration_arrays.npz", out / bins.NZ["arrays"])
    (out / bins.NZ["recipe"]).write_text(yaml.safe_dump(NZ_CONFIG))
    before = json.loads((out / "catalogs/response.json").read_text())
    entry = bins.run(built["recipe"], workers=2)
    return dict(built, data=data, entry=entry, before=before)


def in_memory_bins(data, out):
    """Every row's bin, from the n(z) stage's functions and saved model."""
    model = dict(np.load(out / bins.NZ["model"]))
    gold = np.load(out / bins.NZ["arrays"])["gold"]
    mag = np.column_stack([data[name] for name in inputs.MAGS])
    photoz = np.where(data["NUMBER"] < 0, np.nan, data["Z_B"])
    prelim, cell, _ = nz.source_groups(
        mag, photoz, np.ones(len(data)), model, offset=OFFSET
    )
    group = np.where(cell >= 0, model["group_map"][np.maximum(cell, 0)], -1)
    return nz.tom_bin_ids(prelim, group, gold), gold


def preselected(data):
    """The rows the sample's catalogue cuts keep (``test_sample.in_memory``)."""
    comprehensive = append_fields(
        data,
        ["overlap", "mag"],
        [classification_galaxy_overlap_ra_dec(data), data["MAG_AUTO"]],
        usemask=False,
    )
    return masks.catalogue_selection(SELECTION, comprehensive)


def test_each_bins_response_is_the_estimator_on_its_rows(release):
    data, out = release["data"], release["root"] / "release"
    tom, gold = in_memory_bins(data, out)
    assert not gold.all()  # the gold selection removes rows
    keep = preselected(data)
    table = response.read_table(out / "catalogs/des_weights.fits")
    cuts = response.Cuts.from_config(SELECTION["metacal"])
    written = json.loads((out / "catalogs/response.json").read_text())
    assert written["bins"] == json.loads(json.dumps(release["entry"]))
    assert {k: v for k, v in written.items() if k != "bins"} == release["before"]
    assert written["bins"]["column"] == "TOM_BIN_ID"
    records = written["bins"]["records"]
    assert sorted(records) == ["1", "2", "3", "4"]
    for b, record in records.items():
        sums = response.accumulate(
            data,
            cuts,
            table[:2],
            preselection=keep & (tom == int(b)),
            groups=pipeline.tile_key(data["TILE_ID"]),
            labels=pipeline.tile_key(TILES),
        )
        expected = response.estimate(sums, table)
        assert record["branch_counts"] == expected["counts"]
        for key, name in (("R", "total"), ("R_selection", "selection"), ("c_e", "c_e")):
            npt.assert_allclose(record[key], expected[name], rtol=1e-10, atol=1e-14)
        _, _, covariance = response.jackknife(sums, table)
        npt.assert_allclose(record["covariance"], covariance, rtol=1e-8, atol=1e-18)
    # Sheared branches select rows the no-shear branch does not.
    counts = records["1"]["branch_counts"]
    assert len(set(counts.values())) > 1

    checks = json.loads((out / "provenance/bin_response_checks.json").read_text())
    assert checks["without_object_id"] > 0
    assert sum(checks["selected_per_bin"]) == written["selected_rows"]
    assert checks["selected_per_bin"][1:] == [
        records[b]["branch_counts"]["ns"] for b in "1234"
    ]


def test_bin_sums_add_to_the_whole_sample(release):
    data, out = release["data"], release["root"] / "release"
    tom, _ = in_memory_bins(data, out)
    keep = preselected(data)
    table = response.read_table(out / "catalogs/des_weights.fits")
    cuts = response.Cuts.from_config(SELECTION["metacal"])
    tiles = pipeline.tile_key(TILES)
    key = pipeline.tile_key(data["TILE_ID"])
    sums = response.accumulate(
        data,
        cuts,
        table[:2],
        preselection=keep,
        groups=key * 5 + tom,
        labels=(tiles[:, None] * 5 + np.arange(5)).ravel(),
    )
    parts = bins.split(sums, 5)
    whole = response.Sums.load(out / "catalogs/response_sums.npz")
    for name, value in sum(parts).arrays.items():
        npt.assert_allclose(value, whole.arrays[name], rtol=1e-12, atol=1e-12)
    for b, part in enumerate(parts):
        npt.assert_array_equal(part.labels, tiles)
        alone = response.accumulate(
            data,
            cuts,
            table[:2],
            preselection=keep & (tom == b),
            groups=key,
            labels=tiles,
        )
        for name, value in alone.arrays.items():
            npt.assert_array_equal(part.arrays[name], value)


def mutated(release, tmp_path, change):
    """Copy the inputs of the release to ``tmp_path``, ``change`` them, and run."""
    out = release["root"] / "release"
    for name in (
        "catalogs/response.json",
        "catalogs/des_weights.fits",
        "catalogs/response_sums.npz",
        *bins.NZ.values(),
    ):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out / name, tmp_path / name)
    change(tmp_path)
    return bins.run(release["recipe"], tmp_path)


def test_a_galaxy_in_another_bin_fails(release, tmp_path):
    def move(path):
        table = pq.read_table(path / bins.NZ["bins"])
        tom = table["TOM_BIN_ID"].to_numpy().copy()
        tom[np.flatnonzero(tom > 0)[0]] = 0
        index = table.schema.get_field_index("TOM_BIN_ID")
        table = table.set_column(index, "TOM_BIN_ID", pa.array(tom))
        pq.write_table(table, path / bins.NZ["bins"])

    with pytest.raises(
        ValueError, match=r"1 galaxies have another bin here \(1 from 0 to "
    ):
        mutated(release, tmp_path, move)


def test_galaxies_the_release_holds_without_an_id_are_not_compared(release, tmp_path):
    """A galaxy the n(z) stage had no ID for is in its bin 0 whatever its
    photometry; the run counts it and keeps the bin of its photometry."""

    def forget(path):
        table = pq.read_table(path / bins.NZ["bins"])
        row = np.flatnonzero(table["TOM_BIN_ID"].to_numpy() > 0)[0]
        ids, tom = table[OBJECT_ID].to_pylist(), table["TOM_BIN_ID"].to_numpy().copy()
        ids[row], tom[row] = None, 0
        table = table.set_column(
            table.schema.get_field_index(OBJECT_ID),
            OBJECT_ID,
            pa.array(ids, pa.int64()),
        ).set_column(
            table.schema.get_field_index("TOM_BIN_ID"), "TOM_BIN_ID", pa.array(tom)
        )
        pq.write_table(table, path / bins.NZ["bins"])

    entry = mutated(release, tmp_path, forget)
    assert entry == json.loads(json.dumps(release["entry"]))
    checks = json.loads((tmp_path / "provenance/bin_response_checks.json").read_text())
    unnamed = checks["unnamed_per_bin"]
    assert checks["released_without_object_id"] == unnamed[0] + 1
    assert sum(unnamed[1:]) == 1


def test_sums_that_are_not_the_releases_fail(release, tmp_path):
    def drop(path):
        sums = response.Sums.load(path / "catalogs/response_sums.npz")
        sums.arrays["count"][0, 0] -= 1
        (path / "catalogs/response_sums.npz").unlink()
        sums.save(path / "catalogs/response_sums.npz")

    with pytest.raises(ValueError, match="do not add"):
        mutated(release, tmp_path, drop)
