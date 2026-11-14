"""The calibrated sample: its chunked build equals the in-memory calibration.

A synthetic shape catalogue over four neighbouring tiles is calibrated twice:
by ``calibration.metacal`` on all rows at once, the definition, and by the
sample build in chunks and from a HATS catalogue. Both must give the same
galaxies, weights, response and calibrated shapes.
"""

import hashlib
import json

import lsdb
import numpy as np
import numpy.testing as npt
import pandas as pd
import pyarrow.parquet as pq
import pytest
import yaml
from cs_util import cfis
from hats.io.validation import is_valid_catalog
from numpy.lib.recfunctions import append_fields

from sp_validation import calibration, io, masks, response
from sp_validation.galaxy import classification_galaxy_overlap_ra_dec
from sp_validation.object_id import OBJECT_ID, object_id
from sp_validation.sample import compare, pipeline, snapshot

TILES = ["200.290", "201.290", "201.291", "202.290"]
SELECTION = yaml.safe_load(
    (pipeline.ROOT / "config/calibration/mask_v2.0.yaml").read_text()
)
N = 6000


def shapes(seed=3):
    """Rows with every column the build reads, over ``TILES``. Positions reach
    past the tile edges, so neighbours share objects."""
    rng = np.random.default_rng(seed)
    names = pipeline.source_columns(SELECTION)
    integer = {"FLAGS": "i2", "NUMBER": "i4", "N_EPOCH": "i2", "NGMIX_N_EPOCH": "i2"}
    integer["NGMIX_MCAL_TYPES_FAIL"] = "i2"
    dtype = [
        (name, "?" if name.startswith("MASK_") else integer.get(name, "f8"))
        for name in names
    ]
    d = np.zeros(N, dtype=dtype)
    tile = rng.integers(0, len(TILES), N)
    d["TILE_ID"] = np.array(TILES, dtype=float)[tile]
    d["NUMBER"] = np.arange(N)
    for i, name in enumerate(TILES):
        ra, dec = cfis.get_tile_coord_from_nixy(*([n] for n in name.split(".")))
        here = tile == i
        scale = np.cos(np.radians(dec[0].value))
        d["XWIN_WORLD"][here] = (
            ra[0].value + rng.uniform(-0.27, 0.27, here.sum()) / scale
        )
        d["YWIN_WORLD"][here] = dec[0].value + rng.uniform(-0.27, 0.27, here.sum())
    d["FLAGS"] = rng.choice([0, 1, 2, 3, 16, 18, 19], N)
    for name in names:
        if name.startswith("MASK_"):
            d[name] = rng.uniform(size=N) < 0.03
    d["N_EPOCH"] = rng.integers(1, 5, N)
    d["NGMIX_N_EPOCH"] = rng.integers(0, 4, N)
    d["NGMIX_MCAL_TYPES_FAIL"] = rng.uniform(size=N) < 0.02
    d["MAG_AUTO"] = rng.uniform(14, 31, N)
    for i in (1, 2):
        d[f"NGMIX_G{i}_PSF_ORIG_NOSHEAR"] = np.where(
            rng.uniform(size=N) < 0.02, -10, rng.normal(0, 0.02, N)
        )
    intrinsic = rng.normal(0, 0.25, (2, N))
    flux = 10 ** rng.uniform(0.8, 2.9, N)
    size = 10 ** rng.uniform(-0.3, 1.1, N)
    matrix = np.array([[0.8, 0.02], [-0.01, 0.75]])
    for shear, g in zip(
        ("NOSHEAR", "1P", "1M", "2P", "2M"),
        ([0, 0], [0.01, 0], [-0.01, 0], [0, 0.01], [0, -0.01]),
    ):
        e = intrinsic + (matrix @ g)[:, None] + 0.003
        # Flux and size respond to the shear, so objects cross the cuts and
        # change weight cell between branches.
        gain = 1 + 3 * (intrinsic[0] * g[0] + intrinsic[1] * g[1])
        d[f"NGMIX_G1_{shear}"], d[f"NGMIX_G2_{shear}"] = e
        d[f"NGMIX_FLUX_{shear}"] = flux * gain
        d[f"NGMIX_FLUX_ERR_{shear}"] = 1
        d[f"NGMIX_T_{shear}"] = size / gain
        d[f"NGMIX_T_PSF_RECONV_{shear}"] = 1
        d[f"NGMIX_FLAGS_{shear}"] = rng.uniform(size=N) < 0.01
    return d


def in_memory(data):
    """The definition: the selection and ``calibration.metacal`` on all rows."""
    comprehensive = append_fields(
        data,
        ["overlap", "mag"],
        [classification_galaxy_overlap_ra_dec(data), data["MAG_AUTO"]],
        usemask=False,
    )
    keep = masks.catalogue_selection(SELECTION, comprehensive)
    gm = calibration.metacal_from_config(data, keep, SELECTION["metacal"])
    ns = gm.mask_dict["ns"]
    g = np.array([gm.ns["g1"][ns], gm.ns["g2"][ns]])
    e = np.linalg.inv(gm.R) @ (g - gm.c_e[:, None])
    ids = object_id(data["TILE_ID"][keep][ns], data["NUMBER"][keep][ns]).to_numpy()
    order = np.argsort(ids)
    return gm, dict(id=ids[order], e=e[:, order], w=gm.ns["w_des"][ns][order])


def chunked(data, parts, rng):
    """The build's sums over a random split of ``data`` into ``parts``."""
    part = rng.integers(0, parts, len(data))
    results = [pipeline.select(data[part == i], SELECTION, TILES) for i in range(parts)]
    rows = {
        name: np.concatenate([r["rows"][name] for r in results])
        for name in results[0]["rows"]
    }
    total = {key: results[0][key] for key in ("flow", "moments", "sums")}
    for result in results[1:]:
        total = response.add_sums(total, {key: result[key] for key in total})
    metacal = SELECTION["metacal"]
    table = response.table_from_moments(
        *total["moments"], *pipeline.table_edges(metacal), min_count=3
    )
    return rows, table, response.responses_from_sums(total["sums"], table, 0.01)


def test_chunked_build_equals_the_in_memory_calibration():
    data = shapes()
    gm, expected = in_memory(data)
    assert 500 < len(expected["id"]) < 0.5 * N
    assert len(set(map(len, gm.mask_dict.values()))) > 1  # branches select differently
    rows, table, estimate = chunked(data, 7, np.random.default_rng(1))
    npt.assert_allclose(table[2], gm.W[2], rtol=1e-12)
    npt.assert_array_equal(table[3], gm.W[3])
    for name, value in (
        ("total", gm.R),
        ("shear", gm.R_shear_global),
        ("selection", gm.R_selection),
        ("joint", gm.R_joint),
        ("c_e", gm.c_e),
    ):
        npt.assert_allclose(estimate[name], value, rtol=1e-10, atol=1e-14, err_msg=name)
    assert abs(estimate["selection"][0, 0]) > 1e-3  # the selection term is exercised
    assert estimate["counts"] == gm.branch_counts
    out = pipeline.calibrate(rows, table, estimate).to_pydict()
    order = np.argsort(out[OBJECT_ID])
    npt.assert_array_equal(np.array(out[OBJECT_ID])[order], expected["id"])
    npt.assert_allclose(np.array(out["w_des"])[order], expected["w"], rtol=1e-12)
    for i, name in enumerate(("e1", "e2")):
        npt.assert_allclose(np.array(out[name])[order], expected["e"][i], atol=1e-12)


def test_overlap_neighbours_come_from_the_tile_set():
    data = shapes()
    whole = classification_galaxy_overlap_ra_dec(data)
    assert 0.5 < whole.mean() < 0.98
    one = data["TILE_ID"] == 201.290
    alone = classification_galaxy_overlap_ra_dec(data[one])
    listed = classification_galaxy_overlap_ra_dec(data[one], tiles=TILES)
    # Alone, the tile has no neighbour and keeps the objects it shares.
    assert alone.sum() > whole[one].sum()
    npt.assert_array_equal(listed, whole[one])


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A snapshot, its shapes as HATS, a recipe, and the sample built from it."""
    root = tmp_path_factory.mktemp("sample")
    data = shapes()
    snap = root / "snap"
    snap.mkdir()
    counts = {t: int((data["TILE_ID"] == float(t)).sum()) for t in TILES}
    manifest = dict(
        status="complete", tiles={t: {"objects": n} for t, n in counts.items()}
    )
    (snap / "MANIFEST.json").write_text(json.dumps(manifest))
    sha = hashlib.sha256((snap / "MANIFEST.json").read_bytes()).hexdigest()
    frame = pd.DataFrame({name: data[name] for name in data.dtype.names})
    lsdb.from_dataframe(
        frame, ra_column="XWIN_WORLD", dec_column="YWIN_WORLD", partition_rows=1500
    ).write_catalog(root / "shapes", progress_bar=False)
    primary = io.hats_catalogue_dir(root / "shapes")
    for name in ("hats.properties", "properties"):
        with open(f"{primary}/{name}", "a") as handle:
            handle.write(f"source_manifest_sha256={sha}\n")
    recipe = root / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            dict(
                snapshot=str(snap),
                catalogue=str(root / "shapes"),
                selection="config/calibration/mask_v2.0.yaml",
                footprint_nside=4096,
                output=str(root / "release"),
            )
        )
    )
    result = pipeline.run(recipe, workers=2)
    return dict(root=root, data=data, recipe=recipe, result=result, sha=sha)


def test_run_writes_the_in_memory_sample_as_hats(built):
    release = built["root"] / "release"
    target = release / "catalogs" / "calibrated"
    assert is_valid_catalog(target, strict=True)
    with io.Catalogue(target) as catalogue:
        assert len(catalogue.chunks) > 1
        out = catalogue.read()
    assert out.dtype.names == pipeline.COLUMNS
    gm, expected = in_memory(built["data"])
    order = np.argsort(out[OBJECT_ID])
    npt.assert_array_equal(out[OBJECT_ID][order], expected["id"])
    npt.assert_allclose(out["w_des"][order], expected["w"], rtol=1e-12)
    npt.assert_allclose(out["e1"][order], expected["e"][0], atol=1e-12)
    npt.assert_allclose(out["e2"][order], expected["e"][1], atol=1e-12)
    npt.assert_array_equal(out["TILE_KEY"], np.rint(out["TILE_ID"] * 1000))

    written = json.loads((release / "catalogs/response.json").read_text())
    npt.assert_allclose(written["R"], gm.R, rtol=1e-10)
    assert written["selected_rows"] == len(out) == built["result"]["selected_rows"]
    assert written["raw_rows"] == N
    product = dict(name="test", path=release / "catalogs/des_weights.fits")
    table, min_count = calibration.load_des_weight_product(product)
    npt.assert_allclose(table[2], gm.W[2], rtol=1e-12)
    assert min_count == 3
    survey = json.loads((release / "catalogs/survey.json").read_text())
    assert survey["rows"] == len(out) and survey["area_deg2"] > 0
    flow = json.loads(
        (release / "provenance/calibrated_tile_accounting.json").read_text()
    )
    assert flow["totals"]["catalogue"] == N
    assert flow["totals"][pipeline._METACAL_STAGE] == len(out)
    assert (release / "provenance/calibrated_recipe.yaml").read_bytes() == built[
        "recipe"
    ].read_bytes()


def test_sample_records_its_snapshot_and_an_entry_is_checked_against_it(built):
    target = built["root"] / "release/catalogs/calibrated"
    record = dict(name="snap", tiles=len(TILES), manifest_sha256=built["sha"])
    assert snapshot.recorded(target) == record
    io.validate_calibrated_entry(dict(path=target, snapshot=record))
    io.validate_calibrated_entry(dict(path=target))  # nothing expected, no check
    for wrong in (dict(tiles=3), dict(manifest_sha256="0" * 64), dict(name="other")):
        with pytest.raises(ValueError, match="was built from snapshot"):
            io.validate_calibrated_entry(dict(path=target, snapshot=record | wrong))
    # A catalogue that records no snapshot fails an entry that expects one.
    with pytest.raises(ValueError, match="was built from snapshot None"):
        snapshot.check(dict(path=built["root"] / "shapes", snapshot=record))


def test_run_refuses_a_stale_catalogue_or_an_existing_sample(built):
    root = built["root"]
    with pytest.raises(FileExistsError):
        pipeline.run(built["recipe"])
    config = yaml.safe_load(built["recipe"].read_text())

    def rerun(name, manifest):
        snap = root / name
        snap.mkdir()
        (snap / "MANIFEST.json").write_text(json.dumps(manifest))
        recipe = root / f"{name}.yaml"
        recipe.write_text(yaml.safe_dump(config | dict(snapshot=str(snap))))
        return pipeline.run(recipe, root / f"{name}-out")

    manifest = json.loads((root / "snap/MANIFEST.json").read_text())
    # The snapshot grew after the catalogue was assembled from it.
    grown = json.loads(json.dumps(manifest))
    grown["tiles"]["203.290"] = {"objects": 10}
    with pytest.raises(ValueError, match="was not built from the manifest"):
        rerun("grown", grown)
    with pytest.raises(ValueError, match="not complete"):
        rerun("partial", manifest | dict(status="partial"))


def test_compare_pairs_rows_and_finds_a_changed_shape(built, tmp_path):
    target = built["root"] / "release/catalogs/calibrated"
    same = compare.compare(target, target, workers=2)
    assert same["agree"] and same["only_new"] == same["only_reference"] == 0
    assert same["paired"]["by_id"] == built["result"]["selected_rows"]

    data = io.read_catalogue(target)
    other = data.copy()
    other["e1"][5] += 1e-6
    a, b, by_id = compare.pair(data, other[::-1])
    assert by_id == len(data)
    npt.assert_array_equal(data[OBJECT_ID][a], other[::-1][OBJECT_ID][b])
    # A reference row without an ID is paired on its position.
    legacy = np.empty(
        len(data), dtype=[*data.dtype.descr[:2], ("NUMBER", "i8"), ("TILE_ID", "f8")]
    )
    for name in ("RA", "Dec", "NUMBER", "TILE_ID"):
        legacy[name] = data[name]
    legacy["NUMBER"][:4] = -1
    a, b, by_id = compare.pair(data, legacy)
    assert (by_id, len(a)) == (len(data) - 4, len(data))
    npt.assert_array_equal(data["RA"][a], legacy["RA"][b])

    changed = tmp_path / "changed"
    import shutil

    shutil.copytree(target, changed)
    file = next(changed.glob("dataset/Norder=*/Dir=*/Npix=*.parquet"))
    table = pq.read_table(file)
    e1 = table["e1"].to_numpy().copy()
    e1[0] += 1e-6
    table = table.set_column(table.schema.get_field_index("e1"), "e1", [e1])
    pq.write_table(table, file)
    result = compare.compare(changed, target)
    assert not result["agree"]
    assert result["largest_difference"]["e1"] == pytest.approx(1e-6, rel=1e-6)
    assert result["largest_difference"]["e2"] == 0
