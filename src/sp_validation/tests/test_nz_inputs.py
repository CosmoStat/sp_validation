import fitsio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sp_validation.nz import inputs

MAGS = inputs.MAGS


def _collection(root, object_ids, photoz, status, mags, specz=None):
    """Write a one-pixel collection: shapes with OBJECT_ID, PhotoPipe beside,
    and, given ``specz`` (columns of its rows), a sparse spec-z extension."""
    properties = "hats_npix_suffix=.parquet\n"
    tables = [
        ("shapes", {"OBJECT_ID": object_ids}),
        (
            "photopipe/photopipe",
            {
                "OBJECT_ID": object_ids,
                "PHOTOPIPE_STATUS": pa.array(status, pa.int8()),
                "Z_B": photoz,
                **{m: pa.array(mags[:, k], pa.float32()) for k, m in enumerate(MAGS)},
            },
        ),
    ]
    extensions = ["photopipe"]
    if specz is not None:
        tables.append(("specz/specz", specz))
        extensions.append("specz")
    for name, table in tables:
        directory = root / name
        (directory / "dataset/Norder=0/Dir=0").mkdir(parents=True)
        pq.write_table(
            pa.table(table), directory / "dataset/Norder=0/Dir=0/Npix=0.parquet"
        )
        (directory / "hats.properties").write_text(properties)
        (directory / "partition_info.csv").write_text("Norder,Npix\n0,0\n")
    listed = " ".join(f"{e} {e}.properties" for e in extensions)
    (root / "collection.properties").write_text(
        f"hats_primary_table_url=shapes\nall_extensions={listed}\n"
    )
    columns = {
        "photopipe": f"PHOTOPIPE_STATUS Z_B {' '.join(MAGS)}",
        "specz": "Z_SPEC Z_SPEC_SURVEY Z_SPEC_RA Z_SPEC_DEC",
    }
    for extension in extensions:
        (root / f"{extension}.properties").write_text(
            f"hats_assn_join_table_url={extension}/{extension}\n"
            "hats_col_assn_join=OBJECT_ID\nhats_col_assn_primary=OBJECT_ID\n"
            f"hats_ext_cols={columns[extension]}\n"
        )


def test_calibrators_select_survey_field_photometry_and_deduplicate(tmp_path):
    """A calibrator is a spectrum from a listed survey, inside the field, on an
    object with complete ugriz (z filled from z2), kept once."""
    good = [22.0, 21.5, 21.0, 20.5, 20.0, -99.0]
    z2_only = [22.0, 21.5, 21.0, 20.5, 99.0, 20.1]
    no_u = [-99.0, 21.5, 21.0, 20.5, 20.0, 20.0]
    # Objects 1-7 carry spectra; 8 has none. Object 6 is object 1's spectrum
    # matched again in an overlapping tile.
    ids = pa.array(np.arange(1, 9, dtype=np.int64))
    mags = np.array([good, good, no_u, good, z2_only, good, good, good])
    ra = [214.1, 214.2, 214.3, 214.4, 214.5, 214.1, 252.5]
    dec = [52.5, 52.5, 52.5, 52.5, 52.5, 52.5, 35.0]
    specz = dict(
        OBJECT_ID=pa.array([7, 6, 5, 4, 3, 2, 1], pa.int64()),
        Z_SPEC=pa.array([0.8, 0.5, 0.9, 0.0001, 0.7, 0.6, 0.5], pa.float32()),
        Z_SPEC_SURVEY=[
            "deep2_final",
            "deep2_final",
            "deep3_final",
            "deep2_final",
            "deep3_final",
            "DESI_dr1_iron",
            "deep2_final",
        ],
        Z_SPEC_RA=ra[::-1],
        Z_SPEC_DEC=dec[::-1],
    )
    _collection(tmp_path, ids, np.full(8, 0.5), [0] * 8, mags, specz=specz)
    cal = inputs.calibrators(
        tmp_path,
        ["deep2_final", "deep3_final"],
        {"ra": [213.0, 217.0], "dec": [51.5, 54.0]},
        z_min=0.002,
        offset=-0.1,
    )
    np.testing.assert_array_equal(cal["object_id"], [1, 5])
    np.testing.assert_allclose(cal["z"], [0.5, 0.9], rtol=1e-6)
    np.testing.assert_array_equal(cal["survey"], ["deep2_final", "deep3_final"])
    np.testing.assert_allclose(cal["mag"][1, 4], 20.0, rtol=1e-6)
    assert cal["mag"].shape == (2, 5)


def _catalogue(path, tile, number, weight):
    data = np.zeros(
        len(number), dtype=[("TILE_ID", "f8"), ("NUMBER", "i4"), ("w_des", "f8")]
    )
    data["TILE_ID"], data["NUMBER"], data["w_des"] = tile, number, weight
    fitsio.write(str(path), data)


def test_sources_join_on_object_id_in_catalogue_order(tmp_path):
    """Each catalogue row takes the photometry of its own ID, wherever that row
    sits in the collection, and whatever lies at its position."""
    tile = np.array([1.001, 1.001, 2.002, 2.002, 1.001, 3.003])
    number = np.array([10, 20, 10, 40, -1, 7], np.int32)
    _catalogue(tmp_path / "catalogue.fits", tile, number, [1, 2, 3, 4, 5, 6])
    # The collection holds the first four out of order, plus two objects the
    # catalogue lacks (its OBJECT_ID has no nulls: the index is built on it).
    ids = pa.array(
        [2002000010, 1001000020, 9009000001, 2002000040, 9009000002, 1001000010]
    )
    photoz = np.array([0.3, 0.2, 9.0, 0.4, 8.0, 0.1])
    mags = np.arange(6)[:, None] + np.full((6, 6), 20.0)
    _collection(tmp_path / "collection", ids, photoz, [0, 0, 0, 2, 0, 0], mags)

    out = inputs.sources(tmp_path / "catalogue.fits", tmp_path / "collection")
    np.testing.assert_allclose(out["photoz"], [0.1, 0.2, 0.3, 0.4, np.nan, np.nan])
    np.testing.assert_array_equal(
        out["status"], [0, 0, 0, 2, inputs.NO_ID, inputs.UNMATCHED]
    )
    np.testing.assert_array_equal(out["weight"], [1, 2, 3, 4, 5, 6])
    np.testing.assert_allclose(out["mag"][:4, 0], [25, 21, 20, 23])
    assert np.isnan(out["mag"][4:]).all()
    np.testing.assert_array_equal(out["object_id"][[0, 4]], [1001000010, -1])


def test_sources_refuse_a_repeated_object_id(tmp_path):
    _catalogue(tmp_path / "catalogue.fits", [1.001, 1.001], [10, 10], [1, 1])
    with pytest.raises(ValueError, match="repeats an OBJECT_ID"):
        inputs.sources(tmp_path / "catalogue.fits", tmp_path / "none")
    _catalogue(tmp_path / "single.fits", [1.001], [10], [1])
    ids = pa.array([1001000010, 1001000010])
    _collection(tmp_path / "collection", ids, np.zeros(2), [0, 0], np.zeros((2, 6)))
    with pytest.raises(ValueError, match="repeats 1 OBJECT_IDs"):
        inputs.sources(tmp_path / "single.fits", tmp_path / "collection")


def test_match_report_partitions_the_shear_weight():
    good = [22.0, 21.5, 21.0, 20.5, 20.0, -99.0]
    z2_only = [22.0, 21.5, 21.0, 20.5, 99.0, 20.1]
    no_u = [-99.0, 21.5, 21.0, 20.5, 20.0, 20.0]
    no_z = [22.0, 21.5, 21.0, 20.5, 99.0, -99.0]
    nan = [np.nan] * 6
    mag = np.array([good, z2_only, no_u, no_z, nan, nan, nan, nan])
    status = np.array([0, 0, 0, 0, 1, 2, inputs.UNMATCHED, inputs.NO_ID])
    weight = np.array([1, 2, 3, 4, 5, 6, 7, 8.0])
    report = inputs.match_report(status, mag, weight)
    undetected = report.pop("undetected")
    assert report == dict(
        no_object_id=8 / 36,
        unmatched_by_id=7 / 36,
        tile_without_photopipe=6 / 36,
        object_without_photopipe=5 / 36,
        incomplete=7 / 36,
        complete=3 / 36,
    )
    assert sum(report.values()) == pytest.approx(1)
    assert undetected == dict(u=3 / 36, g=0, r=0, i=0, z=4 / 36)
