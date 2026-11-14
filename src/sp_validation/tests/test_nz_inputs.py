import fitsio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from sp_validation.nz import inputs

MAGS = inputs.MAGS


def _specz_tile(directory, tile, ra, dec, z, mags, surveys):
    n = len(ra)
    data = np.zeros(
        n,
        dtype=[
            ("SeqNr", "i4"),
            ("ALPHA_J2000_spec", "f8"),
            ("DELTA_J2000_spec", "f8"),
            ("z_spec_spec", "f4"),
            ("Z_B", "f8"),
        ]
        + [(m, "f4") for m in MAGS],
    )
    data["SeqNr"] = np.arange(1, n + 1)
    data["ALPHA_J2000_spec"], data["DELTA_J2000_spec"] = ra, dec
    data["z_spec_spec"], data["Z_B"] = z, z
    for k, m in enumerate(MAGS):
        data[m] = mags[:, k]
    fitsio.write(
        str(directory / f"UNIONS.{tile}_ugriz_photoz_ext_specz.cat"), data, clobber=True
    )
    lines = [" ra dec zspec zspec_err wspec ref"]
    lines += [
        f"{i} {r:.10f} {d:.10f} {zz} 0.0 1.0 {s}"
        for i, (r, d, zz, s) in enumerate(zip(ra, dec, z, surveys))
    ]
    (directory / f"UNIONS.{tile}_redshifts-test.asc").write_text("\n".join(lines))


def test_calibrators_select_survey_photometry_and_deduplicate(tmp_path):
    good = np.tile([22.0, 21.5, 21.0, 20.5, 20.0, -99.0], (4, 1))
    z2_only = good[0].copy()
    z2_only[4], z2_only[5] = 99.0, 20.1
    no_u = good[0].copy()
    no_u[0] = -99.0
    ra = np.array([214.1, 214.2, 214.3, 214.4])
    dec = np.full(4, 52.5)
    _specz_tile(
        tmp_path,
        "260.286",
        ra,
        dec,
        np.array([0.5, 0.6, 0.7, 0.0001]),
        np.vstack([good[:2], no_u, good[3]]),
        ["deep2_final", "DESI_dr1_iron", "deep3_final", "deep2_final"],
    )
    # The first spectrum again, in an overlapping tile, and a z2-only one.
    _specz_tile(
        tmp_path,
        "261.286",
        np.array([ra[0], 214.5]),
        np.array([dec[0], 52.5]),
        np.array([0.5, 0.9]),
        np.vstack([good[0], z2_only]),
        ["deep2_final", "deep3_final"],
    )
    cal = inputs.calibrators(
        tmp_path, "test", ["deep2_final", "deep3_final"], z_min=0.002, offset=-0.1
    )
    np.testing.assert_allclose(cal["z"], [0.5, 0.9], rtol=1e-6)
    np.testing.assert_array_equal(cal["survey"], ["deep2_final", "deep3_final"])
    np.testing.assert_allclose(cal["mag"][1, 4], 20.0, rtol=1e-6)
    assert cal["mag"].shape == (2, 5)


def _collection(root, object_ids, photoz, status, mags):
    """Write a one-pixel collection: shapes with OBJECT_ID, PhotoPipe beside."""
    properties = "hats_npix_suffix=.parquet\n"
    for name, table in (
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
    ):
        directory = root / name
        (directory / "dataset/Norder=0/Dir=0").mkdir(parents=True)
        pq.write_table(
            pa.table(table), directory / "dataset/Norder=0/Dir=0/Npix=0.parquet"
        )
        (directory / "hats.properties").write_text(properties)
        (directory / "partition_info.csv").write_text("Norder,Npix\n0,0\n")
    (root / "collection.properties").write_text(
        "hats_primary_table_url=shapes\nall_extensions=photopipe photopipe.properties\n"
    )
    (root / "photopipe.properties").write_text(
        "hats_assn_join_table_url=photopipe/photopipe\nhats_col_assn_join=OBJECT_ID\n"
        "hats_col_assn_primary=OBJECT_ID\n"
        f"hats_ext_cols=PHOTOPIPE_STATUS Z_B {' '.join(MAGS)}\n"
    )


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
    # The collection holds the first four out of order, plus an object the
    # catalogue lacks and one with a null ID.
    ids = pa.array([2002000010, 1001000020, 9009000001, 2002000040, None, 1001000010])
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
