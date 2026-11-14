import fitsio
import healpy as hp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sp_validation import nz_inputs

MAGS = nz_inputs.MAGS


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
    cal = nz_inputs.calibrators(
        tmp_path, "test", ["deep2_final", "deep3_final"], z_min=0.002, offset=-0.1
    )
    np.testing.assert_allclose(cal["z"], [0.5, 0.9], rtol=1e-6)
    np.testing.assert_array_equal(cal["survey"], ["deep2_final", "deep3_final"])
    np.testing.assert_allclose(cal["mag"][1, 4], 20.0, rtol=1e-6)
    assert cal["mag"].shape == (2, 5)


def test_sources_join_by_position_in_catalogue_order(tmp_path):
    ra = np.array([10.0, 10.001, 10.002, 50.0])
    dec = np.array([5.0, 5.0, 5.001, -3.0])
    pixel = hp.ang2pix(2**29, ra, dec, nest=True, lonlat=True)
    for root in ("shapes", "photo"):
        (tmp_path / root / "dataset" / "Norder=0" / "Dir=0").mkdir(parents=True)
    # Shapes hold the catalogue objects (out of order) plus one it lacks.
    shape_rows = [2, 0, 3, 1]
    extra_ra, extra_dec = 10.0005, 5.0
    pq.write_table(
        pa.table(
            {
                "_healpix_29": np.r_[
                    pixel[shape_rows],
                    hp.ang2pix(2**29, extra_ra, extra_dec, nest=True, lonlat=True),
                ],
                "XWIN_WORLD": np.r_[ra[shape_rows], extra_ra],
                "YWIN_WORLD": np.r_[dec[shape_rows], extra_dec],
                "TILE_ID": np.full(5, 1.001),
                "NUMBER": np.array([30, 10, 40, 20, 99], np.int32),
            }
        ),
        tmp_path / "shapes/dataset/Norder=0/Dir=0/Npix=0.parquet",
    )
    photo = {m: np.full(5, 20.0 + k) for k, m in enumerate(MAGS)}
    photo["Z_B"] = np.array([0.3, 0.1, 0.4, 0.2, 9.0])
    photo["PHOTOPIPE_STATUS"] = np.array([0, 0, 2, 0, 0], np.int8)
    pq.write_table(
        pa.table(photo), tmp_path / "photo/dataset/Norder=0/Dir=0/Npix=0.parquet"
    )
    catalogue = np.zeros(4, dtype=[("RA", "f8"), ("Dec", "f8"), ("w_des", "f8")])
    catalogue["RA"], catalogue["Dec"], catalogue["w_des"] = ra, dec, [1, 2, 3, 4]
    path = str(tmp_path / "catalogue.fits")
    fitsio.write(path, catalogue)

    out = nz_inputs.sources(path, tmp_path / "shapes", tmp_path / "photo", workers=1)
    np.testing.assert_array_equal(out["number"], [10, 20, 30, 40])
    np.testing.assert_allclose(out["photoz"], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_array_equal(out["status"], [0, 0, 0, 2])
    np.testing.assert_array_equal(out["weight"], [1, 2, 3, 4])
    assert out["mag"].shape == (4, 6)
