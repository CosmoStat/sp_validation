"""xi+/- and Map^2 must reflect the catalogue they are asked about, not a stale .txt."""

import numpy as np
import pytest
import yaml
from astropy.table import Table

from sp_validation.cosmo_val import CosmologyValidation

TOMO = hasattr(CosmologyValidation, "calculate_2pcf_version")
SCALE = 2.0  # second catalogue: every ellipticity doubled -> xi and Map^2 x4


def _write_inputs(tmp_path, scale):
    rng = np.random.default_rng(901)
    n = 600
    ra = rng.uniform(10, 14, n)
    dec = rng.uniform(10, 14, n)
    w = rng.uniform(0.5, 2, n)
    e1 = scale * rng.normal(0.02, 0.08, n)
    e2 = scale * rng.normal(0.01, 0.08, n)
    Table(dict(RA=ra, Dec=dec, e1=e1, e2=e2, w=w, tomo=np.where(ra < 12, 1, 2))).write(
        tmp_path / "shear.fits", overwrite=True
    )
    Table(dict(RA=ra[:20], Dec=dec[:20], e1=e1[:20], e2=e2[:20])).write(
        tmp_path / "star.fits", overwrite=True
    )
    dndz = tmp_path / "dndz_SP_A.txt"
    np.savetxt(dndz, np.array([[0.1, 1.0], [0.5, 1.0], [1.0, 0.1]]), header="z dn_dz")
    cfg = {
        "nz": {
            "subdir": str(tmp_path),
            "dndz": {
                "path": str(tmp_path / "dndz") if TOMO else str(dndz),
                "blind": "A",
            },
        },
        "paths": {"output": str(tmp_path / "out")},
        "Test": {
            "subdir": str(tmp_path),
            "pipeline": "SP",
            "shear": {
                "path": "shear.fits",
                "redshift_path": str(dndz),
                "e1_col": "e1",
                "e2_col": "e2",
                "w_col": "w",
                "R": 1.0,
                "tomo_bin_col": "tomo",
            },
            "star": {
                "path": "star.fits",
                "ra_col": "RA",
                "dec_col": "Dec",
                "e1_col": "e1",
                "e2_col": "e2",
            },
        },
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))


def _cv(tmp_path, out="cv"):
    cv = CosmologyValidation(
        ["Test"],
        catalog_config=str(tmp_path / "config.yaml"),
        output_dir=str(tmp_path / out),
        npatch=1,
        theta_min=5,
        theta_max=100,
        nbins=8,
    )
    cv.treecorr_config.update(num_threads=2, bin_slop=0, angle_slop=0)
    return cv


def _xip(cv):
    if TOMO:
        return cv.calculate_2pcf_version("Test", npatch=1)[
            "tomo_bin_all_tomo_bin_all"
        ].xip
    return cv.calculate_2pcf("Test", npatch=1).xip


def _mapsq(cv):
    cv.calculate_aperture_mass_dispersion(
        theta_min=4, theta_max=80, nbins=20, nbins_map=3, npatch=4
    )
    m = cv.map2["Test"]
    return (m["tomo_bin_all_tomo_bin_all"] if TOMO else m)["mapsq"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#384: xi text cache ignores changed catalogue inputs",
)
def test_calculate_2pcf_recomputes_xi_after_catalogue_changes(tmp_path):
    """Protects xi+/- (and the SACC part run_2pcf writes from it) against a stale cache.

    ``calculate_2pcf`` reuses ``{ver}_xi_minsep=..._npatch=N.txt`` whenever it
    exists, without checking the catalogue, R, columns or TreeCorr settings.
    Here the catalogue is rewritten with every ellipticity doubled and a fresh
    CosmologyValidation is pointed at the same output directory. Additive-bias
    subtraction and the 1/R calibration are linear, so the calibrated shear
    doubles exactly and xi+ must be exactly 4x the first measurement; returning
    the first vector unchanged means the new catalogue was never read.
    """
    _write_inputs(tmp_path, 1.0)
    xi_before = np.asarray(_xip(_cv(tmp_path)))
    _write_inputs(tmp_path, SCALE)
    xi_after = np.asarray(_xip(_cv(tmp_path)))
    expected = SCALE**2 * xi_before
    np.testing.assert_allclose(
        xi_after,
        expected,
        rtol=1e-6,
        err_msg=(
            f"xi+ after doubling the catalogue: first bin {xi_after[0]:.7g}, "
            f"expected {expected[0]:.7g} (= 4 x {xi_before[0]:.7g}); "
            "equals the old vector to text precision: "
            f"{np.allclose(xi_after, xi_before, rtol=1e-4)}"
        ),
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#384: aperture-mass xi text cache ignores changed inputs",
)
def test_aperture_mass_recomputes_xi_after_catalogue_changes(tmp_path):
    """Protects <M_ap^2> against the ``xi_for_map2_{ver}.txt`` cache.

    ``calculate_aperture_mass_dispersion`` reads ``xi_for_map2_{ver}.txt`` if it
    exists; the name carries only the version, so a changed catalogue (or a
    changed theta_min/theta_max/nbins/npatch) silently reuses the old xi. Map^2 is
    quadratic in the shear, so after doubling every ellipticity it must be
    exactly 4x the first measurement.
    """
    _write_inputs(tmp_path, 1.0)
    m_before = np.asarray(_mapsq(_cv(tmp_path)))
    _write_inputs(tmp_path, SCALE)
    m_after = np.asarray(_mapsq(_cv(tmp_path)))
    expected = SCALE**2 * m_before
    np.testing.assert_allclose(
        m_after,
        expected,
        rtol=1e-6,
        err_msg=(
            f"Map^2 after doubling the catalogue: {m_after.tolist()}, "
            f"expected {expected.tolist()}; "
            "equals the old vector to text precision: "
            f"{np.allclose(m_after, m_before, rtol=1e-4)}"
        ),
    )
