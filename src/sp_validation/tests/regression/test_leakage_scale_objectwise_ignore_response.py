"""PSF-leakage estimators must be invariant to whether shear is pre-calibrated.

CosmologyValidation compares alpha(theta), alpha_mean, C_sys and object-wise
leakage with xi+- built from calibrated shear g = (e - c) / R. A catalogue
holding metacal ellipticities e = R g and response columns must give the same
leakage as one holding g with R = 1. Both contain the same synthetic galaxies.
"""

import numpy as np
import pytest
from astropy.table import Table
from shear_psf_leakage import run_scale

from sp_validation.cosmo_val import CosmologyValidation

R_TRUE = 0.7
ALPHA_TRUE = 0.05


def _psf_pattern(ra, dec):
    # Smooth PSF ellipticity field over a one-square-degree patch.
    e1 = 0.02 * np.cos(2 * np.pi * ra) + 0.01 * np.sin(2 * np.pi * dec)
    e2 = 0.02 * np.sin(2 * np.pi * (ra + dec)) - 0.01 * np.cos(np.pi * dec)
    return e1, e2


@pytest.fixture
def cv(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    n = 2000
    ra, dec = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    ep1, ep2 = _psf_pattern(ra, dec)
    g1 = ALPHA_TRUE * ep1 + rng.normal(0, 0.01, n)
    g2 = ALPHA_TRUE * ep2 + rng.normal(0, 0.01, n)
    base = dict(
        RA=ra,
        Dec=dec,
        w=rng.uniform(0.5, 1.5, n),
        e1_psf=ep1,
        e2_psf=ep2,
        fwhm_PSF=np.full(n, 0.7),
    )
    ref_path = tmp_path / "ref.fits"
    Table(dict(base, e1=g1, e2=g2)).write(ref_path)
    des_path = tmp_path / "des.fits"
    Table(
        dict(
            base,
            e1=R_TRUE * g1,
            e2=R_TRUE * g2,
            R11=np.full(n, R_TRUE),
            R22=np.full(n, R_TRUE),
        )
    ).write(des_path)

    rs, ds = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    s1, s2 = _psf_pattern(rs, ds)
    star_path = tmp_path / "star.fits"
    Table(dict(ra=rs, dec=ds, obs_e1=s1, obs_e2=s2)).write(star_path)

    def entry(path, **shear):
        return {
            "pipeline": "SP",
            "shear": dict(
                path=str(path),
                e1_col="e1",
                e2_col="e2",
                w_col="w",
                e1_PSF_col="e1_psf",
                e2_PSF_col="e2_psf",
                **shear,
            ),
            "star": dict(
                path=str(star_path),
                ra_col="ra",
                dec_col="dec",
                e1_col="obs_e1",
                e2_col="obs_e2",
            ),
        }

    out = tmp_path / "out"
    out.mkdir()
    obj = object.__new__(CosmologyValidation)
    obj.versions = ["REF", "DES"]
    obj.cc = {
        "REF": entry(ref_path, R=1.0),
        "DES": entry(des_path, R=1.0, R11="R11", R22="R22"),
        "nz": {"dndz": {"path": str(tmp_path / "dndz"), "blind": "A"}},
        "paths": {"output": str(out)},
    }
    obj.theta_min, obj.theta_max, obj.nbins = 2.0, 40.0, 6
    obj.treecorr_config = {
        "ra_units": "degrees",
        "dec_units": "degrees",
        "min_sep": 2.0,
        "max_sep": 40.0,
        "sep_units": "arcmin",
        "nbins": 6,
    }
    # Theory only feeds the ratio plot, not any tested leakage estimator.
    monkeypatch.setattr(
        run_scale,
        "get_theo_xi",
        lambda theta, path: (np.ones(len(theta)), np.ones(len(theta))),
    )
    yield obj
    # The plotting dependency leaves figures open; don't leak them to the suite.
    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#382: scale leakage ignores shear response",
)
def test_scale_leakage_alpha_and_csys_use_des_response(cv):
    """Alpha and C_sys must agree for calibrated g and e = 0.7 g with R = 0.7.

    These catalogues represent identical shear, so both ratios must be one.
    Ignoring response scales alpha by R and C_sys by R squared instead.
    """
    cv.calculate_scale_dependent_leakage()
    ref, des = cv.results["REF"], cv.results["DES"]
    ratio_alpha = des.alpha_leak_mean / ref.alpha_leak_mean
    ratio_csys = np.median(des.C_sys_p / ref.C_sys_p)
    assert ratio_alpha == pytest.approx(1.0, rel=1e-6), (
        f"alpha_mean DES/REF = {ratio_alpha:.6f}; "
        f"median C_sys_p DES/REF = {ratio_csys:.6f}; expected both to be 1"
    )
    assert ratio_csys == pytest.approx(1.0, rel=1e-6)


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#382: object-wise leakage ignores shear response",
)
def test_objectwise_leakage_a11_a22_use_des_response(cv):
    """Object-wise a11/a22 must match for two encodings of identical shear.

    Regressing e = 0.7 g on e_PSF must account for R = 0.7, because the
    comparison targets calibrated xi. Both coefficient ratios must be one.
    """
    cv.calculate_objectwise_leakage()
    ref, des = cv.leakage_coeff["REF"], cv.leakage_coeff["DES"]
    r11 = des["a11"].n / ref["a11"].n
    r22 = des["a22"].n / ref["a22"].n
    assert (r11, r22) == pytest.approx((1.0, 1.0), rel=1e-6), (
        f"a11 DES/REF = {r11:.6f}, a22 DES/REF = {r22:.6f}; expected 1"
    )
