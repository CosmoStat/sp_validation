"""Configured ``shear.R`` must calibrate tau and pseudo-Cl as well as xi+/-.

The response is 0.5, so e/R doubles the field and quadruples its spectrum.
"""

import inspect

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.cosmo_val.core import CosmologyValidation
from sp_validation.rho_tau import get_params_rho_tau

RESPONSE = 0.5


def _entry(response, path="unused.fits"):
    psf = {
        "ra_col": "RA",
        "dec_col": "Dec",
        "e1_PSF_col": "e1",
        "e2_PSF_col": "e2",
        "e1_star_col": "e1",
        "e2_star_col": "e2",
        "PSF_size": "T",
        "star_size": "T",
        "PSF_flag": "flags",
        "star_flag": "flags",
    }
    shear = {
        "path": path,
        "ra_col": "RA",
        "dec_col": "Dec",
        "e1_col": "e1",
        "e2_col": "e2",
        "w_col": "w",
        "R": response,
    }
    return {"patch_number": 2, "psf": psf, "shear": shear}


def _catalog(n, rng=None):
    cat = np.zeros(n, dtype=[(k, "f8") for k in ["RA", "Dec", "e1", "e2", "w"]])
    if rng is None:
        cat["e1"] = [-0.1, 0.0, 0.1]
        cat["e2"] = [-0.2, 0.0, 0.2]
    else:
        cat["RA"] = rng.uniform(0, 360, n)
        cat["Dec"] = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        cat["e1"] = rng.normal(0, 0.3, n)
        cat["e2"] = rng.normal(0, 0.3, n)
    cat["w"] = 1.0
    return cat


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#382: tau ignores configured scalar shear.R",
)
def test_tau_galaxy_field_divided_by_scalar_response():
    """Tau must correlate e/R, matching the real-space calibrated galaxy field.

    With raw e1=[-0.1,0,0.1], e2=[-0.2,0,0.2] and ``shear.R=0.5``,
    the expected shears are [-0.2,0,0.2] and [-0.4,0,0.4].
    Their weighted means vanish, so mean subtraction cannot change this result.
    The field is built by the same Catalogs/get_params_rho_tau path used for tau
    and the alpha/beta/eta coefficients fitted from it.
    """
    from shear_psf_leakage.rho_tau_stat import Catalogs

    cat = _catalog(3)
    catalogs = Catalogs(params=get_params_rho_tau(_entry(RESPONSE)))
    _, _, g1, g2, _ = catalogs.get_cat_fields(cat, "gal")
    np.testing.assert_allclose(g1, cat["e1"] / RESPONSE)
    np.testing.assert_allclose(g2, cat["e2"] / RESPONSE)


class _Captured(Exception):
    pass


def _catalog_pseudo_cl_inputs(entry, output_path):
    """Capture pipeline inputs before the estimator, without altering calibration."""
    cv = object.__new__(CosmologyValidation)
    cv.cc = {"fixture": entry}
    seen = {}

    def capture(catalog, params, **kwargs):
        seen["catalog"], seen["params"] = catalog, params
        raise _Captured

    cv.get_pseudo_cls_catalog = capture
    sig = inspect.signature(CosmologyValidation.calculate_pseudo_cl_catalog)
    extra = (
        {"tomo_bin_a": "all", "tomo_bin_b": "all"}
        if "tomo_bin_a" in sig.parameters
        else {}
    )
    with pytest.raises(_Captured):
        cv.calculate_pseudo_cl_catalog("fixture", str(output_path), **extra)
    return seen["catalog"], seen["params"]


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#382: pseudo-Cl ignores scalar shear.R",
)
def test_catalog_pseudo_cl_scales_as_inverse_response_squared(tmp_path):
    """Catalogue pseudo-Cl must use calibrated shear, not raw ellipticity.

    Raw e with ``shear.R=0.5`` must give the same Cl as e/0.5 with ``R=1``:
    EE and BB are quadratic in the field, so both are four times the raw-e Cl.
    Both runs use calculate_pseudo_cl_catalog's own catalogue/parameter wiring
    and the actual NaMaster primitive; no calibration is injected by the test.
    """
    import sp_validation.pseudo_cl as spc

    raw = _catalog(4000, np.random.default_rng(1))
    cal = raw.copy()
    cal["e1"] /= RESPONSE
    cal["e2"] /= RESPONSE
    p_raw, p_cal = tmp_path / "raw.fits", tmp_path / "cal.fits"
    fits.writeto(p_raw, raw)
    fits.writeto(p_cal, cal)

    cls = []
    for entry in (_entry(RESPONSE, str(p_raw)), _entry(1.0, str(p_cal))):
        catalog, params = _catalog_pseudo_cl_inputs(entry, tmp_path / "unused_out")
        _, cl, _ = spc.get_pseudo_cls_catalog(
            catalog, params, 32, "linear", ell_step=16
        )
        cls.append(np.asarray(cl))
    np.testing.assert_allclose(
        cls[0],
        cls[1],
        rtol=1e-6,
        atol=0,
        err_msg="Pseudo-Cl ignores the configured scalar response",
    )
