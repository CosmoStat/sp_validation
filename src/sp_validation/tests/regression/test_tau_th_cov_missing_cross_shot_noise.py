"""CovTauTh shot noise must fill the tau cross blocks (02, 05 and 25)."""

import numpy as np
import pytest
from shear_psf_leakage.rho_tau_cov import CovTauTh

PARAMS = dict(
    e1_col="e1",
    e2_col="e2",
    w_col="w",
    ra_col="ra",
    dec_col="dec",
    ra_PSF_col="ra",
    dec_PSF_col="dec",
    e1_PSF_col="p1",
    e2_PSF_col="p2",
    e1_star_col="s1",
    e2_star_col="s2",
    PSF_size="tp",
    star_size="ts",
    PSF_flag="fp",
    star_flag="fs",
    patch_number=1,
    ra_units="deg",
    dec_units="deg",
    R11=None,
    R22=None,
)
TC_CONFIG = dict(
    min_sep=1.0,
    max_sep=100.0,
    nbins=1,
    sep_units="arcmin",
    ra_units="deg",
    dec_units="deg",
    bin_slop=0,
    angle_slop=0,
)


def _identical_field_catalogue(n=100, seed=8):
    """Make the PSF, residual and size-error fields equal to the same field p.

    Star = 2 * PSF gives residual = PSF. T_star = 2 * T_PSF gives size
    residual = 1/2 and size-error ellipticity = star * 1/2 = PSF.
    """
    rng = np.random.default_rng(seed)
    cols = [
        "ra",
        "dec",
        "e1",
        "e2",
        "w",
        "p1",
        "p2",
        "s1",
        "s2",
        "tp",
        "ts",
        "fp",
        "fs",
    ]
    a = np.zeros(n, dtype=[(k, "f8") for k in cols])
    a["ra"] = rng.uniform(0, 1, n)
    a["dec"] = rng.uniform(0, 1, n)
    a["w"] = 1.0
    for col, scale in [("e1", 0.3), ("e2", 0.3), ("p1", 0.03), ("p2", 0.03)]:
        a[col] = rng.normal(0, scale, n)
    a["s1"], a["s2"] = 2 * a["p1"], 2 * a["p2"]
    a["tp"], a["ts"] = 1.0, 2.0
    return a


@pytest.fixture(scope="module")
def cov_identical():
    a = _identical_field_catalogue()
    c = CovTauTh(a, a, 1, TC_CONFIG, params=PARAMS, use_eta=True, nside=8)
    # Check that the covariance builder really sees identical fields.
    for other in (c.psf_error, c.size_error):
        np.testing.assert_allclose(other.g1, c.psf.g1, rtol=0, atol=1e-15)
        np.testing.assert_allclose(other.g2, c.psf.g2, rtol=0, atol=1e-15)
    return c


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CosmoStat/shear_psf_leakage#49: tau cross blocks omit correlated shot noise",
)
@pytest.mark.parametrize("cross,auto", [("02", "00"), ("05", "00"), ("25", "22")])
def test_tau_cross_shot_noise_equals_auto_for_identical_fields(
    cov_identical, cross, auto
):
    """Cross shot noise must equal auto shot noise for identical star fields.

    tau_a = sum_ij w_i e_i p_a,j / W, so galaxy shape noise contributes
    sigma_e^2 <p_a . p_b> / N_pairs to Cov(tau_a, tau_b). Here the PSF,
    residual and size-error fields are identical by construction: tau_0,
    tau_2 and tau_5 are the same random variable. Every cross shot-noise
    block must therefore equal the auto block, rather than zero.
    """
    sn_auto = np.asarray(cov_identical.compute_sn(auto))
    sn_cross = np.asarray(cov_identical.compute_sn(cross))
    assert sn_auto[0, 0] > 0
    np.testing.assert_allclose(
        sn_cross,
        sn_auto,
        rtol=1e-10,
        err_msg=(
            f"shot noise block {cross} = {sn_cross.ravel()}; identical fields "
            f"require block {auto} = {sn_auto.ravel()}"
        ),
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CosmoStat/shear_psf_leakage#49: missing cross shot noise breaks tau covariance",
)
def test_tau_th_cov_difference_of_identical_taus_has_zero_variance():
    """Full covariance must give Var(tau_0 - tau_2) = 0 for identical fields.

    Residual == PSF implies tau_2 == tau_0 exactly, so the assembled
    covariance must satisfy C00 + C22 - 2 C02 = 0. The mixed and cosmic-
    variance terms already do; omitted cross shot noise breaks this identity.
    """
    a = _identical_field_catalogue()
    c = CovTauTh(a, a, 1, TC_CONFIG, params=PARAMS, use_eta=False, nside=8)
    cov = c.build_cov(nbin_ang=8, nbin_rad=8, compute_minus=False)
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    assert abs(var_diff) <= 1e-6 * cov[0, 0], (
        f"Var(tau0 - tau2) = {var_diff:.4e} for identical estimators "
        f"(Var(tau0) = {cov[0, 0]:.4e}); expected 0"
    )
