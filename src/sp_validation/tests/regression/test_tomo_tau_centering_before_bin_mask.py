"""Tomographic tau statistics must centre galaxy ellipticities within each bin.

The tomography branch passes a bin mask to TauStat. The mean subtracted must
be the selected bin's weighted mean, as in CovTauTh's per-bin covariance.
The shared dependency exposes this mask API on develop as well.
"""

import numpy as np
import pytest

rts = pytest.importorskip("shear_psf_leakage.rho_tau_stat")

# e1 is constant in each bin; correct centring gives g1 == 0. The PSF has
# e2 == 0, so galaxy e2 noise cannot contribute to tau_0+ = <g1 p1 + g2 p2>.
E1_BIN = {1: 0.10, 2: 0.30}
E1_PSF = 0.10


def _params():
    return {
        "e1_col": "e1",
        "e2_col": "e2",
        "w_col": "w",
        "ra_col": "RA",
        "dec_col": "Dec",
        "ra_PSF_col": "RA",
        "dec_PSF_col": "Dec",
        "e1_PSF_col": "E1_PSF",
        "e2_PSF_col": "E2_PSF",
        "e1_star_col": "E1_STAR",
        "e2_star_col": "E2_STAR",
        "PSF_size": "T_PSF",
        "star_size": "T_STAR",
        "patch_number": 2,
        "patch_seed": 1,
        "ra_units": "deg",
        "dec_units": "deg",
    }


def _galaxies(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    gal = np.zeros(
        n,
        dtype=[
            ("RA", "f8"),
            ("Dec", "f8"),
            ("e1", "f8"),
            ("e2", "f8"),
            ("w", "f8"),
            ("bin", "i4"),
        ],
    )
    gal["RA"] = rng.uniform(10, 11, n)
    gal["Dec"] = rng.uniform(0, 1, n)
    gal["bin"] = np.where(np.arange(n) % 2 == 0, 1, 2)
    gal["e1"] = np.where(gal["bin"] == 1, E1_BIN[1], E1_BIN[2])
    gal["e2"] = rng.normal(0, 0.01, n)
    gal["w"] = 1.0
    return gal


def _stars(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    st = np.zeros(
        n,
        dtype=[
            ("RA", "f8"),
            ("Dec", "f8"),
            ("E1_PSF", "f8"),
            ("E2_PSF", "f8"),
            ("E1_STAR", "f8"),
            ("E2_STAR", "f8"),
            ("T_PSF", "f8"),
            ("T_STAR", "f8"),
        ],
    )
    st["RA"] = rng.uniform(10, 11, n)
    st["Dec"] = rng.uniform(0, 1, n)
    st["E1_PSF"] = st["E1_STAR"] = E1_PSF + rng.normal(0, 0.01, n)
    st["E2_STAR"] = rng.normal(0, 0.01, n)  # Non-degenerate tau_2/tau_5 fields.
    st["T_PSF"] = 1.0
    st["T_STAR"] = 1.0 + rng.normal(0, 0.01, n)
    return st


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CosmoStat/shear_psf_leakage#52: centring precedes bin masking",
)
def test_masked_galaxy_catalog_is_centred_on_bin_mean():
    """A bin's TreeCorr galaxy catalogue must have zero weighted mean e1.

    Bin 2 has constant e1 = 0.3 and the full catalogue mean is 0.2.
    Subtracting the bin mean gives zero by construction; centring before
    masking instead leaves g1 = +0.1 on every bin-2 galaxy.
    """
    gal = _galaxies()
    cats = rts.Catalogs(params=_params())
    cats.build_catalog(gal, "gal", "gal_b2", mask=gal["bin"] == 2, npatch=2)
    tc = cats.get_cat("gal_b2")
    mean_g1 = np.average(tc.g1, weights=tc.w)
    assert mean_g1 == pytest.approx(0.0, abs=1e-12), (
        f"bin-2 mean g1 = {mean_g1:.6f}; expected 0 after removing bin mean "
        f"{E1_BIN[2]}, not the full-catalogue mean"
    )


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="CosmoStat/shear_psf_leakage#52: full-catalogue mean biases bin tau",
)
def test_tomographic_tau0_plus_vanishes_for_bin_constant_ellipticity(tmp_path):
    """TauStat must give tau_0+ = 0 for a bin with constant galaxy e1.

    Per-bin centring makes g1 identically zero and the PSF has p2 = 0,
    so <g1 p1 + g2 p2> vanishes at every scale up to TreeCorr round-off.
    Full-catalogue centring leaves (0.3 - 0.2) * 0.1 = 0.01 instead.
    """
    gal, st = _galaxies(), _stars()
    cfg = {
        "ra_units": "deg",
        "dec_units": "deg",
        "sep_units": "arcmin",
        "min_sep": 1.0,
        "max_sep": 30.0,
        "nbins": 5,
    }
    tau = rts.TauStat(params=_params(), output=str(tmp_path), treecorr_config=cfg)
    tau.build_cat_to_compute_tau(st, cat_type="psf", catalog_id="v")
    tau.build_cat_to_compute_tau(
        gal, cat_type="gal", catalog_id="v", mask=gal["bin"] == 2
    )
    tau.compute_tau_stats("v", "tau_v.fits", var_method=None)
    tau0p = np.asarray(tau.tau_stats["tau_0_p"])
    assert np.allclose(tau0p, 0.0, atol=1e-6), (
        f"bin-2 tau_0+ = {np.round(tau0p, 6).tolist()}; expected zero at all scales"
    )
