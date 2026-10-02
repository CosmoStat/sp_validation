"""Rho/tau must exclude stars whose configured HSM PSF or star fit failed.

Clean stars are perfectly modelled, making rho_1 and tau_2 zero by construction.
"""

import inspect

import numpy as np
import pytest
from astropy.io import fits

from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin

VERSION = "SP_synthetic"
N_GOOD = 2000
N_FLAGGED = 100
N_GAL = 3000


def _write_catalogues(tmp_path):
    rng = np.random.default_rng(1)
    n = N_GOOD + N_FLAGGED
    ra = rng.uniform(10.0, 12.0, n)
    dec = rng.uniform(0.0, 2.0, n)
    e1_psf = rng.normal(0.0, 0.02, n)
    e2_psf = rng.normal(0.0, 0.02, n)
    t_psf = rng.uniform(0.4, 0.6, n)

    # Good stars: q = e_star - e_PSF = 0 and T_star = T_PSF exactly.
    e1_star = e1_psf.copy()
    e2_star = e2_psf.copy()
    t_star = t_psf.copy()
    flag_psf = np.zeros(n, dtype=np.int16)
    flag_star = np.zeros(n, dtype=np.int16)

    # Failed fits contribute q = (-0.05, 0) if either flag is ignored.
    bad = slice(N_GOOD, n)
    e1_psf[bad] = 0.05
    e2_psf[bad] = 0.0
    e1_star[bad] = 0.0
    e2_star[bad] = 0.0
    t_star[bad] = 0.069
    flag_star[N_GOOD : N_GOOD + N_FLAGGED // 2] = 1
    flag_psf[N_GOOD + N_FLAGGED // 2 : n] = 1

    psf_path = tmp_path / "psf.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column("RA", "D", array=ra),
            fits.Column("DEC", "D", array=dec),
            fits.Column("HSM_G1_PSF", "D", array=e1_psf),
            fits.Column("HSM_G2_PSF", "D", array=e2_psf),
            fits.Column("HSM_G1_STAR", "D", array=e1_star),
            fits.Column("HSM_G2_STAR", "D", array=e2_star),
            fits.Column("HSM_T_PSF", "D", array=t_psf),
            fits.Column("HSM_T_STAR", "D", array=t_star),
            fits.Column("HSM_FLAG_PSF", "I", array=flag_psf),
            fits.Column("HSM_FLAG_STAR", "I", array=flag_star),
        ]
    ).writeto(psf_path)

    shear_path = tmp_path / "shear.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column("RA", "D", array=rng.uniform(10.0, 12.0, N_GAL)),
            fits.Column("Dec", "D", array=rng.uniform(0.0, 2.0, N_GAL)),
            fits.Column("e1", "D", array=rng.normal(0.0, 0.3, N_GAL)),
            fits.Column("e2", "D", array=rng.normal(0.0, 0.3, N_GAL)),
            fits.Column("w_des", "D", array=np.ones(N_GAL)),
        ]
    ).writeto(shear_path)
    return psf_path, shear_path


class _Validation(PSFSystematicsMixin):
    """Minimal host carrying the configuration read by the real entry point."""

    def __init__(self, cc):
        self.cc = cc
        self.versions = [VERSION]
        self.treecorr_config = {
            "ra_units": "deg",
            "dec_units": "deg",
            "sep_units": "arcmin",
            "min_sep": 2.0,
            "max_sep": 100.0,
            "nbins": 5,
            "num_threads": 2,
        }
        self.cov_estimate_method = "sim"
        self.compute_cov_rho = False
        self.npatch = 4

    def basename(self, version, tomo_bin_a="all", **kwargs):
        return f"{version}_{tomo_bin_a}"

    def _get_galaxy_mask(self, ver, tomo_bin_id):
        return np.ones(N_GAL, dtype=bool)

    def rho_tau_to_sacc_part(self, *args, **kwargs):
        pass

    def print_start(self, *args, **kwargs):
        pass

    print_done = print_cyan = print_magenta = print_start


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#374: rho/tau includes stars with nonzero HSM fit flags",
)
def test_flagged_stars_excluded_from_rho_1_and_tau_2(tmp_path):
    """Both configured HSM flags must exclude failed fits from rho/tau.

    Every unflagged star has e_star == e_PSF exactly, so q is identically zero:
    rho_1 = <q q> and tau_2 = <e_gal q> must be zero in every bin.
    Half of the bad stars have only HSM_FLAG_STAR set, and half have only
    HSM_FLAG_PSF set; their q = (-0.05, 0) catches ignoring either flag.
    This merges the two audit reproductions and exercises the pipeline's
    calculate_rho_tau_stats entry point, not a separately masked reference.
    """
    psf_path, shear_path = _write_catalogues(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    cc = {
        "paths": {"output": str(out)},
        VERSION: {
            "patch_number": 4,
            "cov_th": {"A": 4.0, "n_e": 1.0, "n_psf": 1.0, "sigma_e": 0.3},
            "psf": {
                "path": str(psf_path),
                "hdu": 1,
                "ra_col": "RA",
                "dec_col": "DEC",
                "e1_PSF_col": "HSM_G1_PSF",
                "e2_PSF_col": "HSM_G2_PSF",
                "e1_star_col": "HSM_G1_STAR",
                "e2_star_col": "HSM_G2_STAR",
                "PSF_size": "HSM_T_PSF",
                "star_size": "HSM_T_STAR",
                "PSF_flag": "HSM_FLAG_PSF",
                "star_flag": "HSM_FLAG_STAR",
            },
            "shear": {
                "path": str(shear_path),
                "ra_col": "RA",
                "dec_col": "Dec",
                "w_col": "w_des",
                "e1_col": "e1",
                "e2_col": "e2",
            },
        },
    }
    rho_tau_dir = out / "rho_tau_stats"
    rho_tau_dir.mkdir()
    # The sim method only needs an existing covariance; don't run an estimator.
    np.save(rho_tau_dir / f"cov_tau_{VERSION}_all_th.npy", np.eye(3))

    val = _Validation(cc)
    kwargs = (
        {"tomography": False}
        if "tomography" in inspect.signature(val.calculate_rho_tau_stats).parameters
        else {}
    )
    val.calculate_rho_tau_stats(**kwargs)

    rho_1 = np.asarray(val._rho_stat_handler.rho_stats["rho_1_p"])
    tau_2 = np.asarray(val._tau_stat_handler.tau_stats["tau_2_p"])
    np.testing.assert_allclose(rho_1, 0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(tau_2, 0, atol=1e-12, rtol=0)
