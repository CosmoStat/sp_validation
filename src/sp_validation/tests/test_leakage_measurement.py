"""The joint leakage measurement and its 'eiv' fit through the cosmo_val mixin."""

import json
import os

import numpy as np
import pytest
from astropy.io import fits

from sp_validation import rho_tau
from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin

VERSION = "SP_synthetic"
CIRCLE = {"name": "hole", "ra_deg": 11.0, "dec_deg": 1.0, "radius_deg": 0.3}


def _write_catalogues(tmp_path, n_star=4000, n_gal=6000):
    rng = np.random.default_rng(7)
    ra, dec = rng.uniform(10, 12, n_star), rng.uniform(0, 2, n_star)
    e_psf = rng.normal(0, 0.02, (2, n_star))
    e_star = e_psf + rng.normal(0, 0.01, (2, n_star))
    t_psf = rng.uniform(0.4, 0.6, n_star)
    t_star = t_psf * (1 + rng.normal(0, 0.02, n_star))
    flags = (rng.random(n_star) < 0.05).astype(np.int16)
    psf = tmp_path / "psf.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column("RA", "D", array=ra),
            fits.Column("DEC", "D", array=dec),
            fits.Column("HSM_G1_PSF", "D", array=e_psf[0]),
            fits.Column("HSM_G2_PSF", "D", array=e_psf[1]),
            fits.Column("HSM_G1_STAR", "D", array=e_star[0]),
            fits.Column("HSM_G2_STAR", "D", array=e_star[1]),
            fits.Column("HSM_T_PSF", "D", array=t_psf),
            fits.Column("HSM_T_STAR", "D", array=t_star),
            fits.Column("HSM_FLAG_PSF", "I", array=flags),
            fits.Column("HSM_FLAG_STAR", "I", array=np.zeros(n_star, np.int16)),
        ]
    ).writeto(psf)
    shear = tmp_path / "shear.fits"
    fits.BinTableHDU.from_columns(
        [
            fits.Column("RA", "D", array=rng.uniform(10, 12, n_gal)),
            fits.Column("Dec", "D", array=rng.uniform(0, 2, n_gal)),
            fits.Column("e1", "D", array=rng.normal(0, 0.3, n_gal)),
            fits.Column("e2", "D", array=rng.normal(0, 0.3, n_gal)),
            fits.Column("w_des", "D", array=rng.uniform(0.5, 2, n_gal)),
        ]
    ).writeto(shear)
    return psf, shear


class _Validation(PSFSystematicsMixin):
    """Minimal host with the configuration the leakage methods read."""

    def __init__(self, tmp_path, beta_psf_size=False):
        psf, shear = _write_catalogues(tmp_path)
        (tmp_path / "rho_tau_stats").mkdir()
        mask = tmp_path / "mask.json"
        mask.write_text(json.dumps({"circles": [CIRCLE]}))
        self.output = tmp_path
        self.cc = {
            "paths": {"output": str(tmp_path)},
            VERSION: {
                "patch_number": 30,
                "psf": {
                    "path": str(psf),
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
                    "path": str(shear),
                    "ra_col": "RA",
                    "dec_col": "Dec",
                    "w_col": "w_des",
                    "e1_col": "e1",
                    "e2_col": "e2",
                },
            },
        }
        self.versions = [VERSION]
        self.treecorr_config = {
            "ra_units": "degrees",
            "dec_units": "degrees",
            "sep_units": "arcmin",
            "min_sep": 5.0,
            "max_sep": 60.0,
            "nbins": 3,
            "bin_slop": 0.0,
            "angle_slop": 0.0,
            "num_threads": 1,
        }
        self.psf_leakage = {
            "npatch_fine": 30,
            "patch_counts": [15, 20],
            "fiducial_npatch": 20,
            "beta_psf_size": beta_psf_size,
            "sky_mask": str(mask),
        }
        self.rho_tau_method = "eiv"
        self.force_run = False

    def basename(self, version, tomo_bin_a="all", npatch=None, **kwargs):
        return f"{version}_{tomo_bin_a}_npatch={npatch}"

    def _output_path(self, *parts):
        return os.path.join(self.output, *parts)


def _fresh(val, **psf_leakage):
    """A new host on the same outputs, without in-memory products."""
    fresh = _Validation.__new__(_Validation)
    fresh.__dict__.update(
        {k: v for k, v in val.__dict__.items() if not k.startswith("_leakage")}
    )
    fresh.psf_leakage = {**val.psf_leakage, **psf_leakage}
    return fresh


def test_sky_mask_removes_stars_and_galaxies_inside_the_circles(tmp_path):
    val = _Validation(tmp_path)
    mask_star, mask_gal = val._get_rho_tau_masks(VERSION, "all")
    stars = fits.getdata(val.cc[VERSION]["psf"]["path"])
    gals = fits.getdata(val.cc[VERSION]["shear"]["path"])

    def inside(ra, dec):
        return np.hypot((ra - 11.0) * np.cos(np.deg2rad(1.0)), dec - 1.0) < 0.299

    assert not np.any(mask_star & inside(stars["RA"], stars["DEC"]))
    assert not np.any(mask_gal & inside(gals["RA"], gals["Dec"]))
    assert not np.any(mask_star & (stars["HSM_FLAG_PSF"] != 0))
    outside_stars = ~inside(stars["RA"], stars["DEC"]) & (stars["HSM_FLAG_PSF"] == 0)
    assert mask_star.sum() >= 0.99 * outside_stars.sum()


def test_joint_vector_equals_production_rho_tau(tmp_path):
    """With exact pair counting the joint vector is production's rho/tau."""
    val = _Validation(tmp_path)
    products = val.leakage_jackknife(VERSION, measure=True)
    mask_star, mask_gal = val._get_rho_tau_masks(VERSION, "all")
    (tmp_path / "production").mkdir()
    rho_handler, tau_handler = rho_tau.get_rho_tau(
        val.cc,
        VERSION,
        val.treecorr_config,
        str(tmp_path / "production"),
        "base",
        "base",
        mask_star=mask_star,
        mask_gal=mask_gal,
    )
    expected = np.concatenate(
        [rho_handler.rho_stats[f"rho_{i}_p"] for i in range(6)]
        + [tau_handler.tau_stats[f"tau_{i}_p"] for i in (0, 2, 5)]
    )
    np.testing.assert_allclose(products["vector"], expected, rtol=1e-10, atol=1e-16)
    np.testing.assert_allclose(
        products["rho_xim"],
        [rho_handler.rho_stats[f"rho_{i}_m"] for i in range(6)],
        rtol=1e-10,
        atol=1e-16,
    )
    assert products["rows_30"].shape == (30, 27)
    assert products["rows_15"].shape == (15, 27)
    np.testing.assert_array_equal(np.unique(products["groups_20"]), np.arange(20))


@pytest.mark.parametrize("beta_psf_size", [False, True])
def test_eiv_samples_and_xi_sys_through_the_mixin(tmp_path, monkeypatch, beta_psf_size):
    val = _Validation(tmp_path, beta_psf_size=beta_psf_size)
    val.psf_error_nsamples = 50
    val.leakage_jackknife(VERSION, measure=True)
    samples = val.get_samples(VERSION, {}, "all")
    k = 4 if beta_psf_size else 3
    assert samples.shape == (50, k)
    fits_json = json.loads(
        val.leakage_jackknife_path(VERSION, "all", "leakage_eiv_fit").read_text()
    )
    assert set(fits_json["fits"]) == {"15", "20"}
    assert fits_json["fits"]["20"]["dof"] == 9 - k

    plus, minus = val.get_xi_psf_sys_samples(VERSION, {}, "all", "all")
    assert plus.shape == minus.shape == (50, 3)
    products = val.leakage_jackknife(VERSION)
    rho = products["vector"][:18].reshape(6, 3)
    alpha, beta, eta = samples[0, :3]
    constant = (
        alpha**2 * rho[0]
        + beta**2 * rho[1]
        + 2 * alpha * beta * rho[2]
        + eta**2 * rho[3]
        + 2 * beta * eta * rho[4]
        + 2 * alpha * eta * rho[5]
    )
    if beta_psf_size:
        beta1 = samples[0, 3]
        slope = products["slope_xip"]
        constant += beta1**2 * slope[3] + 2 * beta1 * (
            alpha * slope[0] + beta * slope[1] + eta * slope[2]
        )
    np.testing.assert_allclose(plus[0], constant, rtol=1e-10)

    # A fresh host reads the cached measurement instead of measuring.
    def fail(*args, **kwargs):
        raise AssertionError("cached measurement was not reused")

    monkeypatch.setattr(
        "sp_validation.cosmo_val.psf_systematics.measure_leakage_jackknife", fail
    )
    np.testing.assert_array_equal(
        _fresh(val).leakage_jackknife(VERSION)["vector"], products["vector"]
    )


def test_reading_without_a_measurement_names_the_rule(tmp_path):
    """Only measure=True measures; every other caller needs the cache."""
    val = _Validation(tmp_path)
    with pytest.raises(FileNotFoundError, match="is missing.*cv_leakage_jackknife"):
        val.leakage_jackknife(VERSION)
    with pytest.raises(FileNotFoundError, match="cv_leakage_jackknife"):
        val.get_samples(VERSION, {}, "all")
    assert not val.leakage_jackknife_path(VERSION).exists()


@pytest.mark.parametrize("change", ["sky_mask", "beta_psf_size", "selection"])
def test_cache_is_stale_after_a_change(tmp_path, change):
    """Sky mask, PSF-size slope and star selection each invalidate the cache."""
    val = _Validation(tmp_path)
    val.leakage_jackknife(VERSION, measure=True)
    if change == "sky_mask":
        mask = tmp_path / "wider.json"
        mask.write_text(json.dumps({"circles": [{**CIRCLE, "radius_deg": 0.4}]}))
        fresh = _fresh(val, sky_mask=str(mask))
    elif change == "beta_psf_size":
        fresh = _fresh(val, beta_psf_size=True)
    else:
        fresh = _fresh(val)
        star_mask = PSFSystematicsMixin._get_star_mask

        def fewer_stars(self, ver):
            # A strip far from the masked circle, so the selection changes.
            ra = fits.getdata(self.cc[ver]["psf"]["path"])["RA"]
            return star_mask(self, ver) & (ra > 10.2)

        fresh._get_star_mask = fewer_stars.__get__(fresh)
    with pytest.raises(FileNotFoundError, match="is stale"):
        fresh.leakage_jackknife(VERSION)
    assert _fresh(val).leakage_jackknife(VERSION) is not None


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"npatch_fine": 300}, "strictly between 1 and npatch_fine"),
        ({"patch_counts": [1, 400]}, "strictly between 1 and npatch_fine"),
        ({"fiducial_npatch": 300}, "fiducial_npatch must be one of"),
    ],
)
def test_psf_leakage_settings_are_validated(settings, message):
    from sp_validation.cosmo_val import CosmologyValidation

    with pytest.raises(ValueError, match=message):
        CosmologyValidation(versions=[], psf_leakage=settings)


def test_theory_covariance_follows_the_selection(tmp_path, monkeypatch):
    """cov_tau_*_th.npy is rebuilt when the star or galaxy selection changes."""
    val = _Validation(tmp_path)
    built = []

    class FakeCovTauTh:
        def __init__(self, mask_star=None, mask_gal=None, **kwargs):
            built.append((mask_star.sum(), mask_gal.sum()))

        def build_cov(self, **kwargs):
            return np.eye(2) * len(built)

    monkeypatch.setattr(rho_tau, "CovTauTh", FakeCovTauTh)
    mask_star, mask_gal = val._get_rho_tau_masks(VERSION, "all")
    outdir = tmp_path / "rho_tau_stats"

    def covariance(star, gal):
        rho_tau.get_theory_cov(
            val.cc,
            VERSION,
            val.treecorr_config,
            str(outdir),
            "base",
            mask_star=star,
            mask_gal=gal,
        )
        return np.load(outdir / "cov_tau_base_th.npy")[0, 0]

    assert covariance(mask_star, mask_gal) == 1
    assert covariance(mask_star, mask_gal) == 1
    fewer = mask_gal.copy()
    fewer[np.flatnonzero(fewer)[0]] = False
    assert covariance(mask_star, fewer) == 2
    assert len(built) == 2
