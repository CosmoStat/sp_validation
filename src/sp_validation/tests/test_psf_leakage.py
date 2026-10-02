"""PSF-leakage helpers of the cosmology validation: alpha, xi_psf_sys, summaries.

Synthetic inputs only. The object-wise smoke test runs the real
shear_psf_leakage regression on a toy catalogue and therefore needs the
scientific stack (the container).
"""

from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from sp_validation.cosmo_val.psf_systematics import PSFSystematicsMixin

mixin = PSFSystematicsMixin()

N_THETA = 5
RHO = np.full(N_THETA, 2.0)
TINY = np.diag(np.full(N_THETA, 1e-16))


def _xi_sys(tau_a, tau_b, sig_a, sig_b, same_bin, seed=0, n_samples=40_000):
    return mixin._compute_scale_dependent_xi_psf_sys(
        RHO,
        tau_a,
        tau_b,
        TINY,
        np.diag(np.full(N_THETA, sig_a**2)),
        np.diag(np.full(N_THETA, sig_b**2)),
        n_samples=n_samples,
        same_bin=same_bin,
        seed=seed,
    )


def test_xi_sys_central_value():
    tau_a = np.linspace(1.0, 2.0, N_THETA)
    tau_b = np.linspace(0.5, 3.0, N_THETA)
    xi, _ = _xi_sys(tau_a, tau_b, 0.01, 0.02, same_bin=False)
    np.testing.assert_allclose(xi, tau_a * tau_b / RHO)


def test_xi_sys_auto_pair_reuses_tau_draw():
    mu, sig = 1.0, 0.01
    tau = np.full(N_THETA, mu)
    _, err = _xi_sys(tau, tau, sig, sig, same_bin=True)
    np.testing.assert_allclose(err, 2 * mu * sig / RHO, rtol=0.03)


def test_xi_sys_cross_pair_uses_both_variances():
    mu_a, mu_b, sig_a, sig_b = 1.0, 3.0, 0.01, 0.03
    _, err = _xi_sys(
        np.full(N_THETA, mu_a), np.full(N_THETA, mu_b), sig_a, sig_b, same_bin=False
    )
    expected = np.sqrt(mu_b**2 * sig_a**2 + mu_a**2 * sig_b**2) / RHO
    np.testing.assert_allclose(err, expected, rtol=0.03)


def test_xi_sys_errors_reproducible():
    tau = np.full(N_THETA, 1.0)
    _, err_1 = _xi_sys(tau, tau, 0.1, 0.1, same_bin=False, seed=7)
    _, err_2 = _xi_sys(tau, tau, 0.1, 0.1, same_bin=False, seed=7)
    _, err_3 = _xi_sys(tau, tau, 0.1, 0.1, same_bin=False, seed=8)
    np.testing.assert_array_equal(err_1, err_2)
    assert not np.array_equal(err_1, err_3)


def _handlers(rho, tau, var_rho, var_tau):
    theta = np.geomspace(1, 100, len(rho))
    rho_h = SimpleNamespace(
        rho_stats={"theta": theta, "rho_0_p": rho, "varrho_0_p": var_rho}
    )
    tau_h = SimpleNamespace(tau_stats={"tau_0_p": tau, "vartau_0_p": var_tau})
    return rho_h, tau_h


def test_alpha_leakage_central_value_and_reproducible():
    rho = np.linspace(1.0, 2.0, N_THETA)
    tau = np.linspace(0.01, 0.03, N_THETA)
    rho_h, tau_h = _handlers(rho, tau, np.full(N_THETA, 1e-4), np.full(N_THETA, 1e-6))
    theta, alpha, err = mixin._get_alpha_leakage(rho_h, tau_h, seed=3)
    np.testing.assert_array_equal(theta, rho_h.rho_stats["theta"])
    np.testing.assert_allclose(alpha, tau / rho)
    _, _, err_again = mixin._get_alpha_leakage(rho_h, tau_h, seed=3)
    np.testing.assert_array_equal(err, err_again)
    assert np.all(err > 0)


def test_alpha_summaries_recover_affine_model():
    c, m = 0.01, 2e-4
    theta = np.geomspace(1, 100, 12)
    alpha_err = np.linspace(1e-3, 3e-3, theta.size)
    alpha = c + m * theta

    out = PSFSystematicsMixin._alpha_summaries(theta, alpha, alpha_err)

    assert out["alpha_0"].nominal_value == pytest.approx(c, rel=1e-8)
    assert 0 < out["alpha_0"].std_dev < alpha_err.max()
    assert out["alpha_1"].nominal_value == pytest.approx(alpha[0])
    assert out["alpha_1"].std_dev == pytest.approx(alpha_err[0])
    w = 1 / alpha_err**2
    mean = np.sum(w * alpha) / np.sum(w)
    assert out["alpha_mean"].nominal_value == pytest.approx(mean)
    std = np.sqrt(np.sum(w * (alpha - mean) ** 2) / np.sum(w))
    assert out["alpha_mean"].std_dev == pytest.approx(std)


def _objectwise_config(tmp_path, n_gal=3000, seed=11):
    """Toy shear catalogue with PSF columns and two tomographic bins."""
    from astropy.table import Table

    rng = np.random.default_rng(seed)
    cat_dir = tmp_path / "catalog"
    cat_dir.mkdir()
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    e1_psf = rng.normal(0, 0.03, n_gal)
    e2_psf = rng.normal(0, 0.03, n_gal)
    tomo_bin = rng.integers(1, 3, n_gal)
    leak = np.where(tomo_bin == 1, 0.05, 0.15)
    Table(
        {
            "RA": rng.uniform(10, 12, n_gal),
            "Dec": rng.uniform(10, 12, n_gal),
            "e1": leak * e1_psf + rng.normal(0, 0.02, n_gal),
            "e2": leak * e2_psf + rng.normal(0, 0.02, n_gal),
            "w": np.ones(n_gal),
            "e1_PSF": e1_psf,
            "e2_PSF": e2_psf,
            "fwhm_PSF": rng.uniform(0.6, 0.8, n_gal),
            "tomo_bin": tomo_bin,
        }
    ).write(cat_dir / "shear.fits")

    version = "TestCatalog"
    config = {
        "nz": {"subdir": str(cat_dir), "dndz": {"blind": "A", "path": "dndz"}},
        "paths": {"output": str(output_dir)},
        version: {
            "subdir": str(cat_dir),
            "pipeline": "SP",
            "colour": "C0",
            "marker": "o",
            "shear": {
                "path": "shear.fits",
                "w_col": "w",
                "e1_col": "e1",
                "e2_col": "e2",
                "e1_PSF_col": "e1_PSF",
                "e2_PSF_col": "e2_PSF",
                "tomo_bin_col": "tomo_bin",
                "R": 1.0,
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config, sort_keys=False))
    return version, {
        "catalog_config": str(config_path),
        "output_dir": str(output_dir),
    }


def test_objectwise_leakage_and_plot_smoke(tmp_path, monkeypatch):
    """Object-wise fit per bin and the comparison plot, alpha(theta) stubbed.

    The rho/tau products are replaced by a synthetic alpha(theta), so this
    covers the object-wise regression, the per-bin storage and the plot, not
    the rho/tau loading.
    """
    pytest.importorskip("shear_psf_leakage")
    import matplotlib

    matplotlib.use("Agg")
    # TeX text rendering dominates the run time of the ~20 diagnostic figures
    monkeypatch.setitem(matplotlib.rcParams, "text.usetex", False)
    from sp_validation.cosmo_val import CosmologyValidation

    version, params = _objectwise_config(tmp_path)
    cv = CosmologyValidation(versions=[version], npatch=1, **params)

    theta = np.geomspace(1, 100, 6)
    monkeypatch.setattr(
        cv,
        "_load_alpha_leakage",
        lambda ver, tomo_bin_id, cov_type=None: (
            theta,
            0.01 + 1e-4 * theta,
            np.full(theta.size, 1e-3),
        ),
    )

    cv.plot_objectwise_leakage()
    coeff = cv.leakage_coeff[version]["tomo_bin_all"]
    assert {"a11", "a22", "aii_mean", "alpha_mean", "alpha_1", "alpha_0"} <= set(coeff)
    assert coeff["alpha_0"].nominal_value == pytest.approx(0.01)
    assert (tmp_path / "output" / "leakage_coefficients.png").exists()

    cv.plot_objectwise_leakage(tomography=True)
    for tomo_bin_id, expected in ((1, 0.05), (2, 0.15)):
        bin_coeff = cv.leakage_coeff[version][f"tomo_bin_{tomo_bin_id}"]
        assert bin_coeff["aii_mean"].nominal_value == pytest.approx(expected, abs=0.03)
        assert "alpha_mean" in bin_coeff
    assert (tmp_path / "output" / "leakage_coefficients_tomo.png").exists()


def test_leakage_object_reads_the_selected_rows(tmp_path):
    """The object-wise reader keeps the rows a galaxy mask selects."""
    pytest.importorskip("shear_psf_leakage")
    from sp_validation.cosmo_val import CosmologyValidation
    from sp_validation.cosmo_val.core import _LeakageObject

    version, params = _objectwise_config(tmp_path)
    cv = CosmologyValidation(versions=[version], npatch=1, **params)
    obj = object.__new__(_LeakageObject)
    obj.entries = cv.cc[version]

    mask = cv._get_galaxy_mask(version, 2)
    obj.read_data(selection=mask)
    assert len(obj._dat) == mask.sum()
    assert np.all(obj._dat["tomo_bin"] == 2)

    with pytest.raises(ValueError, match="different length"):
        obj.read_data(selection=mask[1:])


class _ObjectwiseControlFlow(PSFSystematicsMixin):
    """Two versions, one of which lacks a catalogue column."""

    def __init__(self):
        self.versions = ["good", "bad"]
        obj = SimpleNamespace(
            check_params=lambda: None,
            update_params=lambda: None,
            prepare_output=lambda: None,
        )
        self.results_objectwise = {"good": obj, "bad": obj}

    def _get_tomo_bins_for_versions(self, versions, tomography):
        return {v: {"ids": [1, 2] if tomography else ["all"]} for v in versions}

    def _objectwise_leakage_bin(self, obj, ver, *args):
        if ver == "bad":
            raise KeyError("fwhm_PSF")
        return {"aii_mean": 0.1}

    def print_start(self, *args):
        pass

    def print_magenta(self, *args):
        pass


def test_version_missing_a_column_stays_dropped():
    cv = _ObjectwiseControlFlow()
    cv.calculate_objectwise_leakage()
    assert set(cv.results_objectwise) == {"good"}
    assert set(cv.leakage_coeff) == {"good"}
    cv.calculate_objectwise_leakage(tomography=True)
    assert "tomo_bin_1" in cv.leakage_coeff["good"]
