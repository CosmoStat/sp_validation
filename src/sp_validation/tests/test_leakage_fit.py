"""Synthetic and TreeCorr checks for the joint PSF-leakage estimator."""

import numpy as np
import pytest
import treecorr
from scipy.optimize import minimize
from shear_psf_leakage.rho_tau_stat import PSFErrorFit, RhoStat, TauStat

from sp_validation.leakage_fit import (
    fit_joint_eiv,
    keep_sky,
    residual_covariance_model,
    rho_tau_design,
    systematic_correlation_matrix,
    treecorr_jackknife_rows,
    ward_group_map,
    xi_psf_sys,
)

TRUE_PARAMETERS = np.array([0.013, 1.28, 0.96, 1.9])


def _synthetic_joint(seed, nbin=5, npatch=200, slope=False, noise=True):
    """Point vector and delete-one rows for tau = B(rho) p (+ beta1 * template).

    Each row perturbs rho (and the template) and propagates the perturbation
    into tau, plus independent tau noise, so the rows carry a real rho-tau
    cross-covariance. Delete-one rows scatter by sigma / sqrt(N - 1) for a
    point-vector scatter sigma, so the point vector is perturbed by the
    same draws scaled up by sqrt(N - 1). With ``noise=False`` the point vector
    is exact.
    """
    rng = np.random.default_rng(seed)
    parameters = TRUE_PARAMETERS[: 4 if slope else 3]
    rho = rng.normal(size=(6, nbin))
    template = rng.normal(size=3 * nbin) if slope else None

    def draw(scale):
        d_rho = rng.normal(scale=0.08 * scale, size=6 * nbin)
        d_template = rng.normal(scale=0.05 * scale, size=3 * nbin)
        d_tau = rho_tau_design(d_rho.reshape(6, nbin)) @ parameters[:3]
        d_tau += rng.normal(scale=0.04 * scale, size=3 * nbin)
        if slope:
            d_tau += parameters[3] * d_template
        return d_rho, d_template, d_tau

    def tau(rho_values, template_values):
        value = rho_tau_design(rho_values.reshape(6, nbin)) @ parameters[:3]
        return value + (parameters[3] * template_values if slope else 0)

    rows, template_rows = [], []
    for _ in range(npatch):
        d_rho, d_template, d_tau = draw(1 / np.sqrt(npatch - 1))
        row_rho = rho.ravel() + d_rho
        row_template = (template + d_template) if slope else None
        rows.append(np.concatenate((row_rho, tau(rho.ravel(), template) + d_tau)))
        template_rows.append(row_template)
    d_rho, d_template, d_tau = draw(1.0 if noise else 0.0)
    vector = np.concatenate((rho.ravel() + d_rho, tau(rho.ravel(), template) + d_tau))
    template_vector = (template + d_template) if slope else None
    return (
        vector,
        np.array(rows),
        template_vector,
        np.array(template_rows) if slope else None,
        parameters,
    )


@pytest.mark.parametrize("slope", [False, True])
def test_joint_eiv_is_exact_without_noise(slope):
    """An exact point vector gives the true parameters and chi-squared zero."""
    vector, rows, template, template_rows, expected = _synthetic_joint(
        seed=204, slope=slope, noise=False
    )
    result = fit_joint_eiv(
        vector, rows, template_vector=template, template_rows=template_rows
    )
    np.testing.assert_allclose(result["parameters"], expected, rtol=1e-7, atol=1e-9)
    assert result["chi2"] < 1e-12
    assert result["dof"] == 15 - len(expected)
    assert result["hartlap"] == pytest.approx((200 - 15 - 2) / 199)
    assert np.all(np.asarray(result["sigma"]) > 0)


@pytest.mark.parametrize("slope", [False, True])
def test_joint_eiv_recovers_parameters_within_errors(slope):
    """With noise, the fit lands within its errors and chi-squared is sane."""
    pulls, chi2s = [], []
    for seed in range(5):
        vector, rows, template, template_rows, expected = _synthetic_joint(
            seed=seed, slope=slope
        )
        result = fit_joint_eiv(
            vector, rows, template_vector=template, template_rows=template_rows
        )
        pulls.append((np.array(result["parameters"]) - expected) / result["sigma"])
        chi2s.append(result["chi2"])
        assert result["profile_hessian_positive"]
    pulls = np.array(pulls)
    assert np.all(np.abs(pulls) < 4), pulls
    # Five draws of a chi-squared with 11-12 degrees of freedom.
    assert 3 < np.mean(chi2s) < 25, chi2s


def test_joint_eiv_cross_covariance_changes_the_fit():
    """The rho-tau block matters when the delete-one rows correlate."""
    vector, rows, _, _, _ = _synthetic_joint(seed=41)
    full = fit_joint_eiv(vector, rows, cross_term=True)
    no_cross = fit_joint_eiv(vector, rows, cross_term=False)
    # Without the cross term the propagated rho noise is counted twice in V.
    assert full["chi2"] > 1.3 * no_cross["chi2"]
    assert full["chi2"] - no_cross["chi2"] > 2.0


def test_treecorr_rows_match_native_match_jackknife_and_grouped_catalogues():
    """Fine and unioned rows agree with TreeCorr's native design matrix."""
    rng = np.random.default_rng(1234)
    n_fine, n_coarse = 12, 4
    centres = rng.normal(size=(n_fine, 2))
    groups = ward_group_map(centres, n_coarse)
    config = dict(
        min_sep=0.01,
        max_sep=4,
        nbins=10,
        bin_slop=0.0,
        angle_slop=0.0,
        var_method="shot",
        num_threads=1,
        cross_patch_weight="match",
    )

    def catalogue(size):
        return treecorr.Catalog(
            x=rng.normal(size=size),
            y=rng.normal(size=size),
            g1=rng.normal(size=size),
            g2=rng.normal(size=size),
            w=rng.uniform(0.2, 3, size=size),
            patch_centers=centres,
        )

    def coarse(catalogue):
        return treecorr.Catalog(
            x=catalogue.x,
            y=catalogue.y,
            g1=catalogue.g1,
            g2=catalogue.g2,
            w=catalogue.w,
            patch=groups[catalogue.patch],
            npatch=n_coarse,
        )

    first, second = catalogue(1000), catalogue(800)
    for left, right in ((first, second), (first, first)):
        fine_corr = treecorr.GGCorrelation(config)
        fine_corr.process(left, right)
        rows, _ = treecorr_jackknife_rows([fine_corr])
        native, _ = treecorr.build_multi_cov_design_matrix(
            [fine_corr],
            "jackknife",
            func=lambda correlations: correlations[0].xip,
            cross_patch_weight="match",
        )
        np.testing.assert_allclose(rows, native, rtol=1e-11, atol=1e-14)
        centered = rows - rows.mean(axis=0)
        covariance = (len(rows) - 1) / len(rows) * centered.T @ centered
        np.testing.assert_allclose(
            covariance,
            treecorr.estimate_multi_cov(
                [fine_corr],
                "jackknife",
                func=lambda correlations: correlations[0].xip,
                cross_patch_weight="match",
            ),
            rtol=1e-10,
            atol=1e-14,
        )

        coarse_corr = treecorr.GGCorrelation(config)
        coarse_corr.process(coarse(left), coarse(right))
        grouped, _ = treecorr_jackknife_rows(
            [fine_corr], groups=groups, check_rows=False
        )
        native_grouped, _ = treecorr.build_multi_cov_design_matrix(
            [coarse_corr],
            "jackknife",
            func=lambda correlations: correlations[0].xip,
            cross_patch_weight="match",
        )
        np.testing.assert_allclose(
            fine_corr.xip, coarse_corr.xip, rtol=1e-11, atol=1e-14
        )
        np.testing.assert_allclose(grouped, native_grouped, rtol=1e-11, atol=1e-14)


def test_spherical_circle_mask_handles_inside_outside_and_ra_wrap():
    """Circle excision follows great-circle distance across RA=0 degrees."""
    circles = [{"name": "wrap", "ra_deg": 359.8, "dec_deg": 0.0, "radius_deg": 0.5}]
    keep = keep_sky(
        ra=np.array([359.9, 0.1, 359.0, 180.0]),
        dec=np.array([0.0, 0.0, 0.0, 0.0]),
        circles=circles,
    )
    np.testing.assert_array_equal(keep, [False, False, True, True])


def test_sky_mask_rejects_mismatched_coordinates():
    with pytest.raises(ValueError, match="equal length"):
        keep_sky([1.0], [1.0, 2.0], [])


def test_xi_psf_sys_matches_term_by_term_expansion():
    """a^T M b reproduces the production rho-term expansion, plus the slope."""
    rng = np.random.default_rng(3)
    rho = rng.normal(size=(6, 4))
    slope = rng.normal(size=(4, 4))
    a, b = rng.normal(size=4), rng.normal(size=4)
    (al_a, be_a, et_a, b1_a), (al_b, be_b, et_b, b1_b) = a, b
    expected = (
        al_a * al_b * rho[0]
        + be_a * be_b * rho[1]
        + (al_a * be_b + be_a * al_b) * rho[2]
        + et_a * et_b * rho[3]
        + (be_a * et_b + et_a * be_b) * rho[4]
        + (al_a * et_b + et_a * al_b) * rho[5]
    )
    constant = xi_psf_sys(a[:3], b[:3], systematic_correlation_matrix(rho))
    np.testing.assert_allclose(constant[0], expected, rtol=1e-12)

    expected += (
        b1_a * b1_b * slope[3]
        + (al_a * b1_b + b1_a * al_b) * slope[0]
        + (be_a * b1_b + b1_a * be_b) * slope[1]
        + (et_a * b1_b + b1_a * et_b) * slope[2]
    )
    with_slope = xi_psf_sys(a, b, systematic_correlation_matrix(rho, slope))
    np.testing.assert_allclose(with_slope[0], expected, rtol=1e-12)


def test_ward_groups_are_contiguous_unions():
    centres = np.random.default_rng(5).normal(size=(40, 3))
    groups = ward_group_map(centres, 7)
    assert groups.shape == (40,)
    np.testing.assert_array_equal(np.unique(groups), np.arange(7))


def _linear_map(parameters, nbin):
    """L with r = L [rho, tau, template] to first order: tau - B(p) rho - b1 t."""
    alpha_beta_eta = parameters[:3]
    linear = np.zeros((3 * nbin, (9 + 3 * (len(parameters) > 3)) * nbin))
    for block, columns in enumerate(((0, 2, 5), (2, 1, 4), (5, 4, 3))):
        rows = slice(block * nbin, (block + 1) * nbin)
        for value, rho in zip(alpha_beta_eta, columns):
            linear[rows, rho * nbin : (rho + 1) * nbin] -= value * np.eye(nbin)
    linear[:, 6 * nbin : 9 * nbin] = np.eye(3 * nbin)
    if len(parameters) > 3:
        linear[:, 9 * nbin :] = -parameters[3] * np.eye(3 * nbin)
    return linear


def _dense_covariance(rows):
    centred = rows - rows.mean(axis=0)
    return (len(rows) - 1) / len(rows) * centred.T @ centred


def _design_rows(rows, template_rows, nbin):
    return np.stack(
        [
            np.column_stack((rho_tau_design(row[: 6 * nbin].reshape(6, nbin)), t))
            for row, t in zip(rows, template_rows)
        ]
    )


def test_residual_covariance_is_the_propagated_joint_jackknife_covariance():
    """V(p) = L C L^T for the explicit joint covariance of [rho, tau, template]."""
    rng = np.random.default_rng(0)
    nbin, npatch = 4, 50
    rows = rng.normal(size=(npatch, 9 * nbin))
    template_rows = rng.normal(size=(npatch, 3 * nbin))
    covariance, derivative = residual_covariance_model(
        _design_rows(rows, template_rows, nbin), rows[:, 6 * nbin :]
    )
    dense = _dense_covariance(np.concatenate((rows, template_rows), axis=1))
    parameters = np.array([0.02, 1.3, -0.7, 0.5])
    linear = _linear_map(parameters, nbin)
    np.testing.assert_allclose(
        covariance(parameters), linear @ dense @ linear.T, rtol=1e-12, atol=1e-12
    )
    step = 1e-6
    for index in range(4):
        shift = np.eye(4)[index] * step
        numeric = (covariance(parameters + shift) - covariance(parameters - shift)) / (
            2 * step
        )
        np.testing.assert_allclose(
            derivative(parameters)[index], numeric, rtol=1e-6, atol=1e-8
        )


def test_noisy_fit_minimises_the_dense_joint_chi2():
    """The fit's chi2 and minimum are those of the dense r^T (L C L^T)^-1 r."""
    nbin = 5
    vector, rows, template, template_rows, _ = _synthetic_joint(seed=3, slope=True)
    result = fit_joint_eiv(
        vector, rows, template_vector=template, template_rows=template_rows
    )
    dense = _dense_covariance(np.concatenate((rows, template_rows), axis=1))
    x = np.column_stack((rho_tau_design(vector[: 6 * nbin].reshape(6, nbin)), template))

    def dense_chi2(parameters):
        linear = _linear_map(parameters, nbin)
        residual = vector[6 * nbin :] - x @ parameters
        return residual @ np.linalg.solve(linear @ dense @ linear.T, residual)

    fitted = np.array(result["parameters"])
    assert result["chi2_raw"] == pytest.approx(dense_chi2(fitted), rel=1e-10)
    direct = minimize(
        dense_chi2,
        fitted * 1.05,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000},
    )
    np.testing.assert_allclose(direct.x, fitted, rtol=1e-4, atol=1e-6)
    assert result["chi2_raw"] <= direct.fun * (1 + 1e-8)
    assert result["profile_hessian_positive"] and result["bfgs_success"]


def test_unconstrained_parameter_is_flagged_not_given_an_error():
    """A template column of zeros leaves eta unconstrained: no sigma, flagged."""
    nbin = 5
    vector, rows, _, _, _ = _synthetic_joint(seed=8)
    for target in (vector, *rows):
        target[3 * nbin : 6 * nbin] = 0.0  # rho3, rho4, rho5: the eta column
    result = fit_joint_eiv(vector, rows)
    assert not result["profile_hessian_positive"]
    assert result["sigma"] == [None, None, None]
    assert result["parameter_covariance"] is None


def _production_fitter(tmp_path, rho):
    config = dict(min_sep=1.0, max_sep=30.0, nbins=rho.shape[1], sep_units="arcmin")
    fitter = PSFErrorFit(
        RhoStat(output=str(tmp_path), treecorr_config=config),
        TauStat(output=str(tmp_path), treecorr_config=config),
        str(tmp_path),
    )
    stats = {"theta": np.geomspace(1, 30, rho.shape[1])}
    stats.update({f"rho_{i}_p": rho[i] for i in range(6)})
    stats.update({f"rho_{i}_m": -2 * rho[i] for i in range(6)})
    fitter.rho_stat_handler.rho_stats = stats
    return fitter


def test_design_and_xi_sys_match_shear_psf_leakage(tmp_path):
    """The rho-to-tau design and constant-beta xi_sys are PSFErrorFit's."""
    rng = np.random.default_rng(11)
    rho = rng.normal(size=(6, 7))
    fitter = _production_fitter(tmp_path, rho)
    a, b = rng.normal(size=3), rng.normal(size=3)
    np.testing.assert_allclose(
        rho_tau_design(rho) @ a,
        fitter.model((*a, 0.0, 0.0), use_eta=True, use_fourth_moment=False),
        rtol=1e-13,
    )
    for sign, part in ((1, "p"), (-2, "m")):
        np.testing.assert_allclose(
            xi_psf_sys(a, b, systematic_correlation_matrix(sign * rho))[0],
            fitter.compute_xi_psf_sys(a, b, p_or_m=part),
            rtol=1e-12,
        )
