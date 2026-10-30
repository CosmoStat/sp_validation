"""Synthetic and TreeCorr checks for the joint PSF-leakage estimator."""

import numpy as np
import pytest
import treecorr

from sp_validation.leakage_fit import (
    fit_joint_eiv,
    keep_sky,
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
    result, _ = fit_joint_eiv(
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
        result, _ = fit_joint_eiv(
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
    full, _ = fit_joint_eiv(vector, rows, cross_term=True)
    no_cross, _ = fit_joint_eiv(vector, rows, cross_term=False)
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
        rows, weights, _ = treecorr_jackknife_rows([fine_corr])
        native, native_weights = treecorr.build_multi_cov_design_matrix(
            [fine_corr],
            "jackknife",
            func=lambda correlations: correlations[0].xip,
            cross_patch_weight="match",
        )
        np.testing.assert_allclose(rows, native, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(weights, native_weights, rtol=1e-12)
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
        grouped, grouped_weights, _ = treecorr_jackknife_rows(
            [fine_corr], groups=groups, check_rows=False
        )
        native_grouped, native_group_weights = treecorr.build_multi_cov_design_matrix(
            [coarse_corr],
            "jackknife",
            func=lambda correlations: correlations[0].xip,
            cross_patch_weight="match",
        )
        np.testing.assert_allclose(fine_corr.xip, coarse_corr.xip, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(grouped, native_grouped, rtol=1e-11, atol=1e-14)
        np.testing.assert_allclose(grouped_weights, native_group_weights, rtol=1e-12)


def test_spherical_circle_mask_handles_inside_outside_and_ra_wrap():
    """Circle excision follows great-circle distance across RA=0 degrees."""
    circles = [
        {"name": "wrap", "ra_deg": 359.8, "dec_deg": 0.0, "radius_deg": 0.5}
    ]
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
