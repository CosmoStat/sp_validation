"""COSEBIs EE and BB covariance marginals must equal an independent xi± basis sandwich J C Jᵀ.
The estimator's EB covariance remains zero by convention and is checked separately.
"""

import numpy as np
import numpy.testing as npt
import pytest

from sp_validation import b_modes

pytestmark = [pytest.mark.decision("bmodes.cosebis_modes")]


def _scan_covariance_and_basis_reference(monkeypatch):
    """Run the real scan and return it beside a basis-built linear reference."""
    from cosmo_numba.B_modes import cosebis as cosebis_module

    n_nodes = 31
    n_modes = 3
    theta = np.geomspace(1.0, 100.0, n_nodes)
    edges = np.geomspace(1.0, 100.0, n_nodes + 1)

    # Pad one bin below and two above the cut so the scan must slice the
    # xi+ and xi- covariance blocks at offsets that differ from the cut size.
    theta_step = theta[1] / theta[0]
    edge_step = edges[1] / edges[0]
    theta_all = np.concatenate(
        ([theta[0] / theta_step], theta, theta[-1] * theta_step ** np.arange(1, 3))
    )
    edges_all = np.concatenate(
        ([edges[0] / edge_step], edges, edges[-1] * edge_step ** np.arange(1, 3))
    )
    n_all = len(theta_all)
    cut_inds = np.arange(1, 1 + n_nodes)
    cov_inds = np.concatenate((cut_inds, cut_inds + n_all))

    lower = np.eye(2 * n_all)
    lower[np.arange(1, 2 * n_all), np.arange(2 * n_all - 1)] = 0.2
    lower[n_all + np.arange(n_all), np.arange(n_all)] = 0.1
    scale = np.geomspace(1.0, 3.0, 2 * n_all)
    cov_all = np.outer(scale, scale) * (lower @ lower.T)
    np.linalg.cholesky(cov_all)
    cov_xi = cov_all[np.ix_(cov_inds, cov_inds)]
    assert np.any(cov_xi[:n_nodes, n_nodes:] != 0.0)

    rng = np.random.default_rng(20260626)
    xi_all = 1e-5 * rng.standard_normal(2 * n_all)
    xi_plus, xi_minus = xi_all[:n_all], xi_all[n_all:]

    real_cosebis_class = cosebis_module.COSEBIS
    reference = real_cosebis_class(
        theta_min=theta[0], theta_max=theta[-1], N_max=n_modes, precision=120
    )
    jacobian = np.empty((2 * n_modes, 2 * n_nodes))
    for column in range(2 * n_nodes):
        basis = np.zeros(2 * n_nodes)
        basis[column] = 1.0
        e_modes, b_modes_basis = reference.cosebis_from_xipm(
            theta,
            basis[:n_nodes],
            basis[n_nodes:],
            cache=True,
            parallel=False,
        )
        jacobian[:, column] = np.concatenate((e_modes, b_modes_basis))

    cached_tp, cached_tm = reference._Tp_log, reference._Tm_log
    original_transform = reference.cosebis_from_xipm

    def transform_from_cached_filters(
        theta_cut, xi_plus_cut, xi_minus_cut, parallel=True
    ):
        return original_transform(
            theta_cut,
            xi_plus_cut,
            xi_minus_cut,
            cache=True,
            parallel=False,
        )

    def get_cached_tp(theta_cut, roots=None, norms=None):
        npt.assert_array_equal(theta_cut, theta)
        return cached_tp

    def get_cached_tm(theta_cut, roots=None, norms=None):
        npt.assert_array_equal(theta_cut, theta)
        return cached_tm

    reference.cosebis_from_xipm = transform_from_cached_filters
    reference.get_Tp_log = get_cached_tp
    reference.get_Tm_log = get_cached_tm

    def scan_cosebis(**kwargs):
        assert kwargs == {
            "theta_min": theta[0],
            "theta_max": theta[-1],
            "N_max": n_modes,
            "precision": 120,
        }
        return reference

    monkeypatch.setattr(cosebis_module, "COSEBIS", scan_cosebis)
    (result,) = b_modes.cosebis_scan_from_xi(
        theta_all,
        xi_plus,
        xi_minus,
        cov_all,
        edges_all[:-1],
        edges_all[1:],
        nmodes=n_modes,
        scale_cuts=[(edges[0], edges[-1])],
        npatch=None,
    ).values()
    npt.assert_array_equal(result["inds"], cut_inds)

    expected = jacobian @ cov_xi @ jacobian.T
    diagonal_only = np.diag(np.diag(cov_xi))
    expected_diagonal_only = jacobian @ diagonal_only @ jacobian.T
    atol = 1e-12 * np.max(np.abs(expected))
    return result["cov"], expected, expected_diagonal_only, atol


def test_cosebis_covariance_keeps_xi_cross_terms_and_units(monkeypatch):
    """At rtol=1e-10 and atol=1e-12 max|J C Jᵀ|, the deterministic identity has false-alarm probability 0; the tolerance allows only floating-point round-off."""
    actual, expected, expected_diagonal_only, atol = (
        _scan_covariance_and_basis_reference(monkeypatch)
    )
    n_modes = 3
    ee = slice(0, n_modes)
    bb = slice(n_modes, 2 * n_modes)

    npt.assert_allclose(actual[ee, ee], expected[ee, ee], rtol=1e-10, atol=atol)
    npt.assert_allclose(actual[bb, bb], expected[bb, bb], rtol=1e-10, atol=atol)

    # The dependency currently defines the terminal E/B cross block as zero;
    # this is a separate convention, not a general J C Jᵀ identity.
    npt.assert_array_equal(actual[ee, bb], np.zeros((n_modes, n_modes)))
    npt.assert_array_equal(actual[bb, ee], np.zeros((n_modes, n_modes)))

    for block in (actual[ee, ee], actual[bb, bb]):
        npt.assert_allclose(block, block.T, rtol=1e-10, atol=atol)
        assert np.linalg.eigvalsh(block).min() >= -atol

    ee_changed = np.any(
        np.abs(expected[ee, ee] - expected_diagonal_only[ee, ee]) > atol
    )
    bb_changed = np.any(
        np.abs(expected[bb, bb] - expected_diagonal_only[bb, bb]) > atol
    )
    assert ee_changed or bb_changed, "xi+/- cross terms must affect a marginal"
