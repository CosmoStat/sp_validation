"""Pure-E/B MC covariance must describe the data's pair-weighted xi estimator."""

import inspect
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from sp_validation import b_modes as bm

# Reporting bins are exact unions of eight fine log bins, so no partial-bin
# interpolation enters the independently constructed pair-weighted reference.
EDGES_INT = np.geomspace(1.0, 100.0, 41)
THETA_INT = np.sqrt(EDGES_INT[:-1] * EDGES_INT[1:])
REPORT_IDX = np.array([4, 12, 20, 28, 36])
EDGES_REP = EDGES_INT[REPORT_IDX]
WEIGHT_INT = THETA_INT**2  # annular area, hence pair counts, grows as theta^2


def _pair_weighted_binning():
    matrix = np.zeros((len(EDGES_REP) - 1, len(THETA_INT)))
    for i, (lo, hi) in enumerate(zip(REPORT_IDX[:-1], REPORT_IDX[1:])):
        matrix[i, lo:hi] = WEIGHT_INT[lo:hi] / WEIGHT_INT[lo:hi].sum()
    return matrix


def test_mc_reporting_xi_is_pair_weighted_mean_of_fine_bins(monkeypatch, tmp_path):
    """Each MC reporting xi must be sum_j w_j xi_j / sum_j w_j, like TreeCorr.

    Reporting-bin xi feeds the local term of the pure-E/B transform. A uniform
    average of fine bins would give a covariance for a different estimator
    from the pair-weighted data. Exact unions of fine bins make the reference
    independent of the package's binning implementation.
    Develop's explicit operator already uses pair weights, so this test is
    unmarked: it protects against the tomography branch's uniform MC rebinning.
    The transform and theory stubs only expose its inputs, not fix the binning.
    """
    reference = _pair_weighted_binning()
    rng = np.random.default_rng(0)
    if "cov_path_int" in inspect.signature(bm.calculate_pure_eb_correlation).parameters:
        # Tomography API: observe reporting and fine xi passed to the transform.
        n_int = len(THETA_INT)
        gg = SimpleNamespace(
            meanr=np.sqrt(EDGES_REP[:-1] * EDGES_REP[1:]),
            left_edges=EDGES_REP[:-1],
            right_edges=EDGES_REP[1:],
            xip=np.zeros(4),
            xim=np.zeros(4),
        )
        gg_int = SimpleNamespace(
            meanr=THETA_INT,
            left_edges=EDGES_INT[:-1],
            right_edges=EDGES_INT[1:],
            xip=np.zeros(n_int),
            xim=np.zeros(n_int),
            weight=WEIGHT_INT,
        )
        cov_path = tmp_path / "cov_int.txt"
        np.savetxt(cov_path, np.eye(2 * n_int))
        calls = []

        def observe_pure_eb(theta, xip, xim, theta_int, xip_int, xim_int, **kwargs):
            calls.append((np.array([xip, xim]), np.array([xip_int, xim_int])))
            return tuple(np.zeros(len(theta)) for _ in range(6))

        # Replace only the transform dependency, avoiding its eager JIT import.
        # The package's actual MC sampling and rebinning still run unchanged.
        module = types.ModuleType("cosmo_numba.B_modes.schneider2022")
        module.get_pure_EB_modes = observe_pure_eb
        monkeypatch.setitem(sys.modules, "cosmo_numba.B_modes.schneider2022", module)
        monkeypatch.setattr(
            bm, "get_theo_xi", lambda **kwargs: (np.zeros(n_int), np.zeros(n_int))
        )
        # Keep the package's MC sampling but avoid changing global random state.
        monkeypatch.setattr(np.random, "multivariate_normal", rng.multivariate_normal)
        with pytest.warns(UserWarning, match="covariance matrix is not positive"):
            bm.calculate_pure_eb_correlation(
                gg,
                gg_int,
                cov_path_int=str(cov_path),
                cosmo_cov=object(),
                n_samples=3,
                z_dist=np.ones((5, 2)),
            )
        mc = calls[1:]  # First call transforms data, not an MC sample.
        assert len(mc) == 3
        got = np.array([reporting for reporting, fine in mc])
        want = np.array([fine @ reference.T for reporting, fine in mc])
    else:
        # Develop API: rebinning is the explicit linear operator.
        matrix, edges = bm._reporting_binning(WEIGHT_INT, EDGES_INT, EDGES_REP)
        np.testing.assert_allclose(edges, EDGES_REP)
        fine = rng.standard_normal((3, len(THETA_INT)))
        got = fine @ matrix.T
        want = fine @ reference.T

    np.testing.assert_allclose(
        got,
        want,
        rtol=0,
        atol=1e-12,
        err_msg="MC reporting xi is not the pair-weighted mean of fine bins",
    )
