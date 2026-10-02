"""A global minimum-PTE p-value must be valid for a finite mock ensemble."""

import numpy as np
import pytest

from sp_validation import statistics


@pytest.mark.xfail(
    strict=True,
    reason="#392: count/N gives anti-conservative finite-mock p-values",
)
def test_global_pte_rejects_at_most_alpha_with_finite_mocks():
    """The global PTE must control false positives across all N+1 null ranks.

    Under the null, the data minimum PTE is exchangeable with the N mock
    minima, so its rank among N+1 values is uniform. Enumerating one continuous
    statistic at every rank position against N = 20 distinct mocks gives the
    null rejection rate exactly. (count+1)/(N+1) rejects 1/21 at alpha = 0.05;
    count/N instead returns 0 and 0.05 at the two extreme ranks and rejects
    2/21, exceeding alpha. A finite-mock p-value must also never be zero.
    """
    if not hasattr(statistics, "calibrate_min_pte"):
        pytest.skip("calibrate_min_pte is absent on the tomography branch")
    alpha = 0.05
    n_mocks = 20
    mock_ptes = (np.arange(1, n_mocks + 1) / (n_mocks + 1.0))[:, None]
    calibration = statistics.calibrate_min_pte(mock_ptes, alpha=alpha)
    edges = np.concatenate([[0.0], mock_ptes[:, 0], [1.0]])
    data_values = 0.5 * (edges[:-1] + edges[1:])
    assert data_values.size == n_mocks + 1
    pvalues = np.array([calibration.global_pte([d])[0] for d in data_values])
    rejection_rate = np.mean(pvalues <= alpha)
    assert rejection_rate <= alpha, (
        f"Null rejection rate {rejection_rate:.4f} > alpha={alpha}; "
        f"p at the three most extreme ranks: {pvalues[:3].tolist()}"
    )
    assert pvalues.min() > 0, "global_pte returned zero at the most extreme rank"
