"""Joint xi+ B / xi- B statistics must debias the full joint inverse covariance."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from sp_validation.b_modes import calculate_eb_statistics


@pytest.mark.parametrize("start, stop", [(0, 4), (0, 3), (1, 3)])
def test_joint_b_mode_chi2_hartlap_uses_joint_dimension_2n(start, stop):
    """The combined B-mode PTE must apply Hartlap with p = 2n, not p = n.

    An inverse sample covariance from N patches over p entries requires
    (N - p - 2)/(N - 1). For identity covariance the raw joint chi2 is the
    sum of squared xi+ B and xi- B over the cut, so the PTE is known by hand.
    Full and interior cuts check that p is twice the *selected* bin count.
    Develop already uses p = 2n; this unmarked test protects that correct
    behaviour against the tomography branch's single-block p = n regression.
    """
    nbins, npatch = 4, 50
    results = {
        # Both API contracts: develop reads theta/npatch; tomography reads gg.
        "theta": np.geomspace(1.0, 100.0, nbins),
        "npatch": npatch,
        "gg": SimpleNamespace(nbins=nbins, npatch1=npatch, npatch2=npatch),
        "cov": np.eye(6 * nbins),
        "xip_B": np.linspace(0.5, 2.0, nbins),
        "xim_B": np.linspace(-1.5, 1.0, nbins),
    }
    pte = calculate_eb_statistics(results)["pte_matrices"]["combined"]
    n = stop - start
    raw = np.sum(results["xip_B"][start:stop] ** 2) + np.sum(
        results["xim_B"][start:stop] ** 2
    )
    expected = stats.chi2.sf((npatch - 2 * n - 2) / (npatch - 1) * raw, 2 * n)
    assert pte[start, stop - 1] == pytest.approx(expected, rel=1e-10)
