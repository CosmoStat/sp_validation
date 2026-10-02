"""COSEBIs marginal E/B chi2 must debias the dimension actually inverted."""

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import stats

from sp_validation import b_modes

NPATCH, NMODES, NBINS = 100, 20, 24


class _LinearCOSEBIS:
    """Cheap linear filter: take the first NMODES xi+ and xi- bins.

    Only the expensive transform is stubbed; the package's marginal inverses,
    Hartlap correction and PTE calculation run unchanged.
    """

    def __init__(self, theta_min, theta_max, N_max, precision):
        self.n = N_max

    def cosebis_from_xipm(self, theta, xip, xim, parallel=True):
        return xip[: self.n], xim[: self.n]

    def cosebis_covariance_from_xipm_covariance(self, theta, cov):
        indices = np.r_[np.arange(self.n), len(theta) + np.arange(self.n)]
        return cov[np.ix_(indices, indices)]


@pytest.mark.xfail(
    strict=True,
    reason="#378: marginal COSEBIs chi2 uses Hartlap p=2*nmodes",
)
def test_cosebis_marginal_chi2_hartlap_uses_nmodes_not_2_nmodes(monkeypatch):
    """Separate E and B inverses each need Hartlap p = NMODES, not 2*NMODES.

    A Wishart marginal has the same realisation count N and its own dimension
    M, giving (N - M - 2)/(N - 1). The linear filter yields unit modes and an
    identity covariance, so each raw chi2 is exactly 20. With N = 100 patches,
    the correct chi2 is 20 * 78/99 = 15.758 and PTE_B is 0.7315. Using the
    joint dimension instead gives 20 * 58/99 = 11.717 and PTE_B = 0.9255.
    Both the marginal chi2 values and the reported B-mode PTE are protected.
    """
    module = types.ModuleType("cosmo_numba.B_modes.cosebis")
    module.COSEBIS = _LinearCOSEBIS
    monkeypatch.setitem(sys.modules, "cosmo_numba.B_modes.cosebis", module)

    edges = np.geomspace(1.0, 250.0, NBINS + 1)
    gg = SimpleNamespace(
        meanr=np.sqrt(edges[:-1] * edges[1:]),
        xip=np.ones(NBINS),
        xim=np.ones(NBINS),
        cov=np.eye(2 * NBINS),
        npatch1=NPATCH,
        left_edges=edges[:-1],
        right_edges=edges[1:],
    )
    (result,) = b_modes.calculate_cosebis(gg, nmodes=NMODES).values()
    expected_hartlap = (NPATCH - NMODES - 2) / (NPATCH - 1)
    expected_chi2 = NMODES * expected_hartlap
    expected_pte = stats.chi2.sf(expected_chi2, NMODES)
    np.testing.assert_allclose(
        [
            result["hartlap_factor"],
            result["chi2_E"],
            result["chi2_B"],
            result["pte_B"],
        ],
        [expected_hartlap, expected_chi2, expected_chi2, expected_pte],
        rtol=1e-10,
        atol=0,
    )
