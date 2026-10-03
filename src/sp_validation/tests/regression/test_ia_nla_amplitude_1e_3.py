"""NLA intrinsic-alignment amplitude in the GLASS mock convergence."""

import numpy as np
import pytest

from sp_validation.glass_mock import GlassMockConfig, ia_convergence

# Critical density today in units of h^2 Msun / Mpc^3 (3 H0^2 / 8 pi G with
# H0 = 100 h km/s/Mpc); standard value, independent of astropy.
RHO_CRIT_OVER_H2 = 2.77536627e11
# NLA normalisation C1 = 5e-14 h^-2 Msun^-1 Mpc^3 (Hirata & Seljak 2004).
C1_TIMES_H2 = 5e-14


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="#386: NLA convergence amplitude is 1000x too small",
)
def test_ia_nla_prefactor_is_c1_rho_crit_not_1e3_smaller():
    """Protect the NLA amplitude of the IA convergence term in GLASS mocks.

    The NLA model adds kappa_IA = -A_ia * C1 * rho_crit * Om / D(z) * delta.
    C1 * rho_crit is dimensionless and h-independent: 5e-14 * 2.775e11 =
    0.01388 (the familiar "C1 rho_crit ~ 0.0134" of Bridle & King 2007, up to
    the rounding of rho_crit). At z = 0, D = 1, so with A_ia = 1 and delta = 1
    the term must be -0.01388 * Om. Giving Newton's G its SI units twice makes
    rho_crit, and hence the IA term, 1000x too small.
    """
    cfg = GlassMockConfig(ia_bias=1.0)
    expected = -C1_TIMES_H2 * RHO_CRIT_OVER_H2 * cfg.Om
    got = float(np.asarray(ia_convergence(np.array([1.0]), (None, None, 0.0), cfg))[0])
    assert np.isclose(got, expected, rtol=1e-3), (
        f"ia_convergence(delta=1, z=0, A_ia=1) = {got:.4e}, expected "
        f"-C1*rho_crit*Om = {expected:.4e} (ratio {got / expected:.6f})"
    )
