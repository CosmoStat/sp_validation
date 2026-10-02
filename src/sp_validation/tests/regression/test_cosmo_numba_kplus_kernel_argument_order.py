"""Protect the Schneider et al. (2022, arXiv:2110.09774) pure-E/B K+ kernel.

Equation 51 defines K+(vt, th) = (theta_bar/vt)^2 H-(th, vt), with the
integration variable in H-'s first slot. The independent reference below uses
its integral definition instead: H+ is the reproducing kernel of {1, theta^2}
with weight theta/theta_bar^2, and K+ follows by quadrature. A Gaussian pure-E
xi+ with its exact xi- also checks that both transform paths have zero B modes.
"""

import numpy as np
import pytest
from scipy.integrate import quad

# cosmo_numba eagerly JIT-compiles on import (~20-40 s on two cores). Import
# inside each test so selecting "not slow" also avoids that compilation.
pytestmark = pytest.mark.slow

TMIN, TMAX = 1.0, 100.0  # arcmin; K+ matters on an unpadded window
TBAR = 0.5 * (TMIN + TMAX)
B = (TMAX - TMIN) / (TMAX + TMIN)
S = 20.0  # arcmin; Gaussian xi+ width


def _basis(t):
    return np.array([1.0, (t / TBAR) ** 2])


# Compute the Gram matrix independently of cosmo_numba's closed-form kernels.
_GRAM = np.array(
    [
        [
            quad(lambda t: t * _basis(t)[i] * _basis(t)[j], TMIN, TMAX)[0]
            for j in range(2)
        ]
        for i in range(2)
    ]
)


def _h_plus_reference(vt, th):
    """Reproducing kernel under d theta theta / theta_bar^2."""
    return TBAR**2 * _basis(vt) @ np.linalg.solve(_GRAM, _basis(th))


def _k_plus_reference(vt, th):
    """First line of Schneider+22 Eq. 51, by quadrature over phi."""
    integral = quad(
        lambda p: p / vt**2 * _h_plus_reference(p, th) * (4 - 12 * p**2 / vt**2),
        TMIN,
        vt,
        epsabs=1e-13,
        epsrel=1e-12,
    )[0]
    return _h_plus_reference(vt, th) + integral


@pytest.mark.xfail(
    strict=True,
    reason="aguinot/cosmo-numba#22: K_p swaps H_- arguments",
)
def test_kplus_kernel_follows_eq51_argument_order():
    """K_p(vt, th) must equal Eq. 51's independent integral definition.

    H+ projects onto {1, theta^2}; no cosmo_numba closed form enters the
    reference. Swapping H-'s arguments produces order-unity relative errors
    on this [1, 100] arcmin window.
    """
    from cosmo_numba.B_modes import schneider2022_nb as nbk

    vts = np.array([3.0, 10.0, 40.0, 90.0])
    ths = np.array([1.5, 7.0, 25.0, 60.0, 99.0])
    ref = np.array([[_k_plus_reference(v, t) for t in ths] for v in vts])
    got = np.array([[nbk.K_p(v, t, TBAR, B) for t in ths] for v in vts])
    np.testing.assert_allclose(
        got,
        ref,
        rtol=1e-8,
        atol=1e-12,
        err_msg="cosmo_numba K_p(vt, th) != Eq. 51 K+(vt, th)",
    )


@pytest.mark.xfail(
    strict=True,
    reason="aguinot/cosmo-numba#22: K_p swaps H_- arguments",
)
def test_vplus_of_constant_xip_matches_closed_form():
    """For xi+ = 1, V+(vt) = -2 a^2/vt^2 + 3 a^4/vt^4, a = theta_min.

    H+'s normalisation gives int dth th/theta_bar^2 H+(phi, th) = 1.
    Thus V+ = 1 + [2 phi^2/vt^2 - 3 phi^4/vt^4]_a^vt, yielding the exact
    value -0.0197 at vt = 10 arcmin and a = 1 arcmin.
    """
    from cosmo_numba.B_modes import schneider2022_nb as nbk

    vt = 10.0
    expected = -2 * TMIN**2 / vt**2 + 3 * TMIN**4 / vt**4
    got = quad(
        lambda th: th / TBAR**2 * nbk.K_p(vt, th, TBAR, B),
        TMIN,
        TMAX,
        epsabs=1e-14,
        epsrel=1e-12,
    )[0]
    assert got == pytest.approx(expected, rel=1e-10)


def _xip(t):
    return np.exp(-(t**2) / (2 * S**2))


def _xim(t):
    """Exact pure-E xi- for Gaussian xi+ (P(ell) ~ exp(-ell^2 S^2 / 2))."""
    u = t**2 / (2 * S**2)
    return (
        _xip(t)
        + 4 * S**2 / t**2 * (1 - np.exp(-u))
        - 24 * S**4 / t**4 * (1 - np.exp(-u) * (1 + u))
    )


def _narrow_nodes():
    log_edges = np.log(np.geomspace(12.0, 83.0, 401))
    return np.exp(0.5 * (log_edges[:-1] + log_edges[1:]))


@pytest.mark.xfail(
    strict=True,
    reason="aguinot/cosmo-numba#22: K_p swaps H_- arguments",
)
def test_pure_e_input_has_no_xim_b_fixed_quadrature_operator():
    """The fixed-quadrature operator must give zero B for exact pure-E xi+/xi-.

    xi- = xi+ + int_0^theta dphi phi/theta^2 (4 - 12 phi^2/theta^2) xi+
    defines the analytic Gaussian fixture. Without xi- padding, K+ matters
    on the narrow 12-83 arcmin window. Both B rows must vanish to 1e-6;
    the independent xi+ B row already does, while swapped K+ breaks xi- B.
    """
    from cosmo_numba.B_modes.schneider2022_operator import get_pure_EB_operator

    nodes = _narrow_nodes()
    data = np.concatenate([_xip(nodes), _xim(nodes)])
    ops = get_pure_EB_operator(
        nodes,
        nodes,
        tmin=nodes[0] * (1 - 1e-9),
        tmax=nodes[-1] * (1 + 1e-9),
        local_from_int=True,
        pad_xim=False,
    )
    xip_b, xim_b = ops[2] @ data, ops[3] @ data
    assert np.nanmax(np.abs(xip_b)) < 1e-6
    assert np.nanmax(np.abs(xim_b)) < 1e-6


@pytest.mark.xfail(
    strict=True,
    reason="aguinot/cosmo-numba#22: K_p swaps H_- arguments",
)
def test_pure_e_input_has_no_xim_b_adaptive_modes():
    """The adaptive transform must also give zero xi- B for exact pure-E input.

    This is the transform used on the tomography branch. The same analytic
    Gaussian fixture and unpadded 12-83 arcmin window make a swapped K+
    observable; the correct pure-B answer is zero, to quadrature accuracy.
    """
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    nodes = _narrow_nodes()
    theta = np.geomspace(15.0, 70.0, 6)
    res = get_pure_EB_modes(
        theta,
        _xip(theta),
        _xim(theta),
        nodes,
        _xip(nodes),
        _xim(nodes),
        tmin=12.0,
        tmax=83.0,
        pad_xim=False,
    )
    assert np.max(np.abs(np.asarray(res[3]))) < 1e-6
