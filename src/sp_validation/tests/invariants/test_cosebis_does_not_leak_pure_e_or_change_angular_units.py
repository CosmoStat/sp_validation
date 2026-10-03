"""Log-COSEBIs must return zero B modes for analytic pure E and use radians."""

import math

import numpy as np
import pytest
from scipy.integrate import quad

pytestmark = [pytest.mark.fast, pytest.mark.decision("bmodes.cosebis_modes")]


_ARC_MIN_TO_RAD = math.pi / 10800.0
_GAUSSIAN_WIDTH_ARCMIN = 20.0
_THETA_MIN_ARCMIN = 1.0
_THETA_MAX_ARCMIN = 100.0
_NMODES = 3


def _xi_plus(theta_arcmin):
    return np.exp(
        -(np.asarray(theta_arcmin, dtype=float) ** 2) / (2 * _GAUSSIAN_WIDTH_ARCMIN**2)
    )


def _xi_minus(theta_arcmin):
    """Closed-form pure-E partner, evaluated stably at small theta."""
    theta_arcmin = np.asarray(theta_arcmin, dtype=float)
    u = theta_arcmin**2 / (2 * _GAUSSIAN_WIDTH_ARCMIN**2)
    exp_minus_u = np.exp(-u)
    one_minus_exp = -np.expm1(-u)
    one_minus_product = one_minus_exp - u * exp_minus_u
    direct = exp_minus_u + 2 * one_minus_exp / u - 6 * one_minus_product / u**2

    coefficients = np.zeros(15)
    for power in range(2, len(coefficients)):
        coefficients[power] = (
            (-1.0) ** power * power * (power - 1) / math.factorial(power + 2)
        )
    series = np.polynomial.polynomial.polyval(u, coefficients)
    return np.where(u < 0.05, series, direct)


def _cosebis_modes(theta_arcmin, xip, xim, covariance, left_edges, right_edges):
    from sp_validation import b_modes

    (result,) = b_modes.cosebis_scan_from_xi(
        theta_arcmin,
        xip,
        xim,
        covariance,
        left_edges,
        right_edges,
        nmodes=_NMODES,
        npatch=None,
    ).values()
    return result["En"], result["Bn"]


def _independent_e_modes():
    from cosmo_numba.B_modes.cosebis import COSEBIS
    from numba.typed import List

    filters = COSEBIS(
        _THETA_MIN_ARCMIN,
        _THETA_MAX_ARCMIN,
        N_max=_NMODES,
        precision=120,
    )
    filters.compute_roots()
    reference = np.empty(_NMODES)

    for mode in range(_NMODES):
        mode_roots = List()
        mode_roots.append(filters.roots[mode])
        mode_norms = filters.norms[mode : mode + 1]

        def integrand(theta_arcmin):
            theta = np.array([theta_arcmin])
            t_plus = filters.get_Tp_log(theta, roots=mode_roots, norms=mode_norms)[0, 0]
            t_minus = filters.get_Tm_log(theta, roots=mode_roots, norms=mode_norms)[
                0, 0
            ]
            theta_rad = theta_arcmin * _ARC_MIN_TO_RAD
            return (
                theta_rad
                * _ARC_MIN_TO_RAD
                * (t_plus * _xi_plus(theta_arcmin) + t_minus * _xi_minus(theta_arcmin))
                / 2
            )

        reference[mode] = quad(
            integrand,
            _THETA_MIN_ARCMIN,
            _THETA_MAX_ARCMIN,
            epsabs=1e-15,
            epsrel=1e-11,
        )[0]
    return reference


def test_cosebis_does_not_leak_pure_e_or_change_angular_units(tmp_path, monkeypatch):
    """Pure-E Gaussian xi gives B_n = 0 and E_n = radian quadrature, to 1e-4 max|E|.

    The fixture is analytic and noiseless, so the only error is Simpson
    discretisation: at 1001 log nodes |B_n|/max|E| ~ 1e-10 and
    |E_n - E_ref|/max|E| < 1e-8, at least four orders below the tolerance, and
    both shrink ~16x per node doubling. Correct code cannot trip it.
    """
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "numba-cache"))

    modes_by_resolution = {}
    for n_nodes in (1001, 2001):
        theta = np.geomspace(_THETA_MIN_ARCMIN, _THETA_MAX_ARCMIN, n_nodes)
        edges = np.geomspace(
            _THETA_MIN_ARCMIN / 1.001,
            _THETA_MAX_ARCMIN * 1.001,
            n_nodes + 1,
        )
        modes_by_resolution[n_nodes] = _cosebis_modes(
            theta,
            _xi_plus(theta),
            _xi_minus(theta),
            np.eye(2 * n_nodes),
            edges[:-1],
            edges[1:],
        )

    e_reference = _independent_e_modes()
    allowed_error = 1e-4 * np.max(np.abs(e_reference))
    e_1001, b_1001 = modes_by_resolution[1001]
    e_2001, b_2001 = modes_by_resolution[2001]

    max_b = max(np.max(np.abs(b_1001)), np.max(np.abs(b_2001)))
    assert max_b <= allowed_error, (
        f"pure-E COSEBIs leaked into B: max|B|={max_b:.6e}, "
        f"allowed={allowed_error:.6e} (1e-4 max|E_ref|)"
    )

    max_e_error = np.max(np.abs(e_1001 - e_reference))
    assert max_e_error <= allowed_error, (
        f"COSEBIs E disagrees with independent radian quadrature: "
        f"max|E-E_ref|={max_e_error:.6e}, allowed={allowed_error:.6e}"
    )

    refinement_error = max(
        np.max(np.abs(e_2001 - e_1001)),
        np.max(np.abs(b_2001 - b_1001)),
    )
    assert refinement_error < allowed_error / 4, (
        f"doubling log nodes changed a mode by {refinement_error:.6e}; "
        f"required < {allowed_error / 4:.6e}"
    )
