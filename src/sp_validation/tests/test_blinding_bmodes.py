"""B-modes under a blind, on synthetic ξ±.

A blind shifts ξ± by one E-mode signal, so it moves COSEBIs B_n and pure-E/B
ξ_B only by each transform's response to a pure E-mode vector. The input is the
default theory's ξ± at the fiducial on the production reporting and integration
grids, noise-free and with one seeded shape-noise draw, sealed under a blind
whose hidden point is each corner of the envelope in turn. Every tolerance is
the same ceiling: a tenth of a bin's standard deviation.
"""

import numpy as np
import pytest

from sp_validation import b_modes, theory
from sp_validation import blinding as bd
from sp_validation import custody as cu
from sp_validation import sacc_io as sio

REPORTING = (1.0, 250.0, 20)
INTEGRATION = (0.08, 300.0, 1000)
COSEBIS_CUT, NMODES = (12.0, 83.0), 20
# SP_v1.4.6.3's shape noise (its cov_th in cosmo_val/cat_config.yaml).
AREA_DEG2, N_EFF, SIGMA_E = 2894.0, 4.96, 0.378
CEILING = 0.1


def _grid(lo, hi, n):
    edges = np.geomspace(lo, hi, n + 1)
    return np.sqrt(edges[:-1] * edges[1:]), edges


THETA, EDGES = _grid(*REPORTING)
THETA_INT, EDGES_INT = _grid(*INTEGRATION)


def _variance(edges):
    """Var ξ± = σ_e⁴ / (2 N_pairs) (Schneider et al. 2002) in each bin."""
    n_gal = N_EFF * AREA_DEG2 * 3600
    return SIGMA_E**4 / (n_gal * N_EFF * np.pi * np.diff(edges**2))


VARIANCE = np.concatenate([_variance(e) for e in (EDGES, EDGES, EDGES_INT, EDGES_INT)])


def _fiducial_xi():
    """The default theory's ξ± on both grids, with the shape-noise covariance."""
    z = np.linspace(0.01, 3.0, 300)
    s = sio.new_sacc({0: (z, np.exp(-(((z - 0.7) / 0.3) ** 2)))})
    for theta, grid in ((THETA, "reporting"), (THETA_INT, "integration")):
        sio.add_xi(s, (0, 0), theta, 0 * theta, 0 * theta, grid=grid, theta_nom=theta)
    for dp, value in zip(s.data, theory.predict(s, theory.fiducial())):
        dp.value = float(value)
    s.add_covariance(VARIANCE)
    return s


def _vector(s):
    """(ξ+, ξ−) on the reporting grid, then on the integration grid."""
    views = [sio.xi_correlation(s, grid=g) for g in ("reporting", "integration")]
    return np.concatenate([np.r_[v.xip, v.xim] for v in views])


@pytest.fixture(scope="module")
def shifts(tmp_path_factory):
    """Each input's ξ± and, per envelope corner, its sealed ξ±'s shift."""
    root = tmp_path_factory.mktemp("bmodes")
    catalogues = {
        "paths": {"blinds": str(root / "blinds")},
        "SYN": {"shear": {"path": str(root / "SYN.fits")}, "blind": "syn"},
    }
    bd.init("syn", catalogues)
    custody = cu.custody_of(catalogues, "SYN")
    clean = _fiducial_xi()
    noisy = clean.copy()
    noise = np.random.default_rng(3).normal(0.0, np.sqrt(VARIANCE))
    for dp, n in zip(noisy.data, noise):
        dp.value += float(n)
    fiducial = theory.fiducial()
    corners = [
        {**fiducial, "S8": fiducial["S8"] + a, "Omega_m": fiducial["Omega_m"] + b}
        for a in (-bd.ENVELOPE["S8"], bd.ENVELOPE["S8"])
        for b in (-bd.ENVELOPE["Omega_m"], bd.ENVELOPE["Omega_m"])
    ]
    out = {"noise-free": (_vector(clean), []), "noisy": (_vector(noisy), [])}
    with pytest.MonkeyPatch.context() as m:
        for corner in corners:
            m.setattr(bd, "_hidden", lambda blind, corner=corner: corner)
            for name, s in (("noise-free", clean), ("noisy", noisy)):
                out[name][1].append(_vector(sio.seal(s, custody)) - out[name][0])
    return out


def _as_b(delta):
    """The pure-B counterpart of an E-mode ξ± vector: (δξ+, −δξ−) per grid."""
    n, n_int = len(THETA), len(THETA_INT)
    sign = np.r_[np.ones(n), -np.ones(n), np.ones(n_int), -np.ones(n_int)]
    return sign * delta


def _pure_b(x):
    """Pure-E/B (ξ+_B, ξ−_B) on the reporting grid."""
    n, n_int = len(THETA), len(THETA_INT)
    xip, xim = x[:n], x[n : 2 * n]
    xip_int, xim_int = x[2 * n : 2 * n + n_int], x[2 * n + n_int :]
    modes = b_modes.pure_eb_from_xi(
        THETA, xip, xim, THETA_INT, xip_int, xim_int, EDGES[0], EDGES[-1]
    )
    return np.r_[modes["xip_B"], modes["xim_B"]]


@pytest.fixture(scope="module")
def cosebis():
    """The fiducial cut's COSEBIs B_n of an integration-grid ξ±, and σ(B_n).

    The transform :func:`b_modes.cosebis_scan_from_xi` runs, built once.
    """
    from cosmo_numba.B_modes.cosebis import COSEBIS

    n, n_int = len(THETA), len(THETA_INT)
    cut = np.flatnonzero((THETA_INT >= COSEBIS_CUT[0]) & (THETA_INT <= COSEBIS_CUT[1]))
    theta = THETA_INT[cut]
    transform = COSEBIS(
        theta_min=theta.min(), theta_max=theta.max(), N_max=NMODES, precision=120
    )
    rows = np.r_[2 * n + cut, 2 * n + n_int + cut]
    covariance = transform.cosebis_covariance_from_xipm_covariance(
        theta, np.diag(VARIANCE[rows])
    )

    def b_n(x):
        return transform.cosebis_from_xipm(
            theta, x[2 * n + cut], x[2 * n + n_int + cut], parallel=True
        )[1]

    return b_n, np.sqrt(np.diag(covariance)[NMODES:])


@pytest.fixture(scope="module")
def sigma_pure_b(shifts):
    """σ of pure-E/B ξ_B over shape-noise draws about the fiducial."""
    clean, _ = shifts["noise-free"]
    rng = np.random.default_rng(5)
    draws = [_pure_b(clean + rng.normal(0.0, np.sqrt(VARIANCE))) for _ in range(50)]
    return np.std(draws, axis=0)


@pytest.mark.parametrize("noise", ["noise-free", "noisy"])
def test_cosebis_b_modes_move_only_by_the_transforms_response(shifts, cosebis, noise):
    """COSEBIs are linear with fixed weights: ΔB_n = B_n(δ) to round-off, and
    B_n(δ), the transform's B response to a pure E-mode δ, stays under the
    ceiling; δ's pure-B counterpart moves B_n well past it."""
    b_n, sigma = cosebis
    x, deltas = shifts[noise]
    for delta in deltas:
        before, after = b_n(x), b_n(x + delta)
        roundoff = 1e-10 * np.max(np.abs(np.r_[before, after, b_n(delta)]))
        np.testing.assert_allclose(after - before, b_n(delta), rtol=0, atol=roundoff)
        assert np.max(np.abs(b_n(delta)) / sigma) < CEILING
        assert np.max(np.abs(b_n(_as_b(delta))) / sigma) > 10 * CEILING


@pytest.mark.parametrize(
    "noise",
    [
        "noise-free",
        pytest.param(
            "noisy",
            marks=pytest.mark.xfail(
                strict=True,
                reason="pure-eb-bmode-numerics: cosmo_numba's adaptive quadrature "
                "does not converge on a noisy 1000-bin ξ±, so the kernel is not "
                "additive there",
            ),
        ),
    ],
)
def test_pure_eb_b_modes_move_only_by_the_transforms_response(
    shifts, sigma_pure_b, noise
):
    """ΔB = B(x + δ) − B(x) equals B(δ), the kernel's B response to a pure
    E-mode δ, and both stay under the ceiling; δ's pure-B counterpart moves B
    well past it."""
    x, deltas = shifts[noise]
    before = _pure_b(x)
    for delta in deltas:
        moved = _pure_b(x + delta) - before
        response = _pure_b(delta)
        assert np.max(np.abs(moved - response) / sigma_pure_b) < CEILING
        assert np.max(np.abs(response) / sigma_pure_b) < CEILING
        assert np.max(np.abs(_pure_b(_as_b(delta))) / sigma_pure_b) > 10 * CEILING
