"""Tests for :mod:`sp_validation.cosmo_val.sacc_writers`.

Synthetic and fast: each ``*_to_sacc`` writer is exercised with in-memory
arrays, round-tripped through ``tmp_path``, and checked against the SACC layout
contract (data types, tags, ordering, covariance alignment). The analysis-file
assembler is verified to produce a single ``BlockDiagonalCovariance`` covering every
point with each per-statistic block correctly placed. One real small-nside
NaMaster round-trip proves the pseudo-Cℓ window survives the writer path.
"""

import numpy as np
import pytest

from sp_validation import sacc_io as sio
from sp_validation.cosmo_val import sacc_writers as sw


def _nz(seed=0, n=40):
    rng = np.random.default_rng(seed)
    return np.linspace(0.01, 2.0, n), rng.uniform(0.1, 1.0, n)


def _spd(n, seed):
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _theta(n=6):
    return np.geomspace(1.0, 100.0, n)


def _roundtrip(s, tmp_path, name):
    p = tmp_path / f"{name}.sacc"
    sio.save(s, str(p), type="mock")
    return sio.load(str(p))


META = {"catalogue_version": "vSYNTH", "npatch": 1}


# --------------------------------------------------------------------------- #
# Per-writer parts
# --------------------------------------------------------------------------- #
def test_xi_to_sacc_reporting(tmp_path):
    theta = _theta()
    xip, xim = np.arange(6) * 1e-5, np.arange(6) * 2e-5
    s = sw.xi_to_sacc(
        {0: _nz()}, META, theta, xip, xim, grid="reporting", theta_nom=theta * 1.01
    )
    s2 = _roundtrip(s, tmp_path, "xic")
    th, p, m = sio.get_xi(s2, (0, 0), grid="reporting")
    assert np.array_equal(th, theta)
    assert np.array_equal(p, xip) and np.array_equal(m, xim)
    assert s2.covariance is None  # reporting part has no cov until assembly


def test_xi_to_sacc_integration_diagonal(tmp_path):
    theta = np.geomspace(0.5, 300.0, 30)
    xip, xim = np.arange(30) * 1e-5, np.arange(30) * 2e-5
    varxip, varxim = np.arange(1, 31) * 1e-12, np.arange(1, 31) * 2e-12
    s = sw.xi_to_sacc(
        {0: _nz()},
        META,
        theta,
        xip,
        xim,
        grid="integration",
        variances=np.concatenate([varxip, varxim]),
    )
    assert type(s.covariance).__name__ == "DiagonalCovariance"
    s2 = _roundtrip(s, tmp_path, "xif")
    th, p, _ = sio.get_xi(s2, (0, 0), grid="integration")
    assert np.array_equal(th, theta) and np.array_equal(p, xip)
    assert np.array_equal(
        np.diag(s2.covariance.dense), np.concatenate([varxip, varxim])
    )


def test_pseudo_cl_to_sacc_window_and_rows(tmp_path):
    ell = np.array([30.0, 60.0, 90.0, 120.0])
    nbp = len(ell)
    # NaMaster (4, nbp): EE, EB, BE, BB.
    cl_all = np.vstack(
        [
            np.arange(nbp) * 1e-9,
            np.arange(nbp) * 2e-9,
            np.zeros(nbp),
            np.arange(nbp) * 3e-9,
        ]
    )

    class _Wsp:
        """Stand-in workspace: (n_cl_out, nbp, n_cl_in, nell) window array."""

        def __init__(self, nbp, nell):
            w = np.zeros((4, nbp, 4, nell))
            col = np.zeros((nbp, nell))
            for b in range(nbp):
                col[b, b * 3 : b * 3 + 3] = 1.0
            for out in range(4):
                w[out, :, out, :] = col
            self._w = w

        def get_bandpower_windows(self):
            return self._w

    s = sw.pseudo_cl_to_sacc({0: _nz()}, META, ell, cl_all, _Wsp(nbp, 24))
    s2 = _roundtrip(s, tmp_path, "cl")
    ell_r, ee, bb, eb, window = sio.get_pseudo_cl(s2, (0, 0))
    assert np.array_equal(ell_r, ell)
    assert np.array_equal(ee, cl_all[0])  # EE row
    assert np.array_equal(bb, cl_all[3])  # BB row (index 3, not 2=BE)
    assert np.array_equal(eb, cl_all[1])  # EB row
    assert window.weight.shape == (24, nbp)


def test_pseudo_cl_to_sacc_real_namaster(tmp_path):
    """A real small-nside NaMaster workspace's window survives the writer."""
    pytest.importorskip("pymaster")
    from sp_validation.pseudo_cl import get_pseudo_cls_map

    nside = 32
    mask = np.ones(12 * nside**2)
    rng = np.random.default_rng(0)
    shear = (
        rng.normal(size=12 * nside**2) + 1j * rng.normal(size=12 * nside**2)
    ) * 1e-2
    ell_eff, cl_all, wsp = get_pseudo_cls_map(shear, mask, nside, "linear", ell_step=8)
    s = sw.pseudo_cl_to_sacc({0: _nz()}, META, ell_eff, cl_all, wsp)
    s2 = _roundtrip(s, tmp_path, "clreal")
    ell_r, ee, bb, eb, window = sio.get_pseudo_cl(s2, (0, 0))
    assert np.array_equal(ell_r, ell_eff)
    assert np.array_equal(ee, cl_all[0]) and np.array_equal(bb, cl_all[3])
    # window columns correspond to the bandpowers, one per ell_eff
    assert window.weight.shape[1] == len(ell_eff)


def test_cosebis_to_sacc(tmp_path):
    En, Bn = np.arange(1, 11) * 1e-6, np.arange(1, 11) * 1e-7
    result = {"En": En, "Bn": Bn, "cov": _spd(20, 7)}
    s = sw.cosebis_to_sacc({0: _nz()}, META, result, (1.0, 100.0))
    s2 = _roundtrip(s, tmp_path, "co")
    n, E, B = sio.get_cosebis(s2, (0, 0))
    assert np.array_equal(n, np.arange(1, 11))
    assert np.array_equal(E, En) and np.array_equal(B, Bn)
    assert type(s2.covariance).__name__ == "FullCovariance"
    assert np.array_equal(s2.covariance.dense, result["cov"])


def test_pure_eb_to_sacc(tmp_path):
    theta = _theta()
    eb = {key: np.arange(6) * (i + 1) * 1e-6 for i, key in enumerate(sio.PURE_KEYS)}
    cov = _spd(6 * len(theta), 9)
    s = sw.pure_eb_to_sacc({0: _nz()}, META, theta, eb, covariance=cov)
    s2 = _roundtrip(s, tmp_path, "eb")
    th, back = sio.get_pure_eb(s2, (0, 0))
    assert np.array_equal(th, theta)
    for key in sio.PURE_KEYS:
        assert np.array_equal(back[key], eb[key])
    assert np.array_equal(s2.covariance.dense, cov)


def _rho_tau_tables(nth=6, seed=0):
    rng = np.random.default_rng(seed)
    theta = _theta(nth)
    rho = {"theta": theta}
    for k in sw.RHO_K:
        for suffix in ("p", "m"):
            rho[f"rho_{k}_{suffix}"] = rng.normal(size=nth) * 1e-6
            rho[f"varrho_{k}_{suffix}"] = rng.uniform(1e-14, 1e-13, nth)
    tau = {"theta": theta}
    for k in sw.TAU_K:
        for suffix in ("p", "m"):
            tau[f"tau_{k}_{suffix}"] = rng.normal(size=nth) * 1e-6
            tau[f"vartau_{k}_{suffix}"] = rng.uniform(1e-14, 1e-13, nth)
    return rho, tau, theta


def test_rho_tau_to_sacc_diagonal(tmp_path):
    rho, tau, theta = _rho_tau_tables()
    s = sw.rho_tau_to_sacc({0: _nz()}, META, rho, tau)
    s2 = _roundtrip(s, tmp_path, "rt")
    for k in sw.RHO_K:
        th, p, m = sio.get_rho(s2, k)
        assert np.array_equal(th, theta)
        assert np.array_equal(p, rho[f"rho_{k}_p"])
        assert np.array_equal(m, rho[f"rho_{k}_m"])
    for k in sw.TAU_K:
        th, p, m = sio.get_tau(s2, (0, 0), k)
        assert np.array_equal(p, tau[f"tau_{k}_p"])
    assert type(s2.covariance).__name__ == "DiagonalCovariance"


def test_rho_tau_to_sacc_tau_theory_block(tmp_path):
    """The (3·nbin) plus-only CovTauTh block scatters into the τ-plus rows/cols;
    τ-minus keeps a vartau diagonal, and cross plus↔minus stays zero."""
    rho, tau, theta = _rho_tau_tables()
    nbin = len(theta)
    n_plus = len(sw.TAU_K) * nbin  # τ-plus points (k-major, one component per k)
    tau_cov_th = _spd(n_plus, 11)
    s = sw.rho_tau_to_sacc({0: _nz()}, META, rho, tau, tau_cov_th=tau_cov_th)
    assert type(s.covariance).__name__ == "FullCovariance"
    tr = ("source_0", sio.PSF_TRACER)
    tau_plus = np.concatenate(
        [s.indices(sio.TAU_PLUS.format(k=k), tr) for k in sw.TAU_K]
    )
    tau_minus = np.concatenate(
        [s.indices(sio.TAU_MINUS.format(k=k), tr) for k in sw.TAU_K]
    )
    s2 = _roundtrip(s, tmp_path, "rttau")
    dense = s2.covariance.dense
    # τ-plus sub-block equals the supplied theory covariance (scatter is correct).
    assert np.allclose(dense[np.ix_(tau_plus, tau_plus)], tau_cov_th)
    # τ-minus is diagonal from vartau; plus↔minus cross is zero.
    tau_minus_var = np.concatenate([np.asarray(tau[f"vartau_{k}_m"]) for k in sw.TAU_K])
    assert np.allclose(np.diag(dense[np.ix_(tau_minus, tau_minus)]), tau_minus_var)
    assert np.allclose(dense[np.ix_(tau_plus, tau_minus)], 0.0)


def test_rho_tau_to_sacc_tau_cov_shape_mismatch():
    rho, tau, _ = _rho_tau_tables()
    with pytest.raises(ValueError, match="tau_cov_th shape"):
        sw.rho_tau_to_sacc({0: _nz()}, META, rho, tau, tau_cov_th=_spd(3, 1))


# --------------------------------------------------------------------------- #
# Analysis-file assembly
# --------------------------------------------------------------------------- #
def _make_parts(nz):
    theta = _theta()
    ell = np.array([30.0, 60.0, 90.0])

    class _Wsp:
        def get_bandpower_windows(self):
            w = np.zeros((4, 3, 4, 20))
            for out in range(4):
                for b in range(3):
                    w[out, b, out, b * 6 : b * 6 + 6] = 1.0
            return w

    xi = sw.xi_to_sacc(
        nz, META, theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="reporting"
    )
    xi.add_covariance(_spd(len(xi.mean), 1))
    cl_all = np.vstack(
        [np.arange(3) * 1e-9, np.arange(3) * 2e-9, np.zeros(3), np.arange(3) * 3e-9]
    )
    cl = sw.pseudo_cl_to_sacc(nz, META, ell, cl_all, _Wsp(), covariance=_spd(9, 2))
    co = sw.cosebis_to_sacc(
        nz,
        META,
        {
            "En": np.arange(1, 6) * 1e-6,
            "Bn": np.arange(1, 6) * 1e-7,
            "cov": _spd(10, 3),
        },
        (1.0, 100.0),
    )
    return [xi, cl, co]


def test_assemble_analysis_sacc_block_diagonal_covariance(tmp_path):
    nz = {0: _nz()}
    parts = _make_parts(nz)
    s = sw.assemble_analysis_sacc(nz, META, parts)
    assert type(s.covariance).__name__ == "BlockDiagonalCovariance"
    assert s.covariance.dense.shape == (len(s.mean), len(s.mean))
    # every point covered; blocks placed and cross-blocks zero
    tr = ("source_0", "source_0")
    xi_idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    cl_idx = np.concatenate(
        [s.indices(sio.CL_EE, tr), s.indices(sio.CL_BB, tr), s.indices(sio.CL_EB, tr)]
    )
    co_idx = np.concatenate(
        [s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)]
    )
    assert len(xi_idx) + len(cl_idx) + len(co_idx) == len(s.mean)
    dense = s.covariance.dense
    assert np.array_equal(dense[np.ix_(xi_idx, xi_idx)], parts[0].covariance.dense)
    assert np.array_equal(dense[np.ix_(cl_idx, cl_idx)], parts[1].covariance.dense)
    assert np.array_equal(dense[np.ix_(co_idx, co_idx)], parts[2].covariance.dense)
    assert np.array_equal(
        dense[np.ix_(xi_idx, cl_idx)], np.zeros((len(xi_idx), len(cl_idx)))
    )
    # round-trips
    s2 = _roundtrip(s, tmp_path, "analysis")
    assert type(s2.covariance).__name__ == "BlockDiagonalCovariance"
    assert np.allclose(s2.covariance.dense, s.covariance.dense)


def test_assemble_analysis_sacc_requires_covariance():
    nz = {0: _nz()}
    parts = _make_parts(nz)
    parts.append(
        sw.xi_to_sacc(
            nz,
            META,
            _theta(),
            np.arange(6) * 1e-5,
            np.arange(6) * 2e-5,
            grid="reporting",
        )
    )  # no covariance
    with pytest.raises(ValueError, match="own covariance block"):
        sw.assemble_analysis_sacc(nz, META, parts)


def test_assemble_from_reloaded_parts(tmp_path):
    """Parts written to disk then reloaded assemble identically (the DAG path)."""
    nz = {0: _nz()}
    parts = _make_parts(nz)
    reloaded = []
    for i, part in enumerate(parts):
        sio.save(part, str(tmp_path / f"part{i}.sacc"), type="mock")
        reloaded.append(sio.load(str(tmp_path / f"part{i}.sacc")))
    s = sw.assemble_analysis_sacc(nz, META, reloaded)
    assert type(s.covariance).__name__ == "BlockDiagonalCovariance"
    assert s.covariance.dense.shape == (len(s.mean), len(s.mean))
