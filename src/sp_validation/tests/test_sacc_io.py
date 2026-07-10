"""Tests for :mod:`sp_validation.sacc_io`.

All synthetic, all fast: build in memory, round-trip through ``tmp_path``,
and assert arrays/tags/windows/covariances come back bitwise-identical and
correctly aligned. No cluster paths.
"""

import numpy as np
import pytest

from sp_validation import sacc_io as sio


# --------------------------------------------------------------------------- #
# Synthetic builders
# --------------------------------------------------------------------------- #
def _nz(seed, n=50):
    rng = np.random.default_rng(seed)
    z = np.linspace(0.01, 2.0, n)
    return z, rng.uniform(0.1, 1.0, n)


def _theta(nbins=6):
    return np.geomspace(1.0, 100.0, nbins)


def _spd(n, seed):
    """Symmetric positive-definite matrix of size ``n``."""
    a = np.random.default_rng(seed).normal(size=(n, n))
    return a @ a.T + n * np.eye(n)


def _base_sacc(nbins=1):
    """A Sacc with ``nbins`` NZ source tracers plus the PSF tracer."""
    return sio.new_sacc({i: _nz(i) for i in range(nbins)})


# --------------------------------------------------------------------------- #
# 1. Per-writer round-trip (arrays / tags / windows / NZ bitwise)
# --------------------------------------------------------------------------- #
def _roundtrip(s, tmp_path, name="rt"):
    path = tmp_path / f"{name}.sacc"
    sio.save(s, str(path))
    return sio.load(str(path))


def test_nz_roundtrip(tmp_path):
    z, nz = _nz(3)
    s = sio.new_sacc({0: (z, nz)}, metadata={"version": "v1.4.6.3"})
    s2 = _roundtrip(s, tmp_path, "nz")
    z2, nz2 = sio.get_nz(s2, 0)
    assert np.array_equal(z2, z)
    assert np.array_equal(nz2, nz)
    assert s2.metadata["version"] == "v1.4.6.3"
    assert sio.PSF_TRACER in s2.tracers


def test_xi_roundtrip(tmp_path):
    theta = _theta()
    xip, xim = np.arange(6) * 1e-5, np.arange(6) * 2e-5
    npairs, weight = np.arange(6) * 1e3, np.arange(6) * 1.5
    s = _base_sacc()
    sio.add_xi(
        s,
        (0, 0),
        theta,
        xip,
        xim,
        grid="coarse",
        theta_nom=theta * 1.01,
        npairs=npairs,
        weight=weight,
    )
    s2 = _roundtrip(s, tmp_path, "xi")
    th, p, m = sio.get_xi(s2, (0, 0), grid="coarse")
    assert np.array_equal(th, theta)
    assert np.array_equal(p, xip)
    assert np.array_equal(m, xim)
    # extra tags survive
    idx = s2.indices(sio.XI_PLUS, ("source_0", "source_0"), grid="coarse")
    tags = s2.data[idx[0]].tags
    assert tags["grid"] == "coarse"
    assert set(tags) >= {"theta", "theta_nom", "npairs", "weight", "grid"}


def test_pseudo_cl_roundtrip(tmp_path):
    ell_eff = np.array([30.0, 120.0, 210.0, 300.0])
    nell, nbp = 50, len(ell_eff)
    window_ells = np.arange(2, 2 + nell).astype(float)
    W = np.random.default_rng(5).uniform(size=(nell, nbp))
    ee, bb, eb = np.arange(nbp) * 1e-9, np.arange(nbp) * 2e-9, np.arange(nbp) * 3e-9
    s = _base_sacc()
    sio.add_pseudo_cl(
        s,
        (0, 0),
        ell_eff,
        ee,
        bb,
        eb,
        window_ells=window_ells,
        window_weights=W,
    )
    s2 = _roundtrip(s, tmp_path, "cl")
    ell, cl_ee, cl_bb, cl_eb, window = sio.get_pseudo_cl(s2, (0, 0))
    assert np.array_equal(ell, ell_eff)
    assert np.array_equal(cl_ee, ee)
    assert np.array_equal(cl_bb, bb)
    assert np.array_equal(cl_eb, eb)
    assert np.array_equal(window.weight, W)
    assert np.array_equal(window.values, window_ells)


def test_cosebis_roundtrip(tmp_path):
    En, Bn = np.arange(1, 11) * 1e-6, np.arange(1, 11) * 1e-7
    s = _base_sacc()
    sio.add_cosebis(s, (0, 0), En, Bn, (1.0, 100.0))
    s2 = _roundtrip(s, tmp_path, "cosebi")
    n, E, B = sio.get_cosebis(s2, (0, 0))
    assert np.array_equal(n, np.arange(1, 11))
    assert np.array_equal(E, En)
    assert np.array_equal(B, Bn)
    idx = s2.indices(sio.COSEBI_EE, ("source_0", "source_0"))
    assert s2.data[idx[0]].tags["theta_min"] == 1.0
    assert s2.data[idx[0]].tags["theta_max"] == 100.0


def test_pure_eb_roundtrip(tmp_path):
    theta = _theta()
    arrays = {key: np.arange(6) * (i + 1) * 1e-6 for i, key in enumerate(sio.PURE_KEYS)}
    s = _base_sacc()
    sio.add_pure_eb(s, (0, 0), theta, **arrays)
    s2 = _roundtrip(s, tmp_path, "pureeb")
    th, back = sio.get_pure_eb(s2, (0, 0))
    assert np.array_equal(th, theta)
    for key in sio.PURE_KEYS:
        assert np.array_equal(back[key], arrays[key])


def test_rho_roundtrip(tmp_path):
    theta = _theta()
    s = _base_sacc()
    for k in range(6):
        sio.add_rho(
            s, k, theta, np.arange(6) * (k + 1) * 1e-6, np.arange(6) * (k + 1) * 2e-6
        )
    s2 = _roundtrip(s, tmp_path, "rho")
    for k in range(6):
        th, p, m = sio.get_rho(s2, k)
        assert np.array_equal(th, theta)
        assert np.array_equal(p, np.arange(6) * (k + 1) * 1e-6)
        assert np.array_equal(m, np.arange(6) * (k + 1) * 2e-6)


def test_tau_roundtrip(tmp_path):
    theta = _theta()
    s = _base_sacc()
    for k in (0, 2, 5):
        sio.add_tau(
            s,
            (0, 0),
            k,
            theta,
            np.arange(6) * (k + 1) * 1e-6,
            np.arange(6) * (k + 1) * 2e-6,
        )
    s2 = _roundtrip(s, tmp_path, "tau")
    for k in (0, 2, 5):
        th, p, m = sio.get_tau(s2, (0, 0), k)
        assert np.array_equal(th, theta)
        assert np.array_equal(p, np.arange(6) * (k + 1) * 1e-6)
        assert np.array_equal(m, np.arange(6) * (k + 1) * 2e-6)
        assert s2.data[s2.indices(sio.TAU_PLUS.format(k=k))[0]].tracers == (
            "source_0",
            sio.PSF_TRACER,
        )


# --------------------------------------------------------------------------- #
# 2. Covariance alignment
# --------------------------------------------------------------------------- #
def _multi_statistic_sacc():
    """Sacc with ξ+/ξ−, Cℓ (ee/bb/eb) and COSEBIs, ready for a covariance."""
    theta = _theta()
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    ell = np.array([30.0, 120.0, 210.0])
    W = np.random.default_rng(0).uniform(size=(20, 3))
    sio.add_pseudo_cl(
        s,
        (0, 0),
        ell,
        np.arange(3) * 1e-9,
        np.arange(3) * 2e-9,
        np.arange(3) * 3e-9,
        window_ells=np.arange(2, 22).astype(float),
        window_weights=W,
    )
    sio.add_cosebis(
        s, (0, 0), np.arange(1, 6) * 1e-6, np.arange(1, 6) * 1e-7, (1.0, 100.0)
    )
    return s


def test_assemble_covariance_alignment(tmp_path):
    s = _multi_statistic_sacc()
    tr = ("source_0", "source_0")
    # Block selectors in canonical (insertion) order.
    xi = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    cl = np.concatenate(
        [s.indices(sio.CL_EE, tr), s.indices(sio.CL_BB, tr), s.indices(sio.CL_EB, tr)]
    )
    co = np.concatenate([s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)])
    cov_xi, cov_cl, cov_co = _spd(len(xi), 1), _spd(len(cl), 2), _spd(len(co), 3)
    sio.assemble_covariance(s, [(xi, cov_xi), (cl, cov_cl), (co, cov_co)])
    s2 = _roundtrip(s, tmp_path, "cov")
    assert type(s2.covariance).__name__ == "FullCovariance"
    dense = s2.covariance.dense
    # each block's sub-covariance is exactly what went in
    assert np.array_equal(dense[np.ix_(xi, xi)], cov_xi)
    assert np.array_equal(dense[np.ix_(cl, cl)], cov_cl)
    assert np.array_equal(dense[np.ix_(co, co)], cov_co)
    # zero cross-blocks
    assert np.array_equal(dense[np.ix_(xi, cl)], np.zeros((len(xi), len(cl))))
    assert np.array_equal(dense[np.ix_(xi, co)], np.zeros((len(xi), len(co))))
    assert np.array_equal(dense[np.ix_(cl, co)], np.zeros((len(cl), len(co))))


def test_assemble_covariance_selector_tuples():
    """Blocks addressed by (data_type, tracers) tuples, not raw indices."""
    s = _multi_statistic_sacc()
    tr = ("source_0", "source_0")
    xi = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    cl = np.concatenate(
        [s.indices(sio.CL_EE, tr), s.indices(sio.CL_BB, tr), s.indices(sio.CL_EB, tr)]
    )
    sio.assemble_covariance(
        s,
        [
            (xi, _spd(len(xi), 1)),
            (cl, _spd(len(cl), 2)),
            ((sio.COSEBI_EE, tr), _spd(len(s.indices(sio.COSEBI_EE, tr)), 3)),
            ((sio.COSEBI_BB, tr), _spd(len(s.indices(sio.COSEBI_BB, tr)), 4)),
        ],
    )
    assert type(s.covariance).__name__ == "FullCovariance"
    assert s.covariance.dense.shape == (len(s.mean), len(s.mean))


# --------------------------------------------------------------------------- #
# 3. assemble_covariance failure modes
# --------------------------------------------------------------------------- #
def test_assemble_covariance_wrong_dimension():
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), _theta(), np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    idx = np.arange(len(s.mean))
    with pytest.raises(ValueError, match="span"):
        sio.assemble_covariance(s, [(idx, _spd(len(idx) - 1, 1))])


def test_assemble_covariance_non_contiguous():
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), _theta(), np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    idx = np.array([0, 2, 4, 6, 8, 10, 1, 3])  # not contiguous/ascending
    with pytest.raises(ValueError, match="non-contiguous"):
        sio.assemble_covariance(s, [(idx, _spd(len(idx), 1))])


def test_assemble_covariance_missing_coverage():
    s = _multi_statistic_sacc()
    tr = ("source_0", "source_0")
    xi = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    with pytest.raises(ValueError, match="tile"):
        sio.assemble_covariance(
            s, [(xi, _spd(len(xi), 1))]
        )  # leaves cl+cosebi uncovered


def test_assemble_covariance_non_square():
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), _theta(), np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    idx = np.arange(len(s.mean))
    with pytest.raises(ValueError, match="square"):
        sio.assemble_covariance(s, [(idx, np.ones((len(idx), len(idx) - 1)))])


def test_assemble_covariance_overlap():
    s = _multi_statistic_sacc()
    tr = ("source_0", "source_0")
    xi = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    # second block starts before the first ended -> gap/overlap error
    with pytest.raises(ValueError, match="tile|gap|overlap"):
        sio.assemble_covariance(s, [(xi, _spd(len(xi), 1)), (xi, _spd(len(xi), 2))])


# --------------------------------------------------------------------------- #
# 4. Fine file: DiagonalCovariance from 1-D variances
# --------------------------------------------------------------------------- #
def test_diagonal_covariance_roundtrip(tmp_path):
    theta = np.geomspace(0.1, 250.0, 40)
    n = len(theta)
    xip, xim = np.arange(n) * 1e-5, np.arange(n) * 2e-5
    varxip, varxim = np.arange(1, n + 1) * 1e-12, np.arange(1, n + 1) * 2e-12
    s = _base_sacc()
    sio.add_xi(s, (0, 0), theta, xip, xim, grid="fine")
    variances = np.concatenate([varxip, varxim])  # [xip; xim] order
    sio.add_diagonal_covariance(s, variances)
    assert type(s.covariance).__name__ == "DiagonalCovariance"
    s2 = _roundtrip(s, tmp_path, "fine")
    assert type(s2.covariance).__name__ == "DiagonalCovariance"
    assert np.array_equal(np.diag(s2.covariance.dense), variances)


# --------------------------------------------------------------------------- #
# 5. extract(): sub-covariance alignment; original untouched; tag filter
# --------------------------------------------------------------------------- #
def test_extract_subblock_and_original_untouched():
    s = _multi_statistic_sacc()
    tr = ("source_0", "source_0")
    xi = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
    cl = np.concatenate(
        [s.indices(sio.CL_EE, tr), s.indices(sio.CL_BB, tr), s.indices(sio.CL_EB, tr)]
    )
    co = np.concatenate([s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)])
    cov_co = _spd(len(co), 3)
    sio.assemble_covariance(
        s, [(xi, _spd(len(xi), 1)), (cl, _spd(len(cl), 2)), (co, cov_co)]
    )
    n_before = len(s.mean)
    sub = sio.extract(s, data_type=sio.COSEBI_EE, tracers=tr)
    # original untouched
    assert len(s.mean) == n_before
    # subset covariance equals the COSEBI-EE diagonal sub-block
    ee_local = np.arange(len(s.indices(sio.COSEBI_EE, tr)))
    assert np.array_equal(sub.covariance.dense, cov_co[np.ix_(ee_local, ee_local)])


def test_extract_tag_filter():
    theta = _theta()
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    sio.add_xi(s, (0, 0), theta, np.arange(6) * 3e-5, np.arange(6) * 4e-5, grid="fine")
    sub = sio.extract(
        s, data_type=sio.XI_PLUS, tracers=("source_0", "source_0"), grid="fine"
    )
    assert len(sub.mean) == len(theta)
    assert set(sub.get_tag("grid", sio.XI_PLUS)) == {"fine"}


# --------------------------------------------------------------------------- #
# 6. Tomographic case: >=2 bins, >=3 pairs, per-pair selection
# --------------------------------------------------------------------------- #
def test_tomographic_per_pair_selection(tmp_path):
    theta = _theta()
    s = _base_sacc(nbins=2)
    pairs = [(0, 0), (0, 1), (1, 1)]
    for k, (i, j) in enumerate(pairs):
        sio.add_xi(
            s,
            (i, j),
            theta,
            np.arange(6) * (k + 1) * 1e-5,
            np.arange(6) * (k + 1) * 2e-5,
            grid="coarse",
        )
    s2 = _roundtrip(s, tmp_path, "tomo")
    for k, (i, j) in enumerate(pairs):
        th, p, m = sio.get_xi(s2, (i, j), grid="coarse")
        assert np.array_equal(th, theta)
        assert np.array_equal(p, np.arange(6) * (k + 1) * 1e-5)
        assert np.array_equal(m, np.arange(6) * (k + 1) * 2e-5)
    # selecting one pair does not bleed into another
    assert len(s2.indices(sio.XI_PLUS, ("source_0", "source_1"))) == len(theta)
    assert len(s2.indices(sio.XI_PLUS, ("source_1", "source_1"))) == len(theta)


# --------------------------------------------------------------------------- #
# 7. Readers mirror writers on a mixed file
# --------------------------------------------------------------------------- #
def test_readers_on_mixed_file(tmp_path):
    theta = _theta()
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    ell = np.array([30.0, 120.0, 210.0])
    W = np.random.default_rng(0).uniform(size=(20, 3))
    sio.add_pseudo_cl(
        s,
        (0, 0),
        ell,
        np.arange(3) * 1e-9,
        np.arange(3) * 2e-9,
        np.arange(3) * 3e-9,
        window_ells=np.arange(2, 22).astype(float),
        window_weights=W,
    )
    sio.add_cosebis(
        s, (0, 0), np.arange(1, 6) * 1e-6, np.arange(1, 6) * 1e-7, (1.0, 100.0)
    )
    pure = {key: np.arange(6) * (i + 1) * 1e-6 for i, key in enumerate(sio.PURE_KEYS)}
    sio.add_pure_eb(s, (0, 0), theta, **pure)
    for k in range(6):
        sio.add_rho(
            s, k, theta, np.arange(6) * (k + 1) * 1e-7, np.arange(6) * (k + 1) * 2e-7
        )
    for k in (0, 2, 5):
        sio.add_tau(
            s,
            (0, 0),
            k,
            theta,
            np.arange(6) * (k + 1) * 1e-8,
            np.arange(6) * (k + 1) * 2e-8,
        )
    s2 = _roundtrip(s, tmp_path, "mixed")

    _, p, m = sio.get_xi(s2, (0, 0), grid="coarse")
    assert np.array_equal(p, np.arange(6) * 1e-5) and np.array_equal(
        m, np.arange(6) * 2e-5
    )
    ell_r, ee, bb, eb, win = sio.get_pseudo_cl(s2, (0, 0))
    assert np.array_equal(ell_r, ell) and np.array_equal(win.weight, W)
    n, E, B = sio.get_cosebis(s2, (0, 0))
    assert np.array_equal(E, np.arange(1, 6) * 1e-6)
    _, back = sio.get_pure_eb(s2, (0, 0))
    for key in sio.PURE_KEYS:
        assert np.array_equal(back[key], pure[key])
    for k in range(6):
        _, rp, rm = sio.get_rho(s2, k)
        assert np.array_equal(rp, np.arange(6) * (k + 1) * 1e-7)
    for k in (0, 2, 5):
        _, tp, tm = sio.get_tau(s2, (0, 0), k)
        assert np.array_equal(tp, np.arange(6) * (k + 1) * 1e-8)


# --------------------------------------------------------------------------- #
# 8. End-to-end one-file layout for a synthetic catalogue version
# --------------------------------------------------------------------------- #
def test_end_to_end_one_file_layout(tmp_path):
    version = "vSYNTH"
    theta_c = _theta(20)
    theta_f = np.geomspace(0.1, 250.0, 200)

    # one file: analysis products first, fine-grid integration input last
    s = _base_sacc()
    sio.add_xi(
        s, (0, 0), theta_c, np.arange(20) * 1e-5, np.arange(20) * 2e-5, grid="coarse"
    )
    sio.add_cosebis(
        s, (0, 0), np.arange(1, 11) * 1e-6, np.arange(1, 11) * 1e-7, (1.0, 100.0)
    )
    sio.add_xi(
        s, (0, 0), theta_f, np.arange(200) * 1e-5, np.arange(200) * 2e-5, grid="fine"
    )
    tr = ("source_0", "source_0")
    xi_c = np.concatenate(
        [
            s.indices(sio.XI_PLUS, tr, grid="coarse"),
            s.indices(sio.XI_MINUS, tr, grid="coarse"),
        ]
    )
    co = np.concatenate([s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)])
    xi_f = np.concatenate(
        [
            s.indices(sio.XI_PLUS, tr, grid="fine"),
            s.indices(sio.XI_MINUS, tr, grid="fine"),
        ]
    )
    # dense fine block (CosmoCov integration covariance in production)
    fine_block = _spd(len(xi_f), 3)
    sio.assemble_covariance(
        s,
        [(xi_c, _spd(len(xi_c), 1)), (co, _spd(len(co), 2)), (xi_f, fine_block)],
    )
    sio.save(s, str(tmp_path / f"{version}.sacc"))

    a = sio.load(str(tmp_path / f"{version}.sacc"))

    th_c, p_c, _ = sio.get_xi(a, (0, 0), grid="coarse")
    assert np.array_equal(th_c, theta_c) and np.array_equal(p_c, np.arange(20) * 1e-5)
    n, E, B = sio.get_cosebis(a, (0, 0))
    assert np.array_equal(n, np.arange(1, 11))
    assert a.covariance.dense.shape == (len(a.mean), len(a.mean))

    th_f, p_f, _ = sio.get_xi(a, (0, 0), grid="fine")
    assert np.array_equal(th_f, theta_f) and np.array_equal(p_f, np.arange(200) * 1e-5)

    # extract() of the fine selection pulls the aligned dense sub-covariance
    fine = sio.extract(a, sio.XI_PLUS, tr, grid="fine")
    idx_p = a.indices(sio.XI_PLUS, tr, grid="fine")
    assert np.allclose(fine.covariance.dense, a.covariance.dense[np.ix_(idx_p, idx_p)])
    # zero cross-blocks between analysis and fine points
    assert np.all(a.covariance.dense[np.ix_(xi_c, xi_f)] == 0)


def test_one_file_layout_diagonal_fine_fallback(tmp_path):
    """No CosmoCov covariance: the fine block is np.diag(varxip/varxim)."""
    s = _base_sacc()
    theta_f = np.geomspace(0.1, 250.0, 50)
    sio.add_xi(
        s, (0, 0), _theta(6), np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
    )
    sio.add_xi(
        s, (0, 0), theta_f, np.arange(50) * 1e-5, np.arange(50) * 2e-5, grid="fine"
    )
    tr = ("source_0", "source_0")
    xi_c = np.concatenate(
        [
            s.indices(sio.XI_PLUS, tr, grid="coarse"),
            s.indices(sio.XI_MINUS, tr, grid="coarse"),
        ]
    )
    xi_f = np.concatenate(
        [
            s.indices(sio.XI_PLUS, tr, grid="fine"),
            s.indices(sio.XI_MINUS, tr, grid="fine"),
        ]
    )
    variances = np.concatenate([np.arange(1, 51) * 1e-12, np.arange(1, 51) * 2e-12])
    sio.assemble_covariance(s, [(xi_c, _spd(len(xi_c), 1)), (xi_f, np.diag(variances))])
    sio.save(s, str(tmp_path / "vDIAG.sacc"))
    a = sio.load(str(tmp_path / "vDIAG.sacc"))
    assert np.array_equal(np.diag(a.covariance.dense[np.ix_(xi_f, xi_f)]), variances)


# --------------------------------------------------------------------------- #
# 9. Ascending-grid enforcement (writers reject out-of-order grids)
# --------------------------------------------------------------------------- #
def test_add_xi_rejects_non_ascending_theta():
    s = _base_sacc()
    theta = _theta()[::-1]  # descending
    with pytest.raises(ValueError, match="theta must be strictly ascending"):
        sio.add_xi(
            s, (0, 0), theta, np.arange(6) * 1e-5, np.arange(6) * 2e-5, grid="coarse"
        )


def test_add_pseudo_cl_rejects_non_ascending_ell():
    s = _base_sacc()
    ell = np.array([210.0, 30.0, 120.0])  # not ascending
    W = np.random.default_rng(0).uniform(size=(20, 3))
    with pytest.raises(ValueError, match="ell_eff must be strictly ascending"):
        sio.add_pseudo_cl(
            s,
            (0, 0),
            ell,
            np.arange(3) * 1e-9,
            np.arange(3) * 2e-9,
            np.arange(3) * 3e-9,
            window_ells=np.arange(2, 22).astype(float),
            window_weights=W,
        )


def test_add_pure_eb_rho_tau_reject_non_ascending_theta():
    s = _base_sacc()
    theta = _theta()[::-1]
    pure = {key: np.arange(6) * 1e-6 for key in sio.PURE_KEYS}
    with pytest.raises(ValueError, match="theta must be strictly ascending"):
        sio.add_pure_eb(s, (0, 0), theta, **pure)
    with pytest.raises(ValueError, match="theta must be strictly ascending"):
        sio.add_rho(s, 0, theta, np.arange(6) * 1e-6, np.arange(6) * 2e-6)
    with pytest.raises(ValueError, match="theta must be strictly ascending"):
        sio.add_tau(s, (0, 0), 0, theta, np.arange(6) * 1e-6, np.arange(6) * 2e-6)


# --------------------------------------------------------------------------- #
# 10. Bin-pair normalisation: (1, 0) addresses the same points as (0, 1)
# --------------------------------------------------------------------------- #
def test_bin_pair_normalisation():
    theta = _theta()
    xip, xim = np.arange(6) * 1e-5, np.arange(6) * 2e-5
    s = _base_sacc(nbins=2)
    sio.add_xi(s, (0, 1), theta, xip, xim, grid="coarse")
    th01, p01, m01 = sio.get_xi(s, (0, 1), grid="coarse")
    th10, p10, m10 = sio.get_xi(s, (1, 0), grid="coarse")  # reversed order
    assert np.array_equal(th01, th10)
    assert np.array_equal(p01, p10) and np.array_equal(p10, xip)
    assert np.array_equal(m01, m10) and np.array_equal(m10, xim)
    # writing under (1, 0) lands in the same tracer pair, not a new one
    s2 = _base_sacc(nbins=2)
    sio.add_xi(s2, (1, 0), theta, xip, xim, grid="coarse")
    assert len(s2.indices(sio.XI_PLUS, ("source_0", "source_1"))) == len(theta)


# --------------------------------------------------------------------------- #
# 11. Reader/covariance alignment holds for ANY insertion order (the core
#     regression the review caught): a tomographic multi-pair ξ covariance
#     assembled as ONE contiguous pair-major block, per-pair sub-blocks
#     recovered via extract().
# --------------------------------------------------------------------------- #
def test_tomographic_xi_covariance_one_contiguous_block():
    theta = _theta()
    nth = len(theta)
    pairs = [(0, 0), (0, 1), (1, 1)]
    s = _base_sacc(nbins=2)
    for k, (i, j) in enumerate(pairs):
        sio.add_xi(
            s,
            (i, j),
            theta,
            np.arange(nth) * (k + 1) * 1e-5,
            np.arange(nth) * (k + 1) * 2e-5,
            grid="coarse",
        )
    # All ξ points as one contiguous block in insertion (pair-major) order.
    xi_idx = np.arange(len(s.mean))
    assert np.array_equal(xi_idx, np.arange(3 * 2 * nth))  # 3 pairs x [xip; xim]
    cov = _spd(len(xi_idx), 11)  # dense, cross-pair correlations
    sio.assemble_covariance(s, [(xi_idx, cov)])

    # Per-pair xip sub-block: resolve indices, extract, compare to input.
    for i, j in pairs:
        idx_p = s.indices(
            sio.XI_PLUS, ("source_" + str(min(i, j)), "source_" + str(max(i, j)))
        )
        sub = sio.extract(
            s,
            data_type=sio.XI_PLUS,
            tracers=("source_" + str(min(i, j)), "source_" + str(max(i, j))),
        )
        assert np.array_equal(sub.covariance.dense, cov[np.ix_(idx_p, idx_p)])
        # readers stay covariance-aligned: get_xi returns in the same order
        th, xip, _ = sio.get_xi(s, (i, j), grid="coarse")
        assert np.array_equal(th, theta)
        assert np.array_equal(s.mean[idx_p], xip)


# --------------------------------------------------------------------------- #
# 12. get_pseudo_cl window/cl column correspondence: window column j maps to
#     the returned ell_eff[j] (verified through window_ind tags).
# --------------------------------------------------------------------------- #
def test_pseudo_cl_window_column_correspondence(tmp_path):
    ell_eff = np.array([30.0, 120.0, 210.0, 300.0])
    nell, nbp = 40, len(ell_eff)
    window_ells = np.arange(2, 2 + nell).astype(float)
    # Distinct columns so a permutation would be detectable.
    W = np.zeros((nell, nbp))
    for b in range(nbp):
        W[b * 5 : (b + 1) * 5, b] = 1.0
    ee, bb, eb = np.arange(nbp) * 1e-9, np.arange(nbp) * 2e-9, np.arange(nbp) * 3e-9
    s = _base_sacc()
    sio.add_pseudo_cl(
        s,
        (0, 0),
        ell_eff,
        ee,
        bb,
        eb,
        window_ells=window_ells,
        window_weights=W,
    )
    s2 = _roundtrip(s, tmp_path, "clwin")
    ell, cl_ee, _, _, window = sio.get_pseudo_cl(s2, (0, 0))
    # returned ell array is in insertion order
    assert np.array_equal(ell, ell_eff)
    assert np.array_equal(cl_ee, ee)
    # window_ind tag on each EE point indexes the matching window column
    idx = s2.indices(sio.CL_EE, ("source_0", "source_0"))
    for pos, i in enumerate(idx):
        col = s2.data[i].tags["window_ind"]
        assert col == pos  # insertion order preserved => column j <-> ell[j]
        assert np.array_equal(window.weight[:, col], W[:, pos])
