"""Tests for :mod:`sp_validation.blinding_likelihood`.

The likelihood is the blinding box's theory engine (PRD #241 §5). These tests
prove it (1) builds from a :mod:`sp_validation.sacc_io`-written analysis file,
extracting the coarse ξ± subset past COSEBIs/ρ/τ; (2) returns a theory vector
of the right shape, aligned to SACC insertion order via ``sacc_indices``
regardless of statistic order; (3) is deterministic; (4) is generic over N
tomographic bins. All synthetic — deterministic Gaussian n(z), no real data.

The theory-computing tests import CCL/CAMB and are marked ``slow`` (each CCL
cosmology build + ξ± predict runs CAMB's HMCode). The construction/alignment
tests are fast and run in the default suite.
"""

import numpy as np
import pytest

from sp_validation import blinding_likelihood as bl
from sp_validation import sacc_io as sio


# --------------------------------------------------------------------------- #
# Synthetic fixtures — deterministic Gaussian n(z), sacc_io-written files
# --------------------------------------------------------------------------- #
def _gauss_nz(z0, sigma, n=300):
    """Normalised Gaussian n(z) on a fixed grid (deterministic)."""
    z = np.linspace(0.0, 3.0, n)
    nz = np.exp(-0.5 * ((z - z0) / sigma) ** 2)
    return z, nz / np.trapezoid(nz, z)


def _theta(n=10):
    return np.geomspace(5.0, 250.0, n)


def _analysis_sacc(nbins=1, with_extras=True, ntheta=10):
    """A sacc_io analysis file: coarse ξ± for every i≤j pair, plus (optionally)
    COSEBIs/ρ so the extractor is exercised against a realistic mixed file.

    A FullCovariance is assembled block-diagonally (ξ block, then any extras),
    so the file is a valid firecrown ``ConstGaussian`` input.
    """
    theta = _theta(ntheta)
    nz = {i: _gauss_nz(0.5 + 0.3 * i, 0.15 + 0.03 * i) for i in range(nbins)}
    s = sio.new_sacc(nz)
    pairs = [(i, j) for i in range(nbins) for j in range(i, nbins)]
    for k, (i, j) in enumerate(pairs):
        xip = (np.arange(ntheta) + 1) * (k + 1) * 1e-5
        xim = (np.arange(ntheta) + 1) * (k + 1) * 0.5e-5
        sio.add_xi(s, (i, j), theta, xip, xim, grid="coarse")

    blocks = []
    cursor_pairs = pairs
    for i, j in cursor_pairs:
        tr = (sio.source_name(i), sio.source_name(j))
        idx = np.concatenate([s.indices(sio.XI_PLUS, tr), s.indices(sio.XI_MINUS, tr)])
        blocks.append((idx, np.eye(len(idx)) * 1e-12))

    if with_extras:
        sio.add_cosebis(
            s, (0, 0), np.arange(1, 5) * 1e-6, np.arange(1, 5) * 1e-7, (1.0, 100.0)
        )
        for kk in range(3):
            sio.add_rho(
                s, kk, theta, np.arange(ntheta) * 1e-7, np.arange(ntheta) * 2e-7
            )
        tr = ("source_0", "source_0")
        co = np.concatenate(
            [s.indices(sio.COSEBI_EE, tr), s.indices(sio.COSEBI_BB, tr)]
        )
        blocks.append((co, np.eye(len(co)) * 1e-14))
        for kk in range(3):
            ridx = np.concatenate(
                [
                    s.indices(sio.RHO_PLUS.format(k=kk)),
                    s.indices(sio.RHO_MINUS.format(k=kk)),
                ]
            )
            blocks.append((ridx, np.eye(len(ridx)) * 1e-16))

    sio.assemble_covariance(s, blocks)
    return s


# --------------------------------------------------------------------------- #
# 1. TheoryConfig: the single config surface
# --------------------------------------------------------------------------- #
def test_theory_config_defaults_match_cosmosis_fiducial():
    """Defaults reproduce the CosmoSIS v1.4.6.3 `_A_cell` fiducial values."""
    c = bl.TheoryConfig()
    assert c.S8 == 0.80
    assert c.n_s == 0.96
    assert c.h == 0.70
    assert c.m_nu == 0.06
    assert c.halofit_version == "mead2020_feedback"
    assert c.mass_split == "normal"
    # σ8 = S8 / √(Ωm/0.3); at the fiducial Ωm this is ~0.80.
    assert c.sigma8() == pytest.approx(c.S8 / np.sqrt(c.Omega_m / 0.3))
    # Ωc = Ωm − Ωb − Ων, all positive and summing back to Ωm.
    omega_nu = c.m_nu / (93.14 * c.h**2)
    assert c.omega_c() == pytest.approx(c.Omega_m - c.Omega_b - omega_nu)
    assert 0 < c.omega_c() < c.Omega_m


def test_theory_config_overrides_fail_loud():
    """Unknown override keys raise (fail-fast config)."""
    cfg = bl.TheoryConfig.from_overrides({"S8": 0.75, "Omega_m": 0.32})
    assert cfg.S8 == 0.75 and cfg.Omega_m == 0.32
    with pytest.raises(ValueError, match="unknown TheoryConfig fields"):
        bl.TheoryConfig.from_overrides({"S8": 0.75, "sig8": 0.8})


def test_theory_config_is_frozen():
    """The config is immutable — one authoritative fiducial per build."""
    c = bl.TheoryConfig()
    with pytest.raises(Exception):
        c.S8 = 0.9  # frozen dataclass -> FrozenInstanceError


# --------------------------------------------------------------------------- #
# 2. SACC preparation: bin/pair discovery + coarse-ξ extraction
# --------------------------------------------------------------------------- #
def test_source_bins_and_pairs_discovered_from_file():
    s = _analysis_sacc(nbins=3, with_extras=False)
    assert bl.source_bins(s) == [0, 1, 2]
    assert bl.xi_pairs(s, grid="coarse") == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 1),
        (1, 2),
        (2, 2),
    ]


def test_extract_coarse_xi_drops_everything_else():
    """Extraction keeps only coarse ξ±; COSEBIs/ρ and any fine grid are gone.

    Built without a covariance so the fine grid can be appended (SACC forbids
    adding points after ``add_covariance``); extraction logic is
    covariance-independent — the covariance-slicing path is covered by
    :func:`test_extract_coarse_xi_covariance_sliced`.
    """
    theta = _theta(10)
    s = sio.new_sacc({i: _gauss_nz(0.5 + 0.3 * i, 0.15) for i in range(2)})
    for k, (i, j) in enumerate([(0, 0), (0, 1), (1, 1)]):
        sio.add_xi(
            s,
            (i, j),
            theta,
            (np.arange(10) + 1) * (k + 1) * 1e-5,
            (np.arange(10) + 1) * (k + 1) * 0.5e-5,
            grid="coarse",
        )
    sio.add_cosebis(
        s, (0, 0), np.arange(1, 5) * 1e-6, np.arange(1, 5) * 1e-7, (1.0, 100.0)
    )
    for kk in range(3):
        sio.add_rho(s, kk, theta, np.arange(10) * 1e-7, np.arange(10) * 2e-7)
    # a fine grid for (0,0) to confirm it is dropped
    tf = np.geomspace(0.1, 250.0, 40)
    sio.add_xi(s, (0, 0), tf, np.arange(40) * 1e-5, np.arange(40) * 2e-5, grid="fine")

    coarse = bl.extract_coarse_xi(s)
    dtypes = {d.data_type for d in coarse.data}
    grids = {d.tags.get("grid") for d in coarse.data}
    assert dtypes == {sio.XI_PLUS, sio.XI_MINUS}
    assert grids == {"coarse"}
    # 3 coarse pairs x 2 types x 10 theta
    assert len(coarse.mean) == 3 * 2 * 10


def test_extract_coarse_xi_covariance_sliced():
    """Extraction carries an aligned covariance sub-block for the ξ± rows."""
    s = _analysis_sacc(nbins=2, with_extras=True)
    coarse = bl.extract_coarse_xi(s)
    n = len(coarse.mean)
    assert coarse.covariance.dense.shape == (n, n)
    # the ξ blocks in the source file were identity*1e-12; the extracted
    # sub-block is exactly that (diagonal, 1e-12), COSEBI/ρ rows excluded
    assert np.allclose(np.diag(coarse.covariance.dense), 1e-12)


def test_extract_preserves_original():
    s = _analysis_sacc(nbins=1, with_extras=True)
    n_before = len(s.mean)
    _ = bl.extract_coarse_xi(s)
    assert len(s.mean) == n_before


# --------------------------------------------------------------------------- #
# 3. build_likelihood: construction from a sacc_io file
# --------------------------------------------------------------------------- #
def _named(sacc_data, **kw):
    from firecrown.likelihood import NamedParameters

    return NamedParameters({"sacc_data": sacc_data, **kw})


@pytest.mark.slow
def test_build_likelihood_single_bin():
    s = _analysis_sacc(nbins=1, with_extras=True)
    lik, tools = bl.build_likelihood(_named(s))
    # one (ξ+, ξ−) TwoPoint per pair -> 2 statistics for the single-bin case
    assert len(lik.statistics) == 2


@pytest.mark.slow
def test_build_likelihood_requires_coarse_xi():
    """A file with no coarse ξ± is a loud error, not a silent empty likelihood."""
    s = sio.new_sacc({0: _gauss_nz(0.7, 0.2)})
    sio.add_cosebis(
        s, (0, 0), np.arange(1, 4) * 1e-6, np.arange(1, 4) * 1e-7, (1.0, 100.0)
    )
    with pytest.raises(ValueError, match="no coarse"):
        bl.build_likelihood(_named(s))


# --------------------------------------------------------------------------- #
# 4. Theory vector: shape, finiteness, determinism, SACC alignment
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_theory_vector_shape_and_finite():
    s = _analysis_sacc(nbins=1, with_extras=True)
    lik, tools = bl.build_likelihood(_named(s))
    tv = bl.compute_theory_vector(lik, tools)
    assert tv.shape == (2 * 10,)  # ξ+ and ξ− on 10 θ
    assert np.all(np.isfinite(tv))
    assert np.any(tv != 0.0)
    # cosmic-shear ξ+ is positive and falls with θ (monotone over this range)
    xip = tv[:10]
    assert np.all(xip > 0)
    assert xip[0] > xip[-1]


@pytest.mark.slow
def test_theory_vector_deterministic():
    """Same inputs -> bitwise-identical theory vector."""
    s = _analysis_sacc(nbins=1, with_extras=True)
    lik, tools = bl.build_likelihood(_named(s))
    tv1 = bl.compute_theory_vector(lik, tools)
    tv2 = bl.compute_theory_vector(lik, tools)
    assert np.array_equal(tv1, tv2)


@pytest.mark.slow
def test_theory_vector_realigns_to_sacc_order_regardless_of_statistic_order():
    """The `sacc_indices` scatter recovers SACC alignment for any statistic order.

    Two likelihoods over the same file — one with the default (pair-major)
    statistic order, one deliberately reversed — must, after scattering by
    ``sacc_indices``, produce the SAME SACC-aligned theory vector. This is the
    property that lets Smokescreen and the covariance stay aligned without the
    likelihood pinning a particular statistic order.
    """
    from firecrown.likelihood import ConstGaussian, TwoPoint
    from firecrown.likelihood import weak_lensing as wl

    s = _analysis_sacc(nbins=2, with_extras=False)
    coarse = bl.extract_coarse_xi(s)
    n = len(coarse.mean)

    lik_default, tools = bl.build_likelihood(_named(s))
    tv_default = bl.compute_theory_vector(lik_default, tools)
    aligned_default = bl.theory_vector_sacc_order(lik_default, tv_default, n)

    # Rebuild with statistics in reversed order.
    pairs = bl.xi_pairs(coarse, grid="coarse")
    bins = sorted({b for p in pairs for b in p})
    sources = {
        i: wl.WeakLensing(
            sacc_tracer=sio.source_name(i),
            systematics=[wl.LinearAlignmentSystematic(sacc_tracer=None)],
        )
        for i in bins
    }
    stats = []
    for i, j in pairs:
        stats.append(TwoPoint(sio.XI_PLUS, sources[i], sources[j]))
        stats.append(TwoPoint(sio.XI_MINUS, sources[i], sources[j]))
    lik_rev = ConstGaussian(statistics=list(reversed(stats)))
    lik_rev.read(coarse)
    _, tools_rev = bl.build_likelihood(_named(s))
    tv_rev = bl.compute_theory_vector(lik_rev, tools_rev)
    aligned_rev = bl.theory_vector_sacc_order(lik_rev, tv_rev, n)

    assert np.all(np.isfinite(aligned_default))
    assert np.array_equal(aligned_default, aligned_rev)


# --------------------------------------------------------------------------- #
# 5. N-bin generality: a 2-bin file yields 3 pairs / 6 statistics
# --------------------------------------------------------------------------- #
@pytest.mark.slow
def test_two_bin_generality():
    s = _analysis_sacc(nbins=2, with_extras=False)
    lik, tools = bl.build_likelihood(_named(s))
    # pairs (0,0),(0,1),(1,1) -> 3 pairs x {ξ+, ξ−} = 6 statistics
    assert len(lik.statistics) == 6
    tv = bl.compute_theory_vector(lik, tools)
    assert tv.shape == (6 * 10,)
    assert np.all(np.isfinite(tv)) and np.any(tv != 0.0)
    # realignment covers exactly the coarse ξ rows
    coarse = bl.extract_coarse_xi(s)
    aligned = bl.theory_vector_sacc_order(lik, tv, len(coarse.mean))
    assert np.all(np.isfinite(aligned))
