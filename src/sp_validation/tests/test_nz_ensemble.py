"""Known-answer checks for the SOM n(z) realisations."""

import numpy as np
import pytest

from sp_validation import nz_ensemble as ne

GRID = np.linspace(0, 5, 501)


def toy(seed=0, n=600, n_groups=40):
    """Two bins, disjoint calibrators, groups shared across bins."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.1, 1.5, n)
    group = rng.integers(0, n_groups, n)
    in_bin = (z > 0.6).astype(int)
    source = rng.uniform(1, 2, (2, n_groups))
    count = np.stack(
        [np.bincount(group[in_bin == k], minlength=n_groups) for k in range(2)]
    )
    weights = np.zeros((2, n))
    for k in range(2):
        m = in_bin == k
        weights[k, m] = source[k, group[m]] / count[k, group[m]]
    return z, weights, group, source


def test_unit_factors_reproduce_the_calibration():
    z, w, g, source = toy()
    np.testing.assert_allclose(ne.reweight(w, g, np.ones(len(z))), w)
    np.testing.assert_allclose(ne.source_weights(w, g, source.shape[1]), source)


def test_factors_keep_each_group_weight():
    z, w, g, source = toy()
    q = np.random.default_rng(1).exponential(size=len(z))
    np.testing.assert_allclose(ne.source_weights(ne.reweight(w, g, q), g, 40), source)


def test_density_cancels_in_redshift_pure_groups():
    """A group of calibrators at one redshift cell absorbs any δ(z)."""
    rng = np.random.default_rng(2)
    z = np.repeat(np.round(rng.uniform(0.2, 1.4, 30), 2), 10)
    g = np.repeat(np.arange(30), 10)
    w = np.stack([np.where(z < 0.8, 1.0, 0.0), np.where(z >= 0.8, 1.0, 0.0)])
    cov = np.diag(np.full(len(GRID), 0.3))
    real = ne.realisations(z, w, g, GRID, 5, density=cov, seed=3)
    np.testing.assert_allclose(
        real,
        ne.realisations(z, w, g, GRID, 5, density=None, seed=3),
        atol=1e-12,
    )
    assert np.allclose(np.trapezoid(real, GRID, axis=-1), 1)


def test_bootstrap_bins_are_independent():
    z, w, g, _ = toy(n=4000)
    real = ne.realisations(z, w, g, GRID, 400, seed=4)
    mean_z = np.trapezoid(real * GRID, GRID, axis=-1)
    assert abs(np.corrcoef(mean_z.T)[0, 1]) < 0.15


def test_lognormal_moments():
    cov = np.array([[0.2, 0.05], [0.05, 0.1]])
    d = ne.lognormal(cov, 200_000, np.random.default_rng(5))
    np.testing.assert_allclose(d.mean(0), 1, atol=5e-3)
    np.testing.assert_allclose(np.cov(d.T), cov, rtol=0.05)


@pytest.mark.slow
def test_density_variance_matches_box_integral():
    """One cell's variance against a direct 3D integral over its box window."""
    ccl = pytest.importorskip("pyccl")
    cosmo = ccl.CosmologyVanillaLCDM()
    grid = np.linspace(0, 1, 101)
    field = (2.0, 0.25)
    cov = ne.density_covariance(grid, field, 1.0, cosmo)
    i, zz = 50, grid[50]
    chi = lambda x: ccl.comoving_radial_distance(cosmo, 1 / (1 + x))  # noqa: E731
    side = [np.deg2rad(f) * chi(zz) for f in field] + [
        chi(zz + 0.005) - chi(zz - 0.005)
    ]
    u = np.r_[np.linspace(1e-3, 8, 800), np.geomspace(8.01, 3000, 800)]
    du = np.gradient(u)
    k = [2 * u / s for s in side]
    w = [np.sinc(u / np.pi) ** 2 * du * 2 / s for s in side]
    kg = np.geomspace(1e-5, 1e4, 4000)
    lp = np.log(ccl.nonlin_matter_power(cosmo, kg, 1 / (1 + zz)))
    total = 0.0
    for kx, wx in zip(k[0], w[0]):
        kk = np.sqrt(kx**2 + k[1][:, None] ** 2 + k[2][None, :] ** 2)
        pk = np.exp(np.interp(np.log(kk), np.log(kg), lp))
        total += wx * np.einsum("i,j,ij->", w[1], w[2], pk)
    assert cov[i, i] == pytest.approx(8 * total / (2 * np.pi) ** 3, rel=0.03)
