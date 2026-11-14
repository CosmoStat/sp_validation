"""Known-answer checks for the SOM n(z) realisations."""

from dataclasses import replace

import numpy as np
import pytest

from sp_validation import nz_ensemble as ne
from sp_validation import nz_pipeline


def toy_run(tmp_path, n_cal=1500, n_src=30_000, seed=0):
    """A small calibration run through :func:`nz_pipeline.run`."""
    rng = np.random.default_rng(seed)

    def galaxies(n):
        z = rng.uniform(0.15, 1.15, n)
        r = rng.uniform(20, 24, n)
        colour = np.column_stack([1.5 * z, z, 0.6 * z, 0.4 * z])
        colour += rng.normal(0, 0.05, colour.shape)
        g = r + colour[:, 1]
        u = g + colour[:, 0]
        i = r - colour[:, 2]
        zz = i - colour[:, 3]
        return z, np.column_stack([u, g, r, i, zz])

    cz, cmag = galaxies(n_cal)
    sz, smag = galaxies(n_src)
    smag = np.column_stack([smag, smag[:, 4] + 0.1])  # z2 = z + 0.1
    smag[: n_src // 3, 4] = np.nan  # a third measured through z2 only
    np.savez(
        tmp_path / "cal.npz",
        z=cz,
        photoz=cz + rng.normal(0, 0.03, n_cal),
        mag=cmag,
    )
    np.savez(
        tmp_path / "src.npz",
        mag=smag,
        photoz=sz + rng.normal(0, 0.03, n_src),
        weight=rng.uniform(0.5, 1.5, n_src),
        tile=np.zeros(n_src),
        number=np.arange(n_src),
    )
    config = dict(
        name="toy",
        limitation="toy",
        inputs=dict(
            calibrators=dict(output=str(tmp_path / "cal.npz")),
            sources=dict(output=str(tmp_path / "src.npz")),
        ),
        som=dict(side=8, epochs=10, seed=1),
        groups=30,
        edges=[0.1, 0.37, 0.55, 0.76, 1.2],
        gold_qc=dict(floor=0.05, nmad_factor=5),
        z2_offset=-0.1,
        grid=dict(min=0, max=5, step=0.05),
        ensemble=dict(draws=3, seed=0),
    )
    out = tmp_path / "run"
    nz_pipeline.run(config, out)
    return ne.Calibration.load(out), config


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    return toy_run(tmp_path_factory.mktemp("nz"))


def draw(cal, n=40, **terms):
    return ne.realisations(cal, dict(draws=n, seed=1, **terms))


def mean_z(cal, h):
    return np.trapezoid(h * cal.grid, cal.grid, axis=-1)


def test_no_terms_reproduce_the_calibration(run):
    cal, _ = run
    w = ne.calibrator_weights(
        cal.group, cal.member, np.ones(len(cal.z)), cal.source * cal.gold
    )
    np.testing.assert_allclose(w, cal.weights, rtol=1e-12)
    gold = ne.gold_masks(
        cal.z, cal.photoz, cal.group, cal.member, cal.source, cal.source_photoz, 0.05, 5
    )
    np.testing.assert_array_equal(gold, cal.gold)
    _, raw = draw(cal, n=2)
    np.testing.assert_allclose(raw, cal.fiducial(), atol=1e-12)


def test_factors_keep_each_group_weight(run):
    cal, _ = run
    q = np.random.default_rng(1).exponential(size=len(cal.z))
    w = ne.calibrator_weights(cal.group, cal.member, q, cal.source * cal.gold)
    for k in range(len(w)):
        kept = np.bincount(cal.group, weights=w[k], minlength=cal.source.shape[1])
        np.testing.assert_allclose(
            kept, np.where(kept > 0, cal.source[k], 0), rtol=1e-10
        )


def test_offsets_move_catalogue_weight_between_groups_only(run):
    """Every direction keeps each bin's catalogue weight; the z2 one moves some."""
    cal, _ = run
    np.testing.assert_allclose(cal.jacobian.sum(-1), 0, atol=1e-9 * cal.source.sum())
    assert np.abs(cal.jacobian[ne.DIRECTIONS.index("z2")]).sum() > 0


def test_photoz_gradient_recovers_a_plane():
    rng = np.random.default_rng(3)
    mag = rng.uniform(20, 24, (500, 5))
    slope = np.array([0.1, -0.2, 0.05, 0.3, -0.1])
    model = dict(center=np.zeros(11), scale=np.ones(11))
    grad = ne.photoz_gradient(mag, mag @ slope + 0.5, model)
    np.testing.assert_allclose(grad, np.broadcast_to(slope, grad.shape), atol=1e-8)


def test_band_offsets_reproduce_colours():
    b = ne.band_offsets([0.01, 0.02, 0.03, 0.04], 0.05)
    np.testing.assert_allclose(-np.diff(b), [0.01, 0.02, 0.03, 0.04])
    assert b[2] == 0.05


def test_zero_offsets_change_nothing(run):
    cal, _ = run
    _, raw = draw(
        cal,
        n=3,
        zero_points=dict(colour_sigma=[0] * 4, colour_corr=np.eye(4), grey=0),
        z2_offset=0,
    )
    np.testing.assert_allclose(raw, cal.fiducial(), atol=1e-12)


def test_zero_points_spread_the_draws(run):
    cal, _ = run
    real, _ = draw(
        cal, zero_points=dict(colour_sigma=[0.05] * 4, colour_corr=np.eye(4), grey=0.02)
    )
    assert mean_z(cal, real).std(0).min() > 0
    np.testing.assert_allclose(real.mean(0), cal.fiducial(), atol=1e-12)


def test_every_spectrum_failing_erases_colour_information(run):
    """With certain failure, each bin's raw mean n(z) is the pooled calibrator n(z)."""
    cal, _ = run
    _, raw = draw(cal, n=200, specz_failure=1.0)
    np.testing.assert_allclose(mean_z(cal, raw), cal.z.mean(), atol=0.01)


def test_selection_shift_scales_the_draws(run):
    """A selection perturbation spreads physical draws without moving their mean."""
    cal, _ = run
    source = cal.source * cal.gold
    args = (cal.photoz, cal.group, cal.edges, source, cal.grid, cal.fiducial())
    np.testing.assert_allclose(ne.selection_shift(cal.z, *args), 0, atol=1e-12)
    shifted = replace(cal, selection=ne.selection_shift(cal.z + 0.05, *args))
    np.testing.assert_allclose(mean_z(cal, shifted.selection), 0.05, atol=1e-3)
    real, _ = draw(shifted, n=400, specz_selection=1.0)
    narrower, _ = draw(shifted, n=400, specz_selection=0.5)
    assert np.all(real >= 0)
    np.testing.assert_allclose(real.mean(0), cal.fiducial(), atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(real, cal.grid, axis=-1), 1, atol=1e-12)
    assert np.all(mean_z(cal, real).std(0) > mean_z(cal, narrower).std(0))
    # At large amplitude positivity changes the Gaussian input width.
    assert np.all(mean_z(cal, real).std(0) < 0.05)


def test_positive_recentre_preserves_small_perturbation_width():
    grid = np.linspace(0, 1, 101)
    fid = np.ones((1, len(grid)))
    direction = (grid - 0.5)[None, :]
    t = np.random.default_rng(1).normal(0, 0.001, 1000)
    draws = fid + t[:, None, None] * direction
    positive = ne.positive_recentre(draws, fid, grid)
    np.testing.assert_allclose(positive.mean(0), fid, atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(positive, grid, axis=-1), 1, atol=1e-12)
    before = np.trapezoid(draws * grid, grid, axis=-1).std(0)
    after = np.trapezoid(positive * grid, grid, axis=-1).std(0)
    np.testing.assert_allclose(after, before, rtol=1e-5)


def test_positive_recentre_handles_negative_draws_and_zero_support():
    grid = np.linspace(0, 1, 11)
    fid = np.r_[0, np.ones(9), 0][None, :] / 0.9
    draws = fid + np.random.default_rng(3).normal(0, 10, (100, 1, 11))
    assert draws.min() < 0
    positive = ne.positive_recentre(draws, fid, grid)
    assert np.all(positive >= 0)
    assert np.all(positive[..., [0, -1]] == 0)
    np.testing.assert_allclose(positive.mean(0), fid, atol=1e-12)
    np.testing.assert_allclose(np.trapezoid(positive, grid, axis=-1), 1, atol=1e-12)


def test_gold_factor_moves_selection(run):
    cal, _ = run
    loose = replace(cal, gold_qc=dict(floor=0.0, nmad_factor=5))
    real, _ = draw(loose, nmad_factor=[0.5, 2])
    assert mean_z(loose, real).std(0).max() > 0


def test_density_cancels_in_redshift_pure_groups(run):
    """A group of calibrators at one redshift cell absorbs any δ(z)."""
    cal, _ = run
    pure = replace(cal, z=np.round(cal.group * 0.03 + 0.2, 2))
    pure = replace(
        pure,
        weights=ne.calibrator_weights(
            pure.group, pure.member, np.ones(len(pure.z)), cal.source * cal.gold
        ),
    )
    cov = np.diag(np.full(len(cal.grid), 0.3))
    real, _ = ne.realisations(pure, dict(draws=5, seed=3), density=cov)
    np.testing.assert_allclose(
        real, np.broadcast_to(pure.fiducial(), real.shape), atol=1e-12
    )
    assert np.allclose(np.trapezoid(real, cal.grid, axis=-1), 1)


def test_bootstrap_bins_are_independent(run):
    cal, _ = run
    real, _ = draw(cal, n=400, bootstrap=True)
    m = mean_z(cal, real)
    c = np.corrcoef(m.T)
    assert np.abs(c[np.triu_indices(len(c), 1)]).max() < 0.2


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


def test_draws_are_centred_on_the_calibration(run):
    cal, _ = run
    cov = np.diag(np.full(len(cal.grid), 0.5))
    real, _ = ne.realisations(cal, dict(draws=50, seed=6, bootstrap=True), density=cov)
    np.testing.assert_allclose(real.mean(0), cal.fiducial(), atol=1e-12)


def test_offset_response_is_odd(run):
    """A draw at -v mirrors the draw at +v about the unperturbed calibration."""
    cal, _ = run
    q, v = np.ones(len(cal.z)), np.array([0.03, -0.02, 0.01, 0.02, -0.03, 0.02])
    up = cal.calibrate(cal.z, q, cal.gold, v)
    down = cal.calibrate(cal.z, q, cal.gold, -v)
    mid = cal.calibrate(cal.z, q, cal.gold)
    np.testing.assert_allclose(mid, cal.fiducial(), atol=1e-12)
    assert not np.allclose(up, down)
