"""n(z) modes: the algebra on toy ensembles with known answers, and the CLI on a SACC."""

import numpy as np
import pytest

from sp_validation.nz import modes as nz_modes


# --------------------------------------------------------------------------- #
# Algebra
# --------------------------------------------------------------------------- #
def _toy_ensemble(n_real=4000, seed=1):
    """Two bins whose draws are a mean plus three known shapes."""
    rng = np.random.default_rng(seed)
    z = np.linspace(0, 2, 60)
    mean = np.stack([np.exp(-0.5 * ((z - m) / 0.2) ** 2) for m in (0.6, 1.1)])
    shapes = rng.normal(size=(3, 2, 60))
    return z, mean + np.tensordot(rng.normal(size=(n_real, 3)), shapes, 1)


def test_unit_amplitudes_and_round_trip():
    z, real = _toy_ensemble()
    fisher = np.diag(np.linspace(1, 2, 120))
    modes = nz_modes.fit(z, real, nz_modes.fisher_metric(fisher), n_modes=3)

    u = modes.encode(real)
    np.testing.assert_allclose(np.cov(u, rowvar=False), np.eye(3), atol=1e-10)
    # Three shapes span the ensemble: decoding the amplitudes is lossless.
    np.testing.assert_allclose(modes.decode(u), real, atol=1e-8)
    assert modes.chi2_left[3] == pytest.approx(0, abs=1e-8)


def test_eigenvalues_are_expected_chi2():
    """Σ λ_k equals the mean χ² of a draw about the ensemble mean."""
    z, real = _toy_ensemble()
    fisher = np.diag(np.linspace(1, 2, 120))
    modes = nz_modes.fit(z, real, nz_modes.fisher_metric(fisher))
    d = (real - real.mean(0)).reshape(len(real), -1)
    chi2 = np.einsum("ri,ij,rj->r", d, fisher, d)
    assert modes.eigenvalues.sum() == pytest.approx(
        chi2.mean() * len(real) / (len(real) - 1)
    )


def test_observable_metric_matches_fisher_for_linear_theory():
    z, real = _toy_ensemble()
    rng = np.random.default_rng(2)
    jac = rng.normal(size=(30, 120))
    cov = np.diag(rng.uniform(1, 2, 30))
    via_fisher = nz_modes.fit(
        z, real, nz_modes.fisher_metric(jac.T @ np.linalg.solve(cov, jac))
    )
    via_theory = nz_modes.fit(
        z, real, nz_modes.observable_metric(lambda n: jac @ n.ravel(), cov)
    )
    np.testing.assert_allclose(via_theory.eigenvalues, via_fisher.eigenvalues)


def test_threshold_picks_fewest_modes(tmp_path):
    z, real = _toy_ensemble()
    weights = np.ones(120)
    weights[:60] = 100.0  # bin 1 dominates
    metric = nz_modes.fisher_metric(np.diag(weights))
    modes = nz_modes.fit(z, real, metric, threshold=0.05)
    left = modes.chi2_left
    assert left[modes.n_modes] < 0.05 * left[0] <= left[modes.n_modes - 1]

    modes.save(tmp_path / "modes.npz")
    back = nz_modes.NzModes.load(tmp_path / "modes.npz")
    np.testing.assert_array_equal(back.decoder, modes.decoder)


def test_shift_stretch_keeps_normalisation_and_moves_mean():
    z = np.linspace(0, 3, 301)
    nz = np.stack([np.exp(-0.5 * ((z - m) / 0.15) ** 2) for m in (0.5, 1.0)])
    out = nz_modes.shift_stretch(z, nz, [0.05, -0.02], [1.0, 1.1])
    np.testing.assert_allclose(
        np.trapezoid(out, z, axis=-1), np.trapezoid(nz, z, axis=-1)
    )
    mean = np.trapezoid(out * z, z, axis=-1) / np.trapezoid(out, z, axis=-1)
    np.testing.assert_allclose(mean, [0.55, 0.98], atol=5e-4)


# --------------------------------------------------------------------------- #
# The CLI on a SACC
# --------------------------------------------------------------------------- #
def test_cli_on_a_two_bin_xi_sacc(tmp_path):
    from sp_validation import sacc_io as sio

    z = np.linspace(0.0, 3.0, 151)
    nz = np.stack([np.exp(-0.5 * ((z - m) / 0.2) ** 2) for m in (0.5, 0.9)])
    s = sio.new_sacc({i: (z, n) for i, n in enumerate(nz)})
    theta = np.geomspace(5.0, 100.0, 6)
    for pair in [(0, 0), (0, 1), (1, 1)]:
        sio.add_xi(s, pair, theta, 0 * theta, 0 * theta, grid="reporting")
    s.add_covariance(np.diag(np.full(len(s.mean), 1e-12)))
    s.save_fits(tmp_path / "toy.sacc")

    rng = np.random.default_rng(3)
    real = nz_modes.shift_stretch(
        z, nz, rng.normal(0, 0.03, (300, 2)), rng.normal(1, 0.05, (300, 2))
    )
    np.savez(tmp_path / "real.npz", z=z, realisations=real)

    out = tmp_path / "modes.npz"
    nz_modes.main(
        [
            str(tmp_path / "toy.sacc"),
            str(tmp_path / "real.npz"),
            "-o",
            str(out),
            "--scales",
            "8",
            "80",
            "--A-IA",
            "1",
        ]
    )
    modes = nz_modes.NzModes.load(out)
    left = modes.chi2_left
    assert 1 <= modes.n_modes <= 4
    assert left[modes.n_modes] < 0.01 * left[0]
    assert modes.decoder.shape == (modes.n_modes, 2, len(z))
