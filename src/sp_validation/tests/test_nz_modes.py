"""n(z) modes: algebra, and reproduction of William d'Assignies Doumerg's notebook.

The reference (``data/nz_modes_william_reference.npz``) is the output of
William's own ``generate_modes.py`` and ``Fisher_Cell_CS_IA.py`` (9 Nov 2026)
on the two cases of his ``IA modes_2bins.ipynb``: two bins, 2000 realisations,
with and without a low-z bump. The test rebuilds the same inputs (his toy
generator, same seed and draw order) and his toy C_ℓ observable, then runs them
through :func:`sp_validation.nz.modes.observable_metric`, which differentiates along the
ensemble's directions instead of forming his full 400 × 400 Fisher matrix.
"""

from pathlib import Path

import numpy as np
import pytest

from sp_validation.nz import modes as nz_modes

REFERENCE = Path(__file__).parent / "data" / "nz_modes_william_reference.npz"


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
# William's notebook
# --------------------------------------------------------------------------- #
def _william_realisations(amplitude_bumps, sigma_amplitude_bumps, n_real=2000):
    """William's ``generate_nz_realizations`` (two-Gaussian bins), seed 42."""

    def gauss(z, m, s):
        g = np.exp(-0.5 * ((z - m) / s) ** 2)
        return g / np.trapezoid(g, z, axis=-1)[..., None]

    means, sigmas = np.array([0.5, 1.0]), np.array([0.1, 0.2])
    mean_bumps, sigma_bumps = np.array([0.8, 0.4]), np.array([0.15, 0.1])
    amp = np.asarray(amplitude_bumps, float)
    sig_amp = np.asarray(sigma_amplitude_bumps, float)
    sig_shift, sig_shift_bump = np.array([0.02, 0.03]), np.array([0.02, 0.03])
    sig_stretch = sig_stretch_bump = np.array([0.1, 0.1])

    z = np.linspace(0.0, 2.0, 200)
    rng = np.random.default_rng(42)
    size = (n_real, 2)
    shift = rng.normal(0, sig_shift, size)
    shift_bump = rng.normal(0, sig_shift_bump, size)
    stretch = np.clip(rng.normal(1, sig_stretch, size), 1e-3, None)
    stretch_bump = np.clip(rng.normal(1, sig_stretch_bump, size), 1e-3, None)
    amps = np.clip(rng.normal(amp, sig_amp, size), 0, 1)[..., None]

    main = gauss(z, (means + shift)[..., None], (sigmas * stretch)[..., None])
    bump = gauss(
        z,
        (mean_bumps + shift_bump)[..., None],
        (sigma_bumps * stretch_bump)[..., None],
    )
    nz = (1 - amps) * main + amps * bump
    fid = (1 - amp[:, None]) * gauss(z, means[:, None], sigmas[:, None]) + amp[
        :, None
    ] * gauss(z, mean_bumps[:, None], sigma_bumps[:, None])
    return z, fid, nz / np.trapezoid(nz, z, axis=-1)[..., None]


def _william_observable(z, fid, ia):
    """William's toy observable: tomographic C_ℓ (GG+GI+IG+II with NLA A_IA=1
    when ``ia``) at 40 multipoles in 10–1500, Gaussian covariance with
    f_sky 0.1, σ_e 0.26, n_eff 24 arcmin⁻² split across the bins, and the
    total signal plus noise in the covariance."""
    import pyccl as ccl

    cosmo = ccl.Cosmology(
        Omega_c=0.2607, Omega_b=0.04897, h=0.6766, sigma8=0.8102, n_s=0.9665
    )
    ells = np.unique(np.rint(np.linspace(np.sqrt(10), np.sqrt(1500), 40) ** 2))
    ells = ells.astype(int)
    pairs = [(0, 0), (0, 1), (1, 1)]
    ia_bias = (z, np.ones_like(z))

    def spectra(nz, with_ia):
        nz = nz / np.trapezoid(nz, z, axis=-1)[:, None]
        tracers = [
            ccl.WeakLensingTracer(
                cosmo, dndz=(z, n), ia_bias=ia_bias if with_ia else None
            )
            for n in nz
        ]
        return {
            (i, j): ccl.angular_cl(cosmo, tracers[i], tracers[j], ells)
            for i in range(2)
            for j in range(2)
        }

    def theory(nz):
        cl = spectra(nz, ia)
        return np.concatenate([cl[p] for p in pairs])

    # Covariance: always the total (IA-including) signal plus shape noise.
    cl = spectra(fid, True)
    noise = 0.26**2 / (2 * 12.0 * (180 * 60 / np.pi) ** 2)
    for i in range(2):
        cl[i, i] = cl[i, i] + noise
    n_ell = len(ells)
    cov = np.zeros((3 * n_ell, 3 * n_ell))
    for a, (i, j) in enumerate(pairs):
        for b, (p, q) in enumerate(pairs):
            block = (cl[i, p] * cl[j, q] + cl[i, q] * cl[j, p]) / ((2 * ells + 1) * 0.1)
            cov[a * n_ell : (a + 1) * n_ell, b * n_ell : (b + 1) * n_ell] = np.diag(
                block
            )
    return theory, cov


CASES = pytest.mark.parametrize(
    "case, amplitude, sigma_amplitude, threshold",
    [("bump", [0.1, 0.15], [0.05, 0.05], 0.03), ("nobump", [0, 0], [0, 0], 0.01)],
)


def _same_shapes(modes, d_ref):
    """Each kept mode matches William's, up to sign."""
    for mine, his in zip(modes.decoder, d_ref.T.reshape(-1, 2, 200)):
        cos = np.sum(mine * his) / np.sqrt(np.sum(mine**2) * np.sum(his**2))
        assert abs(cos) > 0.99


@CASES
def test_reproduces_william_with_his_fisher(
    case, amplitude, sigma_amplitude, threshold
):
    """His Fisher matrix (``mode="total"``, shear + IA) through :func:`fit`
    gives his eigenvalues, number of modes and mode shapes."""
    ref = np.load(REFERENCE)
    key = f"{case}_total"
    factor = ref[f"{key}_fisher_factor"].astype(float)
    z, _, real = _william_realisations(amplitude, sigma_amplitude)
    modes = nz_modes.fit(
        z, real, nz_modes.fisher_metric(factor.T @ factor), threshold=threshold
    )
    assert modes.n_modes == int(ref[f"{key}_M"])
    np.testing.assert_allclose(
        modes.eigenvalues[:8], ref[f"{key}_lambda_G"], rtol=1e-4, atol=1e-6
    )
    _same_shapes(modes, ref[f"{key}_D"])


@CASES
def test_directional_fisher_agrees_with_william(
    case, amplitude, sigma_amplitude, threshold
):
    """His observable differentiated along the ensemble's directions: the same
    modes, with eigenvalues within 15%.

    The eigenvalues differ because his Jacobian perturbs one z grid point at a
    time. Along a realisation's direction, that Jacobian misses the theory's
    derivative (which two step sizes agree on to 1e-4) by about 7%.
    """
    ref = np.load(REFERENCE)
    key = f"{case}_total"
    z, fid, real = _william_realisations(amplitude, sigma_amplitude)
    theory, cov = _william_observable(z, fid, ia=True)
    modes = nz_modes.fit(
        z, real, nz_modes.observable_metric(theory, cov), threshold=threshold
    )
    assert modes.n_modes == int(ref[f"{key}_M"])
    lam = ref[f"{key}_lambda_G"]
    big = lam > 0.01 * lam[0]
    np.testing.assert_allclose(modes.eigenvalues[: big.sum()], lam[big], rtol=0.15)
    _same_shapes(modes, ref[f"{key}_D"])


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
