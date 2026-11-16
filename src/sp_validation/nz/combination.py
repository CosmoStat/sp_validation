"""One n(z) ensemble per bin from a prior ensemble and clustering redshifts.

A bin's prior is a set of n(z) draws: the SOM realisations (:mod:`.ensemble`)
where spectra represent the sources, or a family of shifted and stretched
templates (:func:`shift_stretch`) where they do not. The clustering measurement
(:mod:`.clustering`) gives y_i = n_u b_u in reference slices i, with jackknife
covariance C. Each draw n_s predicts

    y_i = A  n̄_s,i  ((1 + z_i) / (1 + z_p))^α,

n̄_s,i its slice average, z_p the middle of the fit window, A a free amplitude
and α the slope of the unknown source bias with a Gaussian prior. A is
profiled and α marginalised on a grid, which gives the draw's weight

    W_s ∝ Σ_α π(α) exp(-χ²_s(α) / 2).

The weighted draws are the combined ensemble (:func:`combine`); the posterior of
α from a sample that has a SOM prior can serve as the prior of one that has none.
The bias slope is what limits the result: for a bin of redshift variance σ_z²,
⟨z⟩ moves by about -σ_z² / (1 + z̄) per unit of α.
"""

import numpy as np

from .clustering import slice_average

ALPHAS = np.round(np.arange(-4, 6.001, 0.1), 3)


def slice_matrix(grid, lo, hi):
    """S with n̄ = S n: slice averages of a piecewise-linear gridded n(z)."""
    return np.stack(
        [slice_average(grid, e, lo, hi) for e in np.eye(len(grid))], axis=1
    )


def mean_z(grid, draws):
    return (draws * grid).sum(-1) / draws.sum(-1)


def shift_stretch(template, grid, shift, stretch, n_draw, rng):
    """Draws n((z - z̄ - δ) / s + z̄), δ ~ N(0, shift), ln s ~ N(0, stretch)."""
    centre = mean_z(grid, template)
    delta = rng.normal(0, shift, n_draw)
    s = np.exp(rng.normal(0, stretch, n_draw))
    draws = np.stack(
        [
            np.interp((grid - centre - d) / f + centre, grid, template, left=0, right=0)
            for d, f in zip(delta, s)
        ]
    )
    return draws / np.trapezoid(draws, grid, axis=1)[:, None]


def chi2(y, precision, models, z, alphas=ALPHAS):
    """χ² (draw, α) of y = A model ((1+z)/(1+z_p))^α at the best A ≥ 0."""
    tilt = ((1 + z) / (1 + (z[0] + z[-1]) / 2)) ** alphas[:, None]
    m = models[:, None, :] * tilt
    my = np.clip(m @ (precision @ y), 0, None)
    mm = np.einsum("sai,ij,saj->sa", m, precision, m)
    return y @ precision @ y - my**2 / mm


def combine(grid, draws, y, precision, lo, hi, alpha_prior, alphas=ALPHAS):
    """Weight prior draws by the clustering likelihood of one sample.

    ``draws`` (n_draw, n_grid) on ``grid``; ``y`` and ``precision`` on the
    slices [``lo``, ``hi``) of the fit window; ``alpha_prior`` the (mean, sigma)
    of the bias slope. Returns the weights, their effective number, the prior
    and posterior mean redshift with standard deviations, the posterior of α on
    ``alphas`` and the smallest χ² over draws and α.
    """
    z = (lo + hi) / 2
    x2 = chi2(y, precision, draws @ slice_matrix(grid, lo, hi).T, z, alphas)
    prior = np.exp(-0.5 * ((alphas - alpha_prior[0]) / alpha_prior[1]) ** 2)
    like = np.exp(-0.5 * (x2 - x2.min())) * prior
    weights = like.sum(1) / like.sum()
    alpha_post = like.sum(0) / like.sum()
    zbar = mean_z(grid, draws)

    def moments(x, w):
        m = (w * x).sum() / w.sum()
        return float(m), float(np.sqrt((w * (x - m) ** 2).sum() / w.sum()))

    return dict(
        weights=weights,
        ess=float(1 / (weights**2).sum()),
        prior=moments(zbar, np.ones_like(zbar)),
        posterior=moments(zbar, weights),
        alpha=moments(alphas, alpha_post),
        alpha_posterior=alpha_post,
        chi2=float(x2.min()),
        dof=len(y) - 2,
        nz=weights @ draws,
    )
