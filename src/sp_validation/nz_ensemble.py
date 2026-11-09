"""Realisations of the SOM n(z) calibration, every bin in each draw.

A calibrated bin is a weighted histogram of calibrator redshifts: calibrator
i in SOM group g of bin k carries S_kg / N_kg, the group's source shear weight
over its number of in-bin calibrators (:func:`nz_calibration.calibrate`). A
draw replaces each calibrator's unit count by a random factor q_i and keeps
every group's source weight fixed,

    w_i = S_kg q_i / Σ_{j ∈ g, k} q_j,

so a group only redistributes its weight among its own calibrators, and a
group whose factors all vanish drops out, as an unsupported group does in the
calibration. Two sources enter q_i = m_i (1 + δ(z_i)):

- **calibrator shot noise**: m_i are multinomial bootstrap counts;
- **sample variance of the calibration field**: δ(z) is the galaxy density
  contrast along the line of sight through the spectroscopic footprint, a
  lognormal field on the output grid whose covariance is the nonlinear matter
  power spectrum seen through the footprint and each grid cell, times a
  linear bias squared (:func:`density_covariance`).

Every bin uses its own calibrators (they are split by BPZ), so shot noise is
independent between bins; δ(z) is shared, and is what correlates them. The
SOM, groups, gold masks and source weights are held fixed.
"""

import argparse
from pathlib import Path

import numpy as np

from .nz_calibration import deposit


def source_weights(cal_weights, cal_group, n_groups):
    """Per-(bin, group) source weight S_kg implied by calibrator weights."""
    return np.stack(
        [np.bincount(cal_group, weights=w, minlength=n_groups) for w in cal_weights]
    )


def reweight(cal_weights, cal_group, q):
    """Calibrator weights (n_bin, n_cal) with factors ``q`` at fixed S_kg."""
    n_groups = cal_group.max() + 1
    source = source_weights(cal_weights, cal_group, n_groups)
    out = np.zeros_like(cal_weights)
    for k, w in enumerate(cal_weights):
        member = w > 0
        qk = np.where(member, q, 0.0)
        den = np.bincount(cal_group, weights=qk, minlength=n_groups)
        ratio = np.divide(source[k], den, out=np.zeros(n_groups), where=den > 0)
        out[k] = ratio[cal_group] * qk
    return out


def histograms(cal_z, cal_weights, grid):
    """Unit-integral n(z) per bin from calibrator weights."""
    return np.stack([deposit(cal_z, w / w.sum(), grid) for w in cal_weights])


def density_covariance(grid, field, bias, cosmo, n_phi=4096, max_sep=500.0):
    """Covariance of δ_g averaged over each grid cell of the footprint.

    The footprint is a ``field = (width, height)`` rectangle in degrees; each
    cell is the slab between midpoints of neighbouring grid redshifts. With
    W̃ the cell's Fourier window,

        C_ij = b² ∫ d³k/(2π)³ P(k, z̄_ij) W̃_i(k) W̃_j*(k),

    P the nonlinear matter power at the pair's mean redshift. Separating the
    transverse window (angle-averaged, a function of k⊥χ only) leaves one
    integral over k⊥ and one over k∥. The first grid point (z = 0) has no
    volume and gets zero variance. Cells more than ``max_sep`` Mpc apart are
    uncorrelated.
    """
    import pyccl as ccl

    theta = np.deg2rad(np.asarray(field, float))
    z = np.asarray(grid, float)
    edges = np.r_[z[0], 0.5 * (z[1:] + z[:-1]), z[-1]]
    chi_edges = ccl.comoving_radial_distance(cosmo, 1 / (1 + edges))
    chi = ccl.comoving_radial_distance(cosmo, 1 / (1 + z))
    width = np.diff(chi_edges)

    # Angle-averaged |W̃⊥|² as a function of s = k⊥ χ.
    s = np.geomspace(1e-2, 1e6, 600)
    phi = (np.arange(n_phi) + 0.5) * (np.pi / 2) / n_phi
    a = np.sinc(np.outer(s, np.cos(phi)) * theta[0] / (2 * np.pi)) ** 2
    a *= np.sinc(np.outer(s, np.sin(phi)) * theta[1] / (2 * np.pi)) ** 2
    window_perp = a.mean(1)

    live = np.flatnonzero((width > 0) & (chi > 0))
    # k∥ is linear so cos(k∥ Δχ) is resolved for every pair kept (Δχ ≤
    # max_sep); pairs farther apart, uncorrelated through a pencil beam, are 0.
    kpar = np.r_[np.linspace(0.0, 2.0, 10_001), np.geomspace(2.0005, 50.0, 300)]
    kperp = np.geomspace(1e-4, 200.0, 800)
    klog = np.log(np.geomspace(1e-5, 300.0, 4000))
    kk = np.log(np.sqrt(kperp[None, :] ** 2 + kpar[:, None] ** 2))
    # Q(k∥; z) on a coarse redshift grid, interpolated to each pair's mean.
    zq = np.linspace(z[live].min(), z[live].max(), 60)
    chiq = ccl.comoving_radial_distance(cosmo, 1 / (1 + zq))
    q = np.empty((len(zq), len(kpar)))
    for n, (zz, cc) in enumerate(zip(zq, chiq)):
        lp = np.log(ccl.nonlin_matter_power(cosmo, np.exp(klog), 1 / (1 + zz)))
        pk = np.exp(np.interp(kk, klog, lp))
        wp = np.interp(np.log(kperp * cc), np.log(s), window_perp, left=1.0, right=0.0)
        q[n] = np.trapezoid(kperp * pk * wp, kperp, axis=1) / (2 * np.pi)

    zl, wl, cl = z[live], width[live], chi[live]
    cov = np.zeros((len(z), len(z)))
    sinc = np.sinc(np.outer(wl, kpar) / (2 * np.pi))
    for i in range(len(live)):
        j = i + np.flatnonzero(cl[i:] - cl[i] <= max_sep)
        pos = np.interp(0.5 * (zl[i] + zl[j]), zq, np.arange(len(zq)))
        lo = np.minimum(pos.astype(int), len(zq) - 2)
        t = (pos - lo)[:, None]
        qi = q[lo] * (1 - t) + q[lo + 1] * t
        integrand = np.cos(np.outer(cl[j] - cl[i], kpar)) * sinc[i] * sinc[j] * qi
        row = bias**2 * np.trapezoid(integrand, kpar, axis=1) / np.pi
        cov[live[i], live[j]] = row
        cov[live[j], live[i]] = row
    return cov


def lognormal(cov, n_draw, rng):
    """Draws of 1 + δ with covariance ``cov`` and unit mean in every cell."""
    g = np.log1p(cov)
    lam, vec = np.linalg.eigh(g)
    root = vec * np.sqrt(np.clip(lam, 0, None))
    gauss = rng.standard_normal((n_draw, len(cov))) @ root.T
    return np.exp(gauss - 0.5 * np.diag(g))


def realisations(cal_z, cal_weights, cal_group, grid, n_draw, *, density=None, seed=0):
    """n(z) draws (n_draw, n_bin, n_z), unit integral per bin.

    The draws are recentred so their mean is the calibration's own n(z): the
    within-group ratio is nonlinear in the factors and shifts the raw mean by
    a fraction of the scatter, which is estimator bias, not a redshift shift.
    ``density`` is the covariance of δ on ``grid`` (:func:`density_covariance`);
    ``None`` keeps only calibrator shot noise.
    """
    rng = np.random.default_rng(seed)
    n_cal = len(cal_z)
    counts = rng.multinomial(n_cal, np.full(n_cal, 1 / n_cal), size=n_draw)
    if density is None:
        factor = np.ones((n_draw, n_cal))
    else:
        cell = np.clip(
            np.rint((cal_z - grid[0]) / (grid[1] - grid[0])), 0, len(grid) - 1
        )
        factor = lognormal(density, n_draw, rng)[:, cell.astype(int)]
    draws = np.stack(
        [
            histograms(cal_z, reweight(cal_weights, cal_group, m * f), grid)
            for m, f in zip(counts, factor)
        ]
    )
    return draws - draws.mean(0) + histograms(cal_z, cal_weights, grid)


def run(arrays, output, settings, cosmo=None):
    """Write ``realisations.npz`` from a calibration's ``calibration_arrays.npz``."""
    a = np.load(arrays)
    grid = a["grid"]
    density = None
    if settings.get("bias", 0):
        if cosmo is None:
            from .blinding import cosmology, fiducial

            cosmo = cosmology(fiducial())
        density = density_covariance(grid, settings["field"], settings["bias"], cosmo)
    real = realisations(
        a["z"],
        a["cal_weights"],
        a["cal_group"],
        grid,
        settings["draws"],
        density=density,
        seed=settings.get("seed", 0),
    )
    np.savez_compressed(
        output,
        z=grid,
        realisations=real,
        fiducial=histograms(a["z"], a["cal_weights"], grid),
        field=np.asarray(settings.get("field", [0, 0]), float),
        bias=float(settings.get("bias", 0)),
        seed=settings.get("seed", 0),
    )
    return real


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sp_validation.nz_ensemble")
    p.add_argument("arrays", help="calibration_arrays.npz of a calibration run")
    p.add_argument("-o", "--output", required=True, help="realisations .npz")
    p.add_argument("--draws", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--field",
        type=float,
        nargs=2,
        default=(2.0, 0.25),
        metavar=("WIDTH", "HEIGHT"),
        help="spectroscopic footprint in degrees",
    )
    p.add_argument(
        "--bias", type=float, default=1.3, help="calibrator galaxy bias; 0 drops δ(z)"
    )
    a = p.parse_args(argv)
    run(a.arrays, Path(a.output), vars(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
