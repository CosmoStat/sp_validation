"""Realisations of the SOM n(z) calibration, every bin in each draw.

A calibrated bin is a weighted histogram of calibrator redshifts: calibrator
i in gold SOM group g of bin k carries S_kg / N_kg, the group's source shear
weight over its number of in-bin calibrators (:func:`calibration.calibrate`).
A draw perturbs what enters that sum and recalibrates,

    w_i = S_kg q_i / Σ_{j ∈ g, k} q_j,

so a gold group whose calibrators all leave it drops out, as an unsupported
group does in the calibration. Each term is switched on by its key in the
recipe's ``ensemble`` section.

Statistical terms, through the factors q_i = m_i (1 + δ(z_i)):

- ``bootstrap``: m_i are multinomial bootstrap counts (calibrator shot noise);
- ``density``: δ(z) is the galaxy density contrast along the line of sight
  through the spectroscopic footprint, a lognormal field on the output grid
  whose covariance is the nonlinear matter power seen through the footprint
  and each grid cell, times a linear bias squared (:func:`density_covariance`).

Systematic terms:

- ``zero_points``: a zero-point error δ of the calibration field relative to
  the sources, drawn as Gaussian colour offsets (u-g, g-r, r-i, i-z) with
  standard deviations ``colour_sigma`` and correlations ``colour_corr``, and a
  grey offset of standard deviation ``grey``. The calibrators
  trained the SOM, and training is translation-equivariant, so correcting
  their photometry by -δ moves the cells with them: every source moves by +δ
  against the SOM, shifting the catalogue's weight between groups, and each
  calibrator's BPZ, and so its bin, moves by -δ along its local gradient;
- ``z2_offset``: an error ε in the constant that maps z2 onto z, offsetting
  the z band of every source measured through z2 (the calibrators have none);
- ``specz_failure``: each calibrator's redshift is, with this probability,
  replaced by the redshift of a random calibrator;
- ``nmad_factor``: the gold-QC factor drawn uniformly from a range; gold is
  re-derived at that factor from the calibration's own spectra and sources,
  so the bins' source selection moves with it.

The offsets enter to first order: a draw adds half the difference between
the recalibrations at +v and -v, which cancels what is even in the offsets
(groups losing their last in-bin calibrator at either sign). Every term but
``nmad_factor`` keeps the catalogue's selection: the gold groups' sources, wherever the
offsets move them, calibrated by whichever calibrators share their group.
The SOM and its groups are held fixed.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from . import calibration as nz

#: Offsets the source tables respond to: the calibration field's five bands
#: (as a source offset) and the z2 mapping.
DIRECTIONS = ("u", "g", "r", "i", "z", "z2")


def histograms(cal_z, cal_weights, grid):
    """Unit-integral n(z) per bin from calibrator weights."""
    return np.stack([nz.deposit(cal_z, w / w.sum(), grid) for w in cal_weights])


def members(photoz, edges):
    """Calibrators in each bin's BPZ interval, (n_bin, n_cal)."""
    return np.stack(
        [(photoz >= lo) & (photoz < hi) for lo, hi in zip(edges[:-1], edges[1:])]
    )


def group_tables(prelim, group, weight, photoz, n_bin, n_groups):
    """Source weight and weighted BPZ summed per (bin, group)."""
    index = prelim.astype(np.int64) * n_groups + group
    size = n_bin * n_groups
    w = np.bincount(index, weights=weight, minlength=size)
    p = np.bincount(index, weights=weight * photoz, minlength=size)
    return w.reshape(n_bin, n_groups), p.reshape(n_bin, n_groups)


def ensemble_inputs(
    mag, photoz, weight, prelim, channel, model, gold, *, offset, step=0.02, workers=1
):
    """Per-(bin, group) source tables, and the catalogue's response to offsets.

    ``prelim`` and ``channel`` are :func:`calibration.source_groups` of the
    sources at the recipe's z2 ``offset``; ``gold`` is the calibration's gold
    mask. ``source_weight`` and ``source_photoz`` sum every source with BPZ in
    a bin, as the gold test does. ``catalogue_jacobian`` is the derivative, in
    each of :data:`DIRECTIONS`, of the weight per (bin, group) of the catalogue's
    own sources (those in gold at the fiducial), so the selection stays the
    catalogue's while its sources move between groups. A direction is a
    magnitude offset of the sources against the SOM: one band for every
    source, or z for the sources measured through z2. Sources fill the cells
    densely, so the tables move linearly over a few hundredths of a magnitude;
    central differences at ±``step`` give the derivatives.
    """
    n_bin, n_groups = len(model["edges"]) - 1, int(model["group_map"].max()) + 1
    rows = np.flatnonzero(prelim >= 0)
    m, _ = nz.z_magnitudes(mag[rows], offset)
    x = (nz.features(m) - model["center"]) / model["scale"]
    del m
    via_z2 = channel[rows] == 2
    tree = cKDTree(model["code"])
    bins, weight, photoz = prelim[rows], weight[rows], photoz[rows]
    base = model["group_map"][tree.query(x, workers=workers)[1]]
    catalogue = np.where(gold[bins, base], weight, 0.0)

    def tables(direction, size):
        d = np.zeros((1, 5))
        d[0, min(direction, 4)] = size
        shift = nz.features(d)[0] / model["scale"]
        moved = via_z2 if DIRECTIONS[direction] == "z2" else np.ones(len(rows), bool)
        group = base.copy()
        cell = tree.query(x[moved] + shift, workers=workers)[1]
        group[moved] = model["group_map"][cell]
        return group_tables(bins, group, catalogue, photoz, n_bin, n_groups)[0]

    source, source_photoz = group_tables(bins, base, weight, photoz, n_bin, n_groups)
    jacobian = np.array(
        [
            (tables(k, step) - tables(k, -step)) / (2 * step)
            for k in range(len(DIRECTIONS))
        ]
    )
    return dict(
        directions=np.array(DIRECTIONS),
        source_weight=source,
        source_photoz=source_photoz,
        catalogue_jacobian=jacobian,
    )


def photoz_gradient(mag, photoz, model, k=128):
    """d BPZ / d m (n_cal, 5) of each calibrator, from its k nearest in the SOM space.

    A least-squares plane of BPZ in ugriz through the neighbours.
    """
    x = (nz.features(mag) - model["center"]) / model["scale"]
    _, near = cKDTree(x).query(x, k)
    grad = np.empty((len(mag), 5))
    for i, j in enumerate(near):
        a = np.column_stack([mag[j] - mag[i], np.ones(k)])
        grad[i] = np.linalg.lstsq(a, photoz[j], rcond=None)[0][:5]
    return grad


def calibrator_weights(group, member, q, source):
    """Weights (n_bin, n_cal): S_kg q_i / Σ_{j ∈ g, k} q_j.

    ``source`` is the weight to calibrate per (bin, group), zero outside the
    selection; a group with weight but no in-bin calibrator drops out.
    """
    n_groups = source.shape[1]
    out = np.zeros(member.shape)
    for k in range(len(member)):
        qk = np.where(member[k], q, 0.0)
        den = np.bincount(group, weights=qk, minlength=n_groups)
        ratio = np.divide(source[k], den, out=np.zeros(n_groups), where=den > 0)
        out[k] = ratio[group] * qk
    return out


def gold_masks(z, photoz, group, member, source, source_photoz, floor, factor):
    """Re-derive every bin's gold mask, as :func:`calibration.calibrate` does."""
    n_groups = source.shape[1]
    threshold = nz.gold_threshold(z, photoz, floor, factor)
    masks = []
    for k in range(len(member)):
        g = group[member[k]]
        count = np.bincount(g, minlength=n_groups)
        zsum = np.bincount(g, weights=z[member[k]], minlength=n_groups)
        masks.append(nz.gold_mask(source[k], source_photoz[k], count, zsum, threshold))
    return np.array(masks)


def band_offsets(colours, grey):
    """Offsets of (u, g, r, i, z) from colour offsets (u-g, g-r, r-i, i-z) and r."""
    du_g, dg_r, dr_i, di_z = colours
    g, i = grey + dg_r, grey - dr_i
    return np.array([g + du_g, g, grey, i, i - di_z])


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


@dataclass
class Calibration:
    """A calibration run's state: what a draw perturbs and recalibrates.

    ``source`` and ``source_photoz`` are the (n_bin, n_groups) tables of every
    source with BPZ in a bin, ``jacobian`` the catalogue's response in
    :data:`DIRECTIONS` (:func:`ensemble_inputs`), and ``photoz_gradient`` each
    calibrator's d BPZ / d m.
    """

    z: np.ndarray
    photoz: np.ndarray
    group: np.ndarray
    weights: np.ndarray
    gold: np.ndarray
    edges: np.ndarray
    grid: np.ndarray
    source: np.ndarray
    source_photoz: np.ndarray
    jacobian: np.ndarray
    photoz_gradient: np.ndarray
    gold_qc: dict

    @classmethod
    def load(cls, directory):
        """Read a calibration run's output directory."""
        directory = Path(directory)
        a = np.load(directory / "calibration_arrays.npz")
        e = np.load(directory / "ensemble_inputs.npz")
        return cls(
            z=a["z"],
            photoz=a["cal_photoz"],
            group=a["cal_group"],
            weights=a["cal_weights"],
            gold=a["gold"],
            edges=a["edges"],
            grid=a["grid"],
            source=e["source_weight"],
            source_photoz=e["source_photoz"],
            jacobian=e["catalogue_jacobian"],
            photoz_gradient=e["cal_photoz_gradient"],
            gold_qc=json.loads(str(a["gold_qc"])),
        )

    @property
    def member(self):
        return members(self.photoz, self.edges)

    def fiducial(self):
        """The calibration's own unit-integral n(z) per bin."""
        return histograms(self.z, self.weights, self.grid)

    def calibrate(self, z, q, gold, v=None):
        """Unit-integral n(z) per bin of the selection ``gold``.

        ``v`` holds offsets in :data:`DIRECTIONS`: the calibration field's
        (u, g, r, i, z) zero-point error and the z2 mapping error. The
        catalogue's sources move between groups along the Jacobian, and each
        calibrator's BPZ moves by -grad·δ, the change correcting its
        photometry would make.
        """
        source, photoz = self.source * gold, self.photoz
        if v is not None:
            source = np.maximum(source + np.tensordot(v, self.jacobian, 1), 0)
            photoz = photoz - self.photoz_gradient @ v[:5]
        w = calibrator_weights(self.group, members(photoz, self.edges), q, source)
        return histograms(z, w, self.grid)


def realisations(cal, settings, density=None):
    """Draws (n_draw, n_bin, n_z) and their raw mean before recentring.

    ``settings`` is the recipe's ``ensemble`` section; ``density`` is the
    covariance of δ on the grid (:func:`density_covariance`) when the density
    term is on. The draws are recentred so their mean is the calibration's own
    n(z): the terms are zero-mean in their inputs, but the calibration is
    nonlinear in them, and the raw mean is returned so its offset can be
    reported.

    @sc [decision:nz.ensemble_terms]
    """
    rng = np.random.default_rng(settings.get("seed", 0))
    n_draw, n_cal = settings["draws"], len(cal.z)
    zp = settings.get("zero_points")
    if zp:
        sigma = np.asarray(zp["colour_sigma"], float)
        colour_cov = np.outer(sigma, sigma) * np.asarray(zp["colour_corr"], float)
    if density is not None:
        overdensity = lognormal(density, n_draw, rng)
        step = cal.grid[1] - cal.grid[0]
    draws = []
    for n in range(n_draw):
        z, gold = cal.z, cal.gold
        if settings.get("specz_failure"):
            fail = rng.random(n_cal) < settings["specz_failure"]
            z = z.copy()
            z[fail] = rng.choice(cal.z, fail.sum())
        if settings.get("nmad_factor"):
            gold = gold_masks(
                cal.z,
                cal.photoz,
                cal.group,
                cal.member,
                cal.source,
                cal.source_photoz,
                cal.gold_qc["floor"],
                rng.uniform(*settings["nmad_factor"]),
            )
        v = np.zeros(len(DIRECTIONS))
        if zp:
            colours = rng.multivariate_normal(np.zeros(4), colour_cov)
            v[:5] = band_offsets(colours, rng.normal(0, zp["grey"]))
        if settings.get("z2_offset"):
            v[5] = rng.normal(0, settings["z2_offset"])
        q = np.ones(n_cal)
        if settings.get("bootstrap"):
            q = rng.multinomial(n_cal, np.full(n_cal, 1 / n_cal)).astype(float)
        if density is not None:
            cell = np.clip(np.rint((z - cal.grid[0]) / step), 0, len(cal.grid) - 1)
            q = q * overdensity[n, cell.astype(int)]
        h = cal.calibrate(z, q, gold)
        if v.any():
            h = h + 0.5 * (cal.calibrate(z, q, gold, v) - cal.calibrate(z, q, gold, -v))
        draws.append(h)
    draws = np.array(draws)
    raw = draws.mean(0)
    return draws - raw + cal.fiducial(), raw


def run(directory, output, settings, cosmo=None):
    """Write realisations of the calibration run in ``directory`` to ``output``."""
    cal = Calibration.load(directory)
    density = None
    if settings.get("density"):
        if cosmo is None:
            from ..blinding import cosmology, fiducial

            cosmo = cosmology(fiducial())
        d = settings["density"]
        density = density_covariance(cal.grid, d["field"], d["bias"], cosmo)
    real, raw = realisations(cal, settings, density)
    np.savez_compressed(
        output,
        z=cal.grid,
        realisations=real,
        fiducial=cal.fiducial(),
        raw_mean=raw,
        settings=yaml.safe_dump(settings),
    )
    return real


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m sp_validation.nz.ensemble")
    p.add_argument("directory", help="output directory of a calibration run")
    p.add_argument("-o", "--output", required=True, help="realisations .npz")
    p.add_argument(
        "--settings",
        help="YAML ensemble section; the run's recipe ensemble by default",
    )
    a = p.parse_args(argv)
    settings = a.settings or Path(a.directory) / "recipe.yaml"
    settings = yaml.safe_load(Path(settings).read_text())
    run(a.directory, Path(a.output), settings.get("ensemble", settings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
