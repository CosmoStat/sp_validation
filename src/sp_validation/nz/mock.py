"""Mock galaxies with known redshifts, for the SOM n(z) simulation ladder.

A galaxy is a redshift ``z``, an SED type ``t`` in [0, 1] (0 red, 1 blue)
and a true r magnitude. Its ugriz magnitudes are a smooth function of
(z, t) (:func:`sed`), seen through flux noise that grows with magnitude; its
BPZ is a surrogate (:func:`bpz`), not a fit to the noisy photometry. Sources
and calibrators come from the same :func:`population`, so on every rung the
truth is the sources' own shear-weighted redshifts.

Rungs (:func:`draw`), one change each:

- **0, representative.** The calibrators are a random draw from the source
  population with the same noise. Prediction: Δ⟨z⟩ = ⟨z⟩_calibrated −
  ⟨z⟩_true of the ``TOM_BIN_ID`` sources is consistent with zero in every
  bin, scattering by the calibrators' shot noise, which the bootstrap term
  of the ensemble reproduces (σ_boot / scatter ≈ 1).
  Found (40 realisations at the v2 recipe): σ_boot / scatter 0.84–1.27, but
  Δ⟨z⟩ is −0.0007 to −0.0021 (2–7 standard errors), because the spectra that
  calibrate also decide gold (bins 1–3) and train the SOM (bin 4).
- **1, magnitude-selected spectra.** Calibrators are kept with probability
  falling smoothly through r = 23.5 (DEEP2-like), a selection on r alone.
  Prediction: the raw calibrator n(z) of each BPZ bin is biased low, so the
  naive estimator (unweighted mean spec-z of the in-bin calibrators) fails;
  direct calibration still recovers ⟨z⟩ because r is a SOM feature, up to
  the residual dependence of z on r inside a group, small against the scatter.
  Found: that residual is about −0.002 in bins 1–3, as large as the scatter,
  and grows as the groups coarsen.
"""

import numpy as np
from scipy.special import expit

#: Band centres (nm), u g r i z.
WAVELENGTH = np.array([355.0, 475.0, 620.0, 765.0, 900.0])
#: 10σ galaxy magnitude per band, roughly UNIONS (CFIS u, r; HSC g, z; PS1 i).
DEPTH = np.array([23.0, 23.8, 24.0, 23.2, 22.6])
#: The recipe's z2 -> z constant: z2 = z − offset.
Z2_OFFSET = -0.1104888916015625


def redshift(n, rng, z0=0.23):
    """Smail n(z) ∝ z² exp(−z/z0): mean 3 z0 ≈ 0.7, ~4% above z = 1.5."""
    return z0 * rng.gamma(3.0, size=n)


def sed(z, t):
    """Magnitudes (n, 5) relative to r of type ``t`` at redshift ``z``.

    Two breaks move through the bands with (1 + z): the 4000 Å break,
    strong for red types, and the Lyman break, which a u dropout at z ≳ 2
    makes look like a 4000 Å break at z ≈ 0.3 (1 + z) − 1 (the degeneracy
    :func:`bpz` aliases). Above z ≈ 1.4 the 4000 Å break has left the z band
    and the colours stop tracking z. A continuum slope sets the colour of
    each type away from the breaks.
    """
    z, t = np.asarray(z)[:, None], np.asarray(t)[:, None]
    balmer = (1.3 - 1.0 * t) * expit((400 * (1 + z) - WAVELENGTH) / 60)
    lyman = 2.5 * expit((121.6 * (1 + z) - WAVELENGTH) / 30)
    slope = (0.25 - 0.2 * t) * (620 - WAVELENGTH) / 100
    m = balmer + lyman + slope
    return m - m[:, 2:3]


def bpz(z, r, rng):
    """BPZ surrogate: z + scatter growing with r, outliers, quantised to 0.01.

    Gaussian scatter (0.03 + 0.03 × 10^0.4(r−24)) (1 + z): 0.035 (1 + z) at
    r = 22, 0.1 (1 + z) at r = 25. Catastrophic outliers, 2% + 2% × 10^0.4(r−24)
    of galaxies, land uniformly in [0, 1.6]; half of galaxies above z = 1.8
    take the Lyman/4000 Å alias of :func:`sed` instead.
    """
    faint = 10 ** (0.4 * (r - 24))
    p = z + (0.03 + 0.03 * faint) * (1 + z) * rng.standard_normal(len(z))
    out = rng.random(len(z)) < 0.02 + 0.02 * faint
    p[out] = rng.uniform(0, 1.6, out.sum())
    alias = (z > 1.8) & (rng.random(len(z)) < 0.5)
    p[alias] = 0.304 * (1 + z[alias]) - 1 + 0.03 * rng.standard_normal(alias.sum())
    return np.round(np.clip(p, 0, None), 2)


def observe(m, rng):
    """Noisy magnitudes: flux noise of 1/10 the 10σ-depth flux, 1% floor.

    A non-positive flux is undetected (NaN), as a GAaP non-detection is.
    """
    flux = 10 ** (-0.4 * (m - DEPTH))
    flux = flux * (1 + 0.01 * rng.standard_normal(m.shape))
    flux += 0.1 * rng.standard_normal(m.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(flux > 0, DEPTH - 2.5 * np.log10(flux), np.nan)


def population(n, rng, r_max=25.0):
    """``n`` galaxies of the shear sample (r < ``r_max``, z < 4), with truth.

    r scatters by 0.8 around 22 + 3 z / (z + 0.6), so fainter galaxies are
    farther; the type is uniform and independent of z and r. Each band gets
    0.03 mag of intrinsic SED scatter. The z2 band is a second, independent
    observation of z shifted by −:data:`Z2_OFFSET`. Shear weights fall with
    r as shape-measurement noise 0.15 × 10^0.4(r−24) joins σ_e = 0.26.
    """
    z = np.empty(0)
    r = np.empty(0)
    while len(z) < n:
        zz = redshift(2 * n, rng)
        rr = 22 + 3 * zz / (zz + 0.6) + 0.8 * rng.standard_normal(2 * n)
        keep = (rr < r_max) & (zz < 4)  # the output grid ends at z = 5
        z, r = np.r_[z, zz[keep]], np.r_[r, rr[keep]]
    z, r = z[:n], r[:n]
    t = rng.random(n)
    true = r[:, None] + sed(z, t) + 0.03 * rng.standard_normal((n, 5))
    mag = observe(true, rng)
    z2 = observe(true[:, 4:5], rng)[:, 0] - Z2_OFFSET
    sigma_m = 0.15 * 10 ** (0.4 * (r - 24))
    return dict(
        z=z,
        t=t,
        r=r,
        mag=np.column_stack([mag, z2]),
        photoz=bpz(z, r, rng),
        weight=1 / (0.26**2 + sigma_m**2),
    )


def spectroscopic(r, rung):
    """Probability that a galaxy of true r magnitude gets a spectrum."""
    if rung == 0:
        return np.ones_like(r)
    return expit((23.5 - r) / 0.3)


def draw(rung, seed, n_cal, n_src, z2_fraction=0.2):
    """Calibrators, sources and the sources' true redshifts for one realisation.

    The sources depend on ``seed`` only, so both rungs of a seed calibrate
    the same sources. A fraction ``z2_fraction`` of sources has z undetected
    and is measured through z2. Calibrators keep complete ugriz and BPZ, as
    :func:`.inputs.calibrators` does.
    """
    rng = np.random.default_rng(seed)
    s = population(n_src, rng)
    s["mag"][rng.random(n_src) < z2_fraction, 4] = np.nan
    src = dict(
        mag=s["mag"],
        photoz=s["photoz"],
        weight=s["weight"],
        status=np.zeros(n_src, np.int8),
        object_id=np.arange(n_src),
    )
    cal_rng = np.random.default_rng([seed, rung])
    parts, size = [], 0
    while size < n_cal:
        c = population(4 * n_cal, cal_rng)
        mag = c["mag"][:, :5]
        keep = (cal_rng.random(len(c["z"])) < spectroscopic(c["r"], rung)) & (
            np.isfinite(mag).all(1)
        )
        parts.append(dict(z=c["z"][keep], photoz=c["photoz"][keep], mag=mag[keep]))
        size += keep.sum()
    cal = {k: np.concatenate([p[k] for p in parts])[:n_cal] for k in parts[0]}
    return cal, src, s["z"]
