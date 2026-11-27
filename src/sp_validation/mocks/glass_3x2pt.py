"""GLASS 3x2pt mocks of a harmonic survey (:class:`sp_validation.cloe.survey.Survey`).

Lognormal matter shells, every pair of shells correlated, combine linearly into
every field: the lens density contrast of each lens bin (linear bias times the
shells it spans, plus magnification by the shells in front) and the convergence of
each source bin (lensing through the shells in front, plus NLA). Lenses are Poisson counts at the survey's density,
mean ``b (1 + delta)`` clipped at zero, inside the lens footprint; sources are E-mode
shear from the convergence with Gaussian shape noise inside the source footprint.
Each mock is measured with the forecast's estimator (:mod:`sp_validation.cloe.covariance`:
coupled pseudo-C_ell over the mean mask product, binned, noise bias subtracted).

Three things the model of a mock must match (pando
``unions/first-3x2pt-chain-run-the/glass-lens-clustering-offsets``), each at the
percent level:

- CAMB's number-count shells are cold matter (CDM + baryons) while the theory's
  spectra are total matter: the shell spectra are rescaled by sqrt[r(k_i) r(k_j)],
  r = P_mm / P_cb at k = (l + 1/2) / chi(z_eff);
- the theory regrids n(z) with an Akima spline, so the mock does too;
- for thin overlapping shells the lognormal Gaussian-space spectra are not
  positive definite at some ell; their negative eigenvalues are clipped, and the
  mock's expectation is the spectra the clipped fields realise.

Steps, in the forecast's output directory ``OUT`` (it holds ``masks.npz``,
``gaussian.npz`` and ``forecast.sacc``)::

    python -m sp_validation.mocks.glass_3x2pt shells RECIPE OUT   # glass/shells.npz
    python -m sp_validation.mocks.glass_3x2pt gls    RECIPE OUT   # glass/gls.npz
    python -m sp_validation.mocks.glass_3x2pt mock   RECIPE OUT SEED

``mock`` writes ``glass/seed_SEED.sacc``, the forecast SACC with the mock's bands
(covariance, windows and n(z) unchanged), and ``glass/seed_SEED.npz`` with the
bands with and without noise.

This mock calls GLASS directly rather than :mod:`sp_validation.mocks.glass`:
that module's configuration is the shear-only Paper II mock (sigma8-normalised
Planck18, 200 Mpc shells, convergence accumulated shell by shell), while narrow
lens bins need shells 0.01 wide in z and every tracer is a fixed linear
combination of the shells (:func:`mixing`). Its shell spectra are the same
``glass.ext.camb.matter_cls`` that module wraps, and its GLASS cosmology is that
module's :func:`~sp_validation.mocks.glass.Cosmology_from_camb`.
"""

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .glass import Cosmology_from_camb

#: Triangular-window nodes of the shells: 0.01 wide to z = 0.75, 0.05 to 2.1.
ZNODES = np.r_[np.arange(0.0, 0.75, 0.01), np.arange(0.75, 2.10001, 0.05)]
#: Redshifts on which n(z) are regridded (Akima) and partitioned into shells.
Z_FINE = np.linspace(0.002, 2.1, 2500)


def camb_params(survey):
    """CAMB at the survey's input cosmology, HMcode-2020 with feedback."""
    import camb

    c = survey["cosmology"]
    return camb.set_params(
        H0=100 * c["h"],
        ombh2=c["ombh2"],
        omch2=c["omch2"],
        mnu=c["m_nu"],
        num_massive_neutrinos=1,
        As=1e-10 * np.exp(c["logA"]),
        ns=c["n_s"],
        halofit_version="mead2020_feedback",
        HMCode_logT_AGN=c["HMCode_logT_AGN"],
        NonLinear=camb.model.NonLinear_both,
        WantTransfer=True,
        kmax=50.0,
        redshifts=list(np.linspace(0, 3, 61)[::-1]),
    )


def shells():
    import glass

    return glass.linear_windows(ZNODES, dz=1e-3)


def spin_factor(lmax):
    """sqrt[(l+2)(l-1) / (l(l+1))]: shear E from convergence."""
    ell = np.arange(lmax + 1.0)
    return np.sqrt(
        np.clip((ell + 2) * (ell - 1), 0, None) / np.clip(ell * (ell + 1), 1, None)
    )


def mixing(survey, params, windows):
    """(n_field, n_shell): each field as a combination of the shell density
    contrasts: lens density b X + (5s - 2) X W (linear bias and magnification),
    source convergence X W + X F(z) (X the n(z) partition into shells, W GLASS's
    multi-plane lensing matrix, F the NLA factor with its (1 + z) power about the
    probe's pivot). The shells must reach the top of every n(z)."""
    import camb
    import glass
    from scipy.interpolate import Akima1DInterpolator

    cosmo = Cosmology_from_camb(params)
    zeff = np.array([w.zeff for w in windows])
    lin = camb.get_matter_power_interpolator(
        params, nonlinear=False, hubble_units=False, k_hunit=False, zmax=3, kmax=10
    )

    def growth(z):
        p = lin.P(np.atleast_1d(z), [1e-4]).ravel()
        return np.sqrt(p / lin.P([0.0], [1e-4]).ravel())

    from sp_validation.cloe.theory.cloe import C1_RHO_CRIT

    c, pivot = survey["cosmology"], survey["probe"]["ia_z_pivot"]
    nla = -c["A_IA"] * C1_RHO_CRIT * float(cosmo.Omega_m0) / growth(zeff)
    nla = nla * ((1 + zeff) / (1 + pivot)) ** c["eta_IA"]
    lensing = np.asarray(glass.multi_plane_matrix(windows, cosmo))
    magnification = 5 * survey["lenses"]["mag_slope"] - 2
    out = np.zeros((len(survey.fields), len(windows)))
    for a, f in enumerate(survey.fields):
        if np.any(survey.nz[f][survey.z > ZNODES[-1]] > 0):
            raise ValueError(
                f"n(z) of {f} reaches above the top shell, z = {ZNODES[-1]}"
            )
        n = np.nan_to_num(Akima1DInterpolator(survey.z, survey.nz[f])(Z_FINE))
        n[(Z_FINE < survey.z[0]) | (Z_FINE > survey.z[-1])] = 0
        x = np.asarray(glass.partition(Z_FINE, n / np.trapezoid(n, Z_FINE), windows))
        x = x / x.sum()
        if f[0] == "g":
            out[a] = survey["lenses"]["bias"][int(f[1:])] * x
            out[a] = out[a] + magnification * (x @ lensing)
        else:
            out[a] = x @ lensing + x * nla
    return out, zeff


def shell_spectra(survey, params, windows, zeff):
    """CAMB's shell spectra (GLASS order), rescaled from cold to total matter."""
    import camb
    import glass.ext.camb

    lmax = survey["lmax"]
    cls = np.array(glass.ext.camb.matter_cls(params, lmax, windows, limber=True))
    kw = dict(nonlinear=True, hubble_units=False, k_hunit=False, zmax=3, kmax=50)
    kw["extrap_kmax"] = 1e4
    tot = camb.get_matter_power_interpolator(
        params, var1="delta_tot", var2="delta_tot", **kw
    )
    cold = camb.get_matter_power_interpolator(
        params, var1="delta_nonu", var2="delta_nonu", **kw
    )
    chi = camb.get_background(params).comoving_radial_distance(zeff)
    r = np.ones((lmax + 1, len(windows)))
    for s in range(len(windows)):
        k = (np.arange(1, lmax + 1) + 0.5) / chi[s]
        r[1:, s] = (tot.P([zeff[s]], k) / cold.P([zeff[s]], k)).ravel()
    r[0] = r[1]
    k = 0
    for i in range(len(windows)):
        for j in range(i, -1, -1):
            cls[k] = cls[k] * np.sqrt(r[:, i] * r[:, j])
            k += 1
    return cls


def field_spectra(survey, cls, mix):
    """(n_field, n_field, lmax + 1) spectra of the fields from shell spectra
    ``cls`` (GLASS order), shear legs with their spin factor."""
    import glass

    lmax, n = survey["lmax"], mix.shape[1]
    shell = np.zeros((lmax + 1, n, n))
    for k, (i, j, _) in enumerate(glass.fields.enumerate_spectra(list(cls))):
        shell[:, i, j] = shell[:, j, i] = cls[k][: lmax + 1]
    out = np.einsum("as,lst,bt->abl", mix, shell, mix, optimize=True)
    spin = np.array(
        [spin_factor(lmax) if f[0] == "l" else np.ones(lmax + 1) for f in survey.fields]
    )
    return out * spin[:, None] * spin[None]


def bands(survey, cl, op):
    """Bands of field spectra ``cl`` (n_field, n_field, lmax + 1) through ``op``."""
    index = {f: i for i, f in enumerate(survey.fields)}
    return np.array([o @ cl[index[a], index[b]] for o, (a, b) in zip(op, survey.pairs)])


# ---- the Gaussian-space spectra of the lognormal shells -------------------- #
_cls = None


def _solve(k):
    import glass

    t = glass.grf.Lognormal(1.0)
    cl = _cls[k]
    gl, _, info = glass.grf.solve(
        cl, t, t, pad=2 * len(cl), monopole=0.0 if cl[0] == 0 else None
    )
    return gl


def _realise(gl):
    import glass
    from transformcl import cltocorr, corrtocl

    t = glass.grf.Lognormal(1.0)
    corr = glass.grf.corr(t, t, cltocorr(np.pad(gl, (0, 2 * len(gl)))))
    return corrtocl(corr)[: len(gl)]


def gaussian_spectra(cls, processes):
    """The lognormal shells' Gaussian spectra, negative eigenvalues clipped per
    ell where they are not positive definite, and the spectra they realise."""
    import glass

    global _cls
    _cls = cls
    with mp.get_context("fork").Pool(processes) as pool:
        gls = np.array(pool.map(_solve, range(len(cls)), chunksize=8))
    n = glass.fields.nfields_from_nspectra(len(cls))
    try:
        for _ in glass.fields.iternorm(
            glass.fields.cls2cov(list(gls), gls.shape[1], n, n - 1)
        ):
            pass
    except ValueError:
        gls = np.array(glass.regularized_spectra(list(gls), method="clip"))
    with mp.get_context("fork").Pool(processes) as pool:
        realised = np.array(pool.map(_realise, list(gls), chunksize=8))
    return gls, realised


# ---- one mock -------------------------------------------------------------- #
def mock(survey, gls, mix, masks, op, noise, seed):
    """The bands (n_pair, n_band) of one mock, with and without noise."""
    import glass
    import healpy as hp
    import pymaster as nmt

    rng = np.random.default_rng(seed)
    nside, lmax = survey["nside"], survey["lmax"]
    nf, ns = mix.shape
    fields = np.zeros((nf, hp.nside2npix(nside)))
    lognormal = [glass.grf.Lognormal(1.0)] * ns
    for s, delta in enumerate(glass.generate(lognormal, list(gls), nside, rng=rng)):
        for a in np.flatnonzero(mix[:, s]):
            fields[a] += mix[a, s] * delta
    pix = hp.nside2pixarea(nside)
    maps = {}
    for a, f in enumerate(survey.fields):
        w = masks[f[0]]
        inside = w > 0
        if f[0] == "g":
            nbar = survey.density(f) * pix
            mu = np.clip(1 + fields[a], 0, None)
            count = rng.poisson(nbar * w * mu)
            noisy = np.zeros_like(w)
            noisy[inside] = count[inside] / (nbar * w[inside]) - 1
            maps[f] = dict(noisy=[noisy], signal=[np.where(inside, mu - 1, 0.0)])
        else:
            kappa = hp.map2alm(fields[a], lmax=lmax, iter=3)
            e = hp.almxfl(kappa, spin_factor(lmax))
            q, u = hp.alm2map_spin([e, 0 * e], nside, 2, lmax)
            sd = np.zeros_like(w)
            sd[inside] = np.sqrt(
                survey.variance(f) / (survey.density(f) * pix * w[inside])
            )
            noisy = [x + sd * rng.standard_normal(len(w)) for x in (q, u)]
            maps[f] = dict(noisy=noisy, signal=[q, u])
    del fields
    means = {(x, y): float(np.mean(masks[x] * masks[y])) for x in masks for y in masks}
    lm = min(2 * lmax, 3 * nside - 1)
    B = survey.binning
    out = {}
    for kind in ("noisy", "signal"):
        nmt_fields = {
            f: nmt.NmtField(
                masks[f[0]],
                maps[f][kind],
                spin=2 if f[0] == "l" else 0,
                lmax=lmax,
                lmax_mask=lm,
            )
            for f in survey.fields
        }
        x = []
        for a, b in survey.pairs:
            cl = nmt.compute_coupled_cell(nmt_fields[a], nmt_fields[b])[0]
            bias = noise[a] if (a == b and kind == "noisy") else 0.0
            x.append(B @ cl[: lmax + 1] / means[a[0], b[0]] - bias)
        out[kind] = np.array(x)
    return out["noisy"], out["signal"]


def main(argv=None):
    from sp_validation import sacc_io
    from sp_validation.blinding import NONE
    from sp_validation.cloe.survey import Survey

    ap = argparse.ArgumentParser(prog="python -m sp_validation.mocks.glass_3x2pt")
    ap.add_argument("step", choices=("shells", "gls", "mock"))
    ap.add_argument("recipe")
    ap.add_argument("out", type=Path)
    ap.add_argument("seed", nargs="?", type=int)
    ap.add_argument("--processes", type=int, default=96)
    a = ap.parse_args(argv)
    survey, out = Survey.load(a.recipe), a.out / "glass"
    out.mkdir(parents=True, exist_ok=True)
    op = np.load(a.out / "gaussian.npz")["op"]
    if a.step == "shells":
        params, windows = camb_params(survey), shells()
        mix, zeff = mixing(survey, params, windows)
        cls = shell_spectra(survey, params, windows, zeff)
        np.savez(out / "shells.npz", cls=cls, mixing=mix, zeff=zeff)
    elif a.step == "gls":
        d = np.load(out / "shells.npz")
        gls, realised = gaussian_spectra(d["cls"], a.processes)
        cl = field_spectra(survey, realised, d["mixing"])
        np.savez(out / "gls.npz", gls=gls, expectation=bands(survey, cl, op))
    else:
        start = time.time()
        masks = dict(np.load(a.out / "masks.npz"))
        noise = dict(zip(survey.fields, np.load(a.out / "gaussian.npz")["noise"]))
        gls = np.load(out / "gls.npz")["gls"]
        mix = np.load(out / "shells.npz")["mixing"]
        noisy, signal = mock(survey, gls, mix, masks, op, noise, a.seed)
        np.savez(
            out / f"seed_{a.seed:04d}.npz",
            cl=noisy,
            cl_signal=signal,
            wall=time.time() - start,
        )
        # forecast.sacc holds the pairs in survey order, bands in order
        s = sacc_io.load(a.out / "forecast.sacc", blind="none")
        for point, value in zip(s.data, noisy.ravel()):
            point.value = float(value)
        s.metadata["origin"] = f"GLASS 3x2pt mock, seed {a.seed}"
        del s.metadata[sacc_io.STAMP_KEY]
        sacc_io.save(s, out / f"seed_{a.seed:04d}.sacc", blind=NONE)


if __name__ == "__main__":
    main()
