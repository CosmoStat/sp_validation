"""Coupled pseudo-C_ell bandpowers of a :class:`~.survey.Survey` and their
covariance: Gaussian (improved narrow-kernel approximation, iNKA) plus
super-sample.

The estimator is the coupled pseudo-spectrum of the mask-weighted fields, E modes
only, divided by the mean of the product of the two masks and binned
(:attr:`Survey.binning`), less the noise bias on autos. Its expectation is a fixed
linear map of the true C_ell, :func:`operators`, B @ M_ab / <W_a W_b>, with M_ab
NaMaster's EE coupling matrix of the two masks; that map is the bandpower window
the data vector carries. No coupling matrix is inverted, so every one of the
45 spectra of a 3x2pt vector, with spins 0 and 2 mixed, has its window and
covariance in closed form from the mask couplings.

Gaussian covariance (iNKA, Garcia-Garcia et al. 2019): for bands of spectra ab and
cd,

    Cov = [T(ac, bd) + T(ad, bc)] / (<W_a W_b> <W_c W_d>),
    T(ac, bd) = 1/2 B [Xi(W_a W_c, W_b W_d) * (u v^T + v u^T)] B^T,

with u, v the coupled spectra (plus noise) of ac and bd at each ell, divided by
their mask means, and Xi the coupling matrix of the two mask products over
(2l+1) (EE for two shear legs, spin-0 otherwise). White shape noise needs no E-B
term: its pseudo-E covariance is the ++ coupling of the squared mask.

Super-sample covariance: pyccl's ``angular_cl_cov_SSC`` with a halo-model
response (``halomod_Tk3D_SSC_linear_bias``, linear lens bias and the number-counts
counter-term) and sigma_B^2 of the window products of each pair, on log-spaced
multipole nodes, interpolated in ln ell and projected with :func:`operators`.
"""

import multiprocessing as mp
import os
from pathlib import Path

import numpy as np


def _spin(field):
    return 2 if field[0] == "l" else 0


class Couplings:
    """NaMaster EE coupling matrices of mask products, cached as npz in ``cache``.

    ``masks``: {"g": lens map, "l": source map}. A window is a tuple of those keys
    (``("g", "l")`` is their product)."""

    def __init__(self, masks, lmax, cache):
        self.masks, self.lmax = masks, lmax
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self._fields = {}

    def field(self, window, spin):
        import pymaster as nmt

        key = (tuple(sorted(window)), spin)
        if key not in self._fields:
            m = np.prod([self.masks[w] for w in key[0]], axis=0)
            nside = int(np.sqrt(len(m) / 12))
            self._fields[key] = nmt.NmtField(
                m,
                None,
                spin=spin,
                lmax=self.lmax,
                lmax_mask=min(2 * self.lmax, 3 * nside - 1),
                n_iter=0,
            )
        return self._fields[key]

    def __call__(self, wa, wb, sa, sb):
        """(EE <- EE, EE <- BB or None) coupling of windows ``wa``, ``wb`` at spins
        ``sa``, ``sb``; the second only for two spin-2 fields."""
        import pymaster as nmt

        wa, wb = tuple(sorted(wa)), tuple(sorted(wb))
        if (wb, sb) < (wa, sa):
            wa, wb, sa, sb = wb, wa, sb, sa
        path = self.cache / f"{''.join(wa)}_{''.join(wb)}_{sa}{sb}.npz"
        if not path.exists():
            w = nmt.NmtWorkspace()
            b = nmt.NmtBin.from_lmax_linear(self.lmax, self.lmax // 4)
            w.compute_coupling_matrix(self.field(wa, sa), self.field(wb, sb), b)
            m = w.get_coupling_matrix()
            n = m.shape[0] // (self.lmax + 1)
            out = dict(ee=m[0::n, 0::n].astype(np.float32))
            if n == 4:
                out["bb"] = m[0::4, 3::4].astype(np.float32)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **out)
            os.replace(tmp, path)
        d = np.load(path)
        return d["ee"], d["bb"] if "bb" in d else None


def mask_means(masks):
    """<W_x W_y> for every pair of mask keys."""
    return {(x, y): float(np.mean(masks[x] * masks[y])) for x in masks for y in masks}


def noise(survey, masks):
    """Noise bias of each field's estimator: the white-noise power on the
    weighted sky, <W>/<W^2> / density (times sigma_e^2 for shear)."""
    means = mask_means(masks)
    out = {}
    for f in survey.fields:
        w = f[0]
        ratio = float(np.mean(masks[w])) / means[w, w]
        out[f] = ratio * survey.variance(f) / survey.density(f)
    return out


def operators(survey, masks, couplings):
    """(n_pair, n_band, lmax + 1): the bandpower window of each spectrum."""
    means = mask_means(masks)
    B = survey.binning
    out = []
    for a, b in survey.pairs:
        m = couplings((a[0],), (b[0],), _spin(a), _spin(b))[0]
        out.append(B @ m / means[a[0], b[0]])
    return np.array(out)


def gaussian(survey, masks, cl, couplings, processes=48):
    """The Gaussian (iNKA) covariance of the bands, (n_pair n_band)^2 in pair-major
    order. ``cl``: (n_pair, lmax + 1) true spectra in :attr:`Survey.pairs` order."""
    pairs, B = survey.pairs, survey.binning
    edges, lmax = survey.edges, survey["lmax"]
    means, nb = mask_means(masks), noise(survey, masks)
    ell = np.arange(lmax + 1)
    index = {p: i for i, p in enumerate(pairs)}

    def spectrum(a, b):
        return cl[index[(a, b)] if (a, b) in index else index[(b, a)]]

    # coupled spectra plus noise, over the mask means
    eff = {}
    for a in survey.fields:
        for b in survey.fields:
            m = couplings((a[0],), (b[0],), _spin(a), _spin(b))[0]
            eff[a, b] = m @ spectrum(a, b) / means[a[0], b[0]]
            eff[a, b] = eff[a, b] + (nb[a] if a == b else 0.0)

    def kind(a, c):
        return 2 if _spin(a) == 2 and _spin(c) == 2 else 0

    def key(a, c, b, d):
        x = (tuple(sorted((a[0], c[0]))), kind(a, c))
        y = (tuple(sorted((b[0], d[0]))), kind(b, d))
        return (x, y) if x <= y else (y, x)

    need = set()
    for i, (a, b) in enumerate(pairs):
        for c, d in pairs[i:]:
            need |= {key(a, c, b, d), key(a, d, b, c)}
    lo, hi = int(edges[0]), int(edges[-1])
    starts = np.array(edges[:-1]) - lo
    w = np.concatenate([B[i, edges[i] : edges[i + 1]] for i in range(len(B))])
    xi = {}
    for (p, s), (q, t) in sorted(need):
        ee = couplings(p, q, s, t)[0] / (2 * ell + 1)
        xi[(p, s), (q, t)] = np.asarray(ee[lo:hi, lo:hi], float) * w[:, None] * w

    def rebin(m):
        return np.add.reduceat(np.add.reduceat(m, starts, 0), starts, 1)

    def term(a, c, b, d):
        u, v = eff[a, c][lo:hi], eff[b, d][lo:hi]
        return 0.5 * rebin(xi[key(a, c, b, d)] * (u[:, None] * v + v[:, None] * u))

    def row(i):
        a, b = pairs[i]
        return [
            (term(a, c, b, d) + term(a, d, b, c))
            / (means[a[0], b[0]] * means[c[0], d[0]])
            for c, d in pairs[i:]
        ]

    n, k = len(pairs), len(B)
    cov = np.zeros((n, k, n, k))
    global _row
    _row = row
    with mp.get_context("fork").Pool(processes) as pool:
        for i, blocks in enumerate(pool.imap(_call_row, range(n))):
            for j, m in enumerate(blocks):
                cov[i, :, i + j, :] = m
                cov[i + j, :, i, :] = m.T
    return cov.reshape(n * k, n * k)


_row = None


def _call_row(i):
    return _row(i)


def super_sample(survey, masks, op, processes=24):
    """The super-sample covariance of the bands (layout of :func:`gaussian`);
    ``op`` from :func:`operators`."""
    import healpy as hp
    import pyccl as ccl
    from scipy.interpolate import CubicSpline

    c = survey["cosmology"]
    cosmo = ccl.Cosmology(
        Omega_c=c["omch2"] / c["h"] ** 2,
        Omega_b=c["ombh2"] / c["h"] ** 2,
        h=c["h"],
        n_s=c["n_s"],
        A_s=1e-10 * np.exp(c["logA"]),
        m_nu=c["m_nu"],
        mass_split="single",
        transfer_function="boltzmann_camb",
        matter_power_spectrum="camb",
        extra_parameters=dict(
            camb=dict(
                halofit_version="mead2020_feedback",
                HMCode_logT_AGN=c["HMCode_logT_AGN"],
                kmax=50,
            )
        ),
    )
    # tracers: unit bias (bias is in the response), NLA (pyccl's sign), no
    # magnification
    tracers, bias = {}, {}
    for f in survey.fields:
        nz, z = survey.nz[f], survey.z
        if f[0] == "g":
            tracers[f] = ccl.NumberCountsTracer(
                cosmo, has_rsd=False, dndz=(z, nz), bias=(z, np.ones_like(z))
            )
            bias[f] = survey["lenses"]["bias"][int(f[1:])]
        else:
            ia = (z, c["A_IA"] * np.ones_like(z))
            tracers[f] = ccl.WeakLensingTracer(cosmo, dndz=(z, nz), ia_bias=ia)
            bias[f] = 1.0

    # response pieces on an (a, k) grid: R0 = (47/21 - dlnP/dlnk / 3) P_L + I_1^2,
    # P_hm = P_L + I_0^2 (pyccl's halomod_Tk3D_SSC_linear_bias, term by term)
    a_arr = np.linspace(1 / (1 + 4.0), 1.0, 48)
    lk = np.linspace(np.log(1e-4), np.log(1e2), 160)
    k = np.exp(lk)
    md = ccl.halos.MassDef200m
    hmc = ccl.halos.HMCalculator(
        mass_function=ccl.halos.MassFuncTinker08(mass_def=md),
        halo_bias=ccl.halos.HaloBiasTinker10(mass_def=md),
        mass_def=md,
    )
    prof = ccl.halos.HaloProfileNFW(
        mass_def=md,
        concentration=ccl.halos.ConcentrationDuffy08(mass_def=md),
        fourier_analytic=True,
    )
    pl, p2 = cosmo.get_linear_power(), ccl.halos.Profile2pt()
    r0, phm = np.zeros((len(a_arr), len(k))), np.zeros((len(a_arr), len(k)))
    for i, a in enumerate(a_arr):
        norm = prof.get_normalization(cosmo, a, hmc=hmc) ** 2
        pk = pl(k, a)
        i1 = hmc.I_1_2(cosmo, k, a, prof, prof2=prof, prof_2pt=p2)
        i0 = hmc.I_0_2(cosmo, k, a, prof, prof2=prof, prof_2pt=p2)
        r0[i] = (47 / 21 - pl(k, a, derivative=True) / 3) * pk + i1 / norm
        phm[i] = pk + i0 / norm

    def response(a, b):
        counter = (a[0] == "g") * bias[a] + (b[0] == "g") * bias[b]
        return bias[a] * bias[b] * (r0 - counter * phm)

    # sigma_B^2 of each pair's window, the product of its two masks
    nside = survey["nside"]
    products = ("gg", "gl", "ll")
    alm = {w: hp.map2alm(masks[w[0]] * masks[w[1]], iter=0) for w in products}
    area = {w: 4 * np.pi * np.mean(masks[w[0]] * masks[w[1]]) for w in products}
    a_s2 = np.linspace(1 / (1 + 4.5), 1.0, 96)
    sigma2 = {}
    for u in products:
        for v in products:
            wl = (2 * np.arange(3 * nside) + 1) * hp.alm2cl(alm[u], alm[v])
            wl = wl / (area[u] * area[v])
            sigma2[u, v] = (a_s2, ccl.sigma2_B_from_mask(cosmo, a_s2, mask_wl=wl))

    # multipole nodes, and the spline of l(l+1) l'(l'+1) Cov in ln l to every ell
    lmax = survey["lmax"]
    nodes = np.geomspace(2, lmax, 36)
    ell = np.arange(lmax + 1.0)
    interp = np.zeros((lmax + 1, len(nodes)))
    for n in range(len(nodes)):
        spline = CubicSpline(np.log(nodes), np.eye(len(nodes))[n])
        interp[2:, n] = (
            spline(np.log(ell[2:]))
            * nodes[n]
            * (nodes[n] + 1)
            / (ell[2:] * (ell[2:] + 1))
        )
    opi = np.einsum("ibl,ln->ibn", op, interp)
    pairs = survey.pairs

    def window(a, b):
        return "".join(sorted((a[0], b[0])))

    def block(ij):
        i, j = ij
        (a, b), (c, d) = pairs[i], pairs[j]
        tk = ccl.Tk3D(
            a_arr=a_arr,
            lk_arr=lk,
            pk1_arr=response(a, b),
            pk2_arr=response(c, d),
            is_logt=False,
        )
        cov = ccl.angular_cl_cov_SSC(
            cosmo,
            tracers[a],
            tracers[b],
            tracer3=tracers[c],
            tracer4=tracers[d],
            ell=nodes,
            t_of_kk_a=tk,
            sigma2_B=sigma2[window(a, b), window(c, d)],
        ).T
        return i, j, opi[i] @ cov @ opi[j].T

    n, nb = len(pairs), op.shape[1]
    out = np.zeros((n, nb, n, nb))
    global _block
    _block = block
    todo = [(i, j) for i in range(n) for j in range(i, n)]
    with mp.get_context("fork").Pool(processes) as pool:
        for i, j, m in pool.imap_unordered(_call_block, todo, chunksize=4):
            out[i, :, j, :] = m
            out[j, :, i, :] = m.T
    out = out.reshape(n * nb, n * nb)
    return 0.5 * (out + out.T)


_block = None


def _call_block(ij):
    return _block(ij)
