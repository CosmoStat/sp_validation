"""Clustering redshifts of the tomographic bins against spectroscopic tracers.

An independent check of the SOM n(z) (:mod:`sp_validation.nz_pipeline`): the
gold sources of each bin are cross-correlated with a spectroscopic reference
sample cut into thin redshift slices. Run from the recipe's ``clustering``
section:

- ``prepare`` restricts the reference catalogues and the gold sources to their
  common footprint and draws the jackknife patches;
- ``measure`` counts pairs per slice: the reference autocorrelation
  (Landy-Szalay) and its cross-correlation with every bin (D_u D_r / D_u R_r - 1),
  in an annulus fixed in comoving separation, and the same for the gate, where
  a random half of one tracer plays the unknown sample;
- ``analyse`` turns the counts into n(z) b(z) per bin, runs the known-answer
  gate, fits the calibration's n(z) (and the alternative calibrators' of the
  ensemble's ``specz_selection`` term) and writes ``clustering.json``.

Estimator. With w the mean of ξ over the annulus' log-spaced bins, Δz the slice
width and M = (H/c) ⟨Π(r)⟩, Π(r) = ∫ dk k/2π P_nl(k) J0(kr), the thin-slice
Limber kernel of the matter averaged over the same bins:

    b_r² = w_rr Δz / M,    n_u b_u = w_ur / sqrt(w_rr Δz M).

The source bias is unknown, b_u ∝ (1+z)^α, so a bin's n(z) is tested through
its shape inside a core window, with amplitude and α free. χ² uses the
jackknife variances: slices are nearly uncorrelated.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .data_roots import load_yaml

# Comoving separations, Mpc/h: six bins per decade over 0.5-23.
R_EDGES = 0.5 * 10 ** (np.arange(11) / 6)
ALPHAS = np.round(np.arange(-3, 4.001, 0.05), 3)
SHIFTS = np.round(np.arange(-0.2, 0.2001, 0.0025), 4)
SHIFT_ALPHAS = (-1.0, 0.0, 1.0, 2.0)
GATE_SEED = 2026


def fiducial_cosmology():
    from .blinding import cosmology, fiducial

    return cosmology(fiducial())


def slice_edges(lo, hi, dz):
    return np.round(np.arange(lo, hi + dz / 2, dz), 6)


def _data(config):
    return Path(config["output"]) / "data"


# -- prepare ----------------------------------------------------------------


def _read_reference(directory, tracer, kind, n_random):
    """One tracer's clustering catalogue, both caps: RA, DEC, Z, WEIGHT."""
    import fitsio

    names = (
        ["clustering.dat"]
        if kind == "dat"
        else [f"{i}_clustering.ran" for i in range(n_random)]
    )
    return np.concatenate(
        [
            fitsio.read(
                f"{directory}/{tracer}_{cap}_{name}.fits",
                columns=["RA", "DEC", "Z", "WEIGHT"],
            )
            for cap in ("NGC", "SGC")
            for name in names
        ]
    )


def prepare(config):
    """Cut reference and sources to their overlap; draw the jackknife patches.

    Reference points are kept inside the survey footprint (a boolean HEALPix
    map), gold sources inside ``source_nside`` pixels that hold a reference random.
    Patches are k-means clusters of the nside-1024 pixels that hold a random
    and are more than half inside the footprint.
    """
    import fitsio
    import healpy as hp
    import pyarrow.parquet as pq
    import treecorr

    cl = config["clustering"]
    data = _data(cl)
    data.mkdir(parents=True, exist_ok=True)
    foot = hp.read_map(cl["footprint"]).astype(bool)
    nside = hp.npix2nside(len(foot))
    source_nside = cl.get("source_nside", 256)
    coarse = np.zeros(hp.nside2npix(source_nside), bool)
    fine = np.zeros(hp.nside2npix(1024), bool)
    ref = cl["reference"]
    for tracer in ref["slices"]:
        for kind in ("dat", "ran"):
            d = _read_reference(ref["directory"], tracer, kind, ref["randoms"])
            if kind == "ran":
                coarse[hp.ang2pix(source_nside, d["RA"], d["DEC"], lonlat=True)] = True
                fine[hp.ang2pix(1024, d["RA"], d["DEC"], lonlat=True)] = True
            d = d[foot[hp.ang2pix(nside, d["RA"], d["DEC"], lonlat=True)]]
            np.savez(
                data / f"{tracer}_{kind}.npz",
                ra=d["RA"].astype("f8"),
                dec=d["DEC"].astype("f8"),
                z=d["Z"].astype("f8"),
                w=d["WEIGHT"].astype("f8"),
            )

    cat = fitsio.read(
        config["inputs"]["sources"]["catalogue"], ext=1, columns=["RA", "Dec", "w_des"]
    )
    tom = pq.read_table(
        Path(config["output"]) / "tom_bin.parquet", columns=["TOM_BIN_ID"]
    )["TOM_BIN_ID"].to_numpy()
    keep = (tom > 0) & coarse[hp.ang2pix(source_nside, cat["RA"], cat["Dec"], lonlat=True)]
    np.savez(
        data / "sources.npz",
        ra=cat["RA"][keep].astype("f8"),
        dec=cat["Dec"][keep].astype("f8"),
        w=cat["w_des"][keep].astype("f4"),
        bin=tom[keep],
    )

    pix = np.flatnonzero(fine & (hp.ud_grade(foot.astype(float), 1024) > 0.5))
    ra, dec = hp.pix2ang(1024, pix, lonlat=True)
    patches = treecorr.Catalog(
        ra=ra,
        dec=dec,
        ra_units="deg",
        dec_units="deg",
        npatch=cl["patches"],
        kmeans_alt=True,
        rng=np.random.default_rng(1),
    )
    np.savez(data / "patches.npz", centers=patches.patch_centers)
    return dict(
        overlap_deg2=float(len(pix) * hp.nside2pixarea(1024, degrees=True)),
        sources=np.bincount(tom[keep]).tolist()[1:],
    )


# -- measure ----------------------------------------------------------------


def correlate(dat, ran, unknowns, edges, centers, cosmo, threads=1):
    """Auto- and cross-correlations of a reference sample in redshift slices.

    ``dat`` and ``ran`` hold ra, dec, z, w of the reference and its randoms;
    ``unknowns`` maps a name to (ra, dec, w). In each slice the angular bins
    are :data:`R_EDGES` at the slice centre's comoving distance. Returns ``xi``
    (slice, sample, r bin) with sample 0 the autocorrelation, and ``jackknife``,
    the leave-one-patch-out estimates in the same layout, patch first.
    """
    import pyccl
    import treecorr

    def catalog(ra, dec, w):
        return treecorr.Catalog(
            ra=ra, dec=dec, w=w, ra_units="deg", dec_units="deg", patch_centers=centers
        )

    samples = {name: catalog(*radecw) for name, radecw in unknowns.items()}
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        a = 1 / (1 + (lo + hi) / 2)
        chi = pyccl.comoving_radial_distance(cosmo, a) * cosmo["h"]
        theta = np.degrees(R_EDGES / chi) * 60
        kw = dict(
            min_sep=theta[0],
            max_sep=theta[-1],
            nbins=len(theta) - 1,
            sep_units="arcmin",
            var_method="jackknife",
            cross_patch_weight="match",
            num_threads=threads,
        )
        sd = (dat["z"] >= lo) & (dat["z"] < hi)
        sr = (ran["z"] >= lo) & (ran["z"] < hi)
        D = catalog(dat["ra"][sd], dat["dec"][sd], dat["w"][sd])
        R = catalog(ran["ra"][sr], ran["dec"][sr], ran["w"][sr])
        dd, dr, rr = (treecorr.NNCorrelation(**kw) for _ in range(3))
        dd.process(D)
        dr.process(D, R)
        rr.process(R)
        dd.calculateXi(rr=rr, dr=dr)
        row = [dd]
        for U in samples.values():
            ud, ur = treecorr.NNCorrelation(**kw), treecorr.NNCorrelation(**kw)
            ud.process(U, D)
            ur.process(U, R)
            ud.calculateXi(rr=ur)
            row.append(ud)
        rows.append(row)
    design, _ = treecorr.build_multi_cov_design_matrix(
        [c for row in rows for c in row], "jackknife"
    )
    shape = (len(rows), len(samples) + 1, len(R_EDGES) - 1)
    return dict(
        z_edges=edges,
        samples=["auto", *samples],
        xi=np.array([[c.xi for c in row] for row in rows]),
        jackknife=design.reshape(len(design), *shape),
    )


def measure(config, workers=1):
    """Write ``xcorr_<tracer>.npz`` for the source bins and ``xcorr_gate.npz``."""
    cl = config["clustering"]
    data = _data(cl)
    cosmo = fiducial_cosmology()
    centers = np.load(data / "patches.npz")["centers"]
    src = np.load(data / "sources.npz")
    bins = {
        f"bin{k}": tuple(src[c][src["bin"] == k] for c in ("ra", "dec", "w"))
        for k in range(1, src["bin"].max() + 1)
    }
    for tracer, (lo, hi, dz) in cl["reference"]["slices"].items():
        dat, ran = np.load(data / f"{tracer}_dat.npz"), np.load(data / f"{tracer}_ran.npz")
        out = correlate(dat, ran, bins, slice_edges(lo, hi, dz), centers, cosmo, workers)
        np.savez(data / f"xcorr_{tracer}.npz", **out)
        if cl.get("randoms_null", False):
            null_data, null_random = randoms_split(ran)
            out = correlate(
                null_data, null_random, bins, slice_edges(lo, hi, dz),
                centers, cosmo, workers,
            )
            np.savez(data / f"null_{tracer}.npz", **out)

    tracer, lo, hi, dz = cl["gate"]
    dat, ran = dict(np.load(data / f"{tracer}_dat.npz")), np.load(data / f"{tracer}_ran.npz")
    half = _gate_half(len(dat["z"]))
    unknown = {"half": tuple(dat[c][half] for c in ("ra", "dec", "w"))}
    dat = {k: v[~half] for k, v in dat.items()}
    out = correlate(dat, ran, unknown, slice_edges(lo, hi, dz), centers, cosmo, workers)
    np.savez(data / "xcorr_gate.npz", **out)


def randoms_split(ran):
    """Disjoint random-derived data and denominator, retaining Z and WEIGHT.

    A seeded Bernoulli split preserves the angular/redshift selection and weight
    distribution in both halves. Pair-count normalization uses their weight
    sums, just as for the data; reusing the same points would suppress noise.
    """
    half = np.random.default_rng(2027).random(len(ran["z"])) < 0.5
    return ({k: ran[k][half] for k in ran}, {k: ran[k][~half] for k in ran})


def _gate_half(n):
    return np.random.default_rng(GATE_SEED).random(n) < 0.5


# -- estimator --------------------------------------------------------------


def kernel(cosmo, z, r_lo, r_hi):
    """M = (H/c) Π(r), averaged uniformly in log r over [r_lo, r_hi] Mpc/h."""
    import pyccl
    from scipy.special import j0

    h = cosmo["h"]
    k = np.logspace(-4, 2, 6000)  # 1/Mpc
    r = np.logspace(np.log10(r_lo), np.log10(r_hi), 64) / h  # Mpc
    a = 1 / (1 + z)
    pk = pyccl.nonlin_matter_power(cosmo, k, a)
    pi = np.trapezoid(k**2 * pk * j0(k * r[:, None]) / (2 * np.pi), np.log(k), axis=1)
    return pyccl.h_over_h0(cosmo, a) * h / 2997.92458 * pi.mean()


def jackknife_covariance(rows):
    """Covariance of a (patch, n) array of leave-one-out estimates."""
    n = len(rows)
    d = rows - rows.mean(0)
    return (n - 1) / n * d.T @ d


class Measurement:
    """Annulus averages of one :func:`correlate` output, with jackknife rows."""

    def __init__(self, xcorr, annulus, cosmo):
        sel = (R_EDGES[:-1] >= annulus[0] * 0.99) & (R_EDGES[1:] <= annulus[1] * 1.01)
        r = R_EDGES[:-1][sel][0], R_EDGES[1:][sel][-1]
        self.edges = xcorr["z_edges"]
        self.z = (self.edges[:-1] + self.edges[1:]) / 2
        self.dz = np.diff(self.edges)
        self.samples = list(xcorr["samples"])
        self.w = xcorr["xi"][..., sel].mean(-1)  # (slice, sample)
        self.jk = xcorr["jackknife"][..., sel].mean(-1)  # (patch, slice, sample)
        self.M = np.array([kernel(cosmo, z, *r) for z in self.z])

    def bias(self, jk=False):
        """Reference bias per slice."""
        w = self.jk[..., 0] if jk else self.w[:, 0]
        return np.sqrt(np.clip(w * self.dz / self.M, 1e-6, None))

    def nb(self, sample, jk=False):
        """n_u b_u per slice, per unit redshift."""
        s = self.samples.index(sample)
        w, auto = (self.jk[..., s], self.jk[..., 0]) if jk else (self.w[:, s], self.w[:, 0])
        return w / np.sqrt(np.clip(auto, 1e-6, None) * self.dz * self.M)


def slice_average(grid, nz, lo, hi, shift=0.0):
    """Mean over each slice [lo, hi) of a gridded n(z) moved by ``shift``."""
    fine = np.linspace(grid[0], grid[-1], 50001)
    f = np.interp(fine - shift, grid, nz, left=0, right=0)
    c = np.r_[0, np.cumsum((f[1:] + f[:-1]) / 2 * np.diff(fine))]
    return (np.interp(hi, fine, c) - np.interp(lo, fine, c)) / (hi - lo)


def amplitude_fit(y, ivar, model):
    """χ² of y = A model at the best amplitude."""
    A = (model * ivar * y).sum() / (model * ivar * model).sum()
    return float(((y - A * model) ** 2 * ivar).sum())


def shift_fit(y, ivar, model_at):
    """Shift minimising χ² of ``model_at(shift)``, with its Δχ² = 1 interval."""
    prof = np.array([amplitude_fit(y, ivar, model_at(s)) for s in SHIFTS])
    i = int(np.argmin(prof))
    inside = SHIFTS[prof <= prof[i] + 1]
    return dict(
        best=float(SHIFTS[i]),
        lo=float(inside.min()),
        hi=float(inside.max()),
        chi2=float(prof[i]),
    )


# -- analyse ----------------------------------------------------------------


def gate(config, cosmo):
    """Recover the n(z) of half a tracer from its other half: a known answer.

    Both halves share one bias, so n = (n_u b_u) / b_r. Reports χ² against
    the half's own redshift histogram, its mean redshift, and the shift of the
    true n(z) the measurement prefers, fitted as for the source bins.
    """
    cl = config["clustering"]
    data = _data(cl)
    m = Measurement(np.load(data / "xcorr_gate.npz"), cl["annulus"], cosmo)
    dat = np.load(data / f"{cl['gate'][0]}_dat.npz")
    half = _gate_half(len(dat["z"]))
    z, w = dat["z"][half], dat["w"][half]
    n, rows = m.nb("half") / m.bias(), m.nb("half", jk=True) / m.bias(jk=True)
    var = np.diag(jackknife_covariance(rows))
    truth = np.histogram(z, m.edges, weights=w)[0] / w.sum() / m.dz
    fine = np.linspace(0, m.edges[-1] + 1, 20001)
    density = np.histogram(z, fine, weights=w)[0] / w.sum() / np.diff(fine)
    centres = (fine[:-1] + fine[1:]) / 2
    lo, hi = m.edges[:-1], m.edges[1:]

    def mean(x):
        return (x * m.z).sum(-1) / x.sum(-1)

    return dict(
        z=m.z.tolist(),
        n=n.tolist(),
        err=np.sqrt(var).tolist(),
        truth=truth.tolist(),
        chi2=float(((n - truth) ** 2 / var).sum()),
        dof=len(n),
        mean_z=float(mean(n)),
        mean_z_err=float(np.sqrt(jackknife_covariance(mean(rows)[:, None])[0, 0])),
        mean_z_truth=float(mean(truth)),
        shift=shift_fit(n, 1 / var, lambda s: slice_average(centres, density, lo, hi, s)),
    )


def candidates(run):
    """Unit-integral n(z) per bin of a calibration run, by calibrator set."""
    arrays = np.load(Path(run) / "calibration_arrays.npz")
    out = {"calibration": arrays["bin_hists"]}
    inputs = Path(run) / "ensemble_inputs.npz"
    if inputs.exists() and "selection_shift" in (tables := np.load(inputs)):
        out["alternative calibrators"] = arrays["bin_hists"] + tables["selection_shift"]
    return arrays["grid"], out


def compare(config, cosmo):
    """n(z) b(z) of every bin on the reference slices, and its fits.

    Per bin and candidate n(z), inside the bin's core window: χ² with amplitude
    and bias slope α free, and the shift n(z - Δz) preferred at fixed α.
    """
    cl = config["clustering"]
    data = _data(cl)
    lo, hi, tracer, nb, rows, bias = [], [], [], [], [], []
    for name in cl["reference"]["slices"]:
        m = Measurement(np.load(data / f"xcorr_{name}.npz"), cl["annulus"], cosmo)
        bins = m.samples[1:]
        lo += list(m.edges[:-1])
        hi += list(m.edges[1:])
        tracer += [name] * len(m.z)
        nb.append([m.nb(b) for b in bins])
        rows.append([m.nb(b, jk=True) for b in bins])
        bias.append(m.bias())
    lo, hi = np.array(lo), np.array(hi)
    z = (lo + hi) / 2
    nb, rows = np.concatenate(nb, 1), np.concatenate(rows, 2)  # (bin, [patch,] slice)
    grid, cands = candidates(config["output"])
    out = dict(
        z_lo=lo.tolist(),
        z_hi=hi.tolist(),
        tracer=tracer,
        reference_bias=np.concatenate(bias).tolist(),
        bins=[],
    )
    for k, window in enumerate(cl["core"]):
        var = np.diag(jackknife_covariance(rows[k]))
        core = (lo >= window[0] - 1e-6) & (hi <= window[1] + 1e-6)
        ivar = core / var
        fits = {}
        for name, hists in cands.items():
            base = slice_average(grid, hists[k], lo, hi)
            chi2 = np.array([amplitude_fit(nb[k], ivar, base * (1 + z) ** a) for a in ALPHAS])
            fits[name] = dict(
                chi2=float(chi2.min()),
                alpha=float(ALPHAS[chi2.argmin()]),
                dof=int(core.sum()) - 2,
                shift={
                    f"{a:g}": shift_fit(
                        nb[k],
                        ivar,
                        lambda s, a=a: slice_average(grid, hists[k], lo, hi, s) * (1 + z) ** a,
                    )
                    for a in SHIFT_ALPHAS
                },
            )
        out["bins"].append(
            dict(nb=nb[k].tolist(), nb_err=np.sqrt(var).tolist(), core=list(window), fits=fits)
        )
    return out


def randoms_null(config, cosmo):
    """Raw angular cross-correlation nulls, not divided by a null auto-bias."""
    cl = config["clustering"]
    out = {}
    for tracer in cl["reference"]["slices"]:
        m = Measurement(np.load(_data(cl) / f"null_{tracer}.npz"), cl["annulus"], cosmo)
        bins = {}
        for i, sample in enumerate(m.samples[1:], 1):
            cov = jackknife_covariance(m.jk[:, :, i])
            bins[sample] = dict(
                w=m.w[:, i].tolist(), err=np.sqrt(np.diag(cov)).tolist(),
                covariance=cov.tolist(),
            )
        out[tracer] = dict(z_lo=m.edges[:-1].tolist(), z_hi=m.edges[1:].tolist(), bins=bins)
    return out


def analyse(config, figures=None):
    """Write ``clustering.json`` (gate and bins) and, optionally, its figures."""
    cosmo = fiducial_cosmology()
    cl = config["clustering"]
    result = dict(annulus=cl["annulus"], gate=gate(config, cosmo), **compare(config, cosmo))
    if cl.get("randoms_null", False):
        result["randoms_null"] = randoms_null(config, cosmo)
    (Path(cl["output"]) / "clustering.json").write_text(json.dumps(result, indent=1) + "\n")
    if figures:
        render(result, *candidates(config["output"]), figures)
    return result


def render(result, grid, cands, directory):
    """``clustering_gate.png`` and ``clustering_nz.png``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .nz_diagnostics import COLORS, _save

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    r_lo, r_hi = result["annulus"]

    g = result["gate"]
    z, dz = np.array(g["z"]), np.diff(g["z"]).mean()
    fig, ax = plt.subplots(figsize=(6.5, 3.8), layout="constrained")
    ax.step(np.r_[z - dz / 2, z[-1] + dz / 2], np.r_[g["truth"], g["truth"][-1]],
            where="post", color="0.3", lw=1.2, label="true n(z) of the held-out half")
    ax.errorbar(z, g["n"], g["err"], fmt="o", ms=4, color=COLORS[0],
                label=f"clustering, χ² = {g['chi2']:.1f} / {g['dof']}")
    ax.axhline(0, color="k", lw=0.5)
    s = g["shift"]
    ax.set(
        xlabel="z",
        ylabel="n(z)",
        title=f"Known answer: ⟨z⟩ {g['mean_z']:.3f} ± {g['mean_z_err']:.3f} "
        f"(true {g['mean_z_truth']:.3f}), shift {s['best']:+.3f} "
        f"[{s['lo']:+.3f}, {s['hi']:+.3f}]",
    )
    ax.title.set_fontsize(9)
    ax.legend(fontsize=8)
    _save(fig, directory / "clustering_gate.png")

    lo, hi = np.array(result["z_lo"]), np.array(result["z_hi"])
    z = (lo + hi) / 2
    styles = {"calibration": ("k", "-"), "alternative calibrators": (COLORS[1], "--")}
    labels = {"calibration": "SOM n(z)", "alternative calibrators": "SOM n(z), alternative calibrators"}
    fig, axes = plt.subplots(len(result["bins"]), 1, figsize=(7, 8.5), sharex=True,
                             layout="constrained")
    for k, (b, ax) in enumerate(zip(result["bins"], axes)):
        fit = b["fits"]["calibration"]
        core = (lo >= b["core"][0] - 1e-6) & (hi <= b["core"][1] + 1e-6)
        ivar = core / np.array(b["nb_err"]) ** 2
        tilt = (1 + z) ** fit["alpha"]
        model = slice_average(grid, cands["calibration"][k], lo, hi) * tilt
        A = (model * ivar * b["nb"]).sum() / (model * ivar * model).sum()
        ax.errorbar(z, np.array(b["nb"]) / tilt / A, np.array(b["nb_err"]) / tilt / A,
                    fmt="o", ms=3, lw=1, color=COLORS[0], label="clustering × DESI DR1")
        for name, hists in cands.items():
            ax.plot(grid, hists[k], color=styles[name][0], ls=styles[name][1], lw=1.2,
                    label=labels[name])
        ax.axvspan(*b["core"], color=COLORS[0], alpha=0.07, lw=0)
        ax.axhline(0, color="k", lw=0.5)
        shift = fit["shift"]["1"]
        text = f"bin {k + 1}   χ² = " + ", ".join(
            f"{f['chi2']:.1f}" for f in b["fits"].values()
        ) + f" / {fit['dof']}   Δz = {shift['best']:+.3f} [{shift['lo']:+.3f}, {shift['hi']:+.3f}]"
        ax.text(0.99, 0.93, text, transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_ylabel("n(z)")
    axes[0].legend(fontsize=7, loc="center right")
    axes[-1].set(xlabel="z", xlim=(0, 2.6))
    fig.suptitle(f"Gold bins × spectroscopic slices, {r_lo:g}–{r_hi:g} Mpc/h; "
                 "fit window shaded", fontsize=10)
    _save(fig, directory / "clustering_nz.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("recipe")
    parser.add_argument("--prepare", action="store_true", help="build the inputs first")
    parser.add_argument("--measure", action="store_true", help="count the pairs")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--figures", metavar="DIRECTORY")
    args = parser.parse_args(argv)
    config = load_yaml(args.recipe)
    if args.prepare:
        print(json.dumps(prepare(config)))
    if args.measure:
        measure(config, workers=args.workers)
    analyse(config, args.figures)


if __name__ == "__main__":
    main()
