"""Compare the MCCD and PSFEx nautilus chains: marginals, triangle, S8 shift.

Reads cosmosis nautilus text chains (weights from the log_weight column),
prints mean, std and 68% intervals of S8, Omega_m, sigma_8 (and the PSF
leakage parameters), writes a getdist triangle plot overlaying the chains and
a JSON summary, and expresses the S8 shift in units of
sigma_Delta = sqrt(s1^2 + s2^2 - 2 rho s1 s2) for a few assumed correlations
rho between the two estimates (the samples share sky, so rho > 0).

Usage:
    python compare_chains.py --out <dir> LABEL=path/to/samples.txt [...]
The first chain is the reference (PSFEx fiducial).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from getdist import MCSamples, plots  # noqa: E402

PARAMS = {
    "cosmological_parameters--s_8": ("S8", r"S_8"),
    "cosmological_parameters--omega_m": ("Omega_m", r"\Omega_{\rm m}"),
    "cosmological_parameters--sigma_8": ("sigma_8", r"\sigma_8"),
    "psf_leakage_parameters--alpha": ("alpha", r"\alpha"),
    "psf_leakage_parameters--beta": ("beta", r"\beta"),
    "intrinsic_alignment_parameters--a": ("A_IA", r"A_{\rm IA}"),
}


def load_chain(path, label):
    with open(path) as f:
        header = f.readline().lstrip("#").split()
    data = np.loadtxt(path, comments="#")
    cols = [c.lower() for c in header]
    w = np.exp(data[:, cols.index("log_weight")] - data[:, cols.index("log_weight")].max())
    names, labels, arrays = [], [], []
    for key, (name, lab) in PARAMS.items():
        if key in cols:
            names.append(name)
            labels.append(lab)
            arrays.append(data[:, cols.index(key)])
    return MCSamples(
        samples=np.column_stack(arrays),
        weights=w,
        names=names,
        labels=labels,
        label=label,
        settings={"smooth_scale_2D": 0.4, "smooth_scale_1D": 0.4},
    )


def summarize(s):
    out = {}
    for p in s.getParamNames().names:
        m = s.getMargeStats().parWithName(p.name)
        lim = m.limits[0]
        out[p.name] = {
            "mean": float(m.mean),
            "std": float(m.err),
            "lower68": float(lim.lower),
            "upper68": float(lim.upper),
        }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--rho", type=float, nargs="*", default=[0.0, 0.5, 0.8])
    ap.add_argument("chains", nargs="+", help="LABEL=path")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    samples = [load_chain(p, lab) for lab, p in (c.split("=", 1) for c in a.chains)]
    summary = {s.label: summarize(s) for s in samples}

    for s in samples:
        print(s.label)
        for p, v in summary[s.label].items():
            print(f"  {p:8s} {v['mean']:.4f} +/- {v['std']:.4f}  [{v['lower68']:.4f}, {v['upper68']:.4f}]")

    if len(samples) > 1:
        ref, other = summary[samples[0].label], summary[samples[1].label]
        shift = {}
        for p in ("S8", "Omega_m", "sigma_8"):
            d = other[p]["mean"] - ref[p]["mean"]
            s1, s2 = ref[p]["std"], other[p]["std"]
            shift[p] = {
                "delta": d,
                "n_sigma": {
                    f"rho={r}": d / np.sqrt(s1**2 + s2**2 - 2 * r * s1 * s2) for r in a.rho
                },
            }
            print(f"shift {p}: {d:+.4f}  " + "  ".join(f"{k}: {v:+.2f} sigma" for k, v in shift[p]["n_sigma"].items()))
        summary["shift"] = shift

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    g = plots.get_subplot_plotter(width_inch=6.5)
    g.settings.legend_fontsize = 11
    g.triangle_plot(
        samples, ["S8", "Omega_m", "sigma_8"], filled=True,
        contour_colors=["#1f77b4", "#d62728", "#2ca02c"][: len(samples)],
    )
    g.export(str(out / "triangle_cosmo.png"), dpi=150)
    g = plots.get_subplot_plotter(width_inch=8)
    g.triangle_plot(
        samples, ["S8", "Omega_m", "A_IA", "alpha", "beta"], filled=True,
        contour_colors=["#1f77b4", "#d62728", "#2ca02c"][: len(samples)],
    )
    g.export(str(out / "triangle_nuisance.png"), dpi=120)
    print(f"written to {out}")
