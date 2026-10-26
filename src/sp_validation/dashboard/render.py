"""Render the dashboard figure set for a cosmo_val run.

Reads only cached products (see :mod:`sp_validation.dashboard.products`) and
writes the PNGs plus ``figures.json``, the manifest the page generator lays
out: one entry per figure with its section, caption and size class, and the
headline B-mode PTEs. A run with two or more versions also gets the version
comparisons; the first version is the reference for every other figure.

Usage (inside the container)::

    python -m sp_validation.dashboard.render --run-dir RUN --version SP_v2.0 \\
        --maps-dir PRODUCTS --out RUN/dashboard-figures
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from . import covariance as CV
from . import data_vectors as DV
from . import products as P
from . import psf as PSF
from . import scale_cuts as SC
from . import sky
from . import versions as V
from . import weights as W

SECTIONS = [
    ("xi", "Shear correlation functions"),
    ("bmodes", "B-mode null tests"),
    ("scale_cuts", "PTEs across scale cuts"),
    ("covariance", "Covariances"),
    ("psf", "PSF leakage"),
    ("weights", "Weights"),
    ("sky", "Footprint and field maps"),
    ("versions", "Catalogue versions"),
]


def run_config(run_dir):
    """Scale cuts from the run's recorded configuration, when present."""
    prov = Path(run_dir) / "provenance.json"
    cuts = {"theta": (12.0, 83.0), "ell": (300, 1600), "cosebis_full": (1.0, 250.0)}
    if prov.exists():
        cfg = yaml.safe_load(json.loads(prov.read_text())["config"])
        cuts["theta"] = tuple(map(float, cfg["cosmo_val"]["fiducial_scale_cut"]))
        cuts["ell"] = (
            int(cfg["cl"]["fiducial_ell_min"]),
            int(cfg["cl"]["fiducial_ell_max"]),
        )
        cuts["cosebis_full"] = (
            float(cfg["cosebis"]["theta_min"]),
            float(cfg["cosebis"]["theta_max"]),
        )
    return cuts


class Figures:
    """Collects figure entries while rendering."""

    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.entries = []

    def path(self, name):
        return self.out / f"{name}.png"

    def add(self, fid, section, caption, size, narrow=False, alt=None):
        entry = {
            "id": fid,
            "section": section,
            "file": f"{fid}.png",
            "caption": caption,
            "size": size,
            "alt": alt or caption,
        }
        if narrow:
            entry["narrow"] = f"{fid}_narrow.png"
        self.entries.append(entry)


def render(run_dir, versions, out, maps_dir=None):
    cuts = run_config(run_dir)
    ref = P.RunProducts(run_dir, versions[0], maps_dir)
    blind = P.blind_of(ref.assembled)
    figs = Figures(out)
    xi = P.load_xi(ref.assembled, blind)
    pe = P.load_pure_eb(ref.pure_eb, blind)
    co = P.load_cosebis(ref.cosebis, blind)
    cl = P.load_pseudo_cl(ref.assembled, ref.pseudo_cl_cov, blind)
    summary = (
        P.load_json(ref.summary).get(versions[0], {}) if ref.summary.exists() else {}
    )
    theta_cut, ell_cut = cuts["theta"], cuts["ell"]
    tcut = rf"{theta_cut[0]:g}′–{theta_cut[1]:g}′"

    # 1. hero
    DV.xi_pm(xi, figs.path("xi_pm"), blind=blind)
    DV.xi_pm(xi, figs.path("xi_pm_narrow"), layout="narrow", blind=blind)
    figs.add(
        "xi_pm",
        "xi",
        "Blinded ξ± on the 20-bin reporting grid; error bars from the analytic covariance.",
        "full",
        narrow=True,
    )

    # 2. B-mode data vectors
    ptes = {k: summary.get(k) for k in ("xip_B", "xim_B", "combined")}
    DV.pure_eb(pe, figs.path("bmode_pure_eb"), theta_cut, ptes, blind=blind)
    figs.add(
        "bmode_pure_eb",
        "bmodes",
        f"Pure-mode decomposition of ξ±; B (crimson) is the null test. Shaded: outside {tcut}.",
        "third",
    )
    DV.cosebis(co, figs.path("bmode_cosebis"), cuts["cosebis_full"])
    figs.add(
        "bmode_cosebis",
        "bmodes",
        "COSEBIs B-modes in units of their uncertainty, at the fiducial cut and over all scales.",
        "third",
    )
    DV.pseudo_cl(cl, figs.path("bmode_cl"), ell_cut)
    figs.add(
        "bmode_cl",
        "bmodes",
        f"Pseudo-Cℓ BB and EB in units of their uncertainty. Shaded: outside {ell_cut[0]} ≤ ℓ ≤ {ell_cut[1]}.",
        "third",
    )
    fid_co = co["cuts"][co["fiducial"]]["pte_B"]
    headline = [
        {
            "label": "ξ± pure B, joint",
            "symbol": "xi",
            "pte": ptes.get("combined"),
            "range": tcut,
        },
        {
            "label": "COSEBIs B",
            "symbol": "cosebis",
            "pte": fid_co,
            "range": rf"{co['fiducial'][0]:g}′–{co['fiducial'][1]:g}′",
        },
        {
            "label": "pseudo-Cℓ BB",
            "symbol": "cl",
            "pte": DV.cl_pte(cl, "BB", ell_cut),
            "range": f"{ell_cut[0]} ≤ ℓ ≤ {ell_cut[1]}",
        },
    ]

    # 3. scale cuts
    SC.config_space(pe, figs.path("ptes_config"), theta_cut)
    SC.config_space(pe, figs.path("ptes_config_narrow"), theta_cut, layout="narrow")
    figs.add(
        "ptes_config",
        "scale_cuts",
        "Configuration-space B-mode PTE for every (θmin, θmax) cut; the fiducial cut is outlined.",
        "full",
        narrow=True,
    )
    SC.harmonic_and_cosebis(cl, co, figs.path("ptes_harmonic"), ell_cut)
    SC.harmonic_and_cosebis(
        cl, co, figs.path("ptes_harmonic_narrow"), ell_cut, layout="narrow"
    )
    figs.add(
        "ptes_harmonic",
        "scale_cuts",
        "Cℓ BB PTE for every band range, and the COSEBIs B PTE for each cached angular cut.",
        "full",
        narrow=True,
    )

    # 4. covariances
    n = len(xi["theta"])
    CV.correlation(
        xi["cov"], CV.xi_blocks(n), figs.path("cov_xi"), r"$\xi_\pm$ reporting grid"
    )
    figs.add(
        "cov_xi",
        "covariance",
        "ξ± reporting-grid covariance (analytic, masked).",
        "half",
    )
    CV.correlation(
        pe["cov"],
        CV.pure_eb_blocks(len(pe["theta"])),
        figs.path("cov_pure_eb"),
        "pure E/B modes",
    )
    figs.add(
        "cov_pure_eb", "covariance", "Pure E, B and ambiguous modes of ξ±.", "half"
    )
    fc = co["fiducial"]
    CV.correlation(
        co["cuts"][fc]["cov"],
        CV.cosebis_blocks(co["nmodes"]),
        figs.path("cov_cosebis"),
        rf"COSEBIs, ${fc[0]:g}'\!-\!{fc[1]:g}'$",
    )
    figs.add(
        "cov_cosebis",
        "covariance",
        "COSEBIs E and B modes at the fiducial cut.",
        "half",
    )
    CV.correlation(
        CV.cl_matrix(cl["cov"]),
        CV.cl_blocks(len(cl["ell"])),
        figs.path("cov_cl"),
        r"pseudo-$C_\ell$",
    )
    figs.add(
        "cov_cl", "covariance", "Pseudo-Cℓ EE, EB and BB (Gaussian, NaMaster).", "half"
    )

    # 5. PSF leakage
    rt = P.load_rho_tau(ref.assembled, ref.tau_cov, ref.psf_fit)
    PSF.tau_fit(rt, figs.path("psf_tau"))
    figs.add(
        "psf_tau",
        "psf",
        "Galaxy–PSF correlations τ₀,₂,₅ with the posterior band of the α, β, η leakage model.",
        "full",
    )
    PSF.posterior(rt, figs.path("psf_posterior"))
    figs.add(
        "psf_posterior",
        "psf",
        "Posterior of the leakage parameters (68 and 95%).",
        "half",
    )
    leakage = P.load_json(ref.leakage).get(versions[0], {}).get("tomo_bin_all", {})
    PSF.alpha_estimates(leakage, rt, figs.path("psf_alpha"))
    figs.add(
        "psf_alpha",
        "psf",
        "Every estimate of the leakage coefficient: object-wise, scale-dependent and ρ/τ fit.",
        "half",
    )
    PSF.rho_stats(rt, figs.path("psf_rho"))
    figs.add(
        "psf_rho", "psf", "ρ statistics of the PSF model and its residuals.", "full"
    )
    PSF.xi_sys_ratio(rt, xi, figs.path("psf_xi_sys"))
    figs.add(
        "psf_xi_sys",
        "psf",
        "Propagated PSF systematic relative to the blinded ξ±; dotted lines at ±10%.",
        "full",
    )

    # 6. weights
    if ref.weight_cells is not None:
        cells = dict(np.load(ref.weight_cells, allow_pickle=False))
        W.weights(cells, figs.path("weights"))
        s = W.summary(cells)
        figs.add(
            "weights",
            "weights",
            f"Each galaxy carries the weight R²/σₑ² of its SNR × size cell, so the "
            f"{s['n_values']} distinct values are the {s['n_occupied']} occupied cells; "
            f"histograms of w are combs, not clipping.",
            "full",
        )

    # 7. sky
    if ref.field_maps is not None:
        maps = dict(np.load(ref.field_maps, allow_pickle=False))
        footprint = dict(np.load(ref.cv / f"footprint_{versions[0]}.npz"))
        sky.footprint_map(maps, footprint, figs.path("sky_footprint"))
        figs.add(
            "sky_footprint",
            "sky",
            "Effective source density in a zenithal equal-area projection centred on the patch.",
            "full",
        )
        for which in ("galaxies", "psf"):
            sky.field_maps(maps, figs.path(f"sky_{which}"), which, footprint)
            sky.field_maps(
                maps,
                figs.path(f"sky_{which}_narrow"),
                which,
                footprint,
                layout="narrow",
            )
        figs.add(
            "sky_galaxies",
            "sky",
            "Galaxy fields: ellipticity dispersion, mean weight and weighted mean e₁, e₂.",
            "full",
            narrow=True,
        )
        dt = sky.median_quantity(maps, "dT_over_T", 128)
        figs.add(
            "sky_psf",
            "sky",
            "PSF fields: model FWHM at galaxies, and star-minus-model size and ellipticity "
            f"residuals (median size residual {100 * dt:+.2f}%).",
            "full",
            narrow=True,
        )

    # 8. versions
    if len(versions) > 1:
        runs = [P.RunProducts(run_dir, v, maps_dir) for v in versions]
        blinds = [P.blind_of(r.assembled) for r in runs]
        V.pure_eb(
            [
                (v, P.load_pure_eb(r.pure_eb, b))
                for v, r, b in zip(versions, runs, blinds)
            ],
            figs.path("versions_pure_eb"),
            theta_cut,
        )
        figs.add(
            "versions_pure_eb",
            "versions",
            "ξ± and pure-B ξ±/σ across versions; boxes span the versions, bars mark the reference.",
            "full",
        )
        V.cosebis(
            [
                (v, P.load_cosebis(r.cosebis, b))
                for v, r, b in zip(versions, runs, blinds)
            ],
            figs.path("versions_cosebis"),
            cuts["cosebis_full"],
        )
        figs.add("versions_cosebis", "versions", "COSEBIs B/σ across versions.", "full")
        V.pseudo_cl(
            [
                (v, P.load_pseudo_cl(r.assembled, r.pseudo_cl_cov, b))
                for v, r, b in zip(versions, runs, blinds)
            ],
            figs.path("versions_cl"),
            ell_cut,
        )
        figs.add("versions_cl", "versions", "Cℓ BB/σ and EB/σ across versions.", "full")

    manifest = {
        "schema_version": 1,
        "versions": list(versions),
        "blind": blind,
        "sections": [{"id": s, "title": t} for s, t in SECTIONS],
        "headline_ptes": headline,
        "figures": figs.entries,
    }
    (figs.out / "figures.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--version",
        action="append",
        required=True,
        help="catalogue version; repeat for a comparison run",
    )
    parser.add_argument("--maps-dir", help="directory of field_maps/weight_cells NPZs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = render(args.run_dir, args.version, args.out, args.maps_dir)
    print(f"{len(manifest['figures'])} figures -> {args.out}")


if __name__ == "__main__":
    main()
