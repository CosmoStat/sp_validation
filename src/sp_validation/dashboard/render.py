"""Render the dashboard figure set for a cosmo_val run.

Reads only cached products (see :mod:`sp_validation.dashboard.products`) and
writes the PNGs plus ``figures.json``, the manifest the page generator lays
out: the sections in page order, one entry per figure (section, optional
folded group, caption, size class) and the B-mode PTE table. A run with two
or more versions also gets the version comparisons and one pair of PTE rows
per version; the first version is the reference for every other figure.

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
from . import ptes, sky
from . import scale_cuts as SC
from . import versions as V
from . import weights as W

SECTIONS = [
    ("data_vectors", "Data vectors"),
    ("bmodes", "B modes"),
    ("leakage", "PSF leakage"),
    ("survey", "Weights, footprint and covariances"),
]
GROUPS = {
    "scale_cuts": "PTEs across scale cuts",
    "versions": "Catalogue versions",
    "covariance": "Covariances",
}


def run_config(run_dir):
    """Scale cuts from the run's recorded configuration, when present."""
    prov = Path(run_dir) / "provenance.json"
    cuts = {"theta": (12.0, 83.0), "ell": (300, 1600), "theta_full": (1.0, 250.0)}
    if prov.exists():
        cfg = yaml.safe_load(json.loads(prov.read_text())["config"])
        cuts["theta"] = tuple(map(float, cfg["cosmo_val"]["fiducial_scale_cut"]))
        cuts["ell"] = (
            int(cfg["cl"]["fiducial_ell_min"]),
            int(cfg["cl"]["fiducial_ell_max"]),
        )
        cuts["theta_full"] = (
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

    def add(self, fid, section, caption, size, narrow=False, group=None):
        entry = {
            "id": fid,
            "section": section,
            "group": group,
            "file": f"{fid}.png",
            "caption": caption,
            "size": size,
            "alt": caption,
        }
        if narrow:
            entry["narrow"] = f"{fid}_narrow.png"
        self.entries.append(entry)


def load_version(run_dir, version, maps_dir, cat_config):
    r = P.RunProducts(run_dir, version, maps_dir)
    blind = P.declared_blind(cat_config, version)
    return (
        r,
        blind,
        {
            "pe": P.load_pure_eb(r.pure_eb, blind),
            "co": P.load_cosebis(r.cosebis, blind),
            "cl": P.load_cl(r.assembled, r.cl_cov, blind),
        },
    )


def render(run_dir, versions, out, cat_config, maps_dir=None):
    cuts = run_config(run_dir)
    theta_cut, ell_cut, full_cut = cuts["theta"], cuts["ell"], cuts["theta_full"]
    loaded = [load_version(run_dir, v, maps_dir, cat_config) for v in versions]
    ref, blind, d = loaded[0]
    pe, co, cl = d["pe"], d["co"], d["cl"]
    xi = P.load_xi(ref.assembled, blind)
    figs = Figures(out)
    tcut = f"{theta_cut[0]:g}′–{theta_cut[1]:g}′"
    lcut = f"{ell_cut[0]} ≤ ℓ ≤ {ell_cut[1]}"

    # Data vectors
    for layout, suffix in (("wide", ""), ("narrow", "_narrow")):
        DV.xi_pm(xi, figs.path("xi_pm" + suffix), layout=layout, blind=blind)
        DV.ell_cl(cl, figs.path("cl" + suffix), ell_cut, layout=layout, blind=blind)
    figs.add(
        "xi_pm",
        "data_vectors",
        "Blinded ξ± on the 20-bin reporting grid; errors from the analytic covariance.",
        "full",
        narrow=True,
    )
    figs.add(
        "cl",
        "data_vectors",
        f"Cℓ (NaMaster, mode-decoupled) as ℓCℓ: EE (blinded), BB and EB. Shaded: outside {lcut}.",
        "full",
        narrow=True,
    )

    # B modes
    DV.pure_eb(pe, figs.path("bmode_pure_eb"), theta_cut, blind=blind)
    DV.pure_eb(
        pe, figs.path("bmode_pure_eb_narrow"), theta_cut, layout="narrow", blind=blind
    )
    figs.add(
        "bmode_pure_eb",
        "bmodes",
        f"Pure-mode decomposition of ξ±. Shaded: outside the fiducial {tcut}; full range 1′–250′.",
        "full",
        narrow=True,
    )
    DV.cosebis(co, figs.path("bmode_cosebis"), full_cut)
    figs.add(
        "bmode_cosebis",
        "bmodes",
        f"COSEBIs Bₙ/σₙ at the fiducial {tcut} (filled) and over the full range (open). "
        "Shaded: modes beyond n = 6.",
        "half",
    )
    DV.cl_bmodes(cl, figs.path("bmode_cl"), ell_cut)
    figs.add(
        "bmode_cl",
        "bmodes",
        f"Cℓ BB and EB in units of their uncertainty. Shaded: outside {lcut}.",
        "half",
    )
    for layout, suffix in (("wide", ""), ("narrow", "_narrow")):
        SC.config_space(pe, figs.path("ptes_config" + suffix), theta_cut, layout=layout)
        SC.harmonic_and_cosebis(
            cl, co, figs.path("ptes_harmonic" + suffix), ell_cut, layout=layout
        )
    figs.add(
        "ptes_config",
        "bmodes",
        "Pure-B PTE for every (θmin, θmax) cut; the fiducial cut is outlined.",
        "full",
        narrow=True,
        group="scale_cuts",
    )
    figs.add(
        "ptes_harmonic",
        "bmodes",
        "Cℓ BB PTE for every band range, and COSEBIs B PTE (n ≤ 20) for each cached cut.",
        "full",
        narrow=True,
        group="scale_cuts",
    )
    if len(versions) > 1:
        labelled = [(v, dd) for v, (_, _, dd) in zip(versions, loaded)]
        V.pure_eb(
            [(v, dd["pe"]) for v, dd in labelled],
            figs.path("versions_pure_eb"),
            theta_cut,
        )
        V.cosebis(
            [(v, dd["co"]) for v, dd in labelled],
            figs.path("versions_cosebis"),
            full_cut,
        )
        V.cl([(v, dd["cl"]) for v, dd in labelled], figs.path("versions_cl"), ell_cut)
        figs.add(
            "versions_pure_eb",
            "bmodes",
            "θξ± and pure-B ξ±/σ across versions; boxes span the versions, bars mark the reference.",
            "full",
            group="versions",
        )
        figs.add(
            "versions_cosebis",
            "bmodes",
            "COSEBIs Bₙ/σₙ across versions.",
            "full",
            group="versions",
        )
        figs.add(
            "versions_cl",
            "bmodes",
            "Cℓ BB/σ and EB/σ across versions.",
            "full",
            group="versions",
        )

    table = {
        "threshold": 0.05,
        "columns": ptes.COLUMNS,
        "rows": [
            {"version": v, **row}
            for v, (_, _, dd) in zip(versions, loaded)
            for row in ptes.rows(
                dd["pe"], dd["co"], dd["cl"], theta_cut, ell_cut, full_cut
            )
        ],
    }

    # PSF leakage
    rt = P.load_rho_tau(ref.assembled, ref.tau_cov, ref.psf_fit, blind)
    PSF.tau_fit(rt, figs.path("psf_tau"))
    figs.add(
        "psf_tau",
        "leakage",
        "Galaxy–PSF correlations τ₀,₂,₅ with the posterior band of the α, β, η leakage model.",
        "full",
    )
    PSF.posterior(rt, figs.path("psf_posterior"))
    figs.add(
        "psf_posterior",
        "leakage",
        "Posterior of the leakage parameters (68 and 95%).",
        "half",
    )
    leakage = P.load_json(ref.leakage).get(versions[0], {}).get("tomo_bin_all", {})
    PSF.alpha_estimates(leakage, rt, figs.path("psf_alpha"))
    figs.add(
        "psf_alpha",
        "leakage",
        "Every estimate of the leakage coefficient: object-wise, scale-dependent and ρ/τ fit.",
        "half",
    )
    PSF.rho_stats(rt, figs.path("psf_rho"))
    figs.add(
        "psf_rho", "leakage", "ρ statistics of the PSF model and its residuals.", "full"
    )
    PSF.xi_sys_ratio(rt, xi, figs.path("psf_xi_sys"))
    figs.add(
        "psf_xi_sys",
        "leakage",
        "Propagated PSF systematic relative to the blinded ξ±; dotted lines at ±10%.",
        "full",
    )

    # Weights, footprint and field maps, covariances
    if ref.weight_cells is not None:
        cells = dict(np.load(ref.weight_cells, allow_pickle=False))
        W.weights(cells, figs.path("weights"))
        W.weights(cells, figs.path("weights_narrow"), layout="narrow")
        s = W.summary(cells)
        figs.add(
            "weights",
            "survey",
            f"Shape weight on its {int(np.sqrt(s['n_cells']))}×{int(np.sqrt(s['n_cells']))} "
            f"signal-to-noise × size grid: every galaxy in a cell carries that cell's "
            f"weight, so the catalogue has {s['n_values']} distinct weights "
            f"({s['w_min']:.2f} to {s['w_max']:.1f}).",
            "full",
            narrow=True,
        )
    if ref.field_maps is not None:
        maps = dict(np.load(ref.field_maps, allow_pickle=False))
        footprint = dict(np.load(ref.cv / f"footprint_{versions[0]}.npz"))
        sky.footprint_map(maps, footprint, figs.path("sky_footprint"))
        figs.add(
            "sky_footprint",
            "survey",
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
            "survey",
            "Galaxy fields: ellipticity dispersion, mean weight and weighted mean e₁, e₂.",
            "full",
            narrow=True,
        )
        dt = sky.median_quantity(maps, "dT_over_T", 128)
        figs.add(
            "sky_psf",
            "survey",
            "PSF fields: model FWHM at galaxies, and star-minus-model size and ellipticity "
            f"residuals (median size residual {100 * dt:+.2f}%).",
            "full",
            narrow=True,
        )
    n = len(xi["theta"])
    CV.correlation(
        xi["cov"], CV.xi_blocks(n), figs.path("cov_xi"), r"$\xi_\pm$ reporting grid"
    )
    figs.add(
        "cov_xi",
        "survey",
        "ξ± reporting-grid covariance (analytic, masked).",
        "half",
        group="covariance",
    )
    CV.correlation(
        pe["cov"],
        CV.pure_eb_blocks(len(pe["theta"])),
        figs.path("cov_pure_eb"),
        "pure E/B modes",
    )
    figs.add(
        "cov_pure_eb",
        "survey",
        "Pure E, B and ambiguous modes of ξ±.",
        "half",
        group="covariance",
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
        "survey",
        "COSEBIs E and B modes at the fiducial cut.",
        "half",
        group="covariance",
    )
    CV.correlation(
        CV.cl_matrix(cl["cov"]),
        CV.cl_blocks(len(cl["ell"])),
        figs.path("cov_cl"),
        r"$C_\ell$",
    )
    figs.add(
        "cov_cl",
        "survey",
        "Cℓ EE, EB and BB (Gaussian, NaMaster).",
        "half",
        group="covariance",
    )

    manifest = {
        "schema_version": 2,
        "versions": list(versions),
        "blind": blind,
        "sections": [{"id": s, "title": t} for s, t in SECTIONS],
        "groups": GROUPS,
        "pte_table": table,
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
    parser.add_argument(
        "--cat-config",
        required=True,
        help="catalogue config declaring each version's blind (the run's cat_config.yaml)",
    )
    parser.add_argument("--maps-dir", help="directory of field_maps/weight_cells NPZs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = render(
        args.run_dir, args.version, args.out, args.cat_config, args.maps_dir
    )
    print(f"{len(manifest['figures'])} figures -> {args.out}")


if __name__ == "__main__":
    main()
