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
from . import object_bias, ptes, selection_hist, sky, star_residuals
from . import products as P
from . import psf as PSF
from . import scale_cuts as SC
from . import tangential as TG
from . import versions as V
from . import weights as W

SECTIONS = [
    ("data_vectors", "Data vectors"),
    ("bmodes", "B modes"),
    ("leakage", "PSF leakage"),
    ("tangential", "Tangential shear tests"),
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


def load_version(run_dir, version, maps_dir):
    r = P.RunProducts(run_dir, version, maps_dir)
    blind = P.blind_of(r.assembled)
    return (
        r,
        blind,
        {
            "pe": P.load_pure_eb(r.pure_eb, blind),
            "co": P.load_cosebis(r.cosebis, blind),
            "cl": P.load_cl(r.assembled, r.cl_cov, blind),
        },
    )


CORRELATIONS = [
    # (product stem, renderer, figure id, section, caption, size)
    (
        "alpha_mag",
        TG.alpha_mag,
        "psf_alpha_mag",
        "leakage",
        "Scale-dependent leakage α(θ) = τ₀/ρ₀ for all galaxies and in galaxy-magnitude "
        "tertiles (calibrated e₁, e₂ rescaled to each bin's shear response plus the "
        "global selection response, w_des; jackknife errors). The selection "
        "response is larger for the faint bin, so its |α| is likely underestimated "
        "by ~10–20%. The paper's DES-Y3 comparison is omitted.",
        "full",
    ),
    (
        "gamma_stars",
        TG.gamma_stars,
        "tan_psf_stars",
        "tangential",
        "Galaxy γt and γ× around PSF validation stars (valid HSM, one entry per star), "
        "all and by star magnitude; calibrated e₁, e₂ with w_des, no random "
        "subtraction; jackknife errors, χ²/ν against zero.",
        "full",
    ),
    (
        "lambda_stars",
        TG.lambda_stars,
        "tan_lambda",
        "tangential",
        "PSF validation stars around galaxy positions: tangential and cross PSF-model "
        "ellipticity (λ₁) and star-minus-model residual (λ₃), minus the same around "
        "footprint randoms. Not a null test: it traces the PSF field.",
        "full",
    ),
    (
        "gamma_tiles",
        TG.gamma_tiles,
        "tan_tiles",
        "tangential",
        "θγt and θγ× of galaxies around tile centres; jackknife errors.",
        "full",
    ),
    (
        "gamma_gaia",
        TG.gamma_gaia,
        "tan_gaia",
        "tangential",
        "Galaxy γt and γ× around Gaia DR3 stars (RUWE < 1.4) in three G bins, minus "
        "the same around footprint randoms; jackknife errors.",
        "full",
    ),
]


def render_correlations(figs, corr_dir, version):
    """Figures from the correlation products present in ``corr_dir``."""
    for stem, renderer, fid, section, caption, size in CORRELATIONS:
        path = Path(corr_dir) / f"{stem}_{version}.npz"
        if path.exists():
            renderer(P.load_correlation(path), figs.path(fid))
            figs.add(fid, section, caption, size)


def render(run_dir, versions, out, maps_dir=None, corr_dir=None,
           corrected_tau_run=None, corrected_tau_version=None):
    cuts = run_config(run_dir)
    theta_cut, ell_cut, full_cut = cuts["theta"], cuts["ell"], cuts["theta_full"]
    loaded = [load_version(run_dir, v, maps_dir) for v in versions]
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
        f"Blinded Cℓ EE (NaMaster, mode-decoupled) as ℓCℓ. Shaded: outside {lcut}.",
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
    cg = P.load_cosebis_grid(ref.cosebis_grid, blind) if ref.cosebis_grid else None
    for layout, suffix in (("wide", ""), ("narrow", "_narrow")):
        SC.config_space(
            pe, cg, figs.path("ptes_config" + suffix), theta_cut, layout=layout
        )
    SC.harmonic(cl, figs.path("ptes_harmonic"), ell_cut)
    figs.add(
        "ptes_config",
        "bmodes",
        "B-mode PTE for every (θmin, θmax) cut on the 20-bin reporting grid: pure "
        "ξ+B, pure ξ−B"
        + (" and COSEBIs Bₙ (n ≤ 6)" if cg is not None else "")
        + "; the fiducial cut is outlined.",
        "full",
        narrow=True,
        group="scale_cuts",
    )
    figs.add(
        "ptes_harmonic",
        "bmodes",
        f"Cℓ BB PTE for every band range; the fiducial {lcut} is outlined.",
        "half",
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
    star_path = Path(maps_dir or ref.cv) / f"star_residuals_{versions[0]}.npz"
    stars = dict(np.load(star_path, allow_pickle=False)) if star_path.exists() else {}
    star_keys = {
        f"{kind}_{i}" for i in range(5) for kind in ("edges", "hist", "mean", "std")
    }
    if star_keys <= stars.keys():
        star_residuals.plot(stars, figs.path("star_residuals"))
        figs.add(
            "star_residuals",
            "leakage",
            "PSF validation stars (valid HSM): "
            "star-minus-model ellipticity and fourth moments, and half-light "
            "radius residual from √(T★/T_PSF) − 1. Means and σ use all finite detections.",
            "full",
        )
    else:
        print("Skipping star_residuals: product missing")
    rt = P.load_rho_tau(ref.assembled, ref.tau_cov, ref.psf_fit)
    PSF.tau_fit(rt, figs.path("psf_tau"))
    corrected_tau = None
    if corrected_tau_run is not None:
        corrected_tau = P.load_corrected_tau(
            corrected_tau_run, corrected_tau_version or versions[0] + "_leak_corr"
        )
    PSF.theta_tau(rt, figs.path("psf_theta_tau"), corrected=corrected_tau)
    figs.add(
        "psf_theta_tau",
        "leakage",
        "Galaxy–PSF correlations θτ₀,+, θτ₂,+ and θτ₅,+ (θ in arcmin). "
        + ("Raw and object-wise leakage-corrected catalogues overlaid. "
           if corrected_tau is not None else "Raw catalogue only. ")
        + "Band: raw α–β–η posterior (68%); no corrected fit is implied. "
        "Errors: theory τ covariance; bars omitted for nonpositive variances.",
        "full",
    )
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

    try:
        PSF.xi_sys_comparison(rt, figs.path("psf_xi_sys_comparison"))
    except ValueError as exc:
        print(f"Skipping psf_xi_sys_comparison: {exc}")
    else:
        figs.add(
            "psf_xi_sys_comparison",
            "leakage",
            "PSF systematic ξ⁺ from the α–β–η posterior (68% band) compared "
            "with τ₀,+²/ρ₀,+; no cosmological signal is shown.",
            "half",
        )

    cell_path = ref.weight_cells
    if cell_path is not None:
        bias_cells = dict(np.load(cell_path, allow_pickle=False))
        if {
            "leakage_sums",
            "leakage_snr_edges",
            "resolution_edges",
        } <= bias_cells.keys():
            object_bias.leakage(bias_cells, figs.path("object_leakage_bins"))
            figs.add(
                "object_leakage_bins",
                "leakage",
                "Weighted regressions of calibrated e₁, e₂ against the corresponding "
                "PSF model ellipticity at each galaxy, without an object-wise "
                "leakage correction. Eight equal-count resolution bins each have "
                "eight equal-count SNR bins. Hatching marks |α| < 2σ; σ is the "
                "weighted least-squares fit uncertainty.",
                "full",
            )
        else:
            print("Skipping object_leakage_bins: regression sums missing")
        if {"bias_sums", "bias_names"} <= bias_cells.keys() and bias_cells[
            "bias_sums"
        ].shape[2] == 10:
            object_bias.bias(bias_cells, figs.path("additive_bias"))
            figs.add(
                "additive_bias",
                "leakage",
                "Linear fits to weighted means of calibrated ellipticity in "
                "20 fixed bins of each standardised property. "
                "Error bars and significances use inverse-variance fits to binned "
                "means with independent-galaxy shape-noise errors; they omit "
                "spatially correlated noise, notably for RA and Dec. "
                "Galaxy size is √T and the size ratio is T/T_PSF; only objects "
                "inside the fixed property ranges are used.",
                "full",
            )
        else:
            print("Skipping additive_bias: regression sums missing")
    else:
        print("Skipping object_leakage_bins and additive_bias: product missing")

    if corr_dir is not None:
        render_correlations(figs, corr_dir, versions[0])

    # Weights, footprint and field maps, covariances
    selection_path = Path(maps_dir or ref.cv) / f"selection_hist_{versions[0]}.npz"
    if selection_path.exists():
        selection = dict(np.load(selection_path, allow_pickle=False))
        selection_hist.plot(selection, figs.path("selection_hist"))
        figs.add(
            "selection_hist",
            "survey",
            "Cumulative v2 catalogue selections; histograms show magnitudes 15–30. "
            "The paper's chip-defect/IMA_FLAGS, stellar-halo and pointing-count "
            "stages have no corresponding cut in this run. The configured tile "
            "overlap and r-band coverage cuts are included.",
            "full",
        )
    else:
        print("Skipping selection_hist: product missing")
    if ref.weight_cells is not None:
        cells = dict(np.load(ref.weight_cells, allow_pickle=False))
        if "sum_response" not in cells or "sum_uncal_ee" not in cells:
            print("Skipping weights: weight cells lack response and noise sums")
        else:
            W.weights(cells, figs.path("weights"))
            W.weights(cells, figs.path("weights_narrow"), layout="narrow")
            figs.add(
                "weights",
                "survey",
                "Counts, weights, mean diagonal galaxy response and per-component "
                "uncalibrated ellipticity RMS on the weight grid.",
                "full",
                narrow=True,
            )
        if {
            "response_hist",
            "response_edges",
            "response_outside",
            "n_rows",
        } <= cells.keys():
            W.response_histogram(cells, figs.path("response_histogram"))
            figs.add(
                "response_histogram",
                "survey",
                "Galaxy metacal response components over −5 to 5; at most "
                f"{100 * np.max(cells['response_outside']) / int(cells['n_rows']):.2f}% "
                "of entries per component lie outside this range or are non-finite. "
                "v2 has no metacal star shapes.",
                "half",
            )
        else:
            print("Skipping response_histogram: response histogram missing")
    if ref.field_maps is not None:
        maps = dict(np.load(ref.field_maps, allow_pickle=False))
        footprint = dict(np.load(ref.cv / f"footprint_{versions[0]}.npz"))
        if "sum_dT_over_T_psf" in maps:
            sky.paper_map_pair(maps, figs.path("sky_stars"), "stars")
            figs.add(
                "sky_stars",
                "survey",
                "PSF validation stars (valid HSM): star "
                "detections per sky area, counted once per exposure, and "
                "star-minus-model T divided by model T; HEALPix nside 256.",
                "full",
            )
        else:
            print("Skipping sky_stars: model-normalised size residual missing")
        if "sum_mag" in maps:
            sky.paper_map_pair(maps, figs.path("sky_magnitude"), "magnitude")
            figs.add(
                "sky_magnitude",
                "survey",
                "PSF FWHM and unweighted mean galaxy r magnitude per "
                "HEALPix nside-256 pixel.",
                "full",
            )
        else:
            print("Skipping sky_magnitude: magnitude sums missing")
        sky.footprint_map(maps, footprint, figs.path("sky_footprint"))
        figs.add(
            "sky_footprint",
            "survey",
            "Effective source density in a zenithal equal-area projection centred on the patch.",
            "full",
        )
        field_sets = ["galaxies"]
        if "sum_dT_over_T_psf" in maps:
            field_sets.append("psf")
        else:
            print("Skipping sky_psf: field maps lack model-normalised size residuals")
        for which in field_sets:
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
        if "psf" in field_sets:
            dt = sky.median_quantity(maps, "dT_over_T", 128)
            figs.add(
                "sky_psf",
                "survey",
                "PSF fields: model FWHM at galaxies, and star-minus-model size and "
                "ellipticity residuals for PSF validation stars (valid HSM); "
                f"size residual normalised by the model (median {100 * dt:+.2f}%).",
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
    parser.add_argument("--maps-dir", help="directory of field_maps/weight_cells NPZs")
    parser.add_argument("--correlations-dir", help="directory of correlations NPZs")
    parser.add_argument("--corrected-tau-run", help="optional τ-only leakage-corrected run")
    parser.add_argument("--corrected-tau-version", help="default: reference version + _leak_corr")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = render(
        args.run_dir, args.version, args.out, args.maps_dir, args.correlations_dir,
        args.corrected_tau_run, args.corrected_tau_version,
    )
    print(f"{len(manifest['figures'])} figures -> {args.out}")


if __name__ == "__main__":
    main()
