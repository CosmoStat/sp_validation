# CosmologyValidation diagnostic suite, decomposed into Snakemake rules.
#
# The original cosmo_val/run_cosmo_val.py was one linear driver that built a
# single in-memory `cv` (CosmologyValidation) and called ~13 cv.<method>()
# diagnostics in sequence, linked only by lazy properties on that object. Here
# each diagnostic is a rule, and the rules are linked by the SACC parts and
# products they write under COSMO_VAL (= cosmo_val/output):
#
#   catalogue ──→ xi (one job per grid: reporting, integration)
#                   ├─ integration part ─┬──→ pure_eb (part, npz, figures)
#                   │                    └──→ cosebis (part, npz, figures)
#                   └─ reporting part ──→ 2pcf plot, ratio_xi_sys_xi
#   OneCov ξ±, integration grid ──→ pure_eb, cosebis (their covariances)
#   catalogue ──→ pseudo_cl (part) ──┬─→ pseudo-Cl figures
#   OneCov ξ±, reporting grid ───────┤
#   rho/tau (part + FITS) ───────────┼─→ summarize_bmodes (reads the products)
#                                    └─→ assemble_sacc ──→ {version}.sacc
#
# The B-mode rules and the ξ± figures are ingests: they work from the parts and
# the covariance inputs, never from a catalogue, so a blinded part keeps
# everything downstream blinded. The analytic covariances (OneCovariance ξ±,
# NaMaster pseudo-Cℓ) are what assembly puts in the terminal file, replacing
# the estimates a part was born with.
#
# Every compute rule writes a data product. Every figure is declared by a
# separate rule consuming those products, with no catalogue inputs or lazy cv
# state. cosmo_val_plots gathers all figures; cosmo_val_all also assembles the
# analysis products. Cosmological caches contain only already-sealed vectors.
# Plot modules are explicit inputs so edits to them invalidate figures only.

CV = config["cosmo_val"]
CV_VERSIONS = config["versions"]
CV_FIDUCIAL = {
    "min_sep": CV["theta_min"],
    "max_sep": CV["theta_max"],
    "nbins": CV["nbins"],
    "npatch": CV["npatch"],
}

# str form for building wildcard-bearing output patterns (Path / str + str fails)
COSMO_VAL_STR = str(COSMO_VAL)

# Reporting-binning suffix shared by rho/tau and xi outputs.
CV_BINNING = (
    f"minsep={CV['theta_min']}_maxsep={CV['theta_max']}"
    f"_nbins={CV['nbins']}_npatch={CV['npatch']}"
)


def _pure_eb_stub(version):
    """Shared stem of the pure-E/B diagnostic products (npz + figures)."""
    eb = XI_GRIDS["integration"]
    return str(
        COSMO_VAL
        / (
            f"{version}_eb_minsep={CV['theta_min']}_maxsep={CV['theta_max']}"
            f"_nbins={CV['nbins']}_minsepint={eb['min_sep']}"
            f"_maxsepint={eb['max_sep']}_nbinsint={eb['nbins']}"
            f"_npatch={CV['npatch']}_varmethod=analytic"
        )
    )


def cv_pure_eb_npz(version):
    """Pure-E/B data vectors + covariance .npz."""
    return _pure_eb_stub(version) + "_data.npz"


def cv_pure_eb_figures(version):
    """The pure-E/B companion figures, by output key."""
    stub = _pure_eb_stub(version)
    return {
        "figure_integration_vs_reporting": f"{stub}_integration_vs_reporting.png",
        "figure_xis": f"{stub}_xis.png",
        "figure_ptes": f"{stub}_ptes.png",
        "figure_covariance": f"{stub}_covariance.png",
    }


def _grid_cov(version, grid, gaussian):
    """OneCovariance ξ± covariance on a named grid's own binning."""
    binning = XI_GRIDS[grid]
    return covariance_path(
        version,
        gaussian=gaussian,
        min_sep=binning["min_sep"],
        max_sep=binning["max_sep"],
        nbins=binning["nbins"],
        mask_suffix=DEFAULT_MASK_SUFFIX,
    )


def cv_xi_cov_integration(version):
    """OneCovariance Gaussian ξ± covariance on the integration grid.

    The one analytic covariance both B-mode statistics take: the pure-E/B Monte
    Carlo draws from it and COSEBIs carry it through their kernel. Gaussian
    because B-modes only need the scatter a Gaussian field would give.
    """
    return _grid_cov(version, "integration", "g")


def _cosebis_stub(version):
    """Shared stem of the COSEBIs diagnostic products (npz + figures).

    Named by the grid, where the covariance came from (varmethod), the mode
    count and the fiducial scale cut.
    """
    integ = XI_GRIDS["integration"]
    fsc = CV["fiducial_scale_cut"]
    return str(
        COSMO_VAL
        / (
            f"{version}_cosebis_minsep={integ['min_sep']}"
            f"_maxsep={integ['max_sep']}_nbins={integ['nbins']}"
            f"_varmethod=analytic_nmodes={CV['cosebis']['nmodes']}"
            f"_scalecut={fsc[0]}-{fsc[1]}"
        )
    )


def cv_cosebis_npz(version):
    """COSEBIs multi-cut diagnostic .npz (the PTE scan)."""
    return _cosebis_stub(version) + "_data.npz"


def cv_cosebis_figures(version):
    """The COSEBIs companion figures, by output key."""
    stub = _cosebis_stub(version)
    return {
        "figure_modes": f"{stub}_cosebis.png",
        "figure_covariance": f"{stub}_covariance.png",
        "figure_scalecut_ptes": f"{stub}_scalecut_ptes.png",
    }


_PSEUDO_CL_TAG = pseudo_cl_tag(config)


def cv_pseudo_cl_analysis_sacc(version):
    """Tagged pseudo-Cl SACC part: the harmonic block of the analysis file."""
    return str(COSMO_VAL / f"pseudo_cl_{version}_{_PSEUDO_CL_TAG}.sacc")


def cv_pseudo_cl_cov(version):
    """NaMaster pseudo-Cl covariance FITS (COVAR_EE_EE/BB_BB/EB_EB extensions)."""
    return str(COSMO_VAL / f"pseudo_cl_cov_{version}_{_PSEUDO_CL_TAG}.fits")


def cv_xi_cov(version):
    """OneCovariance ξ± covariance on the reporting grid, the terminal file's."""
    return _grid_cov(version, "reporting", "ng")


def cv_cosebis_sacc(version):
    """COSEBIs SACC part, at the fiducial scale cut."""
    return str(COSMO_VAL / f"{version}_cosebis.sacc")


def cv_pure_eb_sacc(version):
    """Pure-E/B SACC part."""
    return str(COSMO_VAL / f"{version}_pure_eb.sacc")


def cv_rho_tau_sacc(version):
    """ρ/τ SACC part."""
    return str(
        COSMO_VAL / "rho_tau_stats" / f"rho_tau_{cv_basename(version, CV_FIDUCIAL)}.sacc"
    )


def cv_xi_sacc(version, grid):
    """ξ± SACC part for a version on a named grid, named by that grid's binning."""
    return str(COSMO_VAL / f"{version}_xi_{xi_binning(grid)}.sacc")


def cv_analysis_sacc(version):
    """Terminal assembled analysis file {version}.sacc."""
    return str(COSMO_VAL / f"{version}.sacc")


# The CosmologyValidation constructor kwargs every cv_runner rule passes.
CV_INIT = cv_init_params(config)
CV_STYLES = {
    v: {k: catalogue_entry(v).get(k, default) for k, default in
        (("colour", "C0"), ("marker", "o"), ("label", v))}
    for v in CV_VERSIONS
}
CV_PLOT_CODE = str(REPO_ROOT / "src/sp_validation/cosmo_val/diagnostic_plots.py")
CV_REGIONS = ("NGC", "SGC", "fullsky")
CV_PSF_PRODUCTS = [str(COSMO_VAL / f"psf_fit_{v}_{CV_BINNING}.npz") for v in CV_VERSIONS]
CV_FOOTPRINT_MAPS = [str(COSMO_VAL / f"footprint_{v}.npz") for v in CV_VERSIONS]
CV_FOOTPRINT_FIGURES = [
    str(COSMO_VAL / f"footprint_{v}_{r}.png") for v in CV_VERSIONS for r in CV_REGIONS
]


def cv_shear_catalogues():
    """Catalogue inputs belong to compute rules only."""
    return [str(Path(catalogue_entry(v)["subdir"]) / catalogue_entry(v)["shear"]["path"])
            for v in CV_VERSIONS]


def cv_tau_covariances():
    method = CV.get("cov_estimate_method", "th")
    return [str(COSMO_VAL / "rho_tau_stats" /
                f"cov_tau_{cv_basename(v, CV_FIDUCIAL)}_{method}.npy")
            for v in CV_VERSIONS]


# ---------------------------------------------------------------------------
# PSF diagnostics: rho/tau statistics and the PSF-error fit
# ---------------------------------------------------------------------------
# The rho/tau FITS and the ξ± parts are produced by the generic compute rules
# `rho_tau_stats` and `xi` in workflow/rules/twopoint.smk. The cosmo_val suite
# consumes those outputs rather than re-declaring colliding rules: the helper
# paths cv_rho_stats / cv_tau_stats / cv_xi_sacc resolve to exactly the files
# those rules write, so requesting them triggers the existing compute.


rule cv_plot_rho_stats:
    """Overlay rho statistics across all versions."""
    input:
        stats=[cv_rho_stats(v, CV_FIDUCIAL) for v in CV_VERSIONS],
        covariances=[],
        plot_code=CV_PLOT_CODE,
    output:
        figure=str(COSMO_VAL / "rho_tau_stats/rho_stats.png"),
    params:
        kind="rho",
        styles=CV_STYLES,
    resources:
        runtime=20,
    script:
        "../scripts/cv_plot_diagnostic.py"


rule cv_plot_tau_stats:
    """Overlay tau statistics across all versions."""
    input:
        stats=[cv_tau_stats(v, CV_FIDUCIAL) for v in CV_VERSIONS],
        covariances=cv_tau_covariances(),
        plot_code=CV_PLOT_CODE,
    output:
        figure=str(COSMO_VAL / "rho_tau_stats/tau_stats.png"),
    params:
        kind="tau",
        styles=CV_STYLES,
    resources:
        runtime=20,
    script:
        "../scripts/cv_plot_diagnostic.py"


rule cv_rho_tau_fits:
    """Fit the PSF-error model (alpha/beta/eta) and propagate to xi_psf_sys."""
    input:
        rho=[cv_rho_stats(v, CV_FIDUCIAL) for v in CV_VERSIONS],
        tau=[cv_tau_stats(v, CV_FIDUCIAL) for v in CV_VERSIONS],
        covariances=cv_tau_covariances(),
        rho_covariances=[str(COSMO_VAL / "rho_tau_stats" /
            f"cov_rho_{cv_basename(v, CV_FIDUCIAL)}_jk.npy") for v in CV_VERSIONS],
    output:
        products=CV_PSF_PRODUCTS,
    params:
        cv_init=CV_INIT,
    resources:
        mem_mb=16000,
        runtime=120,
    script:
        "../scripts/cv_rho_tau_fits.py"


# ---------------------------------------------------------------------------
# Footprints, leakage, weights
# ---------------------------------------------------------------------------

rule cv_footprints:
    """Cache per-version sparse survey count maps."""
    input:
        catalogues=cv_shear_catalogues(),
    output:
        maps=CV_FOOTPRINT_MAPS,
    params:
        cv_init=CV_INIT,
    resources:
        mem_mb=16000,
        runtime=60,
    script:
        "../scripts/cv_footprints.py"


rule cv_objectwise_leakage:
    """Cache object-wise regression and scale-dependent alpha summaries."""
    input:
        catalogues=cv_shear_catalogues(),
        covariances=cv_tau_covariances(),
        rho=[cv_rho_stats(v, CV_FIDUCIAL) for v in CV_VERSIONS],
        tau=[cv_tau_stats(v, CV_FIDUCIAL) for v in CV_VERSIONS],
    output:
        coefficients=str(COSMO_VAL / "leakage_coefficients.json"),
    params:
        cv_init=CV_INIT,
    threads: 12
    resources:
        mem_mb=30000,
        runtime=180,
    script:
        "../scripts/cv_objectwise_leakage.py"


rule cv_weights:
    """Cache weighted shear-weight histogram counts and edges."""
    input:
        catalogues=cv_shear_catalogues(),
    output:
        histograms=str(COSMO_VAL / "weight_hist.npz"),
    params:
        cv_init=CV_INIT,
    resources:
        mem_mb=16000,
        runtime=30,
    script:
        "../scripts/cv_weights.py"


# ---------------------------------------------------------------------------
# Additive bias and two-point correlation functions
# ---------------------------------------------------------------------------
# The xi data vector itself is produced by the generic `xi` rule
# (workflow/rules/twopoint.smk), whose calculate_2pcf already subtracts the
# additive bias c1/c2 internally. cv_additive_bias persists c1/c2 to JSON as a
# standalone diagnostic (the original driver's cv.calculate_additive_bias()
# call); the xi rule does not depend on it.

rule cv_plot_footprints:
    input:
        maps=CV_FOOTPRINT_MAPS,
        plot_code=CV_PLOT_CODE,
    output:
        figures=CV_FOOTPRINT_FIGURES,
    params:
        kind="footprints",
        styles=CV_STYLES,
        regions=CV_REGIONS,
    script:
        "../scripts/cv_plot_diagnostic.py"


rule cv_plot_weights:
    input:
        histograms=rules.cv_weights.output.histograms,
        plot_code=CV_PLOT_CODE,
    output:
        figure=str(COSMO_VAL / "weight_hist.png"),
    params:
        kind="weights",
        styles=CV_STYLES,
    script:
        "../scripts/cv_plot_diagnostic.py"


rule cv_plot_objectwise_leakage:
    input:
        coefficients=rules.cv_objectwise_leakage.output.coefficients,
        plot_code=CV_PLOT_CODE,
    output:
        figure=str(COSMO_VAL / "leakage_coefficients.png"),
    params:
        kind="objectwise_leakage",
        styles=CV_STYLES,
    script:
        "../scripts/cv_plot_diagnostic.py"


rule cv_plot_rho_tau_fits:
    input:
        products=CV_PSF_PRODUCTS,
        plot_code=CV_PLOT_CODE,
    output:
        contours=str(COSMO_VAL / "rho_tau_stats/contours_tau_stat.png"),
        samples=str(COSMO_VAL / "rho_tau_stats/xi_psf_sys_tomography_False_sample.png"),
        summary=str(COSMO_VAL / "rho_tau_stats/xi_psf_sys_tomography_False_mean_and_std.png"),
    params:
        kind="rho_tau_fits",
        styles=CV_STYLES,
    script:
        "../scripts/cv_plot_diagnostic.py"


rule cv_additive_bias:
    """Weighted mean ellipticity c1/c2 per version (standalone diagnostic)."""
    output:
        additive_bias=str(COSMO_VAL / "additive_bias.json"),
    params:
        cv_init=CV_INIT,
    resources:
        mem_mb=16000,
        runtime=30,
    script:
        "../scripts/cv_additive_bias.py"


# LG: DEPRACATED since we do not plot unblinded data vectors.
rule cv_plot_2pcf:
    """Xi± overlays across versions, from the sealed reporting parts."""
    input:
        xi=[cv_xi_sacc(v, "reporting") for v in CV_VERSIONS],
        plot_code=CV_PLOT_CODE,
    output:
        figures=[str(COSMO_VAL / name) for name in
                 ("xi_pm_tomography_False.png", "xi_pm_theta_tomography_False.png")],
    params:
        kind="xi",
        styles=CV_STYLES,
        blinds=[blind_of(v) for v in CV_VERSIONS],
    resources:
        runtime=20,
    script:
        "../scripts/cv_plot_diagnostic.py"

# LG: DEPRACATED since we do not plot unblinded data vectors.
rule cv_ratio_xi_sys_xi:
    """Ratio of PSF systematics (xi_psf_sys) to the cosmic-shear signal (xi+)."""
    input:
        xi=[cv_xi_sacc(v, "reporting") for v in CV_VERSIONS],
        products=CV_PSF_PRODUCTS,
        plot_code=CV_PLOT_CODE,
    output:
        figure=str(COSMO_VAL / "ratio_xi_sys_xi.png"),
    params:
        kind="ratio",
        styles=CV_STYLES,
        blinds=[blind_of(v) for v in CV_VERSIONS],
    resources:
        mem_mb=16000,
        runtime=120,
    script:
        "../scripts/cv_plot_diagnostic.py"


# ---------------------------------------------------------------------------
# Harmonic-space pseudo-Cl
# ---------------------------------------------------------------------------

def cv_pseudo_cl_figures():
    """The pseudo-Cl figures, by output key (one per spectrum, all versions)."""
    return {
        f"figure_{name}": str(COSMO_VAL / f"cell_{name}.png")
        for name in ("ee", "eb", "bb")
    }


rule cv_plot_pseudo_cl:
    """The EE/EB/BB pseudo-Cl figures, from the analysis parts."""
    input:
        pseudo_cl=[cv_pseudo_cl_analysis_sacc(v) for v in CV_VERSIONS],
        pseudo_cl_cov=[cv_pseudo_cl_cov(v) for v in CV_VERSIONS],
        plot_code=str(REPO_ROOT / "src/sp_validation/cosmo_val/pseudo_cl.py"),
    output:
        **cv_pseudo_cl_figures(),
    params:
        versions=CV_VERSIONS,
        blinds=[blind_of(v) for v in CV_VERSIONS],
        # Style is per catalogue, so the derived variants take their parent's.
        markers=[CATALOG_CONFIG[base_version(v)]["marker"] for v in CV_VERSIONS],
        colours=[CATALOG_CONFIG[base_version(v)]["colour"] for v in CV_VERSIONS],
    resources:
        mem_mb=8000,
        runtime=20,
    script:
        "../scripts/cv_plot_pseudo_cl.py"


# ---------------------------------------------------------------------------
# Pure E/B modes and COSEBIs (per version), then the B-mode summary
# ---------------------------------------------------------------------------

rule cv_pure_eb:
    """Pure E/B-mode decomposition for one version, from its integration-grid part.

    The modes are a fixed linear operator on the part's ξ±; the covariance is
    the integration-grid ξ± covariance pushed exactly through it.
    """
    input:
        xi_integration=lambda w: cv_xi_sacc(w.version, "integration"),
        cov_integration=lambda w: cv_xi_cov_integration(w.version),
    output:
        npz=cv_pure_eb_npz("{version}"),
        sacc=cv_pure_eb_sacc("{version}"),
    params:
        version="{version}",
        blind=lambda w: blind_of(w.version),
        min_sep=CV["theta_min"],
        max_sep=CV["theta_max"],
        nbins=CV["nbins"],
        integration=XI_GRIDS["integration"],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
    resources:
        mem_mb=8000,
        runtime=20,
    script:
        "../scripts/cv_pure_eb.py"


rule cv_cosebis:
    """COSEBIs E/B decomposition for one version, from its integration-grid part.

    The modes come from the part; the covariance is the OneCovariance ξ± covariance
    on the same grid through the same kernel as the modes.
    """
    input:
        xi=lambda w: cv_xi_sacc(w.version, "integration"),
        cov=lambda w: cv_xi_cov_integration(w.version),
    output:
        npz=cv_cosebis_npz("{version}"),
        sacc=cv_cosebis_sacc("{version}"),
    params:
        version="{version}",
        blind=lambda w: blind_of(w.version),
        min_sep=XI_GRIDS["integration"]["min_sep"],
        max_sep=XI_GRIDS["integration"]["max_sep"],
        nbins=XI_GRIDS["integration"]["nbins"],
        nmodes=CV["cosebis"]["nmodes"],
        scale_cuts=CV["cosebis"]["scale_cuts"],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
    threads: 24
    resources:
        mem_mb=48000,
        runtime=600,
    script:
        "../scripts/cv_cosebis.py"


rule cv_plot_pure_eb:
    """Render pure E/B figures without rerunning the transform."""
    input:
        npz=lambda w: cv_pure_eb_npz(w.version),
        plot_code=str(REPO_ROOT / "src/sp_validation/b_modes.py"),
    output:
        **cv_pure_eb_figures("{version}"),
    params:
        version="{version}",
        blind=lambda w: blind_of(w.version),
        fiducial_scale_cut=CV["fiducial_scale_cut"],
    script:
        "../scripts/cv_plot_pure_eb.py"


rule cv_plot_cosebis:
    """Render COSEBIs figures without rerunning the transform."""
    input:
        npz=lambda w: cv_cosebis_npz(w.version),
        plot_code=str(REPO_ROOT / "src/sp_validation/b_modes.py"),
    output:
        **cv_cosebis_figures("{version}"),
    params:
        version="{version}",
        blind=lambda w: blind_of(w.version),
        fiducial_scale_cut=CV["fiducial_scale_cut"],
    script:
        "../scripts/cv_plot_cosebis.py"


rule cv_summarize_bmodes:
    """Collect B-mode PTEs (pure E/B, COSEBIs, pseudo-Cl) into one summary."""
    input:
        pure_eb=[cv_pure_eb_npz(v) for v in CV_VERSIONS],
        cosebis=[cv_cosebis_npz(v) for v in CV_VERSIONS],
        pseudo_cl=(
            [cv_pseudo_cl_analysis_sacc(v) for v in CV_VERSIONS]
            if CV.get("include_pseudo_cl", False) else []
        ),
        pseudo_cl_cov=(
            [cv_pseudo_cl_cov(v) for v in CV_VERSIONS]
            if CV.get("include_pseudo_cl", False) else []
        ),
    output:
        summary_json=str(COSMO_VAL / "bmode_summary.json"),
    params:
        versions=CV_VERSIONS,
        blinds=[blind_of(v) for v in CV_VERSIONS],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        include_pseudo_cl=CV.get("include_pseudo_cl", False),
    resources:
        mem_mb=8000,
        runtime=20,
    script:
        "../scripts/cv_summarize_bmodes.py"


# ---------------------------------------------------------------------------
# Terminal analysis file: assemble the per-statistic SACC parts into {version}.sacc
# ---------------------------------------------------------------------------
# The terminal file carries the analysis vector only. The integration-grid ξ± is
# deliberately not gathered: it stays a per-part intermediate. The two blocks
# born without a covariance (ξ± reporting, pseudo-Cℓ) get theirs injected from
# the covariance inputs below.


def cv_assemble_inputs(version):
    """The per-statistic SACC parts + covariance inputs assemble_sacc consumes.

    Each part's filename carries enough to bind its producing rule's wildcards.
    """
    parts = dict(
        xi_reporting=cv_xi_sacc(version, "reporting"),
        xi_cov=cv_xi_cov(version),
        cosebis=cv_cosebis_sacc(version),
        pure_eb=cv_pure_eb_sacc(version),
        rho_tau=cv_rho_tau_sacc(version),
    )
    if CV.get("include_pseudo_cl", False):
        parts["pseudo_cl"] = cv_pseudo_cl_analysis_sacc(version)
        parts["pseudo_cl_cov"] = cv_pseudo_cl_cov(version)
    return parts


rule assemble_sacc:
    """Assemble the terminal {version}.sacc from the per-statistic SACC parts."""
    input:
        unpack(lambda w: cv_assemble_inputs(w.version)),
    output:
        sacc=cv_analysis_sacc("{version}"),
    params:
        version="{version}",
        blind=lambda w: blind_of(w.version),
        # The statistics this rule wired, so a typo'd input keyword cannot
        # silently drop one.
        expected=lambda w: [
            k
            for k in cv_assemble_inputs(w.version)
            if k not in ("xi_cov", "pseudo_cl_cov")
        ],
    resources:
        mem_mb=8000,
        runtime=20,
    script:
        "../scripts/assemble_sacc.py"


rule assemble_sacc_all:
    """Assemble the analysis SACC file for every version."""
    input:
        [cv_analysis_sacc(v) for v in CV_VERSIONS],


# ---------------------------------------------------------------------------
# Aggregate target: the whole validation suite
# ---------------------------------------------------------------------------

rule cosmo_val_plots:
    """Render every validation figure from cached data products."""
    input:
        rules.cv_plot_rho_stats.output,
        rules.cv_plot_tau_stats.output,
        rules.cv_plot_rho_tau_fits.output,
        rules.cv_plot_footprints.output,
        rules.cv_plot_objectwise_leakage.output,
        rules.cv_plot_weights.output,
        rules.cv_plot_2pcf.output,
        rules.cv_ratio_xi_sys_xi.output,
        [path for v in CV_VERSIONS for path in cv_pure_eb_figures(v).values()],
        [path for v in CV_VERSIONS for path in cv_cosebis_figures(v).values()],
        list(cv_pseudo_cl_figures().values()) if CV.get("include_pseudo_cl", False) else [],
        [cv_xi_cov(v).removesuffix(".dat") + "_corr_plot.pdf" for v in CV_VERSIONS],
        [cv_xi_cov_integration(v).removesuffix(".dat") + "_corr_plot.pdf" for v in CV_VERSIONS],
        # Direct dependencies are intentional. With --allowed-rules Snakemake
        # can prune a plot whose figure exists even when its cache is missing.
        # The aggregate must validate caches independently of those plot jobs.
        caches=[
            *rules.cv_plot_rho_stats.input.stats,
            *rules.cv_plot_tau_stats.input.stats,
            *rules.cv_plot_tau_stats.input.covariances,
            *rules.cv_plot_rho_tau_fits.input.products,
            *rules.cv_plot_footprints.input.maps,
            rules.cv_plot_objectwise_leakage.input.coefficients,
            rules.cv_plot_weights.input.histograms,
            *rules.cv_plot_2pcf.input.xi,
            *[cv_pure_eb_npz(v) for v in CV_VERSIONS],
            *[cv_cosebis_npz(v) for v in CV_VERSIONS],
            *[cv_xi_cov(v) for v in CV_VERSIONS],
            *[cv_xi_cov_integration(v) for v in CV_VERSIONS],
            *([cv_pseudo_cl_analysis_sacc(v) for v in CV_VERSIONS]
              if CV.get("include_pseudo_cl", False) else []),
            *([cv_pseudo_cl_cov(v) for v in CV_VERSIONS]
              if CV.get("include_pseudo_cl", False) else []),
        ],


rule cosmo_val_all:
    """Run the full suite, including every plot-only rule."""
    input:
        rules.cosmo_val_plots.input,
        str(COSMO_VAL / "additive_bias.json"),
        str(COSMO_VAL / "bmode_summary.json"),
        [cv_analysis_sacc(v) for v in CV_VERSIONS],
