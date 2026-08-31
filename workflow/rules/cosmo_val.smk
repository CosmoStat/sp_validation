# CosmologyValidation diagnostic suite, decomposed into Snakemake rules.
#
# The original cosmo_val/run_cosmo_val.py was one linear driver that built a
# single in-memory `cv` (CosmologyValidation) and called ~13 cv.<method>()
# diagnostics in sequence, linked only by lazy properties on that object. Here
# each diagnostic is a rule, and the rules are linked by the SACC parts and
# products they write under COSMO_VAL (= cosmo_val/output):
#
#   catalogue ──→ xi (one job per grid: reporting, integration, cosebis)
#                   │
#                   ├─ reporting part ──┬─→ pure_eb (part, npz, figures)
#                   ├─ integration part ┘        │
#                   ├─ cosebis part ─────→ cosebis (part, npz, figures)
#                   └─ reporting .txt ──→ 2pcf plot, ratio_xi_sys_xi
#   catalogue ──→ pseudo_cl (part) ──┬─→ pseudo-Cl figures
#   CosmoCov ──→ covariance ─────────┤
#   rho/tau (part + FITS) ───────────┼─→ summarize_bmodes (reads the products)
#                                    └─→ assemble_sacc ──→ {version}.sacc
#
# The B-mode rules are ingests: pure_eb, cosebis, the pseudo-Cl figures and the
# summary all work from the parts and the covariance inputs, never from a
# catalogue, so a blinded part keeps everything downstream blinded. The
# analytic covariances (CosmoCov ξ±, NaMaster pseudo-Cℓ) are what assembly puts
# in the terminal file, replacing the estimates a part was born with.
#
# Methods that only emit figures, or whose figure paths derive from internal
# handler state (rho/tau plots, rho_tau_fits, objectwise leakage, 2pcf
# overlay), declare a sentinel under COSMO_VAL/snakemake_sentinels so they stay
# DAG-trackable. Lazy cv state the original code never persists (c1/c2,
# xi_psf_sys) is either materialized to a small JSON (additive bias) or
# recomputed in the one rule that needs it — recompute is cheap next to the
# science it depends on. See workflow/scripts/cv_runner.py.

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


def cv_xi_txt(version):
    """Path to the 2pcf data vector calculate_2pcf writes for a version.

    Mirrors the out_fname f-string in cosmo_val.calculate_2pcf:
    {ver}_xi_minsep=..._maxsep=..._nbins=..._npatch=...txt
    """
    return str(COSMO_VAL / f"{version}_xi_{xi_binning('reporting')}.txt")


def cv_rho_stats(version):
    return str(
        COSMO_VAL / "rho_tau_stats" / f"rho_stats_{cv_basename(version, CV_FIDUCIAL)}.fits"
    )


def cv_tau_stats(version):
    return str(
        COSMO_VAL / "rho_tau_stats" / f"tau_stats_{cv_basename(version, CV_FIDUCIAL)}.fits"
    )


def _pure_eb_stub(version):
    """Shared stem of the pure-E/B diagnostic products (npz + figures)."""
    eb = CV["integration"]
    return str(
        COSMO_VAL
        / (
            f"{version}_eb_minsep={CV['theta_min']}_maxsep={CV['theta_max']}"
            f"_nbins={CV['nbins']}_minsepint={eb['min_sep']}"
            f"_maxsepint={eb['max_sep']}_nbinsint={eb['nbins']}"
            f"_npatch={CV['npatch']}_varmethod=semi-analytic"
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


def cv_xi_cov_integration(version):
    """CosmoCov gaussian ξ± covariance on the integration grid.

    The covariance model the pure-E/B Monte Carlo draws from; gaussian because
    the draws only need the scatter a Gaussian field would give.
    """
    integ = CV["integration"]
    return covariance_path(
        version,
        FIDUCIAL["blind"],
        gaussian="g",
        min_sep=integ["min_sep"],
        max_sep=integ["max_sep"],
        nbins=integ["nbins"],
        mask_suffix=DEFAULT_MASK_SUFFIX,
    )


def _cosebis_stub(version):
    """Shared stem of the COSEBIs diagnostic products (npz + figures).

    varmethod names where the covariance came from, and these products are the
    propagated one — which also keeps them clear of the paths plot_cosebis
    builds for its own byproducts, so nothing overwrites a declared output.
    """
    cb = CV["cosebis"]
    fsc = CV["fiducial_scale_cut"]
    return str(
        COSMO_VAL
        / (
            f"{version}_cosebis_minsep={cb['min_sep_int']}"
            f"_maxsep={cb['max_sep_int']}_nbins={cb['nbins_int']}"
            f"_npatch={cb['npatch']}_varmethod=propagated_nmodes={cb['nmodes']}"
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
    """Analysis pseudo-Cl SACC part: the harmonic block of the analysis file."""
    return str(COSMO_VAL / f"{pseudo_cl_analysis_stem(config, version)}.sacc")


def cv_pseudo_cl_cov(version):
    """NaMaster pseudo-Cl covariance FITS (COVAR_EE_EE/BB_BB/EB_EB extensions)."""
    return str(COSMO_VAL / f"pseudo_cl_cov_{version}_{_PSEUDO_CL_TAG}.fits")


def cv_xi_cov(version):
    """CosmoCov-processed ξ± covariance, on the reporting grid's own binning."""
    return covariance_path(
        version,
        FIDUCIAL["blind"],
        gaussian="ng",
        min_sep=CV["theta_min"],
        max_sep=CV["theta_max"],
        nbins=CV["nbins"],
        mask_suffix=DEFAULT_MASK_SUFFIX,
    )


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


# Common params block shared by every cosmo_val rule: the cv constructor kwargs
# plus the run directory the object must be instantiated in.
def cv_params(version_list=None):
    return dict(
        cv_init=cv_init_params(config, version_list=version_list),
        rundir=CV_RUNDIR,
    )


# ---------------------------------------------------------------------------
# PSF diagnostics: rho/tau statistics and the PSF-error fit
# ---------------------------------------------------------------------------
# The rho/tau FITS and the xi data vector are produced by the generic compute
# rules `rho_tau_stats` and `xi` already defined in workflow/rules/twopoint.smk
# (same filename convention, same CosmologyValidation methods). The cosmo_val
# suite consumes those outputs rather than re-declaring colliding rules. The
# helper paths cv_rho_stats / cv_tau_stats / cv_xi_txt resolve to exactly the
# files those rules write, so requesting them triggers the existing compute.


rule cv_plot_rho_stats:
    """Overlay rho statistics across all versions."""
    input:
        rho=[cv_rho_stats(v) for v in CV_VERSIONS],
    output:
        sentinel=str(CV_SENTINELS / "plot_rho_stats.done"),
    params:
        **cv_params(),
    resources:
        runtime=20,
    script:
        "../scripts/cv_plot_rho_stats.py"


rule cv_plot_tau_stats:
    """Overlay tau statistics across all versions."""
    input:
        tau=[cv_tau_stats(v) for v in CV_VERSIONS],
    output:
        sentinel=str(CV_SENTINELS / "plot_tau_stats.done"),
    params:
        **cv_params(),
    resources:
        runtime=20,
    script:
        "../scripts/cv_plot_tau_stats.py"


rule cv_rho_tau_fits:
    """Fit the PSF-error model (alpha/beta/eta) and propagate to xi_psf_sys."""
    input:
        rho=[cv_rho_stats(v) for v in CV_VERSIONS],
        tau=[cv_tau_stats(v) for v in CV_VERSIONS],
    output:
        sentinel=str(CV_SENTINELS / "rho_tau_fits.done"),
    params:
        **cv_params(),
    resources:
        mem_mb=16000,
        runtime=120,
    script:
        "../scripts/cv_rho_tau_fits.py"


# ---------------------------------------------------------------------------
# Footprints, leakage, weights
# ---------------------------------------------------------------------------

rule cv_footprints:
    """Per-version survey footprint maps."""
    output:
        sentinel=str(CV_SENTINELS / "footprints.done"),
    params:
        **cv_params(),
    resources:
        mem_mb=16000,
        runtime=60,
    script:
        "../scripts/cv_footprints.py"


rule cv_objectwise_leakage:
    """Object-wise PSF-leakage regression vs scale-dependent alpha (all versions)."""
    output:
        sentinel=str(CV_SENTINELS / "objectwise_leakage.done"),
    params:
        **cv_params(),
    threads: 12
    resources:
        mem_mb=30000,
        runtime=180,
    script:
        "../scripts/cv_objectwise_leakage.py"


rule cv_weights:
    """Weighted shear-weight histograms across versions."""
    output:
        weight_hist=str(COSMO_VAL / "weight_hist.png"),
    params:
        **cv_params(),
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

rule cv_additive_bias:
    """Weighted mean ellipticity c1/c2 per version (standalone diagnostic)."""
    output:
        additive_bias=str(COSMO_VAL / "additive_bias.json"),
    params:
        **cv_params(),
    resources:
        mem_mb=16000,
        runtime=30,
    script:
        "../scripts/cv_additive_bias.py"


rule cv_plot_2pcf:
    """n_pairs / xi± overlay across versions."""
    input:
        xi=[cv_xi_txt(v) for v in CV_VERSIONS],
    output:
        sentinel=str(CV_SENTINELS / "plot_2pcf.done"),
    params:
        **cv_params(),
    resources:
        runtime=20,
    script:
        "../scripts/cv_plot_2pcf.py"


rule cv_ratio_xi_sys_xi:
    """Ratio of PSF systematics (xi_psf_sys) to the cosmic-shear signal (xi+)."""
    input:
        xi=[cv_xi_txt(v) for v in CV_VERSIONS],
        rho=[cv_rho_stats(v) for v in CV_VERSIONS],
        tau=[cv_tau_stats(v) for v in CV_VERSIONS],
    output:
        ratio=str(COSMO_VAL / "ratio_xi_sys_xi.png"),
    params:
        offset=0.1,
        **cv_params(),
    resources:
        mem_mb=16000,
        runtime=120,
    script:
        "../scripts/cv_ratio_xi_sys_xi.py"


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
    output:
        **cv_pseudo_cl_figures(),
    params:
        versions=CV_VERSIONS,
        # Style is per catalogue, so the derived variants take their parent's.
        markers=[CATALOG_CONFIG[base_version(v)]["marker"] for v in CV_VERSIONS],
        colours=[CATALOG_CONFIG[base_version(v)]["colour"] for v in CV_VERSIONS],
        rundir=CV_RUNDIR,
    resources:
        mem_mb=8000,
        runtime=20,
    script:
        "../scripts/cv_plot_pseudo_cl.py"


# ---------------------------------------------------------------------------
# Pure E/B modes and COSEBIs (per version), then the B-mode summary
# ---------------------------------------------------------------------------

# On a data run these re-derive their E-mode vector from the *blinded* ξ± parts,
# so they are born blinded; the commitment binds only there.
def cv_cosebis_inputs(w):
    return {
        "xi": blindable_part(cv_xi_sacc(w.version, "cosebis")),
        **commitment_input(w.version),
    }


def cv_pure_eb_inputs(w):
    return {
        "xi_reporting": blindable_part(cv_xi_sacc(w.version, "reporting")),
        "xi_integration": blindable_part(cv_xi_sacc(w.version, "integration")),
        "cov_integration": cv_xi_cov_integration(w.version),
        **commitment_input(w.version),
    }


rule cv_pure_eb:
    """Pure E/B-mode decomposition for one version, from its ξ± parts.

    The modes come from the two parts; the covariance is Monte Carlo from the
    integration-grid covariance model, so no patched estimator run is involved.
    """
    input:
        unpack(cv_pure_eb_inputs),
    output:
        npz=cv_pure_eb_npz("{version}"),
        sacc=cv_pure_eb_sacc("{version}"),
        **cv_pure_eb_figures("{version}"),
    params:
        version="{version}",
        type=CV.get("type", "data"),
        min_sep=CV["theta_min"],
        max_sep=CV["theta_max"],
        nbins=CV["nbins"],
        n_samples=CV.get("n_mc_samples", 1000),
        cosmo_params=CV["cosmo_params"],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        rundir=CV_RUNDIR,
    threads: 24
    resources:
        mem_mb=40000,
        runtime=360,
    script:
        "../scripts/cv_pure_eb.py"


rule cv_cosebis:
    """COSEBIs E/B decomposition for one version, from its ξ± part.

    Values, covariance and PTEs all come from the part: the COSEBIs covariance
    is the part's ξ± covariance through the same kernel as the modes.
    """
    input:
        unpack(cv_cosebis_inputs),
    output:
        npz=cv_cosebis_npz("{version}"),
        sacc=cv_cosebis_sacc("{version}"),
        **cv_cosebis_figures("{version}"),
    params:
        version="{version}",
        type=CV.get("type", "data"),
        min_sep=CV["cosebis"]["min_sep_int"],
        max_sep=CV["cosebis"]["max_sep_int"],
        nbins=CV["cosebis"]["nbins_int"],
        nmodes=CV["cosebis"]["nmodes"],
        scale_cuts=CV["cosebis"]["scale_cuts"],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        rundir=CV_RUNDIR,
    threads: 24
    resources:
        mem_mb=48000,
        runtime=600,
    script:
        "../scripts/cv_cosebis.py"


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
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        min_sep=CV["theta_min"],
        max_sep=CV["theta_max"],
        nbins=CV["nbins"],
        include_pseudo_cl=CV.get("include_pseudo_cl", False),
        rundir=CV_RUNDIR,
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
    # blindable_part binds the raw-signal parts to their blinded siblings on a
    # data run. COSEBIs, pure-E/B and ρ/τ are stamped concealed by their own
    # writers, so they bind by name either way.
    parts = dict(
        xi_reporting=blindable_part(cv_xi_sacc(version, "reporting")),
        xi_cov=cv_xi_cov(version),
        cosebis=cv_cosebis_sacc(version),
        pure_eb=cv_pure_eb_sacc(version),
        rho_tau=cv_rho_tau_sacc(version),
    )
    if CV.get("include_pseudo_cl", False):
        parts["pseudo_cl"] = blindable_part(cv_pseudo_cl_analysis_sacc(version))
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
        type=CV.get("type", "data"),
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

rule cosmo_val_all:
    """Run the full CosmologyValidation diagnostic suite."""
    input:
        # PSF diagnostics
        str(CV_SENTINELS / "footprints.done"),
        str(CV_SENTINELS / "plot_rho_stats.done"),
        str(CV_SENTINELS / "plot_tau_stats.done"),
        str(CV_SENTINELS / "rho_tau_fits.done"),
        # Shear diagnostics
        str(CV_SENTINELS / "objectwise_leakage.done"),
        str(COSMO_VAL / "weight_hist.png"),
        str(COSMO_VAL / "additive_bias.json"),
        # Two-point
        str(CV_SENTINELS / "plot_2pcf.done"),
        str(COSMO_VAL / "ratio_xi_sys_xi.png"),
        # B-modes
        str(COSMO_VAL / "bmode_summary.json"),
        list(cv_pseudo_cl_figures().values()) if CV.get("include_pseudo_cl", False) else [],
        # Terminal analysis file: the assembled {version}.sacc per version
        [cv_analysis_sacc(v) for v in CV_VERSIONS],
