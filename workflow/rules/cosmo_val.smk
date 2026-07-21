# CosmologyValidation diagnostic suite, decomposed into Snakemake rules.
#
# The original cosmo_val/run_cosmo_val.py was one linear driver that built a
# single in-memory `cv` (CosmologyValidation) and called ~13 cv.<method>()
# diagnostics in sequence, linked only by lazy properties on that object. Here
# each diagnostic is a rule, and the rules are linked by the *real* data
# products each method writes under COSMO_VAL (= cosmo_val/output):
#
#   rho/tau FITS  ──┬─→ rho/tau plots
#                   ├─→ rho_tau_fits (PSF-error MCMC)
#                   └─────────────────────────────┐
#   additive bias ──→ xi (2pcf) ──┬─→ 2pcf plot   │
#                                 ├─→ ratio_xi_sys_xi ←┘ (also needs xi_psf_sys)
#                                 ├─→ pure_eb (npz) ─┐
#                                 └─→ cosebis (npz) ─┤
#   pseudo_cl FITS ──────────────────────────────────┼─→ summarize_bmodes
#                                                     ┘
#
# Granularity decision: methods that write durable data products
# (calculate_rho_tau_stats, calculate_2pcf, calculate_pseudo_cl, plot_pure_eb,
# plot_cosebis) own a compute rule keyed on those files. Methods that only
# emit figures, or whose figure paths derive from internal handler state
# (rho/tau plots, rho_tau_fits, objectwise leakage, 2pcf overlay), declare a
# sentinel under COSMO_VAL/snakemake_sentinels so they stay DAG-trackable.
# Lazy cv state that the original code never persists (c1/c2, xi_psf_sys) is
# either materialized to a small JSON (additive bias) or recomputed in the one
# rule that needs it (xi_psf_sys in ratio_xi_sys_xi) — recompute is cheap next
# to the science it depends on. See workflow/scripts/cv_runner.py.

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
    return str(
        COSMO_VAL
        / (
            f"{version}_xi_minsep={CV['theta_min']}_maxsep={CV['theta_max']}"
            f"_nbins={CV['nbins']}_npatch={CV['npatch']}.txt"
        )
    )


def cv_rho_stats(version):
    return str(
        COSMO_VAL / "rho_tau_stats" / f"rho_stats_{cv_basename(version, CV_FIDUCIAL)}.fits"
    )


def cv_tau_stats(version):
    return str(
        COSMO_VAL / "rho_tau_stats" / f"tau_stats_{cv_basename(version, CV_FIDUCIAL)}.fits"
    )


def cv_pure_eb_npz(version):
    eb = CV["integration"]
    return str(
        COSMO_VAL
        / (
            f"{version}_eb_minsep={CV['theta_min']}_maxsep={CV['theta_max']}"
            f"_nbins={CV['nbins']}_minsepint={eb['min_sep']}"
            f"_maxsepint={eb['max_sep']}_nbinsint={eb['nbins']}"
            f"_npatch={CV['npatch']}_varmethod=jackknife_data.npz"
        )
    )


def cv_cosebis_npz(version):
    cb = CV["cosebis"]
    fsc = CV["fiducial_scale_cut"]
    # Mirror calculate/plot_cosebis out_stub (cosmo_val.py): a distinct schema
    # from pure_eb — _cosebis_ prefix, integration nbins, plus _nmodes= and
    # _scalecut= segments. Must match save_cosebis_results exactly or
    # verify_outputs raises and cv_summarize_bmodes deadlocks on this input.
    return str(
        COSMO_VAL
        / (
            f"{version}_cosebis_minsep={cb['min_sep_int']}"
            f"_maxsep={cb['max_sep_int']}_nbins={cb['nbins_int']}"
            f"_npatch={cb['npatch']}_varmethod=jackknife_nmodes={cb['nmodes']}"
            f"_scalecut={fsc[0]}-{fsc[1]}_data.npz"
        )
    )


def cv_pseudo_cl_sacc(version):
    """Untagged pseudo-Cl SACC part cv_pseudo_cl writes (B-mode diagnostic).

    This is the harmonic-space BB diagnostic cv_summarize_bmodes reads. The
    *analysis* file's pseudo-Cl part is the tagged, blinded inference product
    instead (see cv_pseudo_cl_analysis_sacc) so {version}.sacc stays byte-
    comparable against today's cosmosis_fitting.py assembly (PR-3's converter).
    """
    return str(COSMO_VAL / f"pseudo_cl_{version}.sacc")


# Fiducial harmonic-binning tag the pseudo-Cl producer (twopoint.smk rules
# pseudo_cl / pseudo_cl_cov) stamps into the analysis-grade filename. Mirrors
# inference.smk's PSEUDO_CL_TAG so the analysis file carries the same pseudo-Cl
# the inference pipeline consumes (canonical: blind=A, powspace, nbins=32).
_HARMONIC_FIDUCIAL = config["harmonic"]["fiducial"]
_PSEUDO_CL_TAG = (
    f"blind={_HARMONIC_FIDUCIAL['blind']}"
    f"_{_HARMONIC_FIDUCIAL['binning']}"
    f"_nbins={_HARMONIC_FIDUCIAL['nbins']}"
)


def cv_pseudo_cl_analysis_sacc(version):
    """Tagged, blinded pseudo-Cl SACC part the analysis file carries."""
    return str(COSMO_VAL / f"pseudo_cl_{version}_{_PSEUDO_CL_TAG}.sacc")


def cv_pseudo_cl_cov(version):
    """NaMaster pseudo-Cl covariance FITS (COVAR_EE_EE/BB_BB/EB_EB extensions)."""
    return str(COSMO_VAL / f"pseudo_cl_cov_{version}_{_PSEUDO_CL_TAG}.fits")


def cv_cosebis_sacc(version):
    """COSEBIs SACC part (fiducial scale cut) the cv_cosebis rule writes."""
    return str(COSMO_VAL / f"{version}_cosebis.sacc")


def cv_pure_eb_sacc(version):
    """Pure-E/B SACC part the cv_pure_eb rule writes."""
    return str(COSMO_VAL / f"{version}_pure_eb.sacc")


def cv_rho_tau_sacc(version):
    """ρ/τ SACC part calculate_rho_tau_stats writes (rho_tau_{base}.sacc)."""
    return str(
        COSMO_VAL / "rho_tau_stats" / f"rho_tau_{cv_basename(version, CV_FIDUCIAL)}.sacc"
    )


def cv_xi_reporting_sacc(version):
    """Reporting ξ± SACC part the xi rule (run_2pcf.py) writes for a version.

    Carries the reporting-binning suffix so requesting it binds the xi job's
    wildcards (the rule's txt + reporting .sacc outputs share one wildcard set).
    """
    return str(
        COSMO_VAL
        / (
            f"{version}_xi_reporting_minsep={CV['theta_min']}_maxsep={CV['theta_max']}"
            f"_nbins={CV['nbins']}_npatch={CV['npatch']}.sacc"
        )
    )


def cv_xi_integration_sacc(version):
    """Integration-grid ξ± SACC part the xi_highres rule writes, per version.

    Intermediate per-statistic part (grid='integration', its own DiagonalCovariance
    from TreeCorr varxip/varxim). NOT folded into the terminal {version}.sacc (see
    #247 ruling) — COSEBIs and pure-E/B consume it directly.
    """
    return str(COSMO_VAL / f"{version}_xi_integration.sacc")


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

rule cv_pseudo_cl:
    """Pseudo-Cl E/B spectra for all versions (NaMaster), born as SACC parts."""
    output:
        pseudo_cl=[cv_pseudo_cl_sacc(v) for v in CV_VERSIONS],
    params:
        **cv_params(),
    threads: 12
    resources:
        mem_mb=32000,
        runtime=180,
    script:
        "../scripts/cv_pseudo_cl.py"


# ---------------------------------------------------------------------------
# Pure E/B modes and COSEBIs (per version), then the B-mode summary
# ---------------------------------------------------------------------------

# On a data run the COSEBIs / pure-E/B parts re-derive their E-mode vector from
# the *blinded* integration ξ± (COSEBIs) or blinded reporting + integration ξ±
# (pure-E/B), and stamp the derived part concealed from the version's
# commitment.json. blindable_part returns the plaintext part on a mock run and
# the blinded part on a data run, so the ξ± inputs bind unconditionally for every
# version; only the commitment is gated on is_data_run().
def cv_cosebis_inputs(w):
    inputs = {
        "xi": cv_xi_txt(w.version),
        "xi_integration": blindable_part(cv_xi_integration_sacc(w.version)),
    }
    if is_data_run():
        inputs["commitment"] = blind_state_paths(w.version)["commitment"]
    return inputs


def cv_pure_eb_inputs(w):
    inputs = {
        "xi": cv_xi_txt(w.version),
        "xi_reporting": blindable_part(cv_xi_reporting_sacc(w.version)),
        "xi_integration": blindable_part(cv_xi_integration_sacc(w.version)),
    }
    if is_data_run():
        inputs["commitment"] = blind_state_paths(w.version)["commitment"]
    return inputs


rule cv_pure_eb:
    """Pure E/B-mode decomposition for one version (config-space)."""
    input:
        unpack(cv_pure_eb_inputs),
    output:
        npz=cv_pure_eb_npz("{version}"),
        sacc=cv_pure_eb_sacc("{version}"),
    params:
        version="{version}",
        min_sep_int=CV["integration"]["min_sep"],
        max_sep_int=CV["integration"]["max_sep"],
        nbins_int=CV["integration"]["nbins"],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        cv_init=lambda w: cv_init_params(config, version_list=[w.version]),
        rundir=CV_RUNDIR,
    threads: 24
    resources:
        mem_mb=40000,
        runtime=360,
    script:
        "../scripts/cv_pure_eb.py"


rule cv_cosebis:
    """COSEBIs E/B decomposition for one version (config-space, fine binning)."""
    input:
        unpack(cv_cosebis_inputs),
    output:
        npz=cv_cosebis_npz("{version}"),
        sacc=cv_cosebis_sacc("{version}"),
    params:
        version="{version}",
        min_sep_int=CV["cosebis"]["min_sep_int"],
        max_sep_int=CV["cosebis"]["max_sep_int"],
        nbins_int=CV["cosebis"]["nbins_int"],
        npatch=CV["cosebis"]["npatch"],
        nmodes=CV["cosebis"]["nmodes"],
        scale_cuts=CV["cosebis"]["scale_cuts"],
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        cv_init=lambda w: cv_init_params(config, version_list=[w.version]),
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
            [cv_pseudo_cl_sacc(v) for v in CV_VERSIONS]
            if CV.get("include_pseudo_cl", False) else []
        ),
    output:
        summary_json=str(COSMO_VAL / "bmode_summary.json"),
    params:
        fiducial_scale_cut=CV["fiducial_scale_cut"],
        pure_eb_min_sep_int=CV["integration"]["min_sep"],
        pure_eb_max_sep_int=CV["integration"]["max_sep"],
        pure_eb_nbins_int=CV["integration"]["nbins"],
        cosebis_min_sep_int=CV["cosebis"]["min_sep_int"],
        cosebis_max_sep_int=CV["cosebis"]["max_sep_int"],
        cosebis_nbins_int=CV["cosebis"]["nbins_int"],
        cosebis_npatch=CV["cosebis"]["npatch"],
        cosebis_nmodes=CV["cosebis"]["nmodes"],
        cosebis_scale_cuts=CV["cosebis"]["scale_cuts"],
        include_pseudo_cl=CV.get("include_pseudo_cl", False),
        **cv_params(),
    threads: 24
    resources:
        mem_mb=48000,
        runtime=600,
    script:
        "../scripts/cv_summarize_bmodes.py"


# ---------------------------------------------------------------------------
# Terminal analysis file: assemble the per-statistic SACC parts into {version}.sacc
# ---------------------------------------------------------------------------
# The five born-as-SACC parts (xi_reporting, pseudo_cl, cosebis, pure_eb, rho_tau)
# are each written by their own rule carrying its own covariance block, except
# ξ± reporting and pseudo-Cℓ which are born cov-less by design. assemble_sacc.py
# loads the parts in canonical order and rebuilds one {version}.sacc with a
# single BlockDiagonalCovariance (point-insertion order = block order).
#
# The integration-grid ξ± (grid='integration') is deliberately NOT gathered here:
# it persists as its own per-part intermediate {version}_xi_integration.sacc,
# consumed by COSEBIs/pure-E/B, with Snakemake provenance covering traceability
# (see #247 ruling). The terminal file carries the analysis vector only.
#
# The pseudo-Cℓ part is the TAGGED, blinded inference product (blind=A, powspace,
# nbins=32) — the same pseudo-Cℓ today's cosmosis_fitting.py consumes — so the
# analysis file stays byte-comparable against it (PR-3's converter). Its real
# NaMaster covariance is injected here from the matching pseudo_cl_cov FITS
# (COVAR_EE_EE/BB_BB/EB_EB → block-diagonal, dropping cross-spectra, matching the
# B-mode PTE's use of COVAR_BB_BB). The ξ± reporting block is the one piece not yet
# sourced from its real covariance: the CosmoCov theory .txt is blind/gaussian/
# mask-keyed and lives deep in the inference tree, so wiring it couples cosmo_val
# to the whole inference covariance DAG — that sourcing is PR-3's converter
# territory. Until then a documented diagonal placeholder keeps the ξ block (and
# so the BlockDiagonalCovariance) structurally valid; it is a flagged stand-in, never a
# science covariance, and plugs out via --xi-cov the moment PR 3 lands.


def cv_assemble_inputs(version):
    """The per-statistic SACC parts + covariance inputs assemble_sacc consumes.

    Each part's filename carries enough to bind its producing rule's wildcards
    (the reporting ξ± and ρ/τ parts their reporting binning; the pseudo-Cℓ part its
    fiducial harmonic tag). pseudo_cl (+ its cov) is included only when the
    config toggles the harmonic-space BB into the analysis.
    """
    # blindable_part binds the raw-signal parts (reporting ξ±, analysis pseudo-Cℓ)
    # to their blinded siblings on a data run and to the plaintext on a mock run.
    # COSEBIs and pure-E/B are born blinded (their writers derive them from the
    # blinded integration ξ± and stamp concealed=True), so they bind by their own
    # name in both cases; ρ/τ is a diagnostic carrying no cosmological vector but
    # is stamped concealed pass-through so the fail-closed load gate admits it on
    # a data run. The integration-grid ξ± itself is blinded at birth (see rule
    # xi_highres) but is NOT gathered into the terminal file — it persists as its
    # own per-part intermediate (see #247 ruling), consumed by COSEBIs/pure-E/B.
    parts = dict(
        xi_reporting=blindable_part(cv_xi_reporting_sacc(version)),
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
        # Run type (data|mock) gates unblinded loading in assemble_sacc.py: a
        # 'data' run fails closed on unblinded parts, a 'mock' run loads freely.
        # Production runs on real catalogues, so the default is 'data'. PR #253's
        # blind-at-birth conceals each data part, letting the 'data' run assemble.
        type=CV.get("type", "data"),
        # Statistics this rule wired (same toggles as cv_assemble_inputs). The
        # script validates part_paths against this so a typo'd input keyword
        # can't silently drop a statistic from the terminal file.
        expected=lambda w: [
            k for k in cv_assemble_inputs(w.version) if k != "pseudo_cl_cov"
        ],
        # ξ± reporting has no real covariance wired yet (its CosmoCov theory block is
        # PR-3's converter territory, plugging in via --xi-cov). By DEFAULT this
        # is fatal: assemble_sacc.py raises rather than ship {version}.sacc — the
        # terminal science file — with a var=1.0 placeholder as its LEADING
        # covariance block (~20 orders off the real ξ± variance → silent
        # catastrophic χ²/PTE for any consumer). Only an explicit config opt-in
        # (cosmo_val.allow_placeholder_cov: true — dry-run / test configs) attaches
        # the flagged diagonal placeholder. The pseudo-Cℓ block is real (from the
        # pseudo_cl_cov input); COSEBIs / pure-E/B / ρ/τ carry their own.
        placeholder_var=(1.0 if CV.get("allow_placeholder_cov", False) else None),
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
        # Terminal analysis file: the assembled {version}.sacc per version
        [cv_analysis_sacc(v) for v in CV_VERSIONS],
