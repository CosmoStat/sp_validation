# Two-point data-vector rules: xi, rho/tau, and pseudo-Cl products.
# WORKFLOW_SCRIPTS (from common.py) is the generic workflow's scripts dir,
# resolved from the running checkout — used by the raw-shell MPI xi_highres rule.


rule xi:
    input:
        catalog=get_shear_catalog,
    output:
        # Raw TreeCorr .txt byproduct (read back by covariance + skip-if-exists)
        # and the born-as-SACC reporting ξ± part (no covariance until the
        # assemble_sacc rule injects the CosmoCov block). Both outputs carry the
        # same reporting-binning wildcards — Snakemake requires every output of a
        # rule to share one wildcard set, and it keeps the reporting .sacc name
        # self-describing so requesting it binds the xi job unambiguously.
        txt=str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.txt"),
        # Blindable part: temp() on a data run so only its blinded sibling
        # persists (blind_part escrows the true vector first). See common.maybe_temp.
        xi_reporting=maybe_temp(str(COSMO_VAL / "{version}_xi_reporting_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc")),
    threads: 24
    params:
        ver="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        # Stamped as the part's SACC `type` — custody state at assembly
        # (see blinding.assert_consistent_blind).
        type=run_type(),
    resources:
        mem_mb=30000,
        disk_mb=20000,
        runtime=360,
    script:
        "../scripts/run_2pcf.py"


# Integration-grid ξ± measured by xi_highres. The cosmo_val paper owns a dedicated
# cosmo_val.integration block ([0.08, 300] @ 1000 bins); other workflows sharing
# this file (e.g. papers/bmodes, whose config carries no cosmo_val section) fall
# back to their fiducial integration grid. Evaluated at parse time, so the lookup
# must not assume the cosmo_val key exists.
_INTEGRATION = config.get("cosmo_val", {}).get("integration") or {
    "min_sep": FIDUCIAL["min_sep_int"],
    "max_sep": FIDUCIAL["max_sep_int"],
    "nbins": FIDUCIAL["nbins_int"],
}


rule xi_highres:
    """High-resolution integration-grid xi for COSEBIs + pure-E/B, per version.

    Intermediate born-as-SACC part: {version}_xi_integration.sacc (a
    DiagonalCovariance from TreeCorr varxip/varxim). COSEBIs and pure-E/B consume
    it; it stays a standalone per-part file and does not join the terminal
    {version}.sacc (see #247 ruling). The raw .txt dump is kept as a convergence
    byproduct.

    In-container single-process TreeCorr: at the config-driven nbins_int=1000 grid
    this is a normal single-node job (the global container: in the Snakefile makes
    a plain shell: run in-container). run_2pcf_highres.py runs its single-process
    path when not launched under mpiexec. The historical 10k-bin bare-host MPI path
    is removed as unnecessary.
    """
    input:
        catalog=get_shear_catalog,
    output:
        # Only the uniquely-named SACC part is tracked. The raw TreeCorr .txt dump
        # run_2pcf_highres.py writes ({version}_xi_minsep=..._nbins=..._npatch=1.txt)
        # is left UNDECLARED: it is a convergence byproduct nothing in the DAG
        # consumes (cv_xi_txt is the reporting grid), and declaring it would collide
        # with rule xi's wildcard txt output (same filename pattern) — an
        # AmbiguousRuleException. Shared integration grid (cosmo_val.integration:
        # [0.08, 300] at 1000 bins) so the single part serves both consumers:
        # pure-E/B needs it to strictly contain its reporting grid down to 0.08;
        # COSEBIs scale-cuts on the same part. Decoupled from covariance.smk.
        # Blindable part: temp() on a data run so blind_part produces the _blinded
        # sibling the COSEBIs/pure-E/B consumers bind (see rule xi / common.maybe_temp).
        xi_integration=maybe_temp(str(COSMO_VAL / "{version}_xi_integration.sacc")),
    params:
        version="{version}",
        cat_config=CAT_CONFIG,
        min_sep=_INTEGRATION["min_sep"],
        max_sep=_INTEGRATION["max_sep"],
        nbins=_INTEGRATION["nbins"],
        out=str(COSMO_VAL),
        scripts=WORKFLOW_SCRIPTS,
        run_type=run_type(),
    threads: 24
    resources:
        mem_mb=40000,
        runtime=600,
    shell:
        "python {params.scripts}/run_2pcf_highres.py "
        "--version {params.version} --cat-config {params.cat_config} "
        "--min-sep {params.min_sep} --max-sep {params.max_sep} "
        "--nbins {params.nbins} --npatch 1 --out {params.out} "
        "--run-type {params.run_type}"


rule run_cosmo_val:
    """Full CosmoVal diagnostic suite."""
    output:
        sentinel=str(COSMO_VAL / "run_cosmo_val.done"),
    threads: 24
    resources:
        mem_mb=60000,
        disk_mb=20000,
        runtime=360,
    shell:
        """
        export PYTHONPATH="/home/cdaley/.local/lib/python3.12/site-packages:${{PYTHONPATH:-}}"
        cd /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val \
        && python run_cosmo_val.py \
        && touch {output.sentinel}
        """


rule rho_tau_stats:
    # ρ/τ has no blindable input; it binds only the commitment, to stamp its
    # part concealed pass-through (see common.commitment_input).
    input:
        unpack(lambda w: commitment_input(w.version)),
    output:
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # Born-as-SACC ρ/τ part (ρ_0…ρ_5 autos + τ_0/τ_2/τ_5 leakage, carrying
        # its own covariance block) that the assemble_sacc rule consumes;
        # calculate_rho_tau_stats writes it alongside the FITS via
        # rho_tau_to_sacc_part.
        rho_tau=str(COSMO_VAL / "rho_tau_stats/rho_tau_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc"),
    threads: 48
    params:
        ver="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        type=run_type(),
    resources:
        mem_mb=30000,
        disk_mb=20000,
    script:
        "../scripts/run_rho_tau.py"


# Pseudo-Cl generation for harmonic-space data vectors and COSEBIS validation.
BASE_VERSIONS = [v.replace("_leak_corr", "") for v in config["versions"]]

wildcard_constraints:
    binning="linear|logspace|powspace",


rule pseudo_cl:
    """Generate pseudo-Cl data vector (born as SACC) with configurable binning.

    NB: the ``blind`` wildcard is the glass-mock A/B/C variant (three mock
    catalogues), NOT Smokescreen blinding — see common.py BLINDS. The Smokescreen
    concealed=True stamp is a separate axis on the SACC file.
    """
    output:
        # This generic rule produces every pseudo-Cℓ variant — the analysis part
        # (blind=A, powspace, nbins=32) folded into {version}.sacc, plus the fine
        # (COSEBIS) and glass-mock variants. Only the analysis part is a terminal
        # blindable, and a data run blinds it via a requested _blinded sibling
        # (blind_part reads this plaintext); the fine/mock variants are B-mode /
        # validation intermediates left untouched here. The output is therefore
        # not temp()'d — see the PR note on residual unblinded pseudo-Cℓ.
        pseudo_cl=str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_{binning}_nbins={nbins}.sacc"),
    wildcard_constraints:
        blind="[ABC]",  # glass-mock variant, not Smokescreen blinding
    params:
        version="{version}",
        blind="{blind}",
        cat_config=CAT_CONFIG,
        nside=1024,
        npatch=1,
        cosmo_params=PLANCK18,
        binning="{binning}",
        nbins=lambda w: int(w.nbins),
        power=0.5,
    resources:
        mem_mb=32000,
        runtime=120,
    threads: 12
    script:
        "../scripts/generate_pseudo_cl.py"


rule pseudo_cl_cov:
    """Generate pseudo-Cl covariance with configurable binning.

    NB: ``blind`` is the glass-mock A/B/C variant, not Smokescreen blinding
    (see common.py BLINDS).
    """
    output:
        pseudo_cl_cov=str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_{binning}_nbins={nbins}.fits"),
    wildcard_constraints:
        blind="[ABC]",  # glass-mock variant, not Smokescreen blinding
    params:
        version="{version}",
        blind="{blind}",
        cat_config=CAT_CONFIG,
        nside=1024,
        npatch=1,
        cosmo_params=PLANCK18,
        binning="{binning}",
        nbins=lambda w: int(w.nbins),
        power=0.5,
    resources:
        mem_mb=16000,
        runtime=180,
    threads: 12
    script:
        "../scripts/generate_pseudo_cl_cov.py"


PSEUDO_CL_VERSIONS = config["versions"]


rule pseudo_cl_all:
    """Generate pseudo-Cls for all versions."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind=A_powspace_nbins=32.sacc"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_cov_all:
    """Generate pseudo-Cl covariances for all versions."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_cov_{version}_blind=A_powspace_nbins=32.fits"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_fine_all:
    """Generate fine pseudo-Cls for COSEBIS."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_linear_nbins=2040.sacc"),
            version=config["versions"],
            blind=BLINDS,
        ),
