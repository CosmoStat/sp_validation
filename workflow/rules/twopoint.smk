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
    resources:
        mem_mb=30000,
        disk_mb=20000,
        runtime=360,
    script:
        "../scripts/run_2pcf.py"


rule xi_highres:
    """High-resolution xi for COSEBIS integration.

    Born-as-SACC part: {version}_xi_integration.sacc (a DiagonalCovariance from
    TreeCorr varxip/varxim). COSEBIs and pure-E/B consume it, and rule
    assemble_sacc folds its integration-grid ξ± rows (grid='integration') into
    the single terminal {version}.sacc. The raw .txt dump is kept as a
    convergence byproduct.
    """
    container: None
    output:
        txt=str(COSMO_VAL / f"{FIDUCIAL['version']}_xi_minsep={FIDUCIAL['min_sep_int']}_maxsep={FIDUCIAL['max_sep_int']}_nbins=10000_npatch=1.txt"),
        # Blindable part: temp() on a data run (see rule xi / common.maybe_temp).
        xi_integration=maybe_temp(str(COSMO_VAL / f"{FIDUCIAL['version']}_xi_integration.sacc")),
    resources:
        tasks=30,
        cpus_per_task=12,
        nodes=6,
        mem_mb_per_cpu=2000,
        runtime=2880,
        slurm_extra="'--exclude=n17,n09,n36 --partition=pscomp'",
        mpi="/softs/openmpi/5.0.5-slurm-CentOS8/bin/mpiexec",
    shell:
        "{resources.mpi} -n {resources.tasks} "
        "apptainer exec "
        "--bind /home,/n09data,/n17data,/n23data1,/softs "
        "--env LD_LIBRARY_PATH=/softs/openmpi/5.0.5-slurm-CentOS8/lib "
        "/n17data/cdaley/containers/containers "
        f"python {WORKFLOW_SCRIPTS}/run_2pcf_highres.py"


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
