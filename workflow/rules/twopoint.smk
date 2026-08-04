# Two-point data-vector rules: xi, rho/tau, and pseudo-Cl products.


rule xi:
    input:
        catalog=get_shear_catalog,
    output:
        str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.txt"),
        str(COSMO_VAL / "xi_plus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        str(COSMO_VAL / "xi_minus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
    threads: 24
    params:
        ver="{version}",
        compute_tomography="{compute_tomography}",
        npatch="{npatch}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
    resources:
        mem_mb=30000,
        disk_mb=20000,
        runtime=360,
    script:
        "../scripts/run_2pcf.py"


rule xi_highres:
    """High-resolution xi for COSEBIS integration."""
    container: None
    output:
        txt=str(COSMO_VAL / f"{FIDUCIAL['version']}_xi_minsep={FIDUCIAL['min_sep_int']}_maxsep={FIDUCIAL['max_sep_int']}_nbins=10000_npatch=1.txt"),
        xi_plus=str(COSMO_VAL / f"xi_plus_{FIDUCIAL['version']}_minsep={FIDUCIAL['min_sep_int']}_maxsep={FIDUCIAL['max_sep_int']}_nbins=10000_npatch=1.fits"),
        xi_minus=str(COSMO_VAL / f"xi_minus_{FIDUCIAL['version']}_minsep={FIDUCIAL['min_sep_int']}_maxsep={FIDUCIAL['max_sep_int']}_nbins=10000_npatch=1.fits"),
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
        "python /automnt/n17data/cdaley/unions/pure_eb/code/sp_validation/workflow/scripts/run_2pcf_highres.py"


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
    """Generate pseudo-Cl data vector with configurable binning."""
    output:
        pseudo_cl=str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_{binning}_nbins={nbins}.fits"),
    wildcard_constraints:
        blind="[ABC]",
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
    """Generate pseudo-Cl covariance with configurable binning."""
    output:
        pseudo_cl_cov=str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_{binning}_nbins={nbins}.fits"),
    wildcard_constraints:
        blind="[ABC]",
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
            str(COSMO_VAL / "pseudo_cl_{version}_blind=A_powspace_nbins=32.fits"),
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
            str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_linear_nbins=2040.fits"),
            version=config["versions"],
            blind=BLINDS,
        ),
