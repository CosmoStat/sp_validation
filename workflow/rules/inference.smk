# Imports from Snakefile: FIDUCIAL, COSMO_INFERENCE, COSMO_VAL, covariance_path, build_redshift_path, fiducial_binning_suffix
# NOTE (2026-08-01): FITS-CONTENT plumbing reconciled -- cosmosis_fitting.py's
# cov_cl_to_fits() now reads COVAR_EE_EE (the block pseudo_cl.py actually
# writes) instead of a nonexistent COVAR_FULL. inference_prep runs end-to-end.
# inference_fiducial itself has no output/shell -- it only verifies inference_prep's
# outputs exist; running the CosmoSIS sampler on the generated .ini is still a
# manual step (cosmosis_fitting.py prints the command), not wired into this DAG.

# Output root for CosmoSIS data products + configs. COSMO_INFERENCE (common.py)
# already resolves to THIS repo's cosmo_inference dir, so the products land
# beside the code that builds them rather than in a contributor's home.
COSMO_INFERENCE_PROD = COSMO_INFERENCE
# Working directory for the cosmosis_fitting.py invocation — the same repo dir.
COSMO_INFERENCE_RUNDIR = str(COSMO_INFERENCE)

# External chain/mock locations are deployment-specific, so they live in config.
INFERENCE = config["inference"]
CHAINS_DIR = INFERENCE["chains_dir"]            # CosmoSIS chain output root (real data)
GLASS_MOCK_DATA_DIR = INFERENCE["glass_mock_data_dir"]    # precomputed mock xi/Cl products
GLASS_MOCK_CHAINS_DIR = INFERENCE["glass_mock_chains_dir"]  # mock chain output root

PSEUDO_CL_DIR = COSMO_VAL  # producer (twopoint.smk) writes pseudo_cl* here
GLASS_MOCK_VERSION = config["glass_mocks"].get("version", "v0")
GLASS_MOCK_SEED_RANGE = config["glass_mocks"]["seed_range"]

GLASS_MOCK_FITS_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"data/glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}"
    / f"cosmosis_glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}.fits"
)
GLASS_MOCK_CONFIG_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"cosmosis_config/output/cosmosis_pipeline_glass_mocks_{GLASS_MOCK_VERSION}_glass_mock_{{mock_id}}.ini"
)

# Fiducial harmonic-binning tag the pseudo-Cl producer (twopoint.smk) stamps
# into the filename. These are NOT inference_prep wildcards, so the consumer
# reads them from config to reconstruct the exact name the producer emits
# (canonical: blind=A, powspace, nbins=32 — see twopoint.smk pseudo_cl_all).
HARMONIC_FIDUCIAL = config["harmonic"]["fiducial"]
PSEUDO_CL_TAG = (
    f"blind={HARMONIC_FIDUCIAL['blind']}"
    f"_{HARMONIC_FIDUCIAL['binning']}"
    f"_nbins={HARMONIC_FIDUCIAL['nbins']}"
)


def pseudo_cl_assets(version):
    """Return pseudo-Cl and covariance paths for the requested catalog version.

    The producer (twopoint.smk rules pseudo_cl / pseudo_cl_cov) writes
    wildcard-tagged names; the consumer reconstructs them from the fiducial
    harmonic-binning config so the requested path matches byte-for-byte.
    """
    cl_path = PSEUDO_CL_DIR / f"pseudo_cl_{version}_{PSEUDO_CL_TAG}.fits"
    cov_path = PSEUDO_CL_DIR / f"pseudo_cl_cov_{version}_{PSEUDO_CL_TAG}.fits"
    return str(cl_path), str(cov_path)

rule inference_prep:
    input:
        # Processed covariance matrix - use centralized covariance_path()
        cov_matrix=lambda w: covariance_path(w.version, w.blind, min_sep=w.min_sep, max_sep=w.max_sep, nbins=w.nbins),
        # Xi FITS files
        xi_plus=str(COSMO_VAL / "xi_plus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        xi_minus=str(COSMO_VAL / "xi_minus_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # n(z) file (using new location with base version mapping)
        nz_file=lambda w: build_redshift_path(w.version, w.blind),
        # rho/tau stats
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # tau covariance (tracked as dependency)
        tau_cov=str(COSMO_VAL / "rho_tau_stats/cov_tau_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}_th.npy"),
        pseudo_cl=lambda w: pseudo_cl_assets(w.version)[0],
        pseudo_cl_cov=lambda w: pseudo_cl_assets(w.version)[1],
    output:
        fits_file=str(
            COSMO_INFERENCE_PROD
            / "data/{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}/cosmosis_{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"
        ),
        config_file=str(
            COSMO_INFERENCE_PROD
            / "cosmosis_config/output/cosmosis_pipeline_{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.ini"
        )
    # keep {npatch} from swallowing the "_nautilus" suffix of inference_prep_nautilus
    wildcard_constraints:
        npatch=r"\d+",
    params:
        cosmosis_root="{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}",
        data_dir=f"{CHAINS_DIR}/{{version}}_{{blind}}_minsep={{min_sep}}_maxsep={{max_sep}}_nbins={{nbins}}_npatch={{npatch}}",
        output_root=str(COSMO_INFERENCE_PROD),
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd {COSMO_INFERENCE_RUNDIR}

        # Run inference preparation step with cosmosis_fitting.py
        python scripts/cosmosis_fitting.py \
            --cosmosis-root {params.cosmosis_root} \
            --nz-file {input.nz_file} \
            --data-dir {params.data_dir} \
            --output-root {params.output_root} \
            --xi {input.xi_plus} {input.xi_minus} \
            --cov-xi {input.cov_matrix} \
            --use-rho-tau \
            --rho-stats {input.rho_stats} \
            --tau-stats {input.tau_stats} \
            --cov-tau {input.tau_cov} \
            --cl-file {input.pseudo_cl} \
            --cov-cl {input.pseudo_cl_cov}
        """


# cosmosis_root as used by cosmosis_fitting.py / the generated ini's [output]
# filename (SCRATCH = {CHAINS_DIR}/{FIDUCIAL_COSMOSIS_ROOT}).
FIDUCIAL_COSMOSIS_ROOT = (
    f"{FIDUCIAL['version']}_{FIDUCIAL['blind']}"
    f"_minsep={FIDUCIAL['min_sep']}_maxsep={FIDUCIAL['max_sep']}"
    f"_nbins={FIDUCIAL['nbins']}_npatch={FIDUCIAL['npatch']}"
)


rule inference_run_fiducial:
    """Run the CosmoSIS PolyChord sampler for the fiducial parameter set.

    cosmosis lives in the user's own ~/.local install, not the apptainer
    image, so this runs directly on the compute node (container: None).
    PolyChord resumes from its checkpoint (resume=T in the generated ini)
    on rerun, so an under-estimated runtime just means a later rerun picks
    up where it left off rather than starting over.
    """
    input:
        config_file=rules.inference_prep.output.config_file.format(
            version=FIDUCIAL["version"], blind=FIDUCIAL["blind"],
            min_sep=FIDUCIAL["min_sep"], max_sep=FIDUCIAL["max_sep"],
            nbins=FIDUCIAL["nbins"], npatch=FIDUCIAL["npatch"]
        ),
    output:
        # ini [output] filename = %(SCRATCH)s/{root}/samples_{root}.txt, and
        # SCRATCH is itself {CHAINS_DIR}/{root} -- {root} appears twice.
        samples=f"{CHAINS_DIR}/{FIDUCIAL_COSMOSIS_ROOT}/{FIDUCIAL_COSMOSIS_ROOT}/samples_{FIDUCIAL_COSMOSIS_ROOT}.txt",
    container: None
    threads: 1
    resources:
        mem_mb=16000,
        runtime=2820,  # ~47h, under the comp/pscomp 2-day cap
    shell:
        """
        set +u
        source ~/.local/bin/cosmosis-configure
        set -u
        cd {COSMO_INFERENCE_RUNDIR}
        cosmosis {input.config_file}
        """


rule inference_fiducial:
    input:
        # Use the same output patterns as inference_prep with FIDUCIAL params
        rules.inference_prep.output.fits_file.format(
            version=FIDUCIAL["version"], blind=FIDUCIAL["blind"],
            min_sep=FIDUCIAL["min_sep"], max_sep=FIDUCIAL["max_sep"],
            nbins=FIDUCIAL["nbins"], npatch=FIDUCIAL["npatch"]
        ),
        rules.inference_prep.output.config_file.format(
            version=FIDUCIAL["version"], blind=FIDUCIAL["blind"],
            min_sep=FIDUCIAL["min_sep"], max_sep=FIDUCIAL["max_sep"],
            nbins=FIDUCIAL["nbins"], npatch=FIDUCIAL["npatch"]
        ),
        rules.inference_run_fiducial.output.samples,


# Ad hoc alternative to the polychord fiducial run above: blind-A inputs at
# the fiducial binning, but the nautilus sampler (48-core SMP, no MPI rebuild
# needed) and chains written locally under COSMO_INFERENCE instead of
# guerrini's shared CHAINS_DIR. Kept as separate rules/root name so this
# doesn't disturb the polychord run above.
#
# The catalogue version is a wildcard, so the same pipeline, priors, sampler
# settings and scale cuts serve every catalogue in inference.nautilus_versions
# (mccd_vs_psfex: PSFEx v1.4.6.3 against MCCD v1.3.6.3). For the fiducial
# version the paths are the ones the fiducial-only rules used to write.
#
# {nz_tag} is empty for a catalogue's own n(z), or "_nz_<version>" to borrow
# another catalogue's n(z) (same blind letter) -- e.g. MCCD with the v1.4.6.3
# n(z), so that the two chains share one blinding offset and one delta-z
# calibration. The xi covariance stays the catalogue's own.
#
# {prior_tag} is empty for the shared priors_psf.ini, or "_psfprior_own" to use
# cosmosis_config/templates/priors_psf_<version>.ini, whose PSF-leakage
# (alpha, beta) prior is centred on that catalogue's own rho/tau fit -- the
# procedure that produced the priors_psf.ini values for PSFEx v1.4.6.3.
NAUTILUS_SUFFIX = (
    f"_{FIDUCIAL['blind']}_minsep={FIDUCIAL['min_sep']}_maxsep={FIDUCIAL['max_sep']}"
    f"_nbins={FIDUCIAL['nbins']}_npatch={FIDUCIAL['npatch']}_nautilus"
)
NAUTILUS_COSMOSIS_ROOT = "{version}{nz_tag}{prior_tag}" + NAUTILUS_SUFFIX
NAUTILUS_TAG_CONSTRAINTS = dict(nz_tag=r"(_nz_SP_v[\d.]+)?", prior_tag=r"(_psfprior_own)?")
NAUTILUS_CHAINS_DIR = str(COSMO_INFERENCE / "chains")
NAUTILUS_VERSIONS = INFERENCE.get("nautilus_versions", [FIDUCIAL["version"]])
# Borrowed-n(z) variants: list of [version, n(z) version] pairs.
NAUTILUS_NZ_VARIANTS = INFERENCE.get("nautilus_nz_variants", [])
# Own-PSF-prior variants: list of versions.
NAUTILUS_PRIOR_VARIANTS = INFERENCE.get("nautilus_psfprior_own_versions", [])


def nautilus_priors_arg(w):
    return f"--priors-file priors_psf_{w.version}.ini" if w.prior_tag else ""


def nautilus_nz_file(w):
    nz_version = w.nz_tag[len("_nz_"):] if w.nz_tag else w.version
    return build_redshift_path(nz_version, FIDUCIAL["blind"])
# Versions whose pseudo-C_ell data vector is packed into the FITS file. The
# real-space likelihood (XI_PLUS XI_MINUS TAU_0_PLUS TAU_2_PLUS) never reads
# CELL_EE, and the C_ell covariance is a separate diagonal block, so this is
# inert for the xi chain; it only keeps the fiducial FITS file as it was.
NAUTILUS_CL_VERSIONS = INFERENCE.get("nautilus_cl_versions", [FIDUCIAL["version"]])


def nautilus_pseudo_cl_args(w):
    if w.version not in NAUTILUS_CL_VERSIONS:
        return ""
    cl, cov = pseudo_cl_assets(w.version)
    return f"--cl-file {cl} --cov-cl {cov}"


rule inference_prep_nautilus:
    input:
        cov_matrix=lambda w: covariance_path(
            w.version, FIDUCIAL["blind"],
            min_sep=FIDUCIAL["min_sep"], max_sep=FIDUCIAL["max_sep"], nbins=FIDUCIAL["nbins"]
        ),
        xi_plus=str(COSMO_VAL / f"xi_plus_{{version}}{fiducial_binning_suffix()}.fits"),
        xi_minus=str(COSMO_VAL / f"xi_minus_{{version}}{fiducial_binning_suffix()}.fits"),
        nz_file=nautilus_nz_file,
        rho_stats=str(COSMO_VAL / f"rho_tau_stats/rho_stats_{{version}}{fiducial_binning_suffix()}.fits"),
        tau_stats=str(COSMO_VAL / f"rho_tau_stats/tau_stats_{{version}}{fiducial_binning_suffix()}.fits"),
        tau_cov=str(COSMO_VAL / f"rho_tau_stats/cov_tau_{{version}}{fiducial_binning_suffix()}_th.npy"),
        pseudo_cl=lambda w: (
            [pseudo_cl_assets(w.version)[0]] if w.version in NAUTILUS_CL_VERSIONS else []
        ),
        pseudo_cl_cov=lambda w: (
            [pseudo_cl_assets(w.version)[1]] if w.version in NAUTILUS_CL_VERSIONS else []
        ),
    output:
        fits_file=str(COSMO_INFERENCE_PROD / f"data/{NAUTILUS_COSMOSIS_ROOT}/cosmosis_{NAUTILUS_COSMOSIS_ROOT}.fits"),
        config_file=str(COSMO_INFERENCE_PROD / f"cosmosis_config/output/cosmosis_pipeline_{NAUTILUS_COSMOSIS_ROOT}.ini"),
    wildcard_constraints:
        **NAUTILUS_TAG_CONSTRAINTS,
    params:
        cosmosis_root=NAUTILUS_COSMOSIS_ROOT,
        data_dir=f"{NAUTILUS_CHAINS_DIR}/{NAUTILUS_COSMOSIS_ROOT}",
        output_root=str(COSMO_INFERENCE_PROD),
        cl_args=nautilus_pseudo_cl_args,
        priors_arg=nautilus_priors_arg,
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd {COSMO_INFERENCE_RUNDIR}

        python scripts/cosmosis_fitting.py \
            --cosmosis-root {params.cosmosis_root} \
            --sampler-template cosmosis_pipeline_A_psf_nautilus.ini \
            --nz-file {input.nz_file} \
            --data-dir {params.data_dir} \
            --output-root {params.output_root} \
            --xi {input.xi_plus} {input.xi_minus} \
            --cov-xi {input.cov_matrix} \
            --use-rho-tau \
            --rho-stats {input.rho_stats} \
            --tau-stats {input.tau_stats} \
            --cov-tau {input.tau_cov} \
            {params.cl_args} \
            {params.priors_arg}
        """


rule inference_run_nautilus:
    """Run the nautilus sampler with 48-core SMP (cosmosis --smp, plain
    Python multiprocessing -- no MPI build needed, unlike polychord).
    """
    input:
        config_file=rules.inference_prep_nautilus.output.config_file,
    output:
        # Same double-nesting as the polychord samples path (see above):
        # ini [output] filename = %(SCRATCH)s/{root}/samples_{root}.txt,
        # and SCRATCH is itself {data_dir} = {NAUTILUS_CHAINS_DIR}/{root}.
        samples=f"{NAUTILUS_CHAINS_DIR}/{NAUTILUS_COSMOSIS_ROOT}/{NAUTILUS_COSMOSIS_ROOT}/samples_{NAUTILUS_COSMOSIS_ROOT}.txt",
    wildcard_constraints:
        **NAUTILUS_TAG_CONSTRAINTS,
    container: None
    threads: 1
    resources:
        mem_mb=64000,
        runtime=2820,  # ~47h, under the comp/pscomp 2-day cap
        cpus_per_task=48,
        # n23 confirmed 2026-08-03: /n23data1 automount broken there (even
        # plain ls/cd fail) -- every SLURM attempt landing there died
        # instantly with no log, before our script could even start.
        slurm_extra="--exclude=n17,n09,n36,n23",
    shell:
        """
        set +u
        source ~/.local/bin/cosmosis-configure
        set -u
        # Cap internal per-process threading: with 48 SMP worker processes,
        # unrestricted OpenBLAS/OMP threading blows past candide's per-user
        # ulimit -u (max processes/threads per node) almost instantly.
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1
        cd {COSMO_INFERENCE_RUNDIR}
        cosmosis --smp 48 {input.config_file}
        """


def nautilus_targets(version, nz_version=None, own_psf_prior=False):
    tags = dict(
        version=version,
        nz_tag=f"_nz_{nz_version}" if nz_version else "",
        prior_tag="_psfprior_own" if own_psf_prior else "",
    )
    return [
        rules.inference_prep_nautilus.output.fits_file.format(**tags),
        rules.inference_prep_nautilus.output.config_file.format(**tags),
        rules.inference_run_nautilus.output.samples.format(**tags),
    ]


rule inference_nautilus_fiducial:
    input:
        nautilus_targets(FIDUCIAL["version"]),


rule inference_nautilus:
    """Nautilus chains for every catalogue in inference.nautilus_versions,
    plus the borrowed-n(z) variants (inference.nautilus_nz_variants) and the
    own-PSF-prior variants (inference.nautilus_psfprior_own_versions)."""
    input:
        [t for v in NAUTILUS_VERSIONS for t in nautilus_targets(v)],
        [t for v, nz in NAUTILUS_NZ_VARIANTS for t in nautilus_targets(v, nz)],
        [t for v in NAUTILUS_PRIOR_VARIANTS for t in nautilus_targets(v, own_psf_prior=True)],


rule inference_glass_mocks:
    input:
        expand(GLASS_MOCK_FITS_PATTERN, mock_id=[f"{i:05d}" for i in range(GLASS_MOCK_SEED_RANGE[0], GLASS_MOCK_SEED_RANGE[1] + 1)])


rule inference_prep_glass_mock:
    input:
        xi=f"{GLASS_MOCK_DATA_DIR}/xi_glass_mock_{{mock_id}}_4096_nbins=20.fits",
        # Use centralized covariance_path() with fiducial mock version
        cov_matrix=covariance_path(FIDUCIAL["mock_version"], "A"),
        # n(z) file
        nz_file=build_redshift_path(FIDUCIAL["mock_version"], "A"),
        # Rho/tau stats: rho from real data, tau sampled
        rho_stats=str(COSMO_VAL / f"rho_tau_stats/rho_stats_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}.fits"),
        tau_stats="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
        # Tau covariance (real data)
        tau_cov=str(COSMO_VAL / f"rho_tau_stats/cov_tau_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}_th.npy"),
        # C_ell data for dual config generation
        cl_file=f"{GLASS_MOCK_DATA_DIR}/cl_glass_mock_{{mock_id}}_4096.npy",
        cl_cov=pseudo_cl_assets(FIDUCIAL["mock_version"])[1],
    output:
        fits_file=GLASS_MOCK_FITS_PATTERN,
        config_file=GLASS_MOCK_CONFIG_PATTERN,
    params:
        cosmosis_root=f"glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        data_dir=f"{GLASS_MOCK_CHAINS_DIR}/glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        output_root=str(COSMO_INFERENCE_PROD),
        output_basename=f"glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}",
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd {COSMO_INFERENCE_RUNDIR}

        mkdir -p {params.data_dir}

        python scripts/cosmosis_fitting.py --mock \
            --cosmosis-root {params.cosmosis_root} \
            --nz-file {input.nz_file} \
            --data-dir {params.data_dir} \
            --output-root {params.output_root} \
            --output-basename {params.output_basename} \
            --xi {input.xi} \
            --cov-xi {input.cov_matrix} \
            --use-rho-tau \
            --rho-stats {input.rho_stats} \
            --tau-stats {input.tau_stats} \
            --cov-tau {input.tau_cov} \
            --cl-file {input.cl_file} \
            --cov-cl {input.cl_cov}
        """

localrules:
    inference_prep,
    inference_prep_glass_mock,
    inference_fiducial,
    inference_glass_mocks,
