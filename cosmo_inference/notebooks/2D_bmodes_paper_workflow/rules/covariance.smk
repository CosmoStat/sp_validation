BLOCK_PAIRS = [("++", "1"), ("--", "2"), ("+-", "3")]




def get_cat_params(version):
    """Extract covariance parameters (area, n_e, sigma_e) from catalog config."""
    base_version = version.replace("_leak_corr", "")
    if base_version not in config:
        raise KeyError(f"Catalog configuration not found for {base_version}")
    cov_th = config[base_version]["cov_th"]
    return cov_th["A"], cov_th["n_e"], cov_th["sigma_e"]


# Wildcard constraints centralized in Snakefile


def covariance_paths(version, blind, gaussian, min_sep, max_sep, nbins, mask_suffix):
    """Return directory path and file prefix for covariance outputs."""
    covariance_root = str(COSMO_INFERENCE / "data/covariance")
    dir_path = (
        f"{covariance_root}/covariance_{version}_{blind}_{gaussian}"
        f"_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}"
    )
    file_prefix = (
        f"{dir_path}/covariance_{version}_{blind}_{gaussian}"
        f"_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}"
    )
    return dir_path, file_prefix


MASK_CLS_FILES = config["covariance"].get("mask_cls_files", {})
DEFAULT_MASK_SUFFIX = "_masked" if config["covariance"].get("default_masked", False) else ""


def get_mask_cls_path(version):
    """Return absolute mask Cl path for the requested catalog version."""
    version_dir = version.replace('_leak_corr', '').replace('SP_', '')
    return MASK_CLS_FILES.get(version_dir, "")


rule covariance_ini:
    input:
        nz_file=lambda w: build_redshift_path(w.version, w.blind),
        mask=lambda w: [] if w.mask_suffix != "_masked" else [get_mask_cls_path(w.version)],
    output:
        str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}.ini")
    params:
        outdir=lambda w: covariance_paths(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix
        )[0],
        ng_value=lambda wildcards: "1" if wildcards.gaussian == "ng" else "0",
        omega_m=config["covariance"]["cosmology"]["Omega_m"],
        sigma_8=config["covariance"]["cosmology"]["sigma_8"],
        n_s=config["covariance"]["cosmology"]["n_s"],
        h=config["covariance"]["cosmology"]["h"],
        omega_b=config["covariance"]["cosmology"]["Omega_b"],
        area=lambda w: get_cat_params(w.version)[0],
        n_e=lambda w: get_cat_params(w.version)[1],
        sigma_e_param=lambda w: get_cat_params(w.version)[2],
        mask=lambda w: get_mask_cls_path(w.version) if w.mask_suffix == "_masked" else "",
    threads: 1
    shell:
        """
        mkdir -p {params.outdir}

        cat > {output} << 'EOF'
#
# Cosmological parameters
#
Omega_m : {params.omega_m}
Omega_v : 0.75
sigma_8 : {params.sigma_8}
n_spec : {params.n_s}
w0 : -1
wa : 0
omb : {params.omega_b}
h0 : {params.h}


# Survey and galaxy parameters
#
# area in degrees
# n_gal,lens_n_gal in gals/arcmin^2

area : {params.area}
sourcephotoz : multihisto
lensphotoz : multihisto
source_tomobins : 1
lens_tomobins : 1
sigma_e : {params.sigma_e_param}
source_n_gal : {params.n_e}
lens_n_gal : {params.n_e}


shear_REDSHIFT_FILE : {input.nz_file}
clustering_REDSHIFT_FILE : {input.nz_file}
c_footprint_file : {params.mask}


# IA parameters
IA : 1
A_ia : 0.0
eta_ia : 0.0


# Covariance paramters
#
# tmin,tmax in arcminutes
tmin : {wildcards.min_sep}
tmax : {wildcards.max_sep}
ntheta : {wildcards.nbins}
ng : {params.ng_value}
cng : {params.ng_value}


#mkdir before running!
outdir : ./
filename : cov_tmp
ss : true
ls : false
ll : false
EOF
        """


rule covariance_cosmocov:
    input:
        rules.covariance_ini.output,
    output:
        str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/cov_tmp_ssss_{block_pm}_cov_Ntheta{nbins}_Ntomo1_{block_i}")
    params:
        block_i="{block_i}",
        outdir=lambda w: covariance_paths(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix
        )[0],
        ini_path=lambda w: covariance_paths(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix
        )[1]
        + ".ini",
    container:
        None
    threads: 1
    shell:
        """
        module load gcc
        module unload intelpython || true
        module load intelpython/3-2024.1.0
        module load openmpi

        cd {params.outdir}
        /n23data1/n06data/lgoh/scratch/UNIONS/CosmoCov/covs/cov {params.block_i} {params.ini_path}
        """


rule covariance_cat:
    input:
        cov_block=lambda w: [
            f"{covariance_paths(w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix)[0]}"
            f"/cov_tmp_ssss_{pm}_cov_Ntheta{w.nbins}_Ntomo1_{idx}"
            for pm, idx in BLOCK_PAIRS
        ],
    threads: 1
    output:
        str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}.txt")
    shell:
        """
        cat {input} > {output}
        """


rule covariance_glass_mock:
    input:
        xi=expand(
            "/n09data/guerrini/glass_mock_v1.4.6/results/xi_glass_mock_{seed:05d}_4096_nbins=20.fits",
            seed=range(config["glass_mocks"]["seed_range"][0], config["glass_mocks"]["seed_range"][1] + 1),
        ),
        cl=expand(
            "/n09data/guerrini/glass_mock_v1.4.6/results/cl_glass_mock_{seed:05d}_4096.npy",
            seed=range(config["glass_mocks"]["seed_range"][0], config["glass_mocks"]["seed_range"][1] + 1),
        ),
    output:
        xi_covariance="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/xi_covariance.npy",
        cl_covariance="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/cl_covariance.npy",
        combined_covariance="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/combined_covariance.npy",
        correlation_plot="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/combined_correlation.png",
        xi_mean="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/xi_mean.npy",
        cl_mean="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/cl_mean.npy",
        combined_mean="/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6/combined_mean.npy",
    script:
        "../scripts/compute_glass_mock_covariance.py"


rule generate_glass_mock_rhotau_samples:
    input:
        cov_rho=lambda w: str(
            COSMO_VAL / f"rho_tau_stats/cov_rho_{config['fiducial']['mock_version']}.npy"
        ),
        cov_tau=lambda w: str(
            COSMO_VAL
            / (
                "rho_tau_stats/"
                f"cov_tau_{config['fiducial']['mock_version']}"
                f"_minsep={config['fiducial']['min_sep']}"
                f"_maxsep={config['fiducial']['max_sep']}"
                f"_nbins={config['fiducial']['nbins']}"
                f"_npatch={config['fiducial']['npatch']}_th.npy"
            )
        ),
        ref_rho=lambda w: str(
            COSMO_VAL
            / (
                "rho_tau_stats/"
                f"rho_stats_{config['fiducial']['mock_version']}"
                f"_minsep={config['fiducial']['min_sep']}"
                f"_maxsep={config['fiducial']['max_sep']}"
                f"_nbins={config['fiducial']['nbins']}"
                f"_npatch={config['fiducial']['npatch']}.fits"
            )
        ),
        ref_tau=lambda w: str(
            COSMO_VAL
            / (
                "rho_tau_stats/"
                f"tau_stats_{config['fiducial']['mock_version']}"
                f"_minsep={config['fiducial']['min_sep']}"
                f"_maxsep={config['fiducial']['max_sep']}"
                f"_nbins={config['fiducial']['nbins']}"
                f"_npatch={config['fiducial']['npatch']}.fits"
            )
        ),
    output:
        rho="results/glass_mock_rhotau_samples/{mock_id}/rho_stats_sampled.fits",
        tau="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
    params:
        mock_id="{mock_id}",
        output_dir="results/glass_mock_rhotau_samples",
    threads: 1
    shell:
        """
        python workflow/scripts/generate_glass_mock_rhotau_samples.py \
            --cov-rho {input.cov_rho} \
            --cov-tau {input.cov_tau} \
            --ref-rho {input.ref_rho} \
            --ref-tau {input.ref_tau} \
            --output-dir {params.output_dir} \
            --mock-ids {params.mock_id} \
            --threads 1
        """


rule covariance_process:
    input:
        str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}.txt")
    output:
        matrix=str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}_processed.txt"),
        gaussian=str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}_processed_g.txt"),
        plot=str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}_processed_plot.pdf")
    params:
        output_stub=str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}_processed")
    threads: 1
    shell:
        """
        python /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference/scripts/cosmocov_process.py {input} {params.output_stub}
        """


def fiducial_covariance_outputs(mask_suffix=""):
    version = config["fiducial"]["version"]
    blind = config["fiducial"]["blind"]

    ng_prefix = covariance_paths(
        version,
        blind,
        "ng",
        config["fiducial"]["min_sep"],
        config["fiducial"]["max_sep"],
        config["fiducial"]["nbins"],
        mask_suffix,
    )[1]

    g_prefix = covariance_paths(
        version,
        blind,
        "g",
        config["fiducial"]["min_sep_int"],
        config["fiducial"]["max_sep_int"],
        config["fiducial"]["nbins_int"],
        mask_suffix,
    )[1]

    return [f"{ng_prefix}_processed.txt", f"{g_prefix}_processed.txt"]


rule covariance:
    input:
        fiducial_covariance_outputs(mask_suffix=DEFAULT_MASK_SUFFIX)


rule covariance_masked:
    input:
        fiducial_covariance_outputs(mask_suffix="_masked")


rule covariance_unmasked:
    input:
        fiducial_covariance_outputs(mask_suffix="")


ruleorder: covariance_ini > covariance_cosmocov > covariance_cat > covariance_process > covariance

rule covariance_blind_consistency:
    """Blind consistency check: covariance matrices for blinds A, B, C."""
    input:
        specs=[
            "workflow/config/covariance_blind_consistency.md",
            "workflow/config/covariance.md",
        ],
        config="workflow/config/config.yaml",
        cov_a=lambda w: str(
            COSMO_INFERENCE / f"data/covariance/covariance_{config['fiducial']['version']}_A_ng"
            f"_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}"
            f"_nbins={config['fiducial']['nbins']}{DEFAULT_MASK_SUFFIX}/"
            f"covariance_{config['fiducial']['version']}_A_ng"
            f"_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}"
            f"_nbins={config['fiducial']['nbins']}{DEFAULT_MASK_SUFFIX}_processed.txt"
        ),
        cov_b=lambda w: str(
            COSMO_INFERENCE / f"data/covariance/covariance_{config['fiducial']['version']}_B_ng"
            f"_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}"
            f"_nbins={config['fiducial']['nbins']}{DEFAULT_MASK_SUFFIX}/"
            f"covariance_{config['fiducial']['version']}_B_ng"
            f"_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}"
            f"_nbins={config['fiducial']['nbins']}{DEFAULT_MASK_SUFFIX}_processed.txt"
        ),
        cov_c=lambda w: str(
            COSMO_INFERENCE / f"data/covariance/covariance_{config['fiducial']['version']}_C_ng"
            f"_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}"
            f"_nbins={config['fiducial']['nbins']}{DEFAULT_MASK_SUFFIX}/"
            f"covariance_{config['fiducial']['version']}_C_ng"
            f"_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}"
            f"_nbins={config['fiducial']['nbins']}{DEFAULT_MASK_SUFFIX}_processed.txt"
        ),
    output:
        evidence="results/claims/covariance_blind_consistency/evidence.json",
        figure="results/claims/covariance_blind_consistency/figure.png",
    script:
        "../scripts/covariance_blind_consistency.py"


localrules:
    covariance_ini,
    covariance_cat,
    covariance_process,
    generate_glass_mock_rhotau_samples,
    covariance_blind_consistency,


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pseudo-Cl Generation (for COSEBIS cross-validation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CAT_CONFIG and BLINDS defined in Snakefile
# Derive version lists from config["versions"] to stay in sync
BASE_VERSIONS = [v.replace("_leak_corr", "") for v in config["versions"]]


rule fine_pseudo_cl:
    """Generate finely-binned pseudo-Cls for accurate C_ℓ → COSEBIS conversion.

    Uses linear binning with configurable ell_step (default 1 for single-ell bins).
    Supports blind parameter (A, B, C) to override n(z) distribution.
    Uses KiDS-Legacy fiducial cosmology for covariance calculation.
    """
    output:
        pseudo_cl=str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_ellstep={ell_step}.fits"),
        pseudo_cl_cov=str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_ellstep={ell_step}.fits"),
    wildcard_constraints:
        blind="[ABC]",
    params:
        version="{version}",
        blind="{blind}",
        cat_config=CAT_CONFIG,
        nside=1024,
        npatch=1,
        cosmo_params=config["covariance"]["cosmology"],
        binning="linear",
        ell_step=lambda w: int(w.ell_step),
    resources:
        mem_mb=32000,
        runtime=120,  # minutes
    threads: 48
    script:
        "../scripts/generate_fine_pseudo_cl.py"


rule pseudo_cl_cov:
    """Generate 32-bin pseudo-Cl covariances for B-mode null tests.

    Uses power-space (sqrt) binning matching Sasha's NaMaster setup.
    Uses KiDS-Legacy fiducial cosmology for consistency with CosmoCov.
    Supports blind parameter (A, B, C) to override n(z) distribution.
    """
    output:
        pseudo_cl_cov=str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_nellbins=32.fits"),
    wildcard_constraints:
        blind="[ABC]",
    params:
        version="{version}",
        blind="{blind}",
        cat_config=CAT_CONFIG,
        nside=1024,
        npatch=1,
        cosmo_params=config["covariance"]["cosmology"],
        binning="powspace",
        n_ell_bins=32,
        power=0.5,
    resources:
        mem_mb=16000,
        runtime=30,  # minutes - faster than fine binning
    threads: 24
    script:
        "../scripts/generate_fine_pseudo_cl.py"


# Both leak-corrected and base versions for pseudo-Cl generation
PSEUDO_CL_VERSIONS = config["versions"] + BASE_VERSIONS

rule pseudo_cl_cov_all:
    """Generate 32-bin pseudo-Cl covariances for all versions and blinds."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_nellbins=32.fits"),
            version=PSEUDO_CL_VERSIONS,
            blind=BLINDS,
        ),


rule fine_pseudo_cl_all:
    """Generate finely-binned pseudo-Cls for all versions and blinds."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_ellstep=1.fits"),
            version=config["versions"],
            blind=BLINDS,
        ),
        expand(
            str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_ellstep=1.fits"),
            version=config["versions"],
            blind=BLINDS,
        ),
