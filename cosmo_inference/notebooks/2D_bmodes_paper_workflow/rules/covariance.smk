# BLOCK_PAIRS, PLANCK18, COSMOLOGY_PARAMS defined in Snakefile


def get_cat_params(version):
    """Extract covariance parameters (area, n_e, sigma_e) from catalog config."""
    base_version = version.replace("_leak_corr", "")
    if base_version not in config:
        raise KeyError(f"Catalog configuration not found for {base_version}")
    cov_th = config[base_version]["cov_th"]
    return cov_th["A"], cov_th["n_e"], cov_th["sigma_e"]


# covariance_dir(), covariance_base(), covariance_path() defined in Snakefile
# Additional wildcard constraints defined locally for pseudo-Cl rules (line 327)

# DEFAULT_MASK_SUFFIX defined in Snakefile
# Footprint mask power spectra (nside=4096, from comprehensive catalog with spatial cuts only)
MASK_CLS_BASE = str(COSMO_INFERENCE / "data/mask")
MASK_CLS_FILES = {
    "footprint": f"{MASK_CLS_BASE}/mask_cls_footprint_nside_4096_norm.txt",
    "footprint_starhalo": f"{MASK_CLS_BASE}/mask_cls_footprint_starhalo_nside_4096_norm.txt",
}

# v1.4.8 uses the star-halo footprint; all other versions use the standard footprint
STARHALO_VERSIONS = {"v1.4.8"}


def get_mask_cls_path(version):
    """Return absolute mask Cl path for the requested catalog version."""
    version_dir = version.replace('_leak_corr', '').replace('SP_', '')
    version_dir = re.sub(r'_ecut\d+', '', version_dir)
    key = "footprint_starhalo" if version_dir in STARHALO_VERSIONS else "footprint"
    return MASK_CLS_FILES[key]


rule cosmology_params:
    """Generate cosmology parameters JSON from sp_validation.

    This decouples snakemake parse-time from sp_validation import.
    Source of truth remains sp_validation.cosmology.PLANCK18.
    """
    output:
        COSMOLOGY_PARAMS
    shell:
        """
        python -c "
from sp_validation.cosmology import PLANCK18
import json
from pathlib import Path

Path('{output}').parent.mkdir(parents=True, exist_ok=True)
params = dict(PLANCK18)
params['Omega_v'] = 1 - PLANCK18['Omega_m']

with open('{output}', 'w') as f:
    json.dump(params, f, indent=2)
"
        """


rule covariance_ini:
    input:
        nz_file=lambda w: build_redshift_path(w.version, w.blind),
        mask=lambda w: [] if w.mask_suffix != "_masked" else [get_mask_cls_path(w.version)],
    output:
        str(COSMO_INFERENCE / "data/covariance/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}/covariance_{version}_{blind}_{gaussian}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}{mask_suffix}.ini")
    params:
        outdir=lambda w: covariance_dir(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix,
            resolve_version=False
        ),
        ng_value=lambda wildcards: "1" if wildcards.gaussian == "ng" else "0",
        omega_m=PLANCK18["Omega_m"],
        omega_v=PLANCK18["Omega_v"],
        sigma_8=PLANCK18["sigma_8"],
        n_s=PLANCK18["n_s"],
        h=PLANCK18["h"],
        omega_b=PLANCK18["Omega_b"],
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
Omega_v : {params.omega_v}
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


# Covariance parameters
#
# tmin,tmax in arcminutes
tmin : {wildcards.min_sep}
tmax : {wildcards.max_sep}
ntheta : {wildcards.nbins}
ng : {params.ng_value}
cng : {params.ng_value}


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
        outdir=lambda w: covariance_dir(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix,
            resolve_version=False
        ),
        ini_path=lambda w: covariance_path(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix,
            suffix=".ini", resolve_version=False
        ),
        cosmocov=config["tools"]["cosmocov_executable"],
    container:
        None
    threads: 1
    shell:
        """
        module unload gcc || true
        module load gcc
        module unload intelpython || true
        module load intelpython/3-2024.1.0
        module load openmpi

        cd {params.outdir}
        {params.cosmocov} {params.block_i} {params.ini_path}
        """


rule covariance_cat:
    input:
        cov_block=lambda w: [
            f"{covariance_dir(w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins, w.mask_suffix, resolve_version=False)}"
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


# fiducial_binning_suffix() defined in Snakefile


rule generate_glass_mock_rhotau_samples:
    """Generate sampled tau statistics for glass mocks.

    Only tau is sampled; inference_prep_glass_mock uses real rho data.
    """
    input:
        cov_tau=str(COSMO_VAL / f"rho_tau_stats/cov_tau_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}_th.npy"),
        ref_tau=str(COSMO_VAL / f"rho_tau_stats/tau_stats_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}.fits"),
    output:
        tau="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
    params:
        mock_id="{mock_id}",
        output_dir="results/glass_mock_rhotau_samples",
    threads: 1
    shell:
        """
        python workflow/scripts/generate_glass_mock_rhotau_samples.py \
            --cov-tau {input.cov_tau} \
            --ref-tau {input.ref_tau} \
            --output-dir {params.output_dir} \
            --mock-ids {params.mock_id}
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
    """Return processed covariance files for fiducial version/blind."""
    ng_path = covariance_path(
        FIDUCIAL["version"], FIDUCIAL["blind"], "ng",
        FIDUCIAL["min_sep"], FIDUCIAL["max_sep"], FIDUCIAL["nbins"], mask_suffix
    )
    g_path = covariance_path(
        FIDUCIAL["version"], FIDUCIAL["blind"], "g",
        FIDUCIAL["min_sep_int"], FIDUCIAL["max_sep_int"], FIDUCIAL["nbins_int"], mask_suffix
    )
    return [ng_path, g_path]


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

localrules:
    cosmology_params,
    covariance_ini,
    covariance_cat,
    covariance_process,
    generate_glass_mock_rhotau_samples,


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Pseudo-Cl Generation (for COSEBIS cross-validation)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CAT_CONFIG and BLINDS defined in Snakefile
# Derive version lists from config["versions"] to stay in sync
BASE_VERSIONS = [v.replace("_leak_corr", "") for v in config["versions"]]


# Wildcard constraints for unified pseudo-Cl rules
wildcard_constraints:
    binning = "linear|logspace|powspace",


rule pseudo_cl:
    """Generate pseudo-Cl data vector with configurable binning.

    Output pattern: pseudo_cl_{version}_blind={blind}_{binning}_nbins={nbins}.fits

    Binning modes:
    - linear: uniform ell bins (nbins determines ell_step)
    - powspace: power-law spaced bins (nbins=32 with power=0.5 for sqrt spacing)

    Uses astropy Planck18 fiducial cosmology.
    """
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
    """Generate pseudo-Cl covariance with configurable binning.

    Output pattern: pseudo_cl_cov_{version}_blind={blind}_{binning}_nbins={nbins}.fits

    Binning modes:
    - linear: uniform ell bins (nbins determines ell_step)
    - powspace: power-law spaced bins (nbins=32 with power=0.5 for sqrt spacing)

    Uses astropy Planck18 fiducial cosmology.
    """
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


# Use same versions as other statistics (leak-corrected only)
# Uncorrected versions are generated on-demand when needed by specific claims
PSEUDO_CL_VERSIONS = config["versions"]

rule pseudo_cl_all:
    """Generate pseudo-Cls for all versions (harmonic preset, blind A only)."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind=A_powspace_nbins=32.fits"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_cov_all:
    """Generate pseudo-Cl covariances for all versions (harmonic preset, blind A only)."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_cov_{version}_blind=A_powspace_nbins=32.fits"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_fine_all:
    """Generate fine pseudo-Cls for COSEBIS (linear preset)."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_linear_nbins=2040.fits"),
            version=config["versions"],
            blind=BLINDS,
        ),
