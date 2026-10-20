# PLANCK18, COSMOLOGY_PARAMS defined in Snakefile
# cov_output(), covariance_dir(), covariance_base(), covariance_path() defined in Snakefile

# DEFAULT_MASK_SUFFIX defined in Snakefile
# Footprint mask power spectra (nside=4096, from comprehensive catalog with spatial cuts only)
MASK_CLS_BASE = str(COSMO_INFERENCE / "data/mask")
MASK_CLS_FILES = {
    "footprint": f"{MASK_CLS_BASE}/mask_cls_footprint_nside_4096_norm.txt",
    "footprint_starhalo": f"{MASK_CLS_BASE}/mask_cls_footprint_starhalo_nside_4096_norm.txt",
}

# SP_v1.4.8 uses the star-halo footprint; every other catalogue the standard one.
STARHALO_CATALOGUES = {"SP_v1.4.8"}


def get_mask_cls_path(version):
    """Return absolute mask Cl path for the requested catalog version."""
    starhalo = base_version(version) in STARHALO_CATALOGUES
    return MASK_CLS_FILES["footprint_starhalo" if starhalo else "footprint"]


# ---------------------------------------------------------------------------
# OneCovariance: real-space ξ± covariance for cosmic shear, optionally
# tomographic. Named by covariance_base() in common.py.
# ---------------------------------------------------------------------------

def get_onecov_cat_params(version):
    """OneCovariance survey parameters (sigma_e, n_e, area) of ``version``'s
    catalogue entry. Tomographic values are comma-separated strings, one entry
    per bin."""
    cov_th = catalogue_entry(version)["cov_th"]
    return cov_th["sigma_e"], cov_th["n_e"], cov_th["A"]


def _onecov_args(w):
    """covariance_base() arguments of one job, from its wildcards."""
    return (
        w.version, w.gaussian, w.min_sep, w.max_sep, w.nbins,
        w.mask_suffix, w.tomo_suffix,
    )


def _onecov_tool(key, default=None):
    """``config["tools"][key]``, read when a job runs rather than at parse time."""
    def lookup(w):
        value = config.get("tools", {}).get(key, default)
        if value is None:
            raise ValueError(f"Set tools.{key} in the workflow config to run OneCovariance.")
        return value
    return lookup


# Number of cores to use to execute OneCovariance
ONECOV_THREADS = 8


rule cosmology_params:
    """Generate cosmology parameters JSON from sp_validation.

    This decouples snakemake parse-time from sp_validation import.
    Source of truth remains cs_util.cosmo.PLANCK18.
    """
    output:
        COSMOLOGY_PARAMS
    shell:
        """
        python -c "
from cs_util.cosmo import PLANCK18
import json
from pathlib import Path

Path('{output}').parent.mkdir(parents=True, exist_ok=True)
params = dict(PLANCK18)
params['Omega_v'] = 1 - PLANCK18['Omega_m']

with open('{output}', 'w') as f:
    json.dump(params, f, indent=2)
"
        """


rule covariance_glass_mock:
    input:
        xi=expand(
            f"{GLASS_MOCK_DIR}/xi_glass_mock_{{seed:05d}}_4096_nbins=20.fits",
            seed=range(config["glass_mocks"]["seed_range"][0], config["glass_mocks"]["seed_range"][1] + 1),
        ),
        cl=expand(
            f"{GLASS_MOCK_DIR}/cl_glass_mock_{{seed:05d}}_4096.npy",
            seed=range(config["glass_mocks"]["seed_range"][0], config["glass_mocks"]["seed_range"][1] + 1),
        ),
    output:
        xi_covariance=f"results/covariance/{GLASS_MOCK_SUITE}/xi_covariance.npy",
        cl_covariance=f"results/covariance/{GLASS_MOCK_SUITE}/cl_covariance.npy",
        combined_covariance=f"results/covariance/{GLASS_MOCK_SUITE}/combined_covariance.npy",
        correlation_plot=f"results/covariance/{GLASS_MOCK_SUITE}/combined_correlation.png",
        xi_mean=f"results/covariance/{GLASS_MOCK_SUITE}/xi_mean.npy",
        cl_mean=f"results/covariance/{GLASS_MOCK_SUITE}/cl_mean.npy",
        combined_mean=f"results/covariance/{GLASS_MOCK_SUITE}/combined_mean.npy",
    script:
        "../scripts/compute_glass_mock_covariance.py"


rule generate_glass_mock_rhotau_samples:
    """Generate sampled tau statistics for glass mocks.

    Only tau is sampled; inference_prep_glass_mock uses real rho data.
    """
    input:
        cov_tau=cv_cov_tau(FIDUCIAL["mock_version"]),
        ref_tau=cv_tau_stats(FIDUCIAL["mock_version"]),
    output:
        tau="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
    params:
        mock_id="{mock_id}",
        output_dir="results/glass_mock_rhotau_samples",
    threads: 1
    script:
        "../scripts/generate_glass_mock_rhotau_samples.py"


rule covariance_ini:
    input:
        nz_file=lambda w: redshift_path(w.version),
        mask=lambda w: [] if w.mask_suffix != "_masked" else [get_mask_cls_path(w.version)],
    output:
        cov_output(".ini"),
    params:
        outdir=lambda w: covariance_dir(*_onecov_args(w)),
        out_filename=lambda w: covariance_base(*_onecov_args(w)) + ".dat",
        out_plot_filename=lambda w: covariance_base(*_onecov_args(w)) + "_corr_plot.pdf",
        ng_value=lambda w: "1" if w.gaussian == "ng" else "0",
        omega_m=PLANCK18["Omega_m"],
        omega_v=PLANCK18["Omega_v"],
        sigma_8=PLANCK18["sigma_8"],
        n_s=PLANCK18["n_s"],
        h=PLANCK18["h"],
        omega_b=PLANCK18["Omega_b"],
        hmcode_logT_AGN=7.75,
        sigma_e_param=lambda w: get_onecov_cat_params(w.version)[0],
        n_e=lambda w: get_onecov_cat_params(w.version)[1],
        area=lambda w: get_onecov_cat_params(w.version)[2],
        # OneCovariance takes directory and file separately.
        mask_dir=lambda w: MASK_CLS_BASE if w.mask_suffix == "_masked" else "",
        mask_file=lambda w: (
            os.path.basename(get_mask_cls_path(w.version))
            if w.mask_suffix == "_masked"
            else ""
        ),
        nz_dir=lambda w: os.path.dirname(redshift_path(w.version)),
        nz_file=lambda w: os.path.basename(redshift_path(w.version)),
        num_cores=ONECOV_THREADS,
    threads: 1
    shell:
        """
        mkdir -p {params.outdir}

        cat > "{output}" << 'EOF'
[covariance terms]
gauss = True
split_gauss = False
nongauss = {params.ng_value}
ssc = {params.ng_value}
sn_only = False

[observables]
cosmic_shear = True
est_shear = xi_pm
clustering = False
ggl = False

[output settings]
directory = {params.outdir}
file = {params.out_filename}
style = matrix
save_configs = False
corrmatrix_plot = {params.out_plot_filename}

[covTHETAspace settings]
theta_min = {wildcards.min_sep}
theta_max = {wildcards.max_sep}
theta_bins = {wildcards.nbins}
theta_type = log
xi_pp = True
xi_mm = True
theta_accuracy = 1e-3
integration_intervals = 400

[survey specs]
mask_directory = {params.mask_dir}
mask_file_lensing = {params.mask_file}
ellipticity_dispersion = {params.sigma_e_param}
survey_area_lensing_in_deg2 = {params.area}
n_eff_lensing = {params.n_e}

[redshift]
zlens_directory = {params.nz_dir}
zlens_file = {params.nz_file}
value_loc_in_lensbin = mid

[cosmo]
sigma8 = {params.sigma_8}
h = {params.h}
omega_m = {params.omega_m}
omega_b = {params.omega_b}
omega_de = {params.omega_v}
w0 = -1.0
wa = 0.0
ns = {params.n_s}
neff = 3.046
m_nu = 0.06
tcmb0 = 2.725

[IA]
A_IA = 0.0
eta_IA = 0.0
z_pivot_IA = 0.3

[powspec evaluation]
non_linear_model = mead2020_feedback
HMCode_logT_AGN = {params.hmcode_logT_AGN}
log10k_bins = 300
log10k_min = -3.46
log10k_max = 3.15

[hod]
model_mor_cen = double_powerlaw
model_mor_sat = double_powerlaw
dpow_logm0_cen = 10.51
dpow_logm1_cen = 11.38
dpow_a_cen = 7.096
dpow_b_cen = 0.2
dpow_norm_cen = 1.0
dpow_norm_sat = 0.56
model_scatter_cen = lognormal
model_scatter_sat = modschechter
logn_sigma_c_cen = 0.35
modsch_logmref_sat = 13.0
modsch_alpha_s_sat = -0.858
modsch_b_sat = -0.024, 1.149

[halomodel evaluation]
m_bins = 900
log10m_min = 6
log10m_max = 18
hmf_model = Tinker10
mdef_model = SOMean
mdef_params = overdensity, 200
disable_mass_conversion = True
delta_c = 1.686
transfer_model = CAMB
small_k_damping_for1h = damped

[misc]
num_cores = {params.num_cores}

EOF
        """


rule covariance_run:
    """Run OneCovariance on its generated .ini.

    `container: None` on purpose: OneCovariance runs from a host checkout under
    environment-modules, set by tools.onecov_executable.
    """
    input:
        rules.covariance_ini.output,
    output:
        cov_output(".dat"),
    params:
        outdir=lambda w: covariance_dir(*_onecov_args(w)),
        onecov=_onecov_tool("onecov_executable"),
        python_executable=_onecov_tool("python_executable", "python"),
    container:
        None
    threads: ONECOV_THREADS
    shell:
        """
        module unload gcc || true
        module load gcc
        module unload intelpython || true
        module load intelpython/3-2024.1.0
        module load openmpi

        cd {params.outdir}
        {params.python_executable} {params.onecov} {input}
        """


def fiducial_covariance_outputs(mask_suffix="", tomo_suffix=""):
    """Return the covariance matrices for the fiducial version: non-Gaussian on
    the reporting grid, Gaussian on the integration grid."""
    ng_path = covariance_path(
        FIDUCIAL["version"], "ng",
        FIDUCIAL["min_sep"], FIDUCIAL["max_sep"], FIDUCIAL["nbins"],
        mask_suffix, tomo_suffix,
    )
    g_path = covariance_path(
        FIDUCIAL["version"], "g",
        FIDUCIAL["min_sep_int"], FIDUCIAL["max_sep_int"], FIDUCIAL["nbins_int"],
        mask_suffix, tomo_suffix,
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


rule covariance_tomo:
    input:
        fiducial_covariance_outputs(mask_suffix=DEFAULT_MASK_SUFFIX, tomo_suffix="_tomo")


ruleorder: covariance_ini > covariance_run > covariance

localrules:
    cosmology_params,
    covariance_ini,
    generate_glass_mock_rhotau_samples,
