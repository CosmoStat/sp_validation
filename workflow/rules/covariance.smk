# BLOCK_PAIRS, PLANCK18, COSMOLOGY_PARAMS defined in Snakefile
import os

def get_cat_params(version):
    """Extract covariance parameters (sigma_e, n_eff, area) from catalog config.

    Returns (sigma_e, (n_e_lens, n_e_clust), (A_lens, A_ggl, A_clust)).
    For probe == "wl" the clustering/ggl slots are empty strings.
    """
    base_version = version.replace("_leak_corr", "")
    if base_version not in config:
        raise KeyError(f"Catalog configuration not found for {base_version}")
    cov_th = config[base_version]["cov_th"]
    if config["probe_3x2pt"] == "wl":
        return cov_th["sigma_e"], (cov_th["n_e"], ""), (cov_th["A"], "", "")
    return (
        cov_th["sigma_e"],
        (cov_th["n_e_lens"], cov_th["n_e_clust"]),
        (cov_th["A_lens"], cov_th["A_ggl"], cov_th["A_clust"]),
    )

# covariance_dir(), covariance_base(), covariance_path() defined in Snakefile
# Additional wildcard constraints defined locally for pseudo-Cl rules (line 327)

# DEFAULT_MASK_SUFFIX defined in Snakefile
# Footprint mask power spectra (nside=4096, from comprehensive catalog with spatial cuts only)
MASK_CLS_BASE = str(COSMO_INFERENCE / "data/mask")
MASK_CLS_FILES = {
    "footprint_lens": "mask_cls_footprint_nside_4096_norm.txt",
    "footprint_ggl": "mask_cls_footprint_nside_4096_norm.txt",
    "footprint_clust": "mask_cls_footprint_nside_4096_norm.txt",
    "footprint_lens_starhalo": "mask_cls_footprint_starhalo_nside_4096_norm.txt",
}

# v1.4.8 uses the star-halo footprint; all other versions use the standard footprint
STARHALO_VERSIONS = {"v1.4.8"}

def get_mask_cls_file(version, kind="lens"):
    """Return the mask Cl *filename* (OneCov wants dir and file separately)."""
    version_dir = version.replace("_leak_corr", "").replace("SP_", "")
    version_dir = re.sub(r"_ecut\d+", "", version_dir)
    if kind == "lens" and version_dir in STARHALO_VERSIONS:
        return MASK_CLS_FILES["footprint_lens_starhalo"]
    return MASK_CLS_FILES[f"footprint_{kind}"]


def _onecov_mask_params(w):
    """Resolve OneCov mask settings for one job.

    Returns dict with keys: dir, lens, ggl, clust. Empty strings when the
    job is unmasked or the probe doesn't use a given field.
    """
    if w.mask_suffix != "_masked":
        return {"dir": "", "lens": "", "ggl": "", "clust": ""}
    out = {
        "dir": MASK_CLS_BASE,
        "lens": get_mask_cls_file(w.version, "lens"),
        "ggl": "",
        "clust": "",
    }
    if w.probe in ("ggl", "3x2pt"):
        out["ggl"] = get_mask_cls_file(w.version, "ggl")
    if w.probe == "3x2pt":
        out["clust"] = get_mask_cls_file(w.version, "clust")
    return out

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

rule covariance_ini_onecov:
    input:
        # Ensures planck18.json exists before get_planck18() is called below.
        cosmo=COSMOLOGY_PARAMS,
    output:
        cov_output(".ini"),
    params:
        outdir=lambda w: covariance_dir(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins,
            w.probe, w.mask_suffix, resolve_version=False,
        ),
        out_filename=lambda w: covariance_base(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins,
            w.probe, w.mask_suffix, resolve_version=False,
        ) + ".dat",
        out_plot_filename=lambda w: covariance_base(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins,
            w.probe, w.mask_suffix, resolve_version=False,
        ) + "_corr_plot.pdf",
        ng_value=lambda w: "1" if w.gaussian == "ng" else "0",
        do_ggl=lambda w: str(w.probe in ("ggl", "3x2pt")),
        do_clustering=lambda w: str(w.probe == "3x2pt"),
        omega_m=PLANCK18["Omega_m"],
        omega_v=PLANCK18["Omega_v"],
        sigma_8=PLANCK18["sigma_8"],
        n_s=PLANCK18["n_s"],
        h=PLANCK18["h"],
        omega_b=PLANCK18["Omega_b"],
        hmcode_logT_AGN=7.75,
        sigma_e_param=lambda w: get_cat_params(w.version)[0],
        n_e_lens_line=lambda w: (
            f"n_eff_lensing = {get_cat_params(w.version)[1][0]}"),
        n_e_clust_line=lambda w: (""
        if w.probe == "wl"
        else f"n_eff_clust = {get_cat_params(w.version)[1][1]}"),
        area_lens_line=lambda w: (
            f"survey_area_lensing_in_deg2 = {get_cat_params(w.version)[2][0]}"),
        area_ggl_line=lambda w: (
        ""
        if w.probe == "wl"
        else f"survey_area_ggl_in_deg2 = {get_cat_params(w.version)[2][1]}"
        ),
        area_clust_line=lambda w: (
        ""
        if w.probe == "wl"
        else f"survey_area_clust_in_deg2 = {get_cat_params(w.version)[2][2]}"
        ),
        mask_dir=lambda w: _onecov_mask_params(w)["dir"],
        mask_lens=lambda w: _onecov_mask_params(w)["lens"],
        mask_ggl=lambda w: _onecov_mask_params(w)["ggl"],
        mask_clust=lambda w: _onecov_mask_params(w)["clust"],
        nz_dir=lambda w: os.path.dirname(build_redshift_dir(w.version)),
        nz_lens=lambda w: os.path.basename(build_redshift_path_lens(w.version, w.blind)),
        nz_clust=lambda w: (
            "" if w.probe == "wl"
            else os.path.basename(build_redshift_path_source(w.version, w.blind))
        ),
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
clustering = {params.do_clustering}
ggl = {params.do_ggl}
est_ggl = gamma_t
est_clust = w

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
mask_file_lensing = {params.mask_lens}
mask_file_clust = {params.mask_clust}
mask_file_ggl = {params.mask_ggl}
ellipticity_dispersion = {params.sigma_e_param}
{params.area_lens_line}
{params.area_clust_line}
{params.area_ggl_line}
{params.n_e_lens_line}
{params.n_e_clust_line}

[redshift]
zlens_directory = {params.nz_dir}
zlens_file = {params.nz_lens}
zclust_file = {params.nz_clust}
value_loc_in_lensbin = mid
value_loc_in_clustbin = mid

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
num_cores = 8

EOF
        """

rule covariance_onecov:
    input:
        rules.covariance_ini_onecov.output,
    output:
        # Matches `[output settings] file = cov_tmp_onecov.dat` in the ini.
        cov_output(".dat")
    params:
        outdir=lambda w: covariance_dir(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins,
            w.probe, w.mask_suffix, resolve_version=False,
        ),
        ini_path=lambda w: covariance_path(
            w.version, w.blind, w.gaussian, w.min_sep, w.max_sep, w.nbins,
            w.probe, w.mask_suffix, suffix=".ini", resolve_version=False,
        ),
        onecov=config["tools"]["onecov_executable"],
        python_executable=config["tools"]["python_executable"],
    container:
        None
    threads: 8
    shell:
        """
        module unload gcc || true
        module load gcc
        module unload intelpython || true
        module load intelpython/3-2024.1.0
        module load openmpi

        cd {params.outdir}
        {params.python_executable} {params.onecov} {params.ini_path}
        """


# Glass-mock roots. Env/config-overridable for the same reason as COSMO_*.
GLASS_MOCK_RESULTS = config.get("glass_mocks", {}).get(
    "results_dir", "/n09data/guerrini/glass_mock_v1.4.6/results"
)
GLASS_MOCK_COV_DIR = config.get("glass_mocks", {}).get(
    "covariance_dir",
    "/automnt/n17data/cdaley/unions/pure_eb/results/covariance/glass_mock_v1.4.6",
)

rule covariance_glass_mock:
    input:
        xi=expand(
            GLASS_MOCK_RESULTS + "/xi_glass_mock_{seed:05d}_4096_nbins=20.fits",
            seed=range(
                config["glass_mocks"]["seed_range"][0],
                config["glass_mocks"]["seed_range"][1] + 1,
            ),
        ),
        cl=expand(
            GLASS_MOCK_RESULTS + "/cl_glass_mock_{seed:05d}_4096.npy",
            seed=range(
                config["glass_mocks"]["seed_range"][0],
                config["glass_mocks"]["seed_range"][1] + 1,
            ),
        ),
    output:
        xi_covariance=f"{GLASS_MOCK_COV_DIR}/xi_covariance.npy",
        cl_covariance=f"{GLASS_MOCK_COV_DIR}/cl_covariance.npy",
        combined_covariance=f"{GLASS_MOCK_COV_DIR}/combined_covariance.npy",
        correlation_plot=f"{GLASS_MOCK_COV_DIR}/combined_correlation.png",
        xi_mean=f"{GLASS_MOCK_COV_DIR}/xi_mean.npy",
        cl_mean=f"{GLASS_MOCK_COV_DIR}/cl_mean.npy",
        combined_mean=f"{GLASS_MOCK_COV_DIR}/combined_mean.npy",
    script:
        "../scripts/compute_glass_mock_covariance.py"

rule generate_glass_mock_rhotau_samples:
    """Generate sampled tau statistics for glass mocks.

    Only tau is sampled; inference_prep_glass_mock uses real rho data.
    """
    input:
        cov_tau=str(
            COSMO_VAL
            / f"rho_tau_stats/cov_tau_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}_th.npy"
        ),
        ref_tau=str(
            COSMO_VAL
            / f"rho_tau_stats/tau_stats_{FIDUCIAL['mock_version']}{fiducial_binning_suffix()}.fits"
        ),
    output:
        tau="results/glass_mock_rhotau_samples/{mock_id}/tau_stats_sampled.fits",
    params:
        mock_id="{mock_id}",
        output_dir="results/glass_mock_rhotau_samples",
        # Resolved relative to this .smk file instead of a hard-coded checkout.
        script=workflow.source_path(
            "../scripts/generate_glass_mock_rhotau_samples.py"
        ),
    threads: 1
    shell:
        """
        python {params.script} \
            --cov-tau {input.cov_tau} \
            --ref-tau {input.ref_tau} \
            --output-dir {params.output_dir} \
            --mock-ids {params.mock_id}
        """


def fiducial_covariance_outputs(mask_suffix="", probe=DEFAULT_PROBE):
    """Return processed covariance files for fiducial version/blind."""
    path = covariance_path(
        FIDUCIAL["version"], FIDUCIAL["blind"], FIDUCIAL["gaussian"],
        FIDUCIAL["min_sep"], FIDUCIAL["max_sep"], FIDUCIAL["nbins"],
        probe, mask_suffix,
    )
    return path

rule covariance:
    input:
        fiducial_covariance_outputs(mask_suffix=DEFAULT_MASK_SUFFIX, probe=DEFAULT_PROBE),

rule covariance_masked:
    input:
        fiducial_covariance_outputs(mask_suffix="_masked", probe=DEFAULT_PROBE),

rule covariance_unmasked:
    input:
        fiducial_covariance_outputs(mask_suffix="", probe=DEFAULT_PROBE),

rule covariance_3x2pt:
    input:
        fiducial_covariance_outputs(mask_suffix="", 
                                    probe=config["probe_3x2pt"]),

# ruleorder: covariance_ini_onecov > covariance_onecov > covariance

localrules:
    cosmology_params,
    covariance_ini_onecov,
    generate_glass_mock_rhotau_samples,
