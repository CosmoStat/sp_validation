import os
from pathlib import Path

PROJECT_ROOT = Path("/n17data/cdaley/unions/pure_eb")
PSEUDO_CL_DIR = Path("/home/guerrini/sp_validation/notebooks/cosmo_val/output")
GLASS_MOCK_VERSION = config["glass_mocks"].get("version", "v0")

GLASS_MOCK_FITS_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"data/glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}"
    / f"cosmosis_glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}.fits"
)
GLASS_MOCK_CONFIG_PATTERN = str(
    COSMO_INFERENCE_PROD
    / f"cosmosis_config/cosmosis_pipeline_glass_mocks_{GLASS_MOCK_VERSION}_glass_mock_{{mock_id}}.ini"
)

def pseudo_cl_assets(version):
    """Return pseudo-Cl and covariance paths for the requested catalog version."""
    cl_path = PSEUDO_CL_DIR / f"pseudo_cl_{version}.fits"
    cov_path = PSEUDO_CL_DIR / f"pseudo_cl_cov_g_ng_iNKA_{version}.fits"
    return str(cl_path), str(cov_path)

rule inference_prep:
    input:
        # Processed covariance matrix
        cov_matrix=lambda w: str(
            COSMO_INFERENCE
            / (
                f"data/covariance/"
                f"covariance_{w.version}_{w.blind}_ng_minsep={w.min_sep}_maxsep={w.max_sep}_nbins={w.nbins}"
                f"{'_masked' if config['covariance']['default_masked'] else ''}/"
                f"covariance_{w.version}_{w.blind}_ng_minsep={w.min_sep}_maxsep={w.max_sep}_nbins={w.nbins}"
                f"{'_masked' if config['covariance']['default_masked'] else ''}_processed.txt"
            )
        ),
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
            / "cosmosis_config/cosmosis_pipeline_{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.ini"
        )
    params:
        cosmosis_root="{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}",
        data_dir="/n09data/guerrini/output_chains/{version}_{blind}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}",
        output_root=str(COSMO_INFERENCE_PROD),
    threads: 1
    shell:
        """
        cd /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference

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


rule inference_fiducial:
    input:
        # Use the same output patterns as inference_prep
        rules.inference_prep.output.fits_file.format(
            version=config["fiducial"]["version"],
            blind=config["fiducial"]["blind"],
            min_sep=config["fiducial"]["min_sep"],
            max_sep=config["fiducial"]["max_sep"],
            nbins=config["fiducial"]["nbins"],
            npatch=config["fiducial"]["npatch"]
        ),
        rules.inference_prep.output.config_file.format(
            version=config["fiducial"]["version"],
            blind=config["fiducial"]["blind"],
            min_sep=config["fiducial"]["min_sep"],
            max_sep=config["fiducial"]["max_sep"],
            nbins=config["fiducial"]["nbins"],
            npatch=config["fiducial"]["npatch"]
        )


rule inference_glass_mocks:
    input:
        expand(GLASS_MOCK_FITS_PATTERN, mock_id=[f"{i:05d}" for i in range(1, 351)])


rule inference_prep_glass_mock:
    input:
        xi="/n09data/guerrini/glass_mock_v1.4.6/results/xi_glass_mock_{mock_id}_4096_nbins=20.fits",
        cov_matrix=lambda w: str(
            COSMO_INFERENCE
            / (
                f"data/covariance/"
                f"covariance_{config['fiducial']['mock_version']}_A_ng_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}_nbins={config['fiducial']['nbins']}"
                f"{'_masked' if config['covariance']['default_masked'] else ''}/"
                f"covariance_{config['fiducial']['mock_version']}_A_ng_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}_nbins={config['fiducial']['nbins']}"
                f"{'_masked' if config['covariance']['default_masked'] else ''}_processed.txt"
            )
        ),
        # n(z) file
        nz_file=lambda w: build_redshift_path(config["fiducial"]["mock_version"], "A"),
        # Rho/tau stats: rho from real data, tau sampled
        rho_stats=str(
            COSMO_VAL
            / (
                "rho_tau_stats/"
                f"rho_stats_{config['fiducial']['mock_version']}_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}_nbins={config['fiducial']['nbins']}_npatch={config['fiducial']['npatch']}.fits"
            )
        ),
        tau_stats=lambda w: str(
            PROJECT_ROOT / f"results/glass_mock_rhotau_samples/{w.mock_id}/tau_stats_sampled.fits"
        ),
        # Tau covariance (real data)
        tau_cov=str(
            COSMO_VAL
            / (
                "rho_tau_stats/"
                f"cov_tau_{config['fiducial']['mock_version']}_minsep={config['fiducial']['min_sep']}_maxsep={config['fiducial']['max_sep']}_nbins={config['fiducial']['nbins']}_npatch={config['fiducial']['npatch']}_th.npy"
            )
        ),
        # C_ell data for dual config generation
        cl_file="/n09data/guerrini/glass_mock_v1.4.6/results/cl_glass_mock_{mock_id}_4096.npy",
        cl_cov=lambda w: pseudo_cl_assets(config["fiducial"]["mock_version"])[1],
    output:
        fits_file=GLASS_MOCK_FITS_PATTERN,
        config_file=GLASS_MOCK_CONFIG_PATTERN,
    params:
        cosmosis_root=f"glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        data_dir=f"/n09data/guerrini/glass_mock_chains/glass_mock_{GLASS_MOCK_VERSION}_{{mock_id}}",
        output_root=str(COSMO_INFERENCE_PROD),
        output_basename=f"glass_mocks/{GLASS_MOCK_VERSION}/glass_mock_{{mock_id}}",
    threads: 1
    resources:
        mem_mb=8000,
        runtime=10,
    shell:
        """
        cd /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_inference

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
    inference_fiducial,
