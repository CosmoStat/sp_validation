# Glass mock validation: fine-binned ξ± and NaMaster pseudo-Cℓ
# for GLASS mock realizations.
#
# Produces inputs for mock_validation_all_b_mode (research notebook).
# Pre-computed 20-bin ξ± and 32-bin Cℓ from Sacha's pipeline are too coarse
# or bypass the MCM — these rules run the full pipeline on mock catalogs.

GLASS_MOCK_DIR = "/n09data/guerrini/glass_mock_v1.4.6/results"
GLASS_MOCK_IDS = [f"{i:05d}" for i in range(1, 101)]
MOCK_RESULTS = "results/glass_mock"

wildcard_constraints:
    cl_nbins=r"\d+",


rule glass_mock_xi_fine:
    """Fine-binned treecorr ξ± for one GLASS mock (1000 bins, 0.5–500 arcmin).

    Required for config-space COSEBIS and pure E/B on mocks.
    ~35M galaxies → ~30 min on 48 cores.
    """
    input:
        catalog=f"{GLASS_MOCK_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        gg=f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
    params:
        min_sep=0.5,
        max_sep=500.0,
        nbins=1000,
    threads: 24
    resources:
        mem_mb=20000,
        runtime=180,
    script:
        "../scripts/run_glass_mock_2pcf.py"


rule glass_mock_pseudo_cl:
    """NaMaster pseudo-Cℓ for one GLASS mock (powspace, configurable nbins).

    Full MCM pipeline on mock catalog — tests mode-coupling correction.
    Matches real data binning (lmin=8, lmax=2048, power=0.5).
    Use nbins wildcard to test different bandpower resolutions.
    """
    input:
        catalog=f"{GLASS_MOCK_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        pseudo_cl=f"{MOCK_RESULTS}/pseudo_cl_glass_mock_{{mock_id}}_powspace_nbins={{cl_nbins}}.fits",
    params:
        nside=1024,
        nbins=lambda wc: int(wc.cl_nbins),
        power=0.5,
        lmin=8,
        lmax=2048,
    threads: 12
    resources:
        mem_mb=32000,
        runtime=120,
    script:
        "../scripts/run_glass_mock_pseudo_cl.py"


rule glass_mock_all_xi:
    """Aggregator: fine-binned ξ± for 5 mocks."""
    input:
        expand(
            f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
            mock_id=GLASS_MOCK_IDS,
        ),


rule glass_mock_all_pseudo_cl:
    """Aggregator: pseudo-Cℓ for 5 mocks at a given nbins."""
    input:
        expand(
            f"{MOCK_RESULTS}/pseudo_cl_glass_mock_{{mock_id}}_powspace_nbins={{cl_nbins}}.fits",
            mock_id=GLASS_MOCK_IDS,
            cl_nbins=[32],  # default; override with --config cl_nbins=[96,128]
        ),


rule glass_mock_validation:
    """Aggregator: all mock validation inputs (ξ± + pseudo-Cℓ for 5 mocks)."""
    input:
        rules.glass_mock_all_xi.input,
        rules.glass_mock_all_pseudo_cl.input,


rule mock_cosebis_scatter:
    """Scatter: compute COSEBIS E_n/B_n for one GLASS mock.

    Reads fine-binned ξ±, applies scale cuts, computes COSEBIS modes.
    Byte-order conversion for numba compatibility handled internally.
    """
    input:
        xi=f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
    params:
        nmodes=config["fiducial"]["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        cosebis=f"{MOCK_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
    resources:
        runtime=30,
    script:
        "../scripts/mock_cosebis_scatter.py"


rule mock_cosebis_bias_test:
    """Gather: 25-mock COSEBIS bias test figure + evidence.

    Collects per-mock COSEBIS from scatter jobs, propagates CosmoCov ξ±
    covariance to COSEBIS space, tests mean B_n = 0 at σ/√N precision.
    """
    input:
        cosebis=expand(
            f"{MOCK_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
            mock_id=GLASS_MOCK_IDS,
        ),
        xi_ref=f"{MOCK_RESULTS}/gg_glass_mock_00001_nbins=1000.fits",
        cov=str(
            COSMO_INFERENCE / "data/covariance"
            / "covariance_SP_v1.4.6_leak_corr_A_g_minsep=0.5_maxsep=500.0_nbins=1000_masked"
            / "covariance_SP_v1.4.6_leak_corr_A_g_minsep=0.5_maxsep=500.0_nbins=1000_masked_processed.txt"
        ),
    params:
        nmodes=config["fiducial"]["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        figure="results/tapestry/mock_cosebis_bias_test/figure.png",
        evidence="results/tapestry/mock_cosebis_bias_test/evidence.json",
    script:
        "../scripts/mock_cosebis_bias_test.py"


localrules:
    glass_mock_all_xi,
    glass_mock_all_pseudo_cl,
    glass_mock_validation,
    mock_cosebis_bias_test,
