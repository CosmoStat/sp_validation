# Glass mock validation: fine-binned ξ± and NaMaster pseudo-Cℓ
# for GLASS mock realizations.
#
# Produces inputs for mock_validation_all_b_mode (research notebook).
# Pre-computed 20-bin ξ± and 32-bin Cℓ from Sacha's pipeline are too coarse
# or bypass the MCM — these rules run the full pipeline on mock catalogs.

GLASS_MOCK_DIR = "/n09data/guerrini/glass_mock_v1.4.6/results"
GLASS_MOCK_IDS = [f"{i:05d}" for i in range(1, 351)]
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
    """Aggregator: fine-binned ξ± for all configured mocks."""
    input:
        expand(
            f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
            mock_id=GLASS_MOCK_IDS,
        ),


rule glass_mock_all_pseudo_cl:
    """Aggregator: pseudo-Cℓ for all configured mocks at a given nbins."""
    input:
        expand(
            f"{MOCK_RESULTS}/pseudo_cl_glass_mock_{{mock_id}}_powspace_nbins={{cl_nbins}}.fits",
            mock_id=GLASS_MOCK_IDS,
            cl_nbins=[32],  # default; override with --config cl_nbins=[96,128]
        ),


rule glass_mock_validation:
    """Aggregator: all configured mock validation inputs (ξ± + pseudo-Cℓ)."""
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


rule mock_pure_eb:
    """Scatter: compute pure ξ± E/B modes for one GLASS mock.

    The external 20-bin GLASS ξ file defines the reporting grid and the local
    1000-bin product supplies the integration grid.  The script writes only
    the six data-vector arrays; analytic covariance is consumed by the gather
    campaign below.
    """
    input:
        fine=f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
        reporting=f"{GLASS_MOCK_DIR}/xi_glass_mock_{{mock_id}}_4096_nbins=20.fits",
    output:
        pure_eb=f"{MOCK_RESULTS}/pure_eb_glass_mock_{{mock_id}}.npz",
    threads: 1
    resources:
        mem_mb=8000,
        runtime=30,
    script:
        "../scripts/mock_pure_eb_scatter.py"


rule mock_covariance_campaign:
    """Gather the six-statistic GLASS mock covariance campaign.

    Directory inputs are intentional: the gather script filters GLASS_MOCK_IDS
    to the complete subset present at execution time, so this rule remains
    usable while fine ξ and pure-EB jobs are arriving asynchronously.
    """
    input:
        cosebis_dir=MOCK_RESULTS,
        pure_dir=MOCK_RESULTS,
        fine_dir=MOCK_RESULTS,
        cl_dir=GLASS_MOCK_DIR,
        pseudo_cl_dir=MOCK_RESULTS,
        cosebis_xi_grid=(
            f"{COSMO_VAL}/SP_v1.4.6.3_leak_corr_xi_minsep=0.5"
            "_maxsep=300.0_nbins=1000_npatch=1.txt"
        ),
        cosebis_cov= (
            f"{COSMO_INFERENCE}/data/covariance/"
            "covariance_SP_v1.4.6.3_leak_corr_A_g_minsep=0.5_"
            "maxsep=300.0_nbins=1000_masked/"
            "covariance_SP_v1.4.6.3_leak_corr_A_g_minsep=0.5_"
            "maxsep=300.0_nbins=1000_masked_processed.txt"
        ),
        pure_cov=(
            "/automnt/n17data/cdaley/unions/analyses/shear_2d/bmodes_2d/"
            "results/paper_plots/intermediate/"
            "SP_v1.4.6.3_leak_corr_A_pure_eb_semianalytic.npz"
        ),
        cl_cov=(
            f"{COSMO_VAL}/pseudo_cl_cov_SP_v1.4.6.3_leak_corr_"
            "blind=A_powspace_nbins=32.fits"
        ),
    params:
        mock_ids=GLASS_MOCK_IDS,
        label="fiducial",
        cl_cov_scale=1.0,
    output:
        campaign=directory(f"{MOCK_RESULTS}/campaign"),
    threads: 1
    resources:
        mem_mb=16000,
        runtime=120,
    script:
        "../scripts/mock_campaign_analysis.py"


rule mock_cosebis_bias_test:
    """Gather: configured-mock COSEBIS bias test figure + evidence.

    Collects per-mock COSEBIS from scatter jobs, propagates the adopted CosmoCov ξ±
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
            / "covariance_SP_v1.4.6.3_leak_corr_A_g_minsep=0.5_maxsep=300.0_nbins=1000_masked"
            / "covariance_SP_v1.4.6.3_leak_corr_A_g_minsep=0.5_maxsep=300.0_nbins=1000_masked_processed.txt"
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
    mock_covariance_campaign,
    mock_cosebis_bias_test,
    mock_cosebis_v2_all,
    glass_mock_masked_all_xi,
    mock_pure_eb_masked_all,
    mock_cosebis_masked_all,


# ── v1.4.6.3_v2 suite (fiducial-catalog mocks, n≈5.80/arcmin², 2026-03) ──────
# The newest GLASS suite has catalogs + Cl products but no ξ; produce both
# the fine integration grid and the paper reporting grid ourselves.

GLASS_MOCK_V2_DIR = "/n09data/guerrini/glass_mock_v1.4.6.3_v2/results"
MOCK_V2_RESULTS = "results/glass_mock_v1.4.6.3_v2"
GLASS_V148_MASK = config["pixel_mask"]["source_file"]
MOCK_MASKED_RESULTS = "results/glass_mock_masked"


rule glass_mock_v2_xi_fine:
    """Fine-binned treecorr ξ± for one v1.4.6.3_v2 GLASS mock (1000 bins, 0.5–500 arcmin)."""
    input:
        catalog=f"{GLASS_MOCK_V2_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        gg=f"{MOCK_V2_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
    params:
        min_sep=0.5,
        max_sep=500.0,
        nbins=1000,
    threads: 24
    resources:
        mem_mb=28000,
        runtime=240,
    script:
        "../scripts/run_glass_mock_2pcf.py"


rule glass_mock_v2_xi_reporting:
    """Paper reporting-grid treecorr ξ± for one v1.4.6.3_v2 GLASS mock (20 bins, 1–250 arcmin)."""
    input:
        catalog=f"{GLASS_MOCK_V2_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        gg=f"{MOCK_V2_RESULTS}/xi_glass_mock_{{mock_id}}_nbins=20.fits",
    params:
        min_sep=1.0,
        max_sep=250.0,
        nbins=20,
    threads: 24
    resources:
        mem_mb=28000,
        runtime=240,
    script:
        "../scripts/run_glass_mock_2pcf.py"


rule glass_mock_v2_all_xi:
    """Aggregator: both ξ binnings for all 350 v2 mocks."""
    input:
        expand(
            f"{MOCK_V2_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
            mock_id=GLASS_MOCK_IDS,
        ),
        expand(
            f"{MOCK_V2_RESULTS}/xi_glass_mock_{{mock_id}}_nbins=20.fits",
            mock_id=GLASS_MOCK_IDS,
        ),


rule mock_cosebis_v2:
    """Scatter: compute COSEBIS E_n/B_n for one v1.4.6.3_v2 GLASS mock."""
    input:
        xi=f"{MOCK_V2_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
    params:
        nmodes=config["fiducial"]["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        cosebis=f"{MOCK_V2_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
    resources:
        runtime=30,
    script:
        "../scripts/mock_cosebis_scatter.py"


rule mock_cosebis_v2_all:
    """Aggregator: COSEBIS outputs for all v1.4.6.3_v2 mocks."""
    input:
        expand(
            f"{MOCK_V2_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
            mock_id=GLASS_MOCK_IDS,
        ),


# ── v1.4.8 geometry suite (v1.4.6.3 GLASS mocks + star/structure mask) ─────

rule glass_mock_masked_xi_fine:
    """Fine-binned ξ± for one mock after applying the v1.4.8 mask."""
    input:
        catalog=f"{GLASS_MOCK_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        gg=f"{MOCK_MASKED_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
    params:
        min_sep=0.5,
        max_sep=500.0,
        nbins=1000,
        mask_file=GLASS_V148_MASK,
    threads: 24
    resources:
        mem_mb=28000,
        runtime=240,
    script:
        "../scripts/run_glass_mock_2pcf.py"


rule glass_mock_masked_xi_reporting:
    """Paper reporting-grid ξ± for one mock after applying the v1.4.8 mask."""
    input:
        catalog=f"{GLASS_MOCK_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        gg=f"{MOCK_MASKED_RESULTS}/xi_glass_mock_{{mock_id}}_nbins=20.fits",
    params:
        min_sep=1.0,
        max_sep=250.0,
        nbins=20,
        mask_file=GLASS_V148_MASK,
    threads: 24
    resources:
        mem_mb=28000,
        runtime=240,
    script:
        "../scripts/run_glass_mock_2pcf.py"


rule glass_mock_masked_all_xi:
    """Aggregator: both ξ binnings for all masked GLASS mocks."""
    input:
        expand(
            f"{MOCK_MASKED_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
            mock_id=GLASS_MOCK_IDS,
        ),
        expand(
            f"{MOCK_MASKED_RESULTS}/xi_glass_mock_{{mock_id}}_nbins=20.fits",
            mock_id=GLASS_MOCK_IDS,
        ),


rule mock_pure_eb_masked:
    """Scatter: pure ξ± E/B modes for one masked GLASS mock."""
    input:
        fine=f"{MOCK_MASKED_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
        reporting=f"{MOCK_MASKED_RESULTS}/xi_glass_mock_{{mock_id}}_nbins=20.fits",
    output:
        pure_eb=f"{MOCK_MASKED_RESULTS}/pure_eb_glass_mock_{{mock_id}}.npz",
    threads: 1
    resources:
        mem_mb=8000,
        runtime=30,
    script:
        "../scripts/mock_pure_eb_scatter.py"


rule mock_pure_eb_masked_all:
    """Aggregator: pure E/B outputs for all masked GLASS mocks."""
    input:
        expand(
            f"{MOCK_MASKED_RESULTS}/pure_eb_glass_mock_{{mock_id}}.npz",
            mock_id=GLASS_MOCK_IDS,
        ),


rule mock_cosebis_masked:
    """Scatter: COSEBIS E_n/B_n for one masked GLASS mock."""
    input:
        xi=f"{MOCK_MASKED_RESULTS}/gg_glass_mock_{{mock_id}}_nbins=1000.fits",
    params:
        nmodes=config["fiducial"]["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        cosebis=f"{MOCK_MASKED_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
    resources:
        runtime=30,
    script:
        "../scripts/mock_cosebis_scatter.py"


rule mock_cosebis_masked_all:
    """Aggregator: COSEBIS outputs for all masked GLASS mocks."""
    input:
        expand(
            f"{MOCK_MASKED_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
            mock_id=GLASS_MOCK_IDS,
        ),
