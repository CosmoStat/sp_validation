# Glass mock validation: fine-binned ξ± and NaMaster pseudo-Cℓ
# for GLASS mock realizations.
#
# Produces inputs for mock_validation_all_b_mode (research notebook).
# Pre-computed 20-bin ξ± and 32-bin Cℓ from Sacha's pipeline are too coarse
# or bypass the MCM — these rules run the full pipeline on mock catalogs.

# GLASS_MOCK_DIR and GLASS_MOCK_SUITE defined in Snakefile.
# @sc [decision:mocks.cosebis_bias_test]
GLASS_MOCK_IDS = [f"{i:05d}" for i in range(1, 101)]
MOCK_RESULTS = f"results/glass_mock/{GLASS_MOCK_SUITE}"

# Mock ξ± are measured on the data's integration grid, node for node, so the
# E/B transforms see the same θ sampling on mocks and data.
# @sc [decision:mocks.mock_two_point_measurements]
# @sc [decision:real_space.integration_grid]
MOCK_XI_GRID = XI_GRIDS["integration"]
MOCK_XI = f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_{xi_binning('integration')}.fits"

wildcard_constraints:
    cl_nbins=r"\d+",


# @sc [decision:mocks.mock_two_point_measurements]
rule glass_mock_xi_fine:
    """Treecorr ξ± for one GLASS mock on the data's integration grid.

    Required for config-space COSEBIS and pure E/B on mocks.
    ~35M galaxies → ~30 min on 48 cores.
    """
    input:
        catalog=f"{GLASS_MOCK_DIR}/unions_glass_sim_{{mock_id}}_4096.fits",
    output:
        gg=MOCK_XI,
    params:
        min_sep=MOCK_XI_GRID["min_sep"],
        max_sep=MOCK_XI_GRID["max_sep"],
        nbins=MOCK_XI_GRID["nbins"],
    threads: 24
    resources:
        mem_mb=20000,
        runtime=180,
    script:
        "../scripts/run_glass_mock_2pcf.py"


# @sc [decision:mocks.mock_two_point_measurements]
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
    """Aggregator: fine-binned ξ± for the first 100 mocks."""
    input:
        expand(MOCK_XI, mock_id=GLASS_MOCK_IDS),


# @sc [decision:mocks.mock_two_point_measurements]
rule glass_mock_all_pseudo_cl:
    """Aggregator: pseudo-Cℓ for the first 100 mocks at a given nbins."""
    input:
        expand(
            f"{MOCK_RESULTS}/pseudo_cl_glass_mock_{{mock_id}}_powspace_nbins={{cl_nbins}}.fits",
            mock_id=GLASS_MOCK_IDS,
            cl_nbins=[32],  # default; override with --config cl_nbins=[96,128]
        ),


rule glass_mock_validation:
    """Aggregator: all mock validation inputs (ξ± + pseudo-Cℓ for 100 mocks)."""
    input:
        rules.glass_mock_all_xi.input,
        rules.glass_mock_all_pseudo_cl.input,


# @sc [decision:mocks.cosebis_bias_test]
rule mock_cosebis_scatter:
    """Scatter: compute COSEBIS E_n/B_n for one GLASS mock.

    Reads fine-binned ξ±, applies scale cuts, computes COSEBIS modes.
    Byte-order conversion for numba compatibility handled internally.
    """
    input:
        xi=MOCK_XI,
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


# @sc [decision:covariance.cosmocov_terms_per_grid]
# @sc [decision:mocks.cosebis_bias_test]
rule mock_cosebis_bias_test:
    """Gather: 100-mock COSEBIS bias test figure + evidence.

    Collects per-mock COSEBIS from scatter jobs, propagates CosmoCov ξ±
    covariance to COSEBIS space, tests mean B_n = 0 at σ/√N precision.
    """
    input:
        cosebis=expand(
            f"{MOCK_RESULTS}/cosebis_glass_mock_{{mock_id}}.npz",
            mock_id=GLASS_MOCK_IDS,
        ),
        xi_ref=MOCK_XI.format(mock_id="00001"),
        cov=covariance_path(
            FIDUCIAL["version"],
            "g",
            MOCK_XI_GRID["min_sep"],
            MOCK_XI_GRID["max_sep"],
            MOCK_XI_GRID["nbins"],
            "_masked",
        ),
    params:
        nmodes=config["fiducial"]["nmodes"],
        theta_min=config["cosebis"]["theta_min"],
        theta_max=config["cosebis"]["theta_max"],
    output:
        figure=f"results/tapestry/mock_cosebis_bias_test/{GLASS_MOCK_SUITE}/figure.png",
        evidence=f"results/tapestry/mock_cosebis_bias_test/{GLASS_MOCK_SUITE}/evidence.json",
    script:
        "../scripts/mock_cosebis_bias_test.py"


localrules:
    glass_mock_all_xi,
    glass_mock_all_pseudo_cl,
    glass_mock_validation,
    mock_cosebis_bias_test,
