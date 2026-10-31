# Glass mock validation: fine-binned ξ± and NaMaster pseudo-Cℓ
# for GLASS mock realizations.
#
# Produces inputs for mock_validation_all_b_mode (research notebook).
# Pre-computed 20-bin ξ± and 32-bin Cℓ from Sacha's pipeline are too coarse
# or bypass the MCM — these rules run the full pipeline on mock catalogs.

# GLASS_MOCK_DIR and GLASS_MOCK_SUITE defined in Snakefile.
GLASS_MOCK_IDS = [f"{i:05d}" for i in range(1, 101)]
MOCK_RESULTS = f"results/glass_mock/{GLASS_MOCK_SUITE}"

# Mock ξ± are measured on the data's integration grid, node for node, so the
# E/B transforms see the same θ sampling on mocks and data.
MOCK_XI_GRID = XI_GRIDS["integration"]
MOCK_XI = f"{MOCK_RESULTS}/gg_glass_mock_{{mock_id}}_{xi_binning('integration')}.fits"

wildcard_constraints:
    cl_nbins=r"\d+",


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


rule mock_cosebis_bias_test:
    """Gather: 100-mock COSEBIS bias test figure + evidence.

    Collects per-mock COSEBIS from scatter jobs, propagates OneCovariance ξ±
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


# Catalogue-painted mocks are opt-in: no real catalogue is a default target.
# See workflow/README.md for the glass_paint configuration and CLI.
GLASS_PAINT = config.get("glass_paint", {})
GLASS_PAINT_ROOT = GLASS_PAINT.get("output_dir", "results/glass_painted")
GLASS_PAINT_MAPS = f"{GLASS_PAINT_ROOT}/seed_{{seed}}/maps"
GLASS_PAINT_CATALOGUES = GLASS_PAINT.get("catalogues", {})
GLASS_PAINT_INJECTIONS = GLASS_PAINT.get("injections", {"none": {}})


rule glass_painted_maps:
    """One n(z)-averaged lognormal sky, shared by all catalogue/injection choices."""
    input:
        nz=lambda wc: GLASS_PAINT["nz"],
    output:
        kappa=f"{GLASS_PAINT_MAPS}/kappa.npy",
        gamma1=f"{GLASS_PAINT_MAPS}/gamma1.npy",
        gamma2=f"{GLASS_PAINT_MAPS}/gamma2.npy",
        spectra=f"{GLASS_PAINT_MAPS}/spectra.npz",
        metadata=f"{GLASS_PAINT_MAPS}/maps.json",
    params:
        output=lambda wc: GLASS_PAINT_MAPS.format(seed=wc.seed),
        cache=GLASS_PAINT.get("cache_dir", f"{GLASS_PAINT_ROOT}/cache"),
        map_config=GLASS_PAINT.get("map_config", {}),
    threads: GLASS_PAINT.get("threads", 28)
    resources:
        mem_mb=160000,
        runtime=240,
    script:
        "../scripts/run_glass_painted_maps.py"


rule glass_painted_catalogue:
    """Rotate calibrated input shapes and paint selected shear in bounded chunks."""
    input:
        catalogue=lambda wc: GLASS_PAINT_CATALOGUES[wc.catalogue]["path"],
        kappa=rules.glass_painted_maps.output.kappa,
        gamma1=rules.glass_painted_maps.output.gamma1,
        gamma2=rules.glass_painted_maps.output.gamma2,
        metadata=rules.glass_painted_maps.output.metadata,
        b_cl=lambda wc: ([GLASS_PAINT_INJECTIONS[wc.injection]["b_cl_path"]]
                        if GLASS_PAINT_INJECTIONS[wc.injection].get("b_cl_path") else []),
    output:
        catalogue=f"{GLASS_PAINT_ROOT}/seed_{{seed}}/{{catalogue}}_{{injection}}.h5",
        metadata=f"{GLASS_PAINT_ROOT}/seed_{{seed}}/{{catalogue}}_{{injection}}.h5.json",
    params:
        columns=lambda wc: {
            key: value for key, value in GLASS_PAINT_CATALOGUES[wc.catalogue].items()
            if key != "path"
        },
        injection=lambda wc: GLASS_PAINT_INJECTIONS[wc.injection],
        chunk_size=GLASS_PAINT.get("chunk_size", 250000),
    threads: 4
    resources:
        mem_mb=32000,
        runtime=60,
    script:
        "../scripts/run_glass_painted_catalogue.py"


localrules:
    glass_mock_all_xi,
    glass_mock_all_pseudo_cl,
    glass_mock_validation,
    mock_cosebis_bias_test,
