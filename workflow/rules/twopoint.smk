# Two-point data-vector rules: xi, rho/tau, and pseudo-Cl products.

# ---------------------------------------------------------------------------
# ξ± angular grids
# ---------------------------------------------------------------------------
# A grid is a binning: (min_sep, max_sep, nbins, npatch). `reporting` is the
# analysis grid, `integration` the fine one the B-mode integrals run over.
# Workflows carrying no cosmo_val block (e.g. papers/bmodes) fall back to their
# fiducial grids.
def _xi_grids():
    cv = config.get("cosmo_val", {})
    reporting = (
        {
            "min_sep": cv["theta_min"],
            "max_sep": cv["theta_max"],
            "nbins": cv["nbins"],
            "npatch": cv["npatch"],
        }
        if cv
        else {k: FIDUCIAL[k] for k in ("min_sep", "max_sep", "nbins", "npatch")}
    )
    integration = dict(
        cv.get("integration")
        or {
            "min_sep": FIDUCIAL["min_sep_int"],
            "max_sep": FIDUCIAL["max_sep_int"],
            "nbins": FIDUCIAL["nbins_int"],
        }
    )
    integration.setdefault("npatch", 1)
    return {"reporting": reporting, "integration": integration}


XI_GRIDS = _xi_grids()
XI_KEYS = ("min_sep", "max_sep", "nbins", "npatch")


def xi_binning(grid):
    """The `minsep=..._maxsep=..._nbins=..._npatch=...` tag of a named grid."""
    g = XI_GRIDS[grid]
    return (
        f"minsep={g['min_sep']}_maxsep={g['max_sep']}"
        f"_nbins={g['nbins']}_npatch={g['npatch']}"
    )


def xi_grid_of(wildcards):
    """Grid label for the binning a job was requested with.

    Compared numerically, so a "300" wildcard matches a 300.0 config value.
    Binnings matching no named grid (e.g. papers/bmodes' nbins=10000
    convergence check) are measured as plain reporting-style measurements.
    """
    key = tuple(float(getattr(wildcards, k)) for k in XI_KEYS)
    for name, g in XI_GRIDS.items():
        if tuple(float(g[k]) for k in XI_KEYS) == key:
            return name
    return "reporting"


rule xi:
    """TreeCorr ξ±(θ) for one version on one angular grid.

    One rule for every grid: outputs are named by their binning, so a request
    binds the wildcards and `xi_grid_of` resolves the grid label from them.
    """
    input:
        catalog=get_shear_catalog,
    output:
        txt=str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.txt"),
        # Blindable part: temp() on a data run so only its blinded sibling persists.
        sacc=maybe_temp(str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc")),
    threads: 24
    params:
        ver="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        cat_config=CAT_CONFIG,
        grid=lambda w: xi_grid_of(w),
        type=run_type(),  # the part's SACC `type` — custody state at assembly
    resources:
        # The fine integration grid needs more memory and wall time than the
        # ~20-bin reporting one; scale on nbins rather than splitting the rule.
        mem_mb=lambda w: 40000 if int(w.nbins) > 100 else 30000,
        disk_mb=20000,
        runtime=lambda w: 600 if int(w.nbins) > 100 else 360,
    script:
        "../scripts/run_2pcf.py"


rule run_cosmo_val:
    """Full CosmoVal diagnostic suite."""
    output:
        sentinel=str(COSMO_VAL / "run_cosmo_val.done"),
    threads: 24
    resources:
        mem_mb=60000,
        disk_mb=20000,
        runtime=360,
    shell:
        """
        export PYTHONPATH="/home/cdaley/.local/lib/python3.12/site-packages:${{PYTHONPATH:-}}"
        cd /n17data/cdaley/unions/pure_eb/code/sp_validation/cosmo_val \
        && python run_cosmo_val.py \
        && touch {output.sentinel}
        """


rule rho_tau_stats:
    # ρ/τ has no blindable input; it binds the commitment only to stamp its part
    # concealed pass-through.
    input:
        unpack(lambda w: commitment_input(w.version)),
    output:
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # Born-as-SACC ρ/τ part, written alongside the FITS.
        rho_tau=str(COSMO_VAL / "rho_tau_stats/rho_tau_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc"),
    threads: 48
    params:
        ver="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        type=run_type(),
        blind_root=blind_root(),
    resources:
        mem_mb=30000,
        disk_mb=20000,
    script:
        "../scripts/run_rho_tau.py"


# Pseudo-Cl generation for harmonic-space data vectors and COSEBIS validation.
BASE_VERSIONS = [v.replace("_leak_corr", "") for v in config["versions"]]

wildcard_constraints:
    binning="linear|logspace|powspace",


HARMONIC_FIDUCIAL = config["harmonic"]["fiducial"]
PSEUDO_CL_PARAMS = dict(
    cat_config=CAT_CONFIG,
    nside=1024,
    npatch=1,
    cosmo_params=PLANCK18,
    power=0.5,
)


rule pseudo_cl:
    """Generate pseudo-Cl data vector (born as SACC) with configurable binning.

    The diagnostic variants (bmodes claims, mocks, the fine COSEBIs binning);
    the analysis part has its own rule below.
    """
    output:
        pseudo_cl=str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_{binning}_nbins={nbins}.sacc"),
    wildcard_constraints:
        blind="[ABC]",
    params:
        version="{version}",
        blind="{blind}",
        binning="{binning}",
        nbins=lambda w: int(w.nbins),
        **PSEUDO_CL_PARAMS,
    resources:
        mem_mb=32000,
        runtime=120,
    threads: 12
    script:
        "../scripts/generate_pseudo_cl.py"


rule pseudo_cl_analysis:
    """The analysis pseudo-Cℓ part, at the fiducial harmonic binning.

    Split from the generic `pseudo_cl` rule because this variant alone is
    blindable: on a data run the plaintext part is temp(), consumed only by
    blind_part, so only the blinded sibling persists.
    """
    output:
        pseudo_cl=maybe_temp(
            str(COSMO_VAL / f"{pseudo_cl_analysis_stem(config, '{version}')}.sacc")
        ),
    params:
        version="{version}",
        blind=HARMONIC_FIDUCIAL["blind"],
        binning=HARMONIC_FIDUCIAL["binning"],
        nbins=int(HARMONIC_FIDUCIAL["nbins"]),
        **PSEUDO_CL_PARAMS,
    resources:
        mem_mb=32000,
        runtime=120,
    threads: 12
    script:
        "../scripts/generate_pseudo_cl.py"


rule pseudo_cl_cov:
    """Generate pseudo-Cl covariance with configurable binning."""
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


PSEUDO_CL_VERSIONS = config["versions"]


rule pseudo_cl_all:
    """Generate pseudo-Cls for all versions."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind=A_powspace_nbins=32.sacc"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_cov_all:
    """Generate pseudo-Cl covariances for all versions."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_cov_{version}_blind=A_powspace_nbins=32.fits"),
            version=PSEUDO_CL_VERSIONS,
        ),


rule pseudo_cl_fine_all:
    """Generate fine pseudo-Cls for COSEBIS."""
    input:
        expand(
            str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_linear_nbins=2040.sacc"),
            version=config["versions"],
            blind=BLINDS,
        ),
