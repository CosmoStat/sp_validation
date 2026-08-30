# Two-point data-vector rules: xi, rho/tau, and pseudo-Cl products.

# ---------------------------------------------------------------------------
# ξ± angular grids
# ---------------------------------------------------------------------------
# A grid IS a binning: (min_sep, max_sep, nbins, npatch) plus how the
# born-as-SACC part carries its covariance. The reporting grid is the analysis
# one (its ξ covariance is injected at assembly from CosmoCov, so the part is
# written bare); the integration grid is the fine grid COSEBIs and pure-E/B
# integrate over, whose only covariance estimate is TreeCorr's shot-noise
# varxip/varxim — attached as a DiagonalCovariance.
#
# Both grids are measured by the single `xi` rule below: files are named by
# binning, so the grid label and the covariance mode are *resolved* from the
# wildcards rather than duplicated into a second rule. Workflows that carry no
# cosmo_val block (e.g. papers/bmodes) fall back to their fiducial grids; a
# binning matching no named grid is measured as a plain reporting-style
# measurement (no covariance).
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
    return {
        "reporting": {**reporting, "covariance": "none"},
        "integration": {**integration, "covariance": "diagonal"},
    }


XI_GRIDS = _xi_grids()
XI_DEFAULT_GRID = ("reporting", "none")


def xi_binning(grid):
    """The `minsep=..._maxsep=..._nbins=..._npatch=...` tag of a named grid."""
    g = XI_GRIDS[grid]
    return (
        f"minsep={g['min_sep']}_maxsep={g['max_sep']}"
        f"_nbins={g['nbins']}_npatch={g['npatch']}"
    )


def xi_grid_of(wildcards):
    """(grid label, covariance mode) for the binning a job was requested with."""
    key = (wildcards.min_sep, wildcards.max_sep, wildcards.nbins, wildcards.npatch)
    for name, g in XI_GRIDS.items():
        if tuple(str(g[k]) for k in ("min_sep", "max_sep", "nbins", "npatch")) == key:
            return name, g["covariance"]
    return XI_DEFAULT_GRID


rule xi:
    """TreeCorr ξ±(θ) for one version on one angular grid.

    Binning-agnostic: the reporting and integration measurements are the same
    job with different wildcards. The raw TreeCorr .txt byproduct (read back by
    the covariance machinery and by the skip-if-exists) and the born-as-SACC
    part are named by that binning, so a request for either binds unambiguously
    — and the grid label + covariance treatment come from XI_GRIDS.
    """
    input:
        catalog=get_shear_catalog,
    output:
        txt=str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.txt"),
        # Blindable part: temp() on a data run so only its blinded sibling
        # persists (blind_part escrows the true vector first). See common.maybe_temp.
        sacc=maybe_temp(str(COSMO_VAL / "{version}_xi_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc")),
    threads: 24
    params:
        ver="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        cat_config=CAT_CONFIG,
        grid=lambda w: xi_grid_of(w)[0],
        covariance=lambda w: xi_grid_of(w)[1],
        # Stamped as the part's SACC `type` — custody state at assembly
        # (see blinding.assert_consistent_blind).
        type=run_type(),
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
    # ρ/τ has no blindable input; it binds only the commitment, to stamp its
    # part concealed pass-through (see common.commitment_input).
    input:
        unpack(lambda w: commitment_input(w.version)),
    output:
        rho_stats=str(COSMO_VAL / "rho_tau_stats/rho_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        tau_stats=str(COSMO_VAL / "rho_tau_stats/tau_stats_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.fits"),
        # Born-as-SACC ρ/τ part (ρ_0…ρ_5 autos + τ_0/τ_2/τ_5 leakage, carrying
        # its own covariance block) that the assemble_sacc rule consumes;
        # calculate_rho_tau_stats writes it alongside the FITS via
        # rho_tau_to_sacc_part.
        rho_tau=str(COSMO_VAL / "rho_tau_stats/rho_tau_{version}_minsep={min_sep}_maxsep={max_sep}_nbins={nbins}_npatch={npatch}.sacc"),
    threads: 48
    params:
        ver="{version}",
        min_sep="{min_sep}",
        max_sep="{max_sep}",
        nbins="{nbins}",
        npatch="{npatch}",
        type=run_type(),
    resources:
        mem_mb=30000,
        disk_mb=20000,
    script:
        "../scripts/run_rho_tau.py"


# Pseudo-Cl generation for harmonic-space data vectors and COSEBIS validation.
BASE_VERSIONS = [v.replace("_leak_corr", "") for v in config["versions"]]

wildcard_constraints:
    binning="linear|logspace|powspace",


rule pseudo_cl:
    """Generate pseudo-Cl data vector (born as SACC) with configurable binning.

    NB: the ``blind`` wildcard is the glass-mock A/B/C variant (three mock
    catalogues), NOT Smokescreen blinding — see common.py BLINDS. The Smokescreen
    concealed=True stamp is a separate axis on the SACC file.
    """
    output:
        # This generic rule produces every pseudo-Cℓ variant — the analysis part
        # (blind=A, powspace, nbins=32) folded into {version}.sacc, plus the fine
        # (COSEBIS) and glass-mock variants. Only the analysis part is a terminal
        # blindable, and a data run blinds it via a requested _blinded sibling
        # (blind_part reads this plaintext); the fine/mock variants are B-mode /
        # validation intermediates left untouched here. The output is therefore
        # not temp()'d — see the PR note on residual unblinded pseudo-Cℓ.
        pseudo_cl=str(COSMO_VAL / "pseudo_cl_{version}_blind={blind}_{binning}_nbins={nbins}.sacc"),
    wildcard_constraints:
        blind="[ABC]",  # glass-mock variant, not Smokescreen blinding
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

    NB: ``blind`` is the glass-mock A/B/C variant, not Smokescreen blinding
    (see common.py BLINDS).
    """
    output:
        pseudo_cl_cov=str(COSMO_VAL / "pseudo_cl_cov_{version}_blind={blind}_{binning}_nbins={nbins}.fits"),
    wildcard_constraints:
        blind="[ABC]",  # glass-mock variant, not Smokescreen blinding
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
