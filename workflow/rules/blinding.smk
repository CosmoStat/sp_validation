# Smokescreen blind-at-birth custody rules (issues #247/#252, PR #253).
#
# Two rules realise the three-verb custody surface of sp_validation.blinding on
# the DAG. They only enter the graph on a `data` run, and only when a consumer
# binds to a *_blinded part through common.blindable_part — a `mock` run never
# requests a blinded file, so blind_part and blind_init stay dormant.
#
#   blind_init  (once per catalogue version) draws the seed, publishes
#               commitment.json + the encrypted seed bundle.
#   blind_part  (once per blindable part, at birth) conceals the part, escrows
#               the true vector beside the blinded output, and lets Snakemake
#               remove the plaintext (temp()) once it is the sole consumer.
#
# The terminal assemble_sacc rule (cosmo_val.smk) asserts the shared commitment
# across parts — the assembly-time custody check of #252.

# The blindable stems: the binning-named ξ± parts (rule xi — one rule, both the
# reporting and the integration grid, told apart only by their binning tag) and
# the analysis pseudo-Cℓ (pseudo_cl). None contains "_blinded", so the generic
# blind_part rule can never blind its own output twice. The version pattern is
# the shared one from common.WILDCARD_CONSTRAINTS.
_V = WILDCARD_CONSTRAINTS["version"]
BLINDABLE_STEM = (
    rf"(?:{_V}_xi_minsep=[0-9.]+_maxsep=[0-9.]+_nbins=\d+_npatch=\d+"
    rf"|pseudo_cl_{_V}_blind=[ABC]_[a-z]+_nbins=\d+)"
)


rule blind_init:
    """Fix the blind for one catalogue version (blind-init).

    Draws an OS-entropy seed, writes the repo-committable commitment.json
    (seed commitment + config digest + the installed fork's draw scheme) and the
    Fernet-encrypted seed bundle. Runs once per version and refuses to overwrite
    existing state — a blind is a one-shot custody event.
    """
    output:
        commitment=str(COSMO_VAL / "blind" / "{version}" / "commitment.json"),
        bundle=str(COSMO_VAL / "blind" / "{version}" / "blind_seed.encrpt"),
        key=str(COSMO_VAL / "blind" / "{version}" / "blind_seed.key"),
    params:
        blind_dir=lambda w: blind_state_dir(w.version),
    resources:
        runtime=5,
    script:
        "../scripts/blind_init.py"


rule blind_part:
    """Blind one intermediate part SACC at birth (blind-part).

    Conceals the plaintext part through its matching theory backend, escrows the
    true vector into a per-part encrypted bundle beside the blinded output, and
    leaves the plaintext for Snakemake to remove (it is a temp() output of the
    producing rule, and this is its only consumer on a data run). Generic over
    the three blindable stems.
    """
    input:
        part=str(COSMO_VAL / "{stem}.sacc"),
        commitment=lambda w: blind_state_paths(version_of(w.stem))["commitment"],
        bundle=lambda w: blind_state_paths(version_of(w.stem))["bundle"],
        key=lambda w: blind_state_paths(version_of(w.stem))["key"],
    output:
        blinded=str(COSMO_VAL / "{stem}_blinded.sacc"),
        escrow=str(COSMO_VAL / "{stem}_escrow.encrpt"),
        escrow_key=str(COSMO_VAL / "{stem}_escrow.key"),
    wildcard_constraints:
        stem=BLINDABLE_STEM,
    params:
        blind_dir=lambda w: blind_state_dir(version_of(w.stem)),
    resources:
        runtime=10,
    script:
        "../scripts/blind_part.py"
