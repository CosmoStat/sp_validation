# Smokescreen blind-at-birth custody rules (sp_validation.blinding).
#
# Dormant unless a consumer binds a *_blinded part through common.blindable_part,
# which only a `data` run does. The terminal assemble_sacc rule (cosmo_val.smk)
# asserts the shared blind across parts.

import re

# The blindable stems, derived from the same name-builders the producing rules
# use so part names have one authority: the binning-named ξ± parts (rule xi, one
# per named grid) and the analysis pseudo-Cℓ. None contains "_blinded", so the
# generic blind_part rule can never blind its own output twice.
_VERSION_SLOT = "0VERSION0"  # regex-inert placeholder, substituted after escaping
BLINDABLE_STEM = "(?:{})".format(
    "|".join(
        re.escape(stem)
        for stem in (
            [f"{_VERSION_SLOT}_xi_{xi_binning(grid)}" for grid in XI_GRIDS]
            + [f"pseudo_cl_{_VERSION_SLOT}_{pseudo_cl_tag(config)}"]
        )
    ).replace(_VERSION_SLOT, WILDCARD_CONSTRAINTS["version"])
)


rule blind_init:
    """Draw the seed and publish the commitment + encrypted bundle for a version."""
    output:
        commitment=str(COSMO_VAL / "blind" / "{version}" / "commitment.json"),
        bundle=str(COSMO_VAL / "blind" / "{version}" / "blind_seed.encrpt"),
        key=str(COSMO_VAL / "blind" / "{version}" / "blind_seed.key"),
    params:
        blind_dir=lambda w: blind_state_dir(w.version),
    resources:
        runtime=5,
    shell:
        "python {REPO_SCRIPTS}/blind_data_vector.py"
        " blind-init {params.blind_dir}"


rule blind_part:
    """Conceal one part, escrowing its true vector beside the blinded output."""
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
    # --keep-input: the plaintext part is the producing rule's temp() output, so
    # Snakemake removes it once this, its only consumer, finishes.
    shell:
        "python {REPO_SCRIPTS}/blind_data_vector.py"
        " blind-part {input.part} --blind-dir {params.blind_dir} --keep-input"
