"""Rule blind_part: blind one intermediate part SACC at birth.

Thin wrapper over :func:`sp_validation.blinding.blind_part`. Conceals the part,
escrows the true vector beside the blinded output, and leaves the plaintext in
place: it is a temp() output of the producing rule, so Snakemake removes it once
this (its only consumer on a data run) finishes. keep_input=True hands that
lifecycle to Snakemake rather than deleting inside the blind step, which keeps
the blinded output and its temp input in one consistent DAG accounting.
"""

from snakemake.script import snakemake

from sp_validation import blinding

blinding.blind_part(
    snakemake.input["part"],
    snakemake.params["blind_dir"],
    keep_input=True,
)
