"""Rule blind_init: fix the blind for one catalogue version.

Thin wrapper over :func:`sp_validation.blinding.blind_init`. Draws the seed,
writes commitment.json + the encrypted seed bundle into the version's blind
directory. The plaintext seed is never written (the encryptor deletes it).
"""

from snakemake.script import snakemake

from sp_validation import blinding

blinding.blind_init(snakemake.params["blind_dir"])
