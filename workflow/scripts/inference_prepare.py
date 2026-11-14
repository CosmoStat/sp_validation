"""Convert a reporting SACC and record the likelihood's input provenance."""

import json
from pathlib import Path

from sp_validation.inference import prepare

run = snakemake.params.run
provenance = prepare(
    snakemake.input.sacc,
    snakemake.output.fits,
    n_bins=run.get("n_bins", 1),
    statistic=run.get("statistic", "xi"),
    covariance=run.get("covariance"),
    rho_stats=run.get("rho_stats"),
    tau_stats=run.get("tau_stats"),
)
Path(snakemake.output.provenance).write_text(json.dumps(provenance, indent=2) + "\n")
