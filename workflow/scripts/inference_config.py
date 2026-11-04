"""Render a run INI from an explicit template, values and priors."""

from sp_validation.inference import render

run = snakemake.params.run
render(
    snakemake.input.template,
    snakemake.output.ini,
    fits_path=snakemake.input.fits,
    values=snakemake.input.values,
    priors=snakemake.input.priors,
    chain=snakemake.params.chain,
    n_bins=run.get("n_bins", 1),
    sampler=run.get("sampler", "nautilus"),
    settings=run.get("sampler_settings"),
    scale_cuts=run.get("scale_cuts"),
    overrides=run.get("overrides"),
)
