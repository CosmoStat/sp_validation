"""Summarize the completed weighted CosmoSIS chain."""

from sp_validation.inference import summarize

summarize(snakemake.input.chain, snakemake.output.summary)
