"""Snakemake entry point for exact-evaluation catalogue painting and injection."""

from pathlib import Path

import healpy as hp

from sp_validation.glass_mock import b_mode_cl, paint_catalogue, read_sky

injection = dict(snakemake.params.injection)
kappa_alm, _ = read_sky(Path(snakemake.input.alm).parent)
b_cl = b_mode_cl(
    hp.Alm.getlmax(kappa_alm.shape[-1]),
    injection.pop("b_amplitude", 0),
    injection.pop("b_cl_path", None),
)
del kappa_alm
paint_catalogue(
    snakemake.input.catalogue,
    snakemake.output.catalogue,
    Path(snakemake.input.alm).parent,
    int(snakemake.wildcards.seed),
    b_cl=b_cl,
    chunk_size=snakemake.params.chunk_size,
    block_size=snakemake.params.block_size,
    nthreads=snakemake.threads,
    overwrite=True,
    **snakemake.params.columns,
    **injection,
)
