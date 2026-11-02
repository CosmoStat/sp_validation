"""Snakemake entry point for chunked catalogue painting and optional injection."""

from pathlib import Path

import healpy as hp
import numpy as np

from sp_validation.glass_mock import paint_catalogue

injection = dict(snakemake.params.injection)
b_cl = None
amplitude = injection.pop("b_amplitude", 0)
b_cl_path = injection.pop("b_cl_path", None)
if amplitude or b_cl_path:
    nside = hp.npix2nside(len(np.load(snakemake.input.kappa, mmap_mode="r")))
    ell = np.arange(2 * nside + 1)
    if b_cl_path:
        supplied = np.loadtxt(b_cl_path)
        b_cl = np.interp(ell, supplied[:, 0], supplied[:, 1], left=0, right=0)
    else:
        b_cl = amplitude * np.exp(-((ell / 2000) ** 2))
    b_cl[:2] = 0
paint_catalogue(
    snakemake.input.catalogue,
    snakemake.output.catalogue,
    Path(snakemake.input.kappa).parent,
    int(snakemake.wildcards.seed),
    b_cl=b_cl,
    chunk_size=snakemake.params.chunk_size,
    overwrite=True,
    **snakemake.params.columns,
    **injection,
)
