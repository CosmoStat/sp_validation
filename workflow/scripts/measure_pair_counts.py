"""Write a non-cosmological pair-count cache on the requested xi grid."""

from sp_validation.pair_counts import measure_pair_counts

measure_pair_counts(
    snakemake.input.catalog,
    snakemake.output.cache,
    min_sep=float(snakemake.wildcards.min_sep),
    max_sep=float(snakemake.wildcards.max_sep),
    nbins=int(snakemake.wildcards.nbins),
    ra_col=snakemake.params.ra_col,
    dec_col=snakemake.params.dec_col,
    w_col=snakemake.params.w_col,
    num_threads=snakemake.threads,
)
