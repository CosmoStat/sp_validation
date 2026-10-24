"""Render every COSEBIs figure from the already-sealed multi-cut NPZ."""

from sp_validation.b_modes import (
    find_conservative_scale_cut_key,
    load_cosebis_results,
    plot_cosebis_covariance_matrix,
    plot_cosebis_modes,
    plot_cosebis_scale_cut_heatmap,
)

p = snakemake.params
results, edges = load_cosebis_results(snakemake.input.npz)
cut = tuple(p.fiducial_scale_cut)
fiducial = results[find_conservative_scale_cut_key(results, cut)]
plot_cosebis_modes(fiducial, p.version, snakemake.output.figure_modes, cut)
plot_cosebis_covariance_matrix(
    fiducial, p.version, "analytic", snakemake.output.figure_covariance
)
plot_cosebis_scale_cut_heatmap(
    results, edges, p.version, snakemake.output.figure_scalecut_ptes, cut
)
