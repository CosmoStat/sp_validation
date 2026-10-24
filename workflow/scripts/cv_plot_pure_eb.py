"""Render pure E/B figures from the already-sealed NPZ only."""

from sp_validation.b_modes import (
    covariance_label,
    load_pure_eb_results,
    plot_eb_covariance_matrix,
    plot_integration_vs_reporting,
    plot_pte_2d_heatmaps,
    plot_pure_eb_correlations,
)

p = snakemake.params
results = load_pure_eb_results(snakemake.input.npz)
kwargs = dict(
    fiducial_xip_scale_cut=tuple(p.fiducial_scale_cut),
    fiducial_xim_scale_cut=tuple(p.fiducial_scale_cut),
)
plot_integration_vs_reporting(
    results, snakemake.output.figure_integration_vs_reporting, p.version
)
plot_pure_eb_correlations(results, snakemake.output.figure_xis, p.version, **kwargs)
plot_pte_2d_heatmaps(results, p.version, snakemake.output.figure_ptes, **kwargs)
plot_eb_covariance_matrix(
    results["cov"],
    covariance_label(results["npatch"]),
    snakemake.output.figure_covariance,
    p.version,
)
