"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

A consumer of the integration-grid ξ± part plus a ξ± covariance on that grid —
nothing here touches a catalogue. The estimator is one fixed linear operator
on the fine ξ± (b_modes.pure_eb_operator), averaged into the reporting bins with
the part's pair counts, so its covariance is the supplied ξ± covariance pushed
exactly through that operator.
"""

import numpy as np
from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation import sacc_io
from sp_validation.b_modes import (
    calculate_eb_statistics,
    calculate_pure_eb_correlation,
    covariance_label,
    log_bin_edges,
    plot_eb_covariance_matrix,
    plot_integration_vs_reporting,
    plot_pte_2d_heatmaps,
    plot_pure_eb_correlations,
    save_pure_eb_results,
)
from sp_validation.cosmo_val.sacc_writers import pure_eb_to_sacc

_unbuffer_streams()
p = snakemake.params
version = p["version"]
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])

part = sacc_io.load(snakemake.input["xi_integration"])
theta_int, xip_int, xim_int = sacc_io.get_xi(part, (0, 0), grid="integration")
npairs_int = sacc_io.get_xi_npairs(part, (0, 0), grid="integration")
left_edges, right_edges = log_bin_edges(p["min_sep"], p["max_sep"], p["nbins"])

results = calculate_pure_eb_correlation(
    theta_int,
    xip_int,
    xim_int,
    npairs_int,
    np.loadtxt(snakemake.input["cov_integration"]),
    left_edges,
    right_edges,
)
results = calculate_eb_statistics(results)

plot_integration_vs_reporting(
    results, snakemake.output["figure_integration_vs_reporting"], version
)
plot_pure_eb_correlations(
    results,
    snakemake.output["figure_xis"],
    version,
    fiducial_xip_scale_cut=fiducial_scale_cut,
    fiducial_xim_scale_cut=fiducial_scale_cut,
)
plot_pte_2d_heatmaps(
    results,
    version,
    snakemake.output["figure_ptes"],
    fiducial_xip_scale_cut=fiducial_scale_cut,
    fiducial_xim_scale_cut=fiducial_scale_cut,
)
plot_eb_covariance_matrix(
    results["cov"],
    covariance_label(results["npatch"]),
    snakemake.output["figure_covariance"],
    version,
)

save_pure_eb_results(results, snakemake.output["npz"])

# The part inherits the ξ± part's provenance; `type` is re-stamped on save.
metadata = {k: v for k, v in part.metadata.items() if k != "type"}
s = pure_eb_to_sacc(
    {0: sacc_io.get_nz(part, 0)},
    metadata,
    results["theta"],
    {key: results[key] for key in sacc_io.PURE_KEYS},
    covariance=results["cov"],
)
sacc_io.save(s, snakemake.output["sacc"], type="data")

verify_outputs(snakemake)
