"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

A consumer of the integration-grid ξ± part plus a ξ± covariance on that grid —
nothing here touches a catalogue. The estimator is one fixed linear operator
on the fine ξ± (b_modes.pure_eb_operator), averaged with the part's TreeCorr
pair weights into reporting bins snapped onto the fine edges, so its covariance
is the supplied ξ± covariance pushed exactly through that operator. The
pure-E/B part is a derivation of the ξ± part and carries its blind stamp.
"""

import numpy as np
from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation import sacc_io
from sp_validation.b_modes import (
    calculate_eb_statistics,
    calculate_pure_eb_correlation,
    covariance_label,
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
gg_int = sacc_io.xi_correlation(part, grid="integration")
# A part stores bin centres; the edges come from the grid it was measured on.
grid = p["integration"]
edges_int = np.geomspace(grid["min_sep"], grid["max_sep"], grid["nbins"] + 1)

results = calculate_pure_eb_correlation(
    gg_int.meanr,
    gg_int.xip,
    gg_int.xim,
    gg_int.weight,
    edges_int,
    np.loadtxt(snakemake.input["cov_integration"]),
    np.geomspace(p["min_sep"], p["max_sep"], p["nbins"] + 1),
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

sacc_io.save(
    pure_eb_to_sacc(
        {0: sacc_io.get_nz(part, 0)},
        part.metadata,
        results["theta"],
        {key: results[key] for key in sacc_io.PURE_KEYS},
        covariance=results["cov"],
    ),
    snakemake.output["sacc"],
    derived_from=[part],
)

verify_outputs(snakemake)
