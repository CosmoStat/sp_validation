"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

A consumer of the two ξ± parts plus one covariance file — nothing here touches
a catalogue. The modes come from the reporting and integration parts through
the pipeline kernel; the covariance is Monte Carlo through that same kernel,
drawn from the CosmoCov integration-grid ξ± covariance around a theory mean, so
it depends on the covariance model and the grids rather than on the measured
vector. A jackknife of the transformed modes would need per-patch realisations,
which are never persisted.

On a data run the parts it reads are the blinded ones, so these modes are born
blinded and the output is stamped under the same commitment.
"""

import numpy as np
from cs_util.cosmo import get_cosmo
from cv_runner import _unbuffer_streams, verify_outputs
from snakemake.script import snakemake

from sp_validation import sacc_io
from sp_validation.b_modes import (
    calculate_eb_statistics,
    log_bin_edges,
    plot_eb_covariance_matrix,
    plot_integration_vs_reporting,
    plot_pte_2d_heatmaps,
    plot_pure_eb_correlations,
    pure_eb_covariance_mc,
    pure_eb_from_xi,
    save_pure_eb_results,
)
from sp_validation.cosmo_val.sacc_writers import pure_eb_to_sacc

_unbuffer_streams()
p = snakemake.params
version = p["version"]
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])

reporting = sacc_io.load(snakemake.input["xi_reporting"])
integration = sacc_io.load(snakemake.input["xi_integration"])
theta, xip, xim = sacc_io.get_xi(reporting, (0, 0), grid="reporting")
theta_int, xip_int, xim_int = sacc_io.get_xi(integration, (0, 0), grid="integration")
left_edges, right_edges = log_bin_edges(p["min_sep"], p["max_sep"], p["nbins"])

# The reporting grid must sit strictly inside the integration grid: a reporting
# point on the boundary has no interior support and comes back NaN.
modes = pure_eb_from_xi(
    theta, xip, xim, theta_int, xip_int, xim_int, left_edges[0], right_edges[-1]
)

z, nz = sacc_io.get_nz(reporting, 0)
cov, eb_samples = pure_eb_covariance_mc(
    theta=theta,
    left_edges=left_edges,
    right_edges=right_edges,
    theta_int=theta_int,
    cov_int=np.loadtxt(snakemake.input["cov_integration"]),
    z=z,
    nz=nz,
    cosmo=get_cosmo(**p["cosmo_params"]),
    n_samples=p["n_samples"],
)

variances = reporting.covariance.dense.diagonal()
results = {
    "theta": theta,
    "left_edges": left_edges,
    "right_edges": right_edges,
    "xip": xip,
    "xim": xim,
    "var_xip": variances[: len(theta)],
    "var_xim": variances[len(theta) :],
    "theta_int": theta_int,
    "xip_int": xip_int,
    "xim_int": xim_int,
    "n_eff": p["n_samples"],
    "cov": cov,
    "eb_samples": eb_samples,
    **modes,
}
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
    cov, "semi-analytic", snakemake.output["figure_covariance"], version
)

save_pure_eb_results(results, snakemake.output["npz"])

# The part inherits the ξ± part's provenance; `type` and the blind stamp are
# re-applied on save, from the run type and the version's commitment.
metadata = {
    k: v
    for k, v in reporting.metadata.items()
    if k not in ("type", "concealed", "blind_commitment", "blind_config_digest")
}
s = pure_eb_to_sacc(
    {0: (z, nz)},
    metadata,
    theta,
    {key: results[key] for key in sacc_io.PURE_KEYS},
    covariance=cov,
)
sacc_io.save(
    s,
    snakemake.output["sacc"],
    type=p["type"],
    commitment=snakemake.input.get("commitment", None),
)

verify_outputs(snakemake)
