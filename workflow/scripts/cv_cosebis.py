"""Rule cv_cosebis: COSEBIs E/B decomposition for one version.

A consumer of the integration-grid ξ± part and the CosmoCov ξ± covariance on
the same grid — nothing here touches a catalogue. The covariance goes through
the same linear kernel as the modes to give the COSEBIs covariance; it is
analytic, so no Hartlap debiasing applies.
"""

import numpy as np
from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation import sacc_io
from sp_validation.b_modes import (
    cosebis_scan_from_xi,
    find_conservative_scale_cut_key,
    log_bin_edges,
    plot_cosebis_covariance_matrix,
    plot_cosebis_modes,
    plot_cosebis_scale_cut_heatmap,
    save_cosebis_results,
)
from sp_validation.cosmo_val.sacc_writers import cosebis_to_sacc

_unbuffer_streams()
p = snakemake.params
version = p["version"]
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])

part = sacc_io.load(snakemake.input["xi"])
theta, xip, xim = sacc_io.get_xi(part, (0, 0), grid="integration")
edges = log_bin_edges(p["min_sep"], p["max_sep"], p["nbins"])

# @sc [decision:bmodes.bmode_covariance]
# @sc [decision:bmodes.cosebis_modes]
results = cosebis_scan_from_xi(
    theta,
    xip,
    xim,
    np.loadtxt(snakemake.input["cov"]),
    *edges,
    nmodes=p["nmodes"],
    scale_cuts=[tuple(sc) for sc in p["scale_cuts"]],
)

# @sc [decision:bmodes.cosebis_modes]
# @sc [decision:bmodes.fiducial_scale_cut]
fiducial_key = find_conservative_scale_cut_key(results, fiducial_scale_cut)
fiducial = results[fiducial_key]

plot_cosebis_modes(
    fiducial,
    version,
    snakemake.output["figure_modes"],
    fiducial_scale_cut=fiducial_scale_cut,
)
plot_cosebis_covariance_matrix(
    fiducial, version, "analytic", snakemake.output["figure_covariance"]
)
plot_cosebis_scale_cut_heatmap(
    results,
    edges,
    version,
    snakemake.output["figure_scalecut_ptes"],
    fiducial_scale_cut=fiducial_scale_cut,
)

save_cosebis_results(results, snakemake.output["npz"], fiducial_scale_cut)

# The part inherits the ξ± part's provenance; `type` is re-stamped on save.
metadata = {k: v for k, v in part.metadata.items() if k != "type"}
s = cosebis_to_sacc({0: sacc_io.get_nz(part, 0)}, metadata, fiducial, fiducial_key)
sacc_io.save(s, snakemake.output["sacc"], type="data")

verify_outputs(snakemake)
