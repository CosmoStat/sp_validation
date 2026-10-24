"""Rule cv_cosebis: COSEBIs E/B decomposition for one version.

A consumer of the integration-grid ξ± part and the OneCovariance ξ± covariance on
the same grid — nothing here touches a catalogue. The covariance goes through
the same linear kernel as the modes to give the COSEBIs covariance; it is
analytic, so no Hartlap debiasing applies. The COSEBIs part is a derivation of
the ξ± part and carries its blind stamp.
"""

import numpy as np
from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation import sacc_io
from sp_validation.b_modes import (
    cosebis_scan_from_xi,
    find_conservative_scale_cut_key,
    log_bin_edges,
    save_cosebis_results,
)
from sp_validation.cosmo_val.sacc_writers import cosebis_to_sacc

_unbuffer_streams()
p = snakemake.params
version = p["version"]
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])

part = sacc_io.load(snakemake.input["xi"], blind=p["blind"])
theta, xip, xim = sacc_io.get_xi(part, (0, 0), grid="integration")
edges = log_bin_edges(p["min_sep"], p["max_sep"], p["nbins"])

results = cosebis_scan_from_xi(
    theta,
    xip,
    xim,
    np.loadtxt(snakemake.input["cov"]),
    *edges,
    nmodes=p["nmodes"],
    scale_cuts=[tuple(sc) for sc in p["scale_cuts"]],
)

fiducial_key = find_conservative_scale_cut_key(results, fiducial_scale_cut)
fiducial = results[fiducial_key]

save_cosebis_results(results, snakemake.output["npz"], fiducial_scale_cut, edges)

s = cosebis_to_sacc({0: sacc_io.get_nz(part, 0)}, part.metadata, fiducial, fiducial_key)
sacc_io.save(s, snakemake.output["sacc"], derived_from=[part])

verify_outputs(snakemake)
