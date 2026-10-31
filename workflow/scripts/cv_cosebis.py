"""Derive COSEBIs from a concealed integration-grid ξ± part.

All pairs use the same fiducial scale cut; tomographic cuts are not applied.
The COSEBIs convention sets E×B covariance blocks to zero across all pairs.
"""

from pathlib import Path

from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation import sacc_io
from sp_validation.b_modes import (
    cosebis_operator,
    cosebis_scan_from_xi,
    find_conservative_scale_cut_key,
    log_bin_edges,
    save_cosebis_results,
)
from sp_validation.cosmo_val.derived_covariance import (
    propagate,
    xi_covariance,
    xi_pairs,
)
from sp_validation.cosmo_val.sacc_writers import cosebis_to_sacc

_unbuffer_streams()
p = snakemake.params
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])
part = sacc_io.load(snakemake.input["xi"], blind=p["blind"])
pairs = xi_pairs(part)
# Read only after the declared blind has been checked.
xis = [sacc_io.get_xi(part, pair, grid="integration") for pair in pairs]
covariance, npatch = xi_covariance(part, snakemake.input["cov"])
edges = log_bin_edges(p["min_sep"], p["max_sep"], p["nbins"])
outputs, operators = [], []
for pair, (theta, xip, xim) in zip(pairs, xis):
    view = sacc_io.xi_correlation(part, pair, grid="integration")
    cov = covariance[view._rows[:, None], view._rows]
    results = cosebis_scan_from_xi(
        theta,
        xip,
        xim,
        cov,
        *edges,
        nmodes=p["nmodes"],
        scale_cuts=[tuple(sc) for sc in p["scale_cuts"]],
        npatch=npatch,
    )
    key = find_conservative_scale_cut_key(results, fiducial_scale_cut)
    outputs.append((pair, results[key]))
    if len(pairs) > 1:
        operators.append(cosebis_operator(theta, *edges, key, p["nmodes"]))
    path = Path(snakemake.output["npz"])
    if len(pairs) > 1:
        path = path.with_name(f"{path.stem}_{pair[0]}_{pair[1]}{path.suffix}")
    save_cosebis_results(results, path, fiducial_scale_cut, edges, derived_from=[part])
    if pair == pairs[0] and len(pairs) > 1:
        save_cosebis_results(
            results,
            snakemake.output["npz"],
            fiducial_scale_cut,
            edges,
            derived_from=[part],
        )

cov = (
    outputs[0][1]["cov"]
    if len(pairs) == 1
    else propagate(operators, covariance, cosebis=True)
)
nz = {i: sacc_io.get_nz(part, i) for pair in pairs for i in pair}
s = cosebis_to_sacc(nz, part.metadata, outputs, key, covariance=cov)
sacc_io.save(s, snakemake.output["sacc"], derived_from=[part])
verify_outputs(snakemake)
