"""Derive pure E/B from the concealed integration-grid ξ± part.

All pairs use the same reporting bins; tomographic scale cuts are not applied.
Their pair-weighted operators propagate the full ξ± covariance without zeroing
any of its blocks.
"""

from pathlib import Path

import numpy as np
from cv_runner import _unbuffer_streams, verify_outputs

from sp_validation import sacc_io
from sp_validation.b_modes import (
    calculate_eb_statistics,
    calculate_pure_eb_correlation,
    pure_eb_operator,
    save_pure_eb_results,
)
from sp_validation.cosmo_val.derived_covariance import (
    propagate,
    xi_covariance,
    xi_pairs,
)
from sp_validation.cosmo_val.sacc_writers import pure_eb_to_sacc

_unbuffer_streams()
p = snakemake.params
part = sacc_io.load(snakemake.input["xi_integration"], blind=p["blind"])
pairs = xi_pairs(part)
views = [sacc_io.xi_correlation(part, pair, grid="integration") for pair in pairs]
covariance, npatch = xi_covariance(part, snakemake.input["cov_integration"])
grid = p["integration"]
edges_int = np.geomspace(grid["min_sep"], grid["max_sep"], grid["nbins"] + 1)
reporting = np.geomspace(p["min_sep"], p["max_sep"], p["nbins"] + 1)
outputs, operators = [], []
for pair, gg in zip(pairs, views):
    cov = covariance[gg._rows[:, None], gg._rows]
    results = calculate_pure_eb_correlation(
        gg.meanr, gg.xip, gg.xim, gg.weight, edges_int, cov, reporting, npatch=npatch
    )
    results = calculate_eb_statistics(results)
    path = Path(snakemake.output["npz"])
    if len(pairs) > 1:
        path = path.with_name(f"{path.stem}_{pair[0]}_{pair[1]}{path.suffix}")
    save_pure_eb_results(results, path, derived_from=[part])
    if pair == pairs[0] and len(pairs) > 1:
        save_pure_eb_results(results, snakemake.output["npz"], derived_from=[part])
    outputs.append((pair, results))
    if len(pairs) > 1:
        operators.append(pure_eb_operator(gg.weight, edges_int, reporting)[0])

cov = outputs[0][1]["cov"] if len(pairs) == 1 else propagate(operators, covariance)
nz = {i: sacc_io.get_nz(part, i) for pair in pairs for i in pair}
s = pure_eb_to_sacc(nz, part.metadata, outputs, covariance=cov)
sacc_io.save(s, snakemake.output["sacc"], derived_from=[part])
verify_outputs(snakemake)
