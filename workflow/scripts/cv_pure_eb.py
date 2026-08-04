"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

Compute + plot rule (per version). plot_pure_eb calls calculate_pure_eb, which
runs two TreeCorr correlations of its own (reporting + integration binning) —
it reads no data vector from disk, so this rule declares no xi input. Writes
the {version}_eb_..._data.npz data product (declared output) plus companion
figures, and the per-version E/B PTEs that cv_summarize_bmodes collects.

Non-tomographic: plot_pure_eb does not forward compute_tomography to
calculate_pure_eb, so the measurement runs on the single "all" bin pair.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
p = snakemake.params
cv.plot_pure_eb(
    versions=[p["version"]],
    min_sep_int=p["min_sep_int"],
    max_sep_int=p["max_sep_int"],
    nbins_int=p["nbins_int"],
    fiducial_xip_scale_cut=tuple(p["fiducial_scale_cut"]),
    fiducial_xim_scale_cut=tuple(p["fiducial_scale_cut"]),
)
verify_outputs(snakemake)
