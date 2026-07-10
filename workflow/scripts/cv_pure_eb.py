"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

Compute + plot rule (per version). plot_pure_eb calls calculate_pure_eb, which
runs two TreeCorr correlations (reporting + integration binning); the reporting
binning reuses the cv_2pcf data vector via calculate_2pcf's skip-if-exists
path. Writes the {version}_eb_..._data.npz data product plus companion figures,
and the per-version E/B PTEs that cv_summarize_bmodes collects. It also writes
the born-as-SACC pure-E/B part ({version}_pure_eb.sacc, the six PURE_KEYS blocks
+ covariance) that the assemble_sacc rule consumes.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
p = snakemake.params
version = p["version"]
cv.plot_pure_eb(
    versions=[version],
    min_sep_int=p["min_sep_int"],
    max_sep_int=p["max_sep_int"],
    nbins_int=p["nbins_int"],
    fiducial_xip_scale_cut=tuple(p["fiducial_scale_cut"]),
    fiducial_xim_scale_cut=tuple(p["fiducial_scale_cut"]),
)
# Born-as-SACC pure-E/B part (plot_pure_eb stored the results on the instance).
cv.pure_eb_to_sacc_part(version, snakemake.output["sacc"], cv._pure_eb_results[version])
verify_outputs(snakemake)
