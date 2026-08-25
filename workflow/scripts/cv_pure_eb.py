"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

Compute + plot rule (per version). plot_pure_eb calls calculate_pure_eb, which
runs two TreeCorr correlations (reporting + integration binning); the reporting
binning reuses the cv_2pcf data vector via calculate_2pcf's skip-if-exists
path. Writes the {version}_eb_..._data.npz data product (declared output) plus
companion figures, and the per-version E/B PTEs that cv_summarize_bmodes
collects.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
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
