"""Rule cv_cosebis: COSEBIs E/B decomposition for one version.

Compute + plot rule (per version). plot_cosebis calls calculate_cosebis over a
fine integration binning (the 2000-bin TreeCorr is the dominant cost) and
evaluates the configured scale cuts. Writes the {version}_eb_..._data.npz
COSEBIs data product (declared output) plus figures, and the per-version
COSEBIs PTE that cv_summarize_bmodes collects.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
p = snakemake.params
cv.plot_cosebis(
    version=p["version"],
    min_sep_int=p["min_sep_int"],
    max_sep_int=p["max_sep_int"],
    nbins_int=p["nbins_int"],
    npatch=p["npatch"],
    nmodes=p["nmodes"],
    scale_cuts=[tuple(sc) for sc in p["scale_cuts"]],
    fiducial_scale_cut=tuple(p["fiducial_scale_cut"]),
)
verify_outputs(snakemake)
