"""Rule cv_cosebis: COSEBIs E/B decomposition for one version.

Compute + plot rule (per version). plot_cosebis calls calculate_cosebis over a
fine integration binning (the 2000-bin TreeCorr is the dominant cost) and
evaluates the configured scale cuts. Writes the {version}_eb_..._data.npz
COSEBIs data product plus figures, and the per-version COSEBIs PTE that
cv_summarize_bmodes collects. It also writes the born-as-SACC COSEBIs part
({version}_cosebis.sacc, the fiducial scale cut's {En,Bn,cov}) that the
assemble_sacc rule consumes — the multi-cut .npz sidecar stays the diagnostic
PTE scan.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
p = snakemake.params
version = p["version"]
cv.plot_cosebis(
    version=version,
    min_sep_int=p["min_sep_int"],
    max_sep_int=p["max_sep_int"],
    nbins_int=p["nbins_int"],
    npatch=p["npatch"],
    nmodes=p["nmodes"],
    scale_cuts=[tuple(sc) for sc in p["scale_cuts"]],
    fiducial_scale_cut=tuple(p["fiducial_scale_cut"]),
)
# Born-as-SACC COSEBIs part at the fiducial scale cut (plot_cosebis stored the
# multi-cut results on the instance).
cv.cosebis_to_sacc_part(
    version,
    snakemake.output["sacc"],
    cv._cosebis_results[version],
    fiducial_scale_cut=tuple(p["fiducial_scale_cut"]),
)
verify_outputs(snakemake)
