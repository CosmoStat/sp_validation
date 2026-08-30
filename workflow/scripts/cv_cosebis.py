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

import numpy as np
from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
p = snakemake.params
version = p["version"]
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])
cv.plot_cosebis(
    version=version,
    min_sep_int=p["min_sep_int"],
    max_sep_int=p["max_sep_int"],
    nbins_int=p["nbins_int"],
    npatch=p["npatch"],
    nmodes=p["nmodes"],
    scale_cuts=[tuple(sc) for sc in p["scale_cuts"]],
    fiducial_scale_cut=fiducial_scale_cut,
)

# Re-derive En from the integration ξ± part via the same kernel; Bn and cov stay raw.
from sp_validation import sacc_io
from sp_validation.b_modes import cosebis_from_xi

integ = sacc_io.load(snakemake.input["xi_integration"])
theta, xip, xim = sacc_io.get_xi(integ, (0, 0), grid="integration")
en_part, _ = cosebis_from_xi(theta, xip, xim, p["nmodes"], scale_cut=fiducial_scale_cut)

# Blinded at birth on a data run: commitment.json is bound only there.
commitment_path = snakemake.input.get("commitment")

# Born-as-SACC COSEBIs part at the fiducial scale cut.
cv.cosebis_to_sacc_part(
    version,
    snakemake.output["sacc"],
    cv._cosebis_results[version],
    fiducial_scale_cut=fiducial_scale_cut,
    en_override=en_part,
    commitment_path=commitment_path,
)

# Sync the npz's En with the part-derived values; Bn / cov / PTE untouched.
npz_path = snakemake.output["npz"]
data = dict(np.load(npz_path, allow_pickle=True))
data["En"] = np.asarray(en_part)
np.savez(npz_path, **data)

verify_outputs(snakemake)
