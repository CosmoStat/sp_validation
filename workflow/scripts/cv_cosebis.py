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

# Born-blinded E-mode on a data run: the En written to the SACC part (and the
# npz sidecar) is re-derived from the *blinded* integration ξ± part, at the
# fiducial scale cut, through the same cosmo_numba kernel the raw path uses
# (b_modes.cosebis_from_xi). Bn and the covariance are blind-invariant and stay
# from the raw plot_cosebis result. The blinded integration part is bound as an
# input only for the fiducial version (xi_highres is fiducial-only); a mock run
# binds neither input, keeping En raw and the part unconcealed.
en_blinded = None
commitment_path = snakemake.input.get("commitment")
if snakemake.input.get("xi_integration"):
    from sp_validation import sacc_io
    from sp_validation.b_modes import cosebis_from_xi

    integ = sacc_io.load(snakemake.input["xi_integration"])
    theta, xip, xim = sacc_io.get_xi(integ, (0, 0), grid="integration")
    en_blinded, _ = cosebis_from_xi(
        theta, xip, xim, p["nmodes"], scale_cut=fiducial_scale_cut
    )

cv.cosebis_to_sacc_part(
    version,
    snakemake.output["sacc"],
    cv._cosebis_results[version],
    fiducial_scale_cut=fiducial_scale_cut,
    en_override=en_blinded,
    commitment_path=commitment_path,
)

# No unblinded E-mode on disk: overwrite the raw fiducial-cut En plot_cosebis
# wrote into the diagnostic npz with the blinded En (identical to the SACC
# part's). Bn / cov / PTE fields are blind-invariant and untouched, so the
# B-mode summary reader is unaffected. Fiducial-only, like the SACC En.
if en_blinded is not None:
    npz_path = snakemake.output["npz"]
    data = dict(np.load(npz_path, allow_pickle=True))
    data["En"] = np.asarray(en_blinded)
    np.savez(npz_path, **data)

verify_outputs(snakemake)
