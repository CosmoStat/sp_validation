"""Rule cv_pure_eb: pure E/B-mode decomposition for one version.

Compute + plot rule (per version). plot_pure_eb calls calculate_pure_eb, which
runs two TreeCorr correlations (reporting + integration binning); the reporting
binning reuses the cv_2pcf data vector via calculate_2pcf's skip-if-exists
path. Writes the {version}_eb_..._data.npz data product plus companion figures,
and the per-version E/B PTEs that cv_summarize_bmodes collects. It also writes
the born-as-SACC pure-E/B part ({version}_pure_eb.sacc, the six PURE_KEYS blocks
+ covariance) that the assemble_sacc rule consumes.
"""

import numpy as np
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
results = cv._pure_eb_results[version]

# Born-blinded pure-E/B on a data run: the six pure-mode arrays written to the
# SACC part (and the npz sidecar) are re-derived from the *blinded* reporting +
# integration ξ± parts through the same cosmo_numba kernel the raw path uses
# (b_modes.pure_eb_from_xi). The covariance is blind-invariant and stays from
# the raw plot_pure_eb result. tmin/tmax are the reporting grid's TreeCorr bin
# edges (from the raw reporting gg, not the SACC — add_xi stores no edges). Both
# blinded inputs are bound only for the fiducial version (xi_highres integration
# is fiducial-only); a mock run binds none, keeping raw modes and no concealment.
eb_blinded = None
commitment_path = snakemake.input.get("commitment")
if snakemake.input.get("xi_integration"):
    from sp_validation import sacc_io
    from sp_validation.b_modes import pure_eb_from_xi

    rep = sacc_io.load(snakemake.input["xi_reporting"])
    integ = sacc_io.load(snakemake.input["xi_integration"])
    theta_r, xip_r, xim_r = sacc_io.get_xi(rep, (0, 0), grid="reporting")
    theta_i, xip_i, xim_i = sacc_io.get_xi(integ, (0, 0), grid="integration")
    gg = results["gg"]
    eb_blinded = pure_eb_from_xi(
        theta_r,
        xip_r,
        xim_r,
        theta_i,
        xip_i,
        xim_i,
        float(gg.left_edges[0]),
        float(gg.right_edges[-1]),
    )

cv.pure_eb_to_sacc_part(
    version,
    snakemake.output["sacc"],
    results,
    eb_override=eb_blinded,
    commitment_path=commitment_path,
)

# No unblinded E-mode on disk: overwrite the raw pure-mode arrays plot_pure_eb
# wrote into the diagnostic npz with the blinded ones (identical to the SACC
# part's). theta / cov / PTE fields are blind-invariant and untouched, so the
# B-mode summary reader is unaffected. Fiducial-only, like the SACC modes.
if eb_blinded is not None:
    npz_path = snakemake.output["npz"]
    data = dict(np.load(npz_path, allow_pickle=True))
    for key, arr in eb_blinded.items():
        data[key] = np.asarray(arr)
    np.savez(npz_path, **data)

verify_outputs(snakemake)
