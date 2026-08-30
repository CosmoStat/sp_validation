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

# Re-derive the pure modes from the ξ± parts via the same kernel; cov stays raw.
# tmin/tmax are the reporting grid's TreeCorr bin edges (add_xi stores no edges).
from sp_validation import sacc_io
from sp_validation.b_modes import pure_eb_from_xi

gg = results["gg"]
tmin, tmax = float(gg.left_edges[0]), float(gg.right_edges[-1])
rep = sacc_io.load(snakemake.input["xi_reporting"])
integ = sacc_io.load(snakemake.input["xi_integration"])
tr, xpr, xmr = sacc_io.get_xi(rep, (0, 0), grid="reporting")
ti, xpi, xmi = sacc_io.get_xi(integ, (0, 0), grid="integration")
modes = pure_eb_from_xi(tr, xpr, xmr, ti, xpi, xmi, tmin, tmax)

# Blinded at birth on a data run: commitment.json is bound only there.
commitment_path = snakemake.input.get("commitment")

# Born-as-SACC pure-E/B part; the six blocks come from the consumed parts.
cv.pure_eb_to_sacc_part(
    version,
    snakemake.output["sacc"],
    results,
    eb_override=modes,
    commitment_path=commitment_path,
)

# Sync the npz's pure modes with the part-derived values; theta / cov / PTE untouched.
npz_path = snakemake.output["npz"]
data = dict(np.load(npz_path, allow_pickle=True))
for key, arr in modes.items():
    data[key] = np.asarray(arr)
np.savez(npz_path, **data)

verify_outputs(snakemake)
