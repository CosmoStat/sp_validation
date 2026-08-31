"""Rule cv_summarize_bmodes: collect B-mode PTEs across all statistics.

The terminal diagnostic, and a reader of what the three B-mode rules already
wrote: the pure-E/B PTE matrices and the COSEBIs B-mode PTE from their .npz
products, and the pseudo-Cℓ BB spectrum from its SACC part against the NaMaster
covariance. Nothing is recomputed and no catalogue is touched, so the summary
cannot disagree with the products it summarises.
"""

import json

import numpy as np
from cv_runner import _unbuffer_streams, verify_outputs
from snakemake.script import snakemake

from sp_validation import sacc_io
from sp_validation.b_modes import _get_pte_from_scale_cut, log_bin_edges
from sp_validation.cosmo_val.core import print_bmode_summary
from sp_validation.statistics import chi2_and_pte

_unbuffer_streams()
p = snakemake.params
fiducial_scale_cut = tuple(p["fiducial_scale_cut"])
edges = log_bin_edges(p["min_sep"], p["max_sep"], p["nbins"])

summary = {}
cov_methods = set()

for i, version in enumerate(p["versions"]):
    row = {}

    pure_eb = np.load(snakemake.input["pure_eb"][i])
    for stat in ("xip_B", "xim_B", "combined"):
        try:
            row[stat] = _get_pte_from_scale_cut(
                pure_eb[f"pte_matrices_{stat}"], edges, fiducial_scale_cut
            )
        except (KeyError, RuntimeError):
            pass
    cov_methods.add(f"pure-E/B: semi-analytic ({int(pure_eb['n_eff'])} draws)")

    # The COSEBIs .npz is written at the fiducial cut, so its PTE is the one
    # this table wants.
    cosebis = np.load(snakemake.input["cosebis"][i])
    row["COSEBIS"] = float(cosebis["pte_B"])
    cov_methods.add("COSEBIs: propagated from the ξ± covariance")

    if p["include_pseudo_cl"]:
        from astropy.io import fits

        part = sacc_io.load(snakemake.input["pseudo_cl"][i])
        _ell, _ee, bb, _eb, _window = sacc_io.get_pseudo_cl(part, (0, 0))
        with fits.open(snakemake.input["pseudo_cl_cov"][i]) as hdul:
            cov_bb = np.asarray(hdul["COVAR_BB_BB"].data, float)
        _chi2, _red, row["C_l_BB"] = chi2_and_pte(bb, cov_bb)
        cov_methods.add("pseudo-Cℓ: Gaussian (NaMaster)")

    summary[version] = row

print_bmode_summary(summary, fiducial_scale_cut, cov_methods)

with open(snakemake.output["summary_json"], "w") as f:
    json.dump(summary, f, indent=2, default=str)

verify_outputs(snakemake)
