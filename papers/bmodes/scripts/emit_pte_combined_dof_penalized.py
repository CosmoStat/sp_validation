"""Emit the degrees-of-freedom-penalized combined B-mode PTE as an ASTRA
`type: metric` output.

The paper footnote notes that if the two fitted scale-cut boundaries are
treated as fitted parameters (reducing the chi-square degrees of freedom by 2),
the minimum combined PTE across all statistics rises from the un-penalized
value to a penalized one, still above the 0.05 threshold.

The combined PTE is `scipy.stats.chi2.sf(chi2_obs, nu)` with nu = 2 * (number
of theta bins in the cut) for the joint xi_+ / xi_- test. The per-cut PTE grid
npz stores the PTE matrix but not the raw chi2/nu; however nu is fully
determined by the fiducial scale cut (12-83 arcmin -> bins [8, 15] -> 8 bins
-> nu = 16), so chi2_obs is recovered exactly by inverting the stored
un-penalized PTE, and the penalized PTE follows with nu-2. No MC re-run needed.
"""

import argparse
import json
import os

import numpy as np
from scipy import stats

# Fiducial scale cut (arcmin); the boundaries are the two "fitted" parameters.
FIDUCIAL_CUT_ARCMIN = (12.0, 83.0)
N_FITTED_BOUNDARIES = 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pte-per-cut-npz",
        required=True,
        help="<version>_A_pure_eb_ptes.npz with theta + pte_combined grid",
    )
    p.add_argument("--out", required=True, help="output directory")
    args = p.parse_args()

    d = np.load(args.pte_per_cut_npz)
    theta = d["theta"]
    s = int(np.argmin(np.abs(theta - FIDUCIAL_CUT_ARCMIN[0])))
    e = int(np.argmin(np.abs(theta - FIDUCIAL_CUT_ARCMIN[1])))
    nbins = (e + 1) - s
    nu = 2 * nbins  # xi_+ and xi_- jointly

    pte_unpen = float(d["pte_combined"][s, e])
    chi2_obs = float(stats.chi2.isf(pte_unpen, nu))
    nu_pen = nu - N_FITTED_BOUNDARIES
    pte_pen = float(stats.chi2.sf(chi2_obs, nu_pen))

    os.makedirs(args.out, exist_ok=True)

    metric = {"value": round(pte_pen, 4), "uncertainty": None, "unit": None}
    with open(os.path.join(args.out, "pte_combined_dof_penalized.json"), "w") as f:
        json.dump(metric, f, indent=2)

    evidence = {
        "spec_id": "pte_combined_dof_penalized",
        "spec_path": "papers/bmodes/scripts/emit_pte_combined_dof_penalized.py",
        "evidence": {
            "description": "Degrees-of-freedom-penalized combined B-mode PTE for the "
            "fiducial catalog at fiducial scale cuts: the un-penalized "
            "combined PTE recomputed with nu-2 (two fitted scale-cut "
            "boundaries). chi2_obs recovered exactly by inverting the "
            "stored un-penalized PTE at the known nu.",
            "statistic": "combined_xitot_B",
            "cut": "fiducial",
            "scale_cut_arcmin": list(FIDUCIAL_CUT_ARCMIN),
            "bin_indices": [s, e],
            "nu": nu,
            "nu_penalized": nu_pen,
            "n_fitted_boundaries": N_FITTED_BOUNDARIES,
            "pte_unpenalized": pte_unpen,
            "chi2_obs": chi2_obs,
            "pte_penalized": pte_pen,
            "source_npz": os.path.abspath(args.pte_per_cut_npz),
        },
        "output": {"metric": "pte_combined_dof_penalized.json"},
    }
    with open(os.path.join(args.out, "evidence.json"), "w") as f:
        json.dump(evidence, f, indent=2)

    print(
        f"nu={nu} chi2={chi2_obs:.4f} pte_unpen={pte_unpen:.5f} pte_penalized={pte_pen:.5f}"
    )


if __name__ == "__main__":
    main()
