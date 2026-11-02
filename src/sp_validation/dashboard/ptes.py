"""B-mode PTE table in the layout of Paper II's PTE table.

One row per catalogue version and scale selection (fiducial, full range);
columns COSEBIs B_n for n <= 6 and n <= 20, pure-B ξ+, ξ− and joint ξ±, and
C_ℓ^BB. Pure E/B PTEs come from the cached scale-cut matrices and the
COSEBIs n <= 20 PTE from the cached transform; the COSEBIs n <= 6 and C_ℓ^BB
PTEs are the same χ² against zero over the corresponding subset of the cached
data vector and covariance (no Hartlap factor: both covariances are
analytic).
"""

import numpy as np
from scipy import stats

from sp_validation.b_modes import bins_from_scale_cut

# Column labels as (symbol, subscript, superscript, qualifier) so the page
# can typeset them with <sub>/<sup>.
COLUMNS = [
    {"key": "cosebis_n6", "group": "COSEBIs", "label": ["B", "n", "", "n ≤ 6"]},
    {"key": "cosebis_n20", "group": "COSEBIs", "label": ["B", "n", "", "n ≤ 20"]},
    {"key": "xip_B", "group": "Pure E/B", "label": ["ξ", "+", "B", ""]},
    {"key": "xim_B", "group": "Pure E/B", "label": ["ξ", "−", "B", ""]},
    {"key": "combined", "group": "Pure E/B", "label": ["ξ", "±", "B", "joint"]},
    {"key": "cl_BB", "group": "Cℓ", "label": ["C", "ℓ", "BB", ""]},
]


def chi2_pte(data, cov):
    return float(stats.chi2.sf(data @ np.linalg.solve(cov, data), len(data)))


def cosebis_pte(cut, nmodes):
    """B_n PTE over the first ``nmodes`` modes of one cached cut."""
    n = len(cut["Bn"])
    block = cut["cov"][n:, n:]
    return chi2_pte(cut["Bn"][:nmodes], block[:nmodes, :nmodes])


def cl_pte(cl, key="BB", ell_cut=None):
    """C_ℓ PTE over bands lying wholly inside ``ell_cut`` (all bands if None)."""
    mask = np.ones(len(cl["ell"]), bool)
    if ell_cut is not None:
        mask = (cl["ell_lo"] >= ell_cut[0]) & (cl["ell_hi"] <= ell_cut[1])
    return chi2_pte(cl[key][mask], cl["cov"][f"{key}_{key}"][np.ix_(mask, mask)])


def rows(pe, co, cl, theta_cut, ell_cut, full_cut):
    """The fiducial and full-range rows for one version."""
    start, stop = bins_from_scale_cut(pe["left_edges"], pe["right_edges"], theta_cut)
    n_theta = len(pe["theta"])
    selections = [
        ("Fiducial", (start, stop - 1), co["fiducial"], ell_cut),
        ("Full range", (0, n_theta - 1), full_cut, None),
    ]
    out = []
    for name, (i, j), co_cut, l_cut in selections:
        cut = co["cuts"][co_cut]
        values = {
            "cosebis_n6": cosebis_pte(cut, 6),
            "cosebis_n20": cosebis_pte(cut, co["nmodes"]),
            **{
                k: float(pe["pte_matrices"][k][i, j])
                for k in ("xip_B", "xim_B", "combined")
            },
            "cl_BB": cl_pte(cl, "BB", l_cut),
        }
        ell_range = (
            f"{l_cut[0]} ≤ ℓ ≤ {l_cut[1]}"
            if l_cut
            else f"{cl['ell_lo'][0]} ≤ ℓ ≤ {cl['ell_hi'][-1]}"
        )
        out.append(
            {
                "selection": name,
                "theta": f"{pe['left_edges'][i]:.0f}′–{pe['right_edges'][j]:.0f}′",
                "cosebis_theta": f"{co_cut[0]:g}′–{co_cut[1]:g}′",
                "ell": ell_range,
                "values": values,
            }
        )
    return out
