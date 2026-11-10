"""Is the change in the B-mode nulls between the old and #468 bins noise resampling?

The #468 bins keep most galaxies where they were, so the two blinded data
vectors share most of their shape noise. Per bin i, f_i = S_i / sqrt(N_i^old N_i^new)
with S_i the galaxies in bin i in both runs. Shared pairs lie in the old area only,
so the shape-noise correlation of pair (i, j) between the runs is
f_i f_j sqrt(A_new / A_old) (unit weights; the real weights are not in the transition
counts). The cross-covariance is modelled as F (C_old + C_new)/2 F with
F = diag(sqrt(f_i f_j) (A_new/A_old)^(1/4)) per row. This is exact for pure shape
noise; E→B leakage and signal terms correlate more strongly, so it is a lower bound.

Given the old vector, the new one under pure noise (H0) is Gaussian with
mean Cx^T C_old^-1 d_old and covariance C_new − Cx^T C_old^-1 Cx. Drawing from it
gives the χ² and PTE the new bins would show if the old failures were noise.
The Cℓ^BB template amplitude (report.additive_tests) is a scalar on both runs, so
its pair (A_old, A_new) tests H0 (A = 0) and constancy (one A in both) directly.

Writes resample.json beside the placement it reads. B modes only; the blind adds none.

    python resample.py --placement <dir of the nz468 placement.json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from report import ADDITIVE_B, DATASETS, ROOT, gls_amp, load_aligned
from placement import KEYS, PAIRS, rows
from sp_validation import sacc_io

TRANSITIONS = ROOT.parent / "cosmo-val-v2-tomo-nz468/catalogue/bin_changes.json"
AREAS = (ROOT.parent / "mock-ensemble-tomo/inputs/catalogue/survey_tomo.json",
         ROOT.parent / "cosmo-val-v2-tomo-nz468/catalogue/survey_tomo.json")


def bin_overlap():
    d = json.loads(TRANSITIONS.read_text())
    f = {}
    for i in range(1, 5):
        shared = d["transitions"][f"{i}->{i}"]
        f[i] = shared / np.sqrt(d["old_counts"][str(i)] * d["new_counts"][str(i)])
    return f


def template_pair(so, sn, co, cn, r, t, scale):
    """GLS template amplitudes on both runs, their correlation and the tests built on it."""
    o, n = co[np.ix_(r, r)], cn[np.ix_(r, r)]
    x = scale[r][:, None] * (o + n) / 2 * scale[r][None, :]
    ao, sig_o = gls_amp(so.mean[r], t, o)
    an, sig_n = gls_amp(sn.mean[r], t, n)
    wo = np.linalg.solve(o, t) * sig_o ** 2
    wn = np.linalg.solve(n, t) * sig_n ** 2
    cov_on = float(wo @ x @ wn)
    rho = cov_on / (sig_o * sig_n)
    pi = np.linalg.inv(np.array([[sig_o ** 2, cov_on], [cov_on, sig_n ** 2]]))
    one = np.ones(2)
    a_joint = float(one @ pi @ np.array([ao, an]) / (one @ pi @ one))
    sig_joint = float((one @ pi @ one) ** -.5)
    # Pure noise: the new amplitude regresses to ρσ_n/σ_o of the old. A fixed component:
    # the two differ by noise only, tested by constant_a_diff_z.
    return dict(
        old=ao, old_sigma=sig_o, new=an, new_sigma=sig_n, rho=rho,
        h0_new_given_old=[rho * sig_n / sig_o * ao, sig_n * np.sqrt(1 - rho ** 2)],
        constant_a_diff_z=(an - ao) / np.sqrt(sig_o ** 2 + sig_n ** 2 - 2 * cov_on),
        joint=a_joint, joint_sigma=sig_joint, joint_z=a_joint / sig_joint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", required=True)
    ap.add_argument("--draws", type=int, default=20000)
    args = ap.parse_args()
    pattern = str(ROOT / "cosmo_val/MOCK_0000[12]_tomo.sacc")
    so, _, _, _ = load_aligned(str(DATASETS["tomo1101"]["sacc"]), pattern)
    sn, _, _, _ = load_aligned(str(DATASETS["nz468"]["sacc"]), pattern)
    placement = json.loads((Path(args.placement) / "placement.json").read_text())
    co, cn = np.asarray(so.covariance.dense), np.asarray(sn.covariance.dense)

    f = bin_overlap()
    area_old, area_new = (json.loads(p.read_text())["area_deg2"] for p in AREAS)
    scale = np.ones(len(so.data))
    for k, x in enumerate(so.data):
        i, j = (int(t.split("_")[-1]) + 1 for t in x.tracers)  # tracers source_0..3 are bins 1..4
        scale[k] = np.sqrt(f[i] * f[j]) * (area_new / area_old) ** .25

    b_rows = np.unique(np.concatenate([placement["statistics"][k]["joint"]["rows"] for k in KEYS]))
    so_sd, sn_sd = np.sqrt(np.diag(co)), np.sqrt(np.diag(cn))
    zo, zn = so.mean[b_rows] / so_sd[b_rows], sn.mean[b_rows] / sn_sd[b_rows]
    out = dict(bin_overlap=f, area_ratio=area_new / area_old, n_b_rows=int(len(b_rows)),
               empirical_corr=float(np.corrcoef(zo, zn)[0, 1]),
               model_corr_mean=float(np.mean(scale[b_rows] ** 2)), tests={})
    rng = np.random.default_rng(468)
    for k in KEYS:
        r = np.asarray(placement["statistics"][k]["joint"]["rows"])
        o, n = co[np.ix_(r, r)], cn[np.ix_(r, r)]
        x = scale[r][:, None] * (o + n) / 2 * scale[r][None, :]
        do, dn = so.mean[r], sn.mean[r]
        gain = np.linalg.solve(o, x).T
        mean, cond = gain @ do, n - gain @ x
        draws = rng.multivariate_normal(mean, (cond + cond.T) / 2, size=args.draws, method="cholesky")
        ni = np.linalg.inv(n)
        chi_new = np.einsum("ij,jk,ik->i", draws, ni, draws)
        data_new = float(dn @ ni @ dn)
        dof = len(r)
        out["tests"][k] = dict(
            dof=dof, chi2_old=float(do @ np.linalg.solve(o, do)), chi2_new=data_new,
            pte_old=float(stats.chi2.sf(do @ np.linalg.solve(o, do), dof)),
            pte_new=float(stats.chi2.sf(data_new, dof)),
            h0_chi2_new_median=float(np.median(chi_new)),
            h0_chi2_new_68=np.percentile(chi_new, [16, 84]).tolist(),
            h0_pte_new_median=float(stats.chi2.sf(np.median(chi_new), dof)),
            h0_frac_chi2_new_le_data=float(np.mean(chi_new <= data_new)))

    rs = {p: rows(so, sacc_io.CL_BB, p) for p in PAIRS}
    for name, keep in (("common", lambda p: True), ("autos", lambda p: p[0] == p[1]),
                       ("crosses", lambda p: p[0] != p[1])):
        r = np.concatenate([rs[p] for p in PAIRS if keep(p)])
        ells = np.array([so.data[i].tags["ell"] for i in r])
        t = ADDITIVE_B * np.exp(-(ells / 2000) ** 2)
        out[f"clbb_template_{name}"] = template_pair(so, sn, co, cn, r, t, scale)
        # The model correlation is a lower bound; the measured B-element correlation an upper one.
        boost = np.sqrt(out["empirical_corr"] / out["model_corr_mean"])
        out[f"clbb_template_{name}_empirical_rho"] = template_pair(so, sn, co, cn, r, t, scale * boost)
    path = Path(args.placement) / "resample.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
