"""Written readings for the sampled-ensemble report, filled from the placement numbers.

    python readings.py --old placement-n66 --new placement-nz468-oldmocks > readings.html

--old is the placement of the first-bin data vector, --new the placement of the
#468-bin vector against the same mocks, with its resample.json.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from placement import KEYS
from report import NAMES, png


def z(chi2, dof):
    return (chi2 - dof) / np.sqrt(2 * dof)


def fig_rebin(old, new, rs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), layout="constrained",
                               gridspec_kw=dict(width_ratios=[1, 1.4]))
    t = old["additive"]["clbb_template_common"]
    a.hist(t["mocks"], bins=14, color="#c9c6be", edgecolor="white", label=f"{old['n_mocks']} mocks (first bins)")
    r = rs["clbb_template_common"]
    mu, sd = r["h0_new_given_old"]
    a.axvspan(mu - sd, mu + sd, color="#8fb3d9", alpha=.5, label="#468 bins if the first were pure noise (68%)")
    a.axvline(r["old"], color="#c0392b", lw=2, label=f"data, first bins {r['old']:.2f} ± {r['old_sigma']:.2f}")
    a.axvline(r["new"], color="#e67e22", lw=2, ls="--", label=f"data, #468 bins {r['new']:.2f} ± {r['new_sigma']:.2f}")
    a.set_xlabel("Cℓ^BB additive template amplitude (all pairs)")
    a.set_yticks([])
    a.legend(fontsize=7, loc="upper left", frameon=False)
    keys = list(KEYS)
    y = np.arange(len(keys))[::-1]
    for yy, k in zip(y, keys):
        s = rs["tests"][k]
        lo, hi = (z(c, s["dof"]) for c in s["h0_chi2_new_68"])
        b.plot([lo, hi], [yy, yy], color="#8fb3d9", lw=6, solid_capstyle="butt")
        b.plot(z(s["chi2_old"], s["dof"]), yy, "o", color="#c0392b", ms=7)
        b.plot(z(s["chi2_new"], s["dof"]), yy, "D", color="#e67e22", ms=6)
    b.axvline(0, color="k", lw=.6)
    b.set_yticks(y, [NAMES[k] for k in keys], fontsize=8)
    b.set_xlabel("(χ² − dof) / √(2 dof), analytic covariance of each run")
    b.set_title("joint B-mode χ²: first bins (red), #468 bins (orange), noise-resampling prediction (blue, 68%)",
                fontsize=8)
    return png(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    old = json.loads((Path(args.old) / "placement.json").read_text())
    new = json.loads((Path(args.new) / "placement.json").read_text())
    rs = json.loads((Path(args.new) / "resample.json").read_text())
    n = old["n_mocks"]
    J = {k: old["statistics"][k]["joint"] for k in KEYS}
    A = old["additive"]
    t, tc = rs["clbb_template_common"], rs["clbb_template_crosses"]
    te = rs["clbb_template_common_empirical_rho"]
    T = rs["tests"]
    s_xi = [J[k]["s"] for k in ("xip_B", "xim_B", "combined", "small")]
    s_cl = [J[k]["s"] for k in ("cl_bb", "cl_eb")]
    f = rs["bin_overlap"]
    noise_kept = rs["model_corr_mean"]
    an = new["additive"]["clbb_template_common"]
    snew = [new["statistics"][k]["joint"]["s"] for k in KEYS]

    print(f"""<h2>Readings</h2>
<p class="lede"><b>The data's B-mode excess is coherent and additive. The #468 rebinning's better B nulls are noise resampling, and they neither strengthen nor weaken that.</b>
On the first bins, {n} sampled GLASS mocks put the Cℓ^BB additive-template amplitude at {A['clbb_template_common']['z']:+.1f}σ
(0 of {n} mocks reach it) and the mean signed Cℓ^BB offset at {A['mean_offset_cl_bb']['z']:+.1f}σ (0 of {n}).
The #468 bins lower the joint χ² by what resampling about a quarter of the shape noise predicts. Their own template amplitude, {t['new']:.2f} ± {t['new_sigma']:.2f}, is again reached by none of the mocks;
the two binnings share too much noise for the comparison to tell a fixed component from a fluctuation.</p>

<h3>First bins, {n} mocks</h3>
<ul>
<li><b>The three χ² failures stay at the edge of the ensemble.</b> ξ−B (12–83′) has χ² {J['xim_B']['data_chi2']:.1f} against a mock maximum of {max(J['xim_B']['mock_chi2']):.1f};
Cℓ^BB {J['cl_bb']['data_chi2']:.1f} against {max(J['cl_bb']['mock_chi2']):.1f}; ξ±B {J['combined']['data_chi2']:.1f} is exceeded by {J['combined']['rank_upper_count']} of {n}.
Recalibrated PTEs are {J['xim_B']['recalibrated_pte']:.3f}, {J['cl_bb']['recalibrated_pte']:.3f} and {J['combined']['recalibrated_pte']:.3f}: tensions, not detections, on χ² alone.</li>
<li><b>The sign is what makes it significant.</b> χ² ignores sign; an additive B field (PSF leakage, a shared shear offset) pushes every band the same way.
The data's Cℓ^BB is high almost everywhere (per-element panel below), the template amplitude is {A['clbb_template_common']['data']:.2f} against mocks at {np.mean(A['clbb_template_common']['mocks']):+.2f} ± {np.std(A['clbb_template_common']['mocks'], ddof=1):.2f},
and the cross-pairs alone, which carry no pseudo-Cℓ noise bias, give {A['clbb_template_crosses']['z']:+.1f}σ.
ξ±B at 2–20′ is high on average by {A['mean_offset_small']['z']:+.1f}σ (0 of {n}), although its χ² passes. Cℓ^EB is symmetric and passes everything.</li>
<li><b>The mocks find the analytic covariance slightly low for ξ_B.</b> The mean mock χ²/dof is s = {min(s_xi):.2f}–{max(s_xi):.2f} for the ξ-based B tests
(bootstrap error ~0.02, so it's resolved), {J['cosebis']['s']:.2f} for COSEBIs and {min(s_cl):.2f}–{max(s_cl):.2f} for Cℓ.
The per-element excess is ~4% below 5′, where shape noise dominates, and ~10% at 20–90′, so it grows with scale.
Candidates I haven't separated: E→B leakage of the finite-range pure-mode decomposition carrying the lognormal E variance, and the mocks' own pair counts
(sampled positions with bias 1.2), which the data's measured-pair-count noise term doesn't describe. The recalibrated PTEs above absorb it;
whether the data share it is the painted ensemble's question, since painted mocks keep the real positions.</li>
</ul>

<h3>The #468 bins against the same mocks</h3>
<p>The remeasured data vector (sp_validation #468 bins and dz = 0.05 n(z), same code) is placed against the first-bin mocks.
<b>The mocks are on the old bins</b>: old n(z), the old footprint (3,815 deg² against 4,065) and the old per-bin noise, so their variance exceeds the new covariance
(s = {min(snew):.2f}–{max(snew):.2f}) and the ensemble PTEs of this placement run high. Comparisons against the mock spread are conservative for the same reason:
the template amplitude {an['data']:.2f} is reached by {an['rank_above']} of {n} mocks ({an['z']:+.1f}σ).</p>
<p><b>Noise resampling, calibrated.</b> In each bin a fraction f = {min(f.values()):.2f}–{max(f.values()):.2f} of the galaxy count is shared between the two binnings
(from <code>bin_changes.json</code>). With the area growth (shared pairs live only in the old {100 * (1 - 1 / rs['area_ratio']):.0f}%-smaller area), the two vectors share about {100 * noise_kept:.0f}% of their shape-noise variance.
That is a lower bound: E→B leakage and any real B signal correlate fully between the runs, and real weights may differ from counts. The measured correlation of the B elements is {rs['empirical_corr']:.2f}.
Given the old vector, the new one is Gaussian under pure noise, and drawing from that conditional gives the χ² the #468 bins would show if the old excess were noise:</p>
<table><tr><th>statistic</th><th>dof</th><th>first bins χ² (PTE)</th><th>#468 bins χ² (PTE)</th><th>predicted by resampling alone, median [68%]</th></tr>
{''.join(f"<tr><td>{NAMES[k]}</td><td>{T[k]['dof']}</td><td>{T[k]['chi2_old']:.1f} ({T[k]['pte_old']:.3f})</td><td>{T[k]['chi2_new']:.1f} ({T[k]['pte_new']:.3f})</td><td>{T[k]['h0_chi2_new_median']:.1f} [{T[k]['h0_chi2_new_68'][0]:.1f}, {T[k]['h0_chi2_new_68'][1]:.1f}]</td></tr>" for k in KEYS)}
</table>
<p>The χ² moved as resampling predicts: the PTE improvements (ξ±B 0.006 → 0.10, Cℓ^BB 0.017 → 0.099) are regression of the fresh quarter of the noise toward its mean.
They would happen whether or not a systematic is present, so they say nothing about whether the first bins were wrong.</p>
<p><b>The template amplitude can't discriminate either.</b> If the first-bin excess were pure noise, the shared noise would still carry {t['old']:.2f} to {t['h0_new_given_old'][0]:.2f} ± {t['h0_new_given_old'][1]:.2f} (about 3σ) on the #468 bins; a fixed component would keep it near {t['old']:.2f}.
The #468 bins give {t['new']:.2f}: {(t['new'] - t['h0_new_given_old'][0]) / t['h0_new_given_old'][1]:+.1f}σ from the noise prediction and {t['constant_a_diff_z']:+.1f}σ from constancy, a likelihood ratio near 1.
The cross-pairs behave the same way ({tc['old']:.2f} → {tc['new']:.2f}; noise alone predicts {tc['h0_new_given_old'][0]:.2f} ± {tc['h0_new_given_old'][1]:.2f}).
Combining the runs with their correlation gives {t['joint']:.2f} ± {t['joint_sigma']:.2f} ({t['joint_z']:.1f}σ), which is the first-bin result restated: the new bins shrink σ by only {100 * (1 - t['joint_sigma'] / t['old_sigma']):.0f}%.
With the measured correlation in place of the model, the numbers barely move (noise prediction {te['h0_new_given_old'][0]:.2f} ± {te['h0_new_given_old'][1]:.2f}, constancy {te['constant_a_diff_z']:+.1f}σ, joint {te['joint_z']:.1f}σ).
So the evidence for an additive component is the first-bin amplitude against the mocks, whose spread ({np.std(A['clbb_template_common']['mocks'], ddof=1):.2f}) calibrates the GLS σ ({t['old_sigma']:.2f}); the rebinning leaves it where it was.</p>
{fig_rebin_img(old, new, rs)}
<p class="mute">Model behind the prediction: for pair (i, j) the shape-noise correlation between the binnings is f_i f_j √(A_new/A_old), with f_i = (galaxies in bin i in both) / √(N_i,old N_i,new) from unit counts.
The cross-covariance is F (C_old + C_new)/2 F with F = √(f_i f_j) (A_new/A_old)^¼ per row. The template shape (ADDITIVE_B e^{{−(ℓ/2000)²}}) was fitted to the non-tomographic run's B modes of largely the same galaxies, so its significance is mildly a posteriori. Numbers: <code>placement-nz468-oldmocks/resample.json</code>.</p>

<h3>What follows</h3>
<ul>
<li>The additive B is a data systematic for the data leg; the fitted PSF model explains ~26–30% of it there. The mocks contain none by construction, so the next mock ingredient is
PSF leakage injected at the measured per-bin α, to see whether the data's amplitude and its pair pattern follow.</li>
<li>A new-bin ensemble (#468 n(z), footprint and per-bin noise) replaces this placement for the #468 vector; it's started on Fir.</li>
</ul>""")


def fig_rebin_img(old, new, rs):
    return f'<img src="{fig_rebin(old, new, rs)}" alt="first vs #468 bins">'


if __name__ == "__main__":
    main()
