"""Place the blinded v2 tomographic data in the sampled ensemble and write report.html.

Run in the container with this checkout's src and the placement tool on PYTHONPATH:

    python report.py --out <dir> [--html <path>]
"""
import argparse
import base64
import glob
import html
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

UNIONS = Path("/project/rrg-kyi/cdaley/unions").resolve()
ROOT = UNIONS / "analyses/mock-ensemble-sampled"
PAINTED = UNIONS / "analyses/mock-ensemble-tomo"
DATA = PAINTED / "inputs/data-current/cosmo_val"
sys.path.insert(0, str(PAINTED / "placement"))

from placement import KEYS, PAIRS, analyze, label, load_edges, rows  # noqa: E402
from sp_validation import sacc_io  # noqa: E402

NAMES = {"xip_B": "ξ+ B (12–83′)", "xim_B": "ξ− B (12–83′)", "combined": "ξ± B (12–83′)",
         "cosebis": "COSEBIs B_n", "cl_bb": "Cℓ^BB", "cl_eb": "Cℓ^EB", "small": "ξ± B (2–20′)"}
E_TYPES = {"xip_E": (sacc_io.XI_PLUS, "theta", (12, 83)), "xim_E": (sacc_io.XI_MINUS, "theta", (12, 83)),
           "cl_ee": (sacc_io.CL_EE, "ell", (300, 1600)), "cosebis_E": (sacc_io.COSEBI_EE, "n", (1, 20))}
B_ELEMENTS = ("xip_B", "xim_B", "cosebis_B", "cl_bb", "cl_eb")


def png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def img(path_or_uri, alt):
    uri = path_or_uri
    if not str(uri).startswith("data:"):
        uri = "data:image/png;base64," + base64.b64encode(Path(uri).read_bytes()).decode()
    return f'<img src="{uri}" alt="{html.escape(alt)}">'


def load_aligned(data_path, pattern):
    """Data SACC trimmed to the mocks' rows, the mock vectors, and how the definitions differ.

    Sampled mocks have their own positions and unit weights, so measured tags
    (mean θ, pair counts and weights) and the pseudo-Cℓ bandpower windows differ
    slightly from the data's, and the pure-E/B covariance blocks inherit the
    measured θ grid. Rows are matched by type, tracers and order; the tests use
    the data's covariance.
    """
    s = sacc_io.load(data_path, blind="unions")
    paths = sorted(glob.glob(pattern))
    first = sacc_io.load(paths[0], blind="mock-ensemble-tomo")
    for dtype in set(s.get_data_types()) - set(first.get_data_types()):
        s.remove_selection(data_type=dtype)
    c = np.asarray(s.covariance.dense)
    sd = np.sqrt(np.diag(c))
    diffs = dict(max_theta_rel=0.0, max_cov_rel_sigma=0.0, max_window_rel=0.0)
    vectors = []
    for path in paths:
        m = sacc_io.load(path, blind="mock-ensemble-tomo")
        if len(m.data) != len(s.data) or any(
                (x.data_type, x.tracers) != (y.data_type, y.tracers) for x, y in zip(s.data, m.data)):
            raise ValueError(f"SACC rows differ: {path}")
        for x, y in zip(s.data, m.data):
            for tag in ("ell", "n"):
                if tag in x.tags and x.tags[tag] != y.tags[tag]:
                    raise ValueError(f"{tag} differs in {path}")
            if "theta" in x.tags:
                diffs["max_theta_rel"] = max(diffs["max_theta_rel"], abs(y.tags["theta"] / x.tags["theta"] - 1))
        dc = np.abs(np.asarray(m.covariance.dense) - c) / np.outer(sd, sd)
        diffs["max_cov_rel_sigma"] = max(diffs["max_cov_rel_sigma"], float(dc.max()))
        if path == paths[0]:
            for p in PAIRS:
                r = rows(s, sacc_io.CL_EE, p)
                wd, wm = s.get_bandpower_windows(r).weight, m.get_bandpower_windows(r).weight
                diffs["max_window_rel"] = max(diffs["max_window_rel"], float(np.abs(wm - wd).max() / np.abs(wd).max()))
        vectors.append(m.mean.copy())
    return s, np.asarray(vectors), paths, diffs


ADDITIVE_B = 2.6e-11  # the plain run's additive B level: C_BB ≈ A·exp[−(ℓ/2000)²]


def gls_amp(d, t, c):
    ci = np.linalg.solve(c, t)
    return float(ci @ d / (t @ ci)), float((t @ ci) ** -.5)


def additive_tests(s, mocks, result):
    """Statistics an additive B component would move, on the data and on every mock.

    The Cℓ^BB template amplitude is the data leg's (emode-clbb `clbb_template`):
    GLS amplitude of ADDITIVE_B·exp(−(ℓ/2000)²) with the data covariance, common
    to all pairs, autos only and crosses only. The mean signed σ-offset averages
    d/σ_analytic over a test's selected elements: an additive component pushes it
    one way, noise averages it out.
    """
    c = np.asarray(s.covariance.dense)
    out = {}
    rs = {p: rows(s, sacc_io.CL_BB, p) for p in PAIRS}
    ells = {p: np.array([s.data[i].tags["ell"] for i in r]) for p, r in rs.items()}
    for name, keep in (("common", lambda p: True), ("autos", lambda p: p[0] == p[1]),
                       ("crosses", lambda p: p[0] != p[1])):
        r = np.concatenate([rs[p] for p in PAIRS if keep(p)])
        t = np.concatenate([ADDITIVE_B * np.exp(-(ells[p] / 2000) ** 2) for p in PAIRS if keep(p)])
        cr = c[np.ix_(r, r)]
        data, sigma = gls_amp(s.mean[r], t, cr)
        out[f"clbb_template_{name}"] = dict(data=data, sigma=sigma,
                                            mocks=[gls_amp(m[r], t, cr)[0] for m in mocks])
    sd = np.sqrt(np.diag(c))
    for k in ("xip_B", "xim_B", "combined", "cl_bb", "small"):
        r = np.asarray(result["statistics"][k]["joint"]["rows"])
        out[f"mean_offset_{k}"] = dict(data=float(np.mean(s.mean[r] / sd[r])),
                                       mocks=(mocks[:, r] / sd[r]).mean(axis=1).tolist())
    for v in out.values():
        m = np.asarray(v["mocks"])
        v["z"] = float((v["data"] - m.mean()) / m.std(ddof=1))
        v["rank_above"] = int(np.count_nonzero(m >= v["data"]))
    return out


def table_additive(add, n):
    names = {"clbb_template_common": "Cℓ^BB template amplitude, all pairs",
             "clbb_template_autos": "Cℓ^BB template amplitude, autos",
             "clbb_template_crosses": "Cℓ^BB template amplitude, crosses",
             "mean_offset_xip_B": "mean d/σ, ξ+ B 12–83′", "mean_offset_xim_B": "mean d/σ, ξ− B 12–83′",
             "mean_offset_combined": "mean d/σ, ξ± B 12–83′", "mean_offset_cl_bb": "mean d/σ, Cℓ^BB",
             "mean_offset_small": "mean d/σ, ξ± B 2–20′"}
    out = ['<table><tr><th>statistic</th><th>data</th><th>mock mean ± std</th><th>data z</th><th>mocks ≥ data</th></tr>']
    for k, label_ in names.items():
        v = add[k]
        m = np.asarray(v["mocks"])
        extra = f" (GLS σ {v['sigma']:.2f})" if "sigma" in v else ""
        out.append(f"<tr><td>{label_}</td><td>{v['data']:.3f}{extra}</td><td>{m.mean():.3f} ± {m.std(ddof=1):.3f}</td>"
                   f"<td>{v['z']:+.2f}</td><td>{v['rank_above']}/{n}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def e_diagnostics(s, mocks):
    """Ensemble/analytic variance per E element, and data z (blinded; see caveat)."""
    c = np.asarray(s.covariance.dense)
    out = {}
    for name, (dtype, tag, (lo, hi)) in E_TYPES.items():
        out[name] = {}
        for p in PAIRS:
            r = rows(s, dtype, p)
            x = np.array([s.data[i].tags[tag] for i in r])
            r, x = r[(x >= lo) & (x <= hi)], x[(x >= lo) & (x <= hi)]
            m = mocks[:, r]
            out[name][label(p)] = dict(
                coordinate=x.tolist(),
                variance_ratio=(m.var(axis=0, ddof=1) / np.diag(c)[r]).tolist(),
                data_z=((s.mean[r] - m.mean(axis=0)) / m.std(axis=0, ddof=1)).tolist())
    return out


def fig_joint(result, n):
    keys = list(KEYS)
    fig, axes = plt.subplots(1, len(keys), figsize=(14, 2.8), layout="constrained")
    for ax, k in zip(axes, keys):
        j = result["statistics"][k]["joint"]
        q = np.asarray(j["mock_chi2"])
        ax.hist(q, bins=max(5, n // 3), color="#8fb3d9", edgecolor="white")
        ax.axvline(j["data_chi2"], color="#c0392b", lw=2)
        xs = np.linspace(min(q.min(), j["data_chi2"]) * .8, max(q.max(), j["data_chi2"]) * 1.1, 200)
        ax.plot(xs, stats.chi2.pdf(xs, j["dof"]) * len(q) * (np.ptp(q) / max(5, n // 3)), color="k", lw=.8)
        ax.set_title(f"{NAMES[k]}\ndof {j['dof']}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    fig.suptitle("Joint χ² (all 10 pairs): mocks (histogram), analytic χ² (line), blinded data (red)", fontsize=9)
    return png(fig)


def fig_pair_ptes(result):
    keys = list(KEYS)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), layout="constrained")
    for ax, field, title in zip(axes, ("data_analytic_pte", "empirical_pte"),
                                ("analytic PTE", "ensemble PTE (fraction of mocks with χ² ≥ data)")):
        a = np.array([[result["statistics"][k]["pairs"][label(p)][field] for p in PAIRS] for k in keys])
        im = ax.imshow(np.log10(np.clip(a, 1e-3, 1)), cmap="RdYlBu", vmin=-3, vmax=0, aspect="auto")
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                ax.text(j, i, f"{a[i, j]:.2f}" if a[i, j] >= .01 else f"{a[i, j]:.3f}",
                        ha="center", va="center", fontsize=6.5)
        ax.set_xticks(range(len(PAIRS)), [label(p) for p in PAIRS], fontsize=7)
        ax.set_yticks(range(len(keys)), [NAMES[k] for k in keys], fontsize=7)
        ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=axes, label="log10 PTE", shrink=.8)
    return png(fig)


def fig_z(result, ediag):
    names = list(B_ELEMENTS) + list(E_TYPES)
    fig, axes = plt.subplots(len(names), 1, figsize=(11, 1.25 * len(names) + 1), layout="constrained")
    for ax, name in zip(axes, names):
        src = result["elements"][name] if name in B_ELEMENTS else ediag[name]
        blocks, ticks, pos = [], [], 0
        for p in PAIRS:
            z = np.asarray(src[label(p)]["data_z"])
            blocks.append(z)
            ticks.append(pos + len(z) / 2)
            pos += len(z)
        z = np.concatenate(blocks)
        ax.bar(np.arange(len(z)), z, width=1, color=np.where(np.abs(z) > 2, "#c0392b", "#7f8c8d"))
        for b in np.cumsum([len(b) for b in blocks])[:-1]:
            ax.axvline(b - .5, color="k", lw=.4)
        ax.axhspan(-2, 2, color="#d6eaf8", zorder=-1)
        ax.set_xticks(ticks, [label(p) for p in PAIRS], fontsize=6)
        ax.set_xlim(-.5, len(z) - .5)
        ax.set_ylabel(name + (" *" if name in E_TYPES else ""), fontsize=7, rotation=0, ha="right")
        ax.tick_params(axis="y", labelsize=6)
    fig.suptitle("Data z against the ensemble, per element (blue band |z|<2; * E-modes are blinded)", fontsize=9)
    return png(fig)


def fig_variance(result, ediag, lo, hi):
    names = list(B_ELEMENTS) + list(E_TYPES)
    a = np.array([[np.mean((result["elements"][n] if n in B_ELEMENTS else ediag[n])[label(p)]
                            ["diagonal_variance_ratio" if n in B_ELEMENTS else "variance_ratio"])
                   for p in PAIRS] for n in names])
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    im = ax.imshow(a, cmap="PuOr_r", vmin=.6, vmax=1.4, aspect="auto")
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_xticks(range(len(PAIRS)), [label(p) for p in PAIRS], fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    fig.colorbar(im, label="mean over elements of ensemble variance / analytic")
    ax.set_title(f"Ensemble vs analytic diagonal variance (one element's 95% sampling band: {lo:.2f}–{hi:.2f})", fontsize=8)
    return png(fig)


def table_joint(result):
    out = ['<table><tr><th>statistic</th><th>dof</th><th>data χ²</th><th>analytic PTE</th>'
           '<th>ensemble PTE</th><th>s = ⟨χ²_mock⟩/dof</th><th>recalibrated PTE</th><th>KS p (mock PTEs uniform)</th></tr>']
    n = result["n_mocks"]
    for k in KEYS:
        j = result["statistics"][k]["joint"]
        emp = f"{j['rank_upper_count']}/{n}" + (" (&lt; resolution)" if j["rank_upper_count"] == 0 else "")
        out.append(f"<tr><td>{NAMES[k]}</td><td>{j['dof']}</td><td>{j['data_chi2']:.1f}</td>"
                   f"<td>{j['data_analytic_pte']:.4f}</td><td>{emp}</td>"
                   f"<td>{j['s']:.3f} ± {j['s_bootstrap_error']:.3f}</td>"
                   f"<td>{j['recalibrated_pte']:.4f} [{j['recalibrated_pte_68'][1]:.4f}, {j['recalibrated_pte_68'][0]:.4f}]</td>"
                   f"<td>{j['mock_pte_ks_p']:.2f}</td></tr>")
    out.append("</table>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "placement"))
    ap.add_argument("--html", default=None)
    ap.add_argument("--readings", default=None, help="HTML fragment with the written readings")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = str(DATA / "SP_v2.0_tomo.sacc")
    edges = glob.glob(str(DATA / "*_eb_minsep*_tomo_data_0_0.npz"))[0]
    s, mocks, paths, diffs = load_aligned(data, str(ROOT / "cosmo_val/MOCK_?????_tomo.sacc"))
    result = analyze(s, mocks, load_edges(edges, "unions"))
    ediag = e_diagnostics(s, mocks)
    result["e_modes"] = ediag
    result["definition_differences"] = diffs
    result["additive"] = additive_tests(s, mocks, result)
    result["inputs"] = dict(data=data, mocks=paths, edges=edges)
    (out / "placement.json").write_text(json.dumps(result, indent=1, allow_nan=False))
    from figures import render
    render(s, mocks, result, out / "figures", "unions")

    n = result["n_mocks"]
    lo, hi = result["sampling_variance_bands_gaussian"]["0.95"]
    readings = Path(args.readings).read_text() if args.readings else ""
    figs = out / "figures"
    bands = "".join(f"<details><summary>{k}</summary>{img(figs / f'band_{k}.png', k)}</details>"
                    for k in ("xip_B", "xim_B", "cosebis_B", "cl_bb", "cl_eb", "xip_E", "xim_E", "cosebis_E", "cl_ee"))
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mocks vs data</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--mute:#6b6b70;--line:#e2e0da;--accent:#2c5d8f;--bad:#c0392b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#16171a;--fg:#e8e6e1;--mute:#9a9893;--line:#2e2f33;--accent:#7fb0e0}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1100px;margin:0 auto;padding:24px 16px}}
h1{{font-size:1.6em;margin:.2em 0}} h2{{border-bottom:1px solid var(--line);padding-bottom:.2em;margin-top:2em}}
.lede{{font-size:1.05em}} .mute{{color:var(--mute);font-size:.9em}}
table{{border-collapse:collapse;font-size:.85em;margin:1em 0;width:100%;overflow-x:auto;display:block}}
td,th{{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left;white-space:nowrap}}
img{{max-width:100%;background:#fff;border-radius:4px;margin:.5em 0}}
details{{margin:.4em 0}} summary{{cursor:pointer;color:var(--accent)}}
.box{{border-left:3px solid var(--accent);padding:.4em 1em;background:color-mix(in srgb,var(--accent) 6%,transparent)}}
</style></head><body>
<p class="mute">Mocks ↔ data, UNIONS v2 tomographic · {n} sampled GLASS mocks · Claude Opus 5.5 on behalf of Cail</p>
<h1>The blinded v2 tomographic data inside a sampled GLASS ensemble</h1>
{readings}
<h2>Joint null tests</h2>
<p>All ten bin pairs together, with the selections and analytic covariance of the blinded report. <b>Ensemble PTE</b> is the fraction of the {n} mocks with χ² at least the data's (resolution 1/{n}). <b>s</b> is the mocks' mean χ²/dof: 1 if the analytic covariance describes the mocks. The <b>recalibrated PTE</b> rescales the covariance by s (68% bootstrap range in brackets).</p>
{table_joint(result)}
{img(fig_joint(result, n), 'joint chi2 histograms')}
<h2>Is the excess additive?</h2>
<p>χ² is blind to sign. An additive B component (PSF leakage, a shared shear offset) also shows as a coherent sign across bands and pairs, and in the data leg's Cℓ^BB template (ADDITIVE_B·e^{{−(ℓ/2000)²}}, GLS amplitude with the data covariance). The mocks give each statistic's null distribution.</p>
{table_additive(result['additive'], n)}
<h2>Per pair</h2>
{img(fig_pair_ptes(result), 'per-pair PTEs')}
<h2>Per element</h2>
<p>Each bar is one band of one pair: (data − mock mean)/mock std. The B-mode rows test whether the data look like a draw from the mocks. The E-mode rows (*) carry two different blinds (the data's and the mocks' throwaway one) and the mocks' Planck18 cosmology, so their offsets are not a measurement of anything. Read only their scatter.</p>
{img(fig_z(result, ediag), 'data z per element')}
<h2>Analytic vs ensemble covariance</h2>
<p>The mocks share the data's analytic covariance (pair-count shape noise, the data run's own files). The map below averages ensemble variance / analytic variance over each pair's elements, inside the same cuts as the tests (ξ± 12–83′, Cℓ 300–1600, all COSEBIs).</p>
{img(fig_variance(result, ediag, lo, hi), 'variance ratios')}
<details><summary>Per-element variance ratios and eigenmode variances</summary>
{''.join(img(figs / f'covariance_{k}.png', k) for k in KEYS)}{img(figs / 'covariance_xi_scale_split.png', 'xi scale split')}</details>
<h2>Bands</h2>
<p>Pointwise 68% and 95% mock bands with the data, per pair.</p>
{bands}
<h2>Provenance</h2>
<p class="mute">Row definitions: sampled mocks have their own positions and unit weights, so their measured mean θ differs from the data's by at most {diffs['max_theta_rel']:.1e} (relative), their pseudo-Cℓ bandpower windows by at most {diffs['max_window_rel']:.1e} of the peak weight, and their pure-E/B covariance blocks (built on each catalogue's θ grid) by at most {diffs['max_cov_rel_sigma']:.1e} σ. All tests use the data's covariance; the data's ρ/τ rows are dropped.</p>
<p class="mute">Data: <code>{html.escape(data)}</code> (run/tomo1101 @ 7081c2d0, blind <code>unions</code>, never unblinded). Mocks: <code>{ROOT}/cosmo_val/MOCK_*_tomo.sacc</code>, generated by <code>scripts/glass_mock/make_glass_sim.py</code> (sp_validation develop @ 08b37ee0) and measured by <code>run/mock-ensemble-sampled</code> (cosmo_val estimators of c8f7b522, equal to the data run's). Numbers: <code>{out}/placement.json</code>.</p>
</body></html>"""
    html_path = Path(args.html) if args.html else out / "report.html"
    html_path.write_text(page)
    print(html_path)


if __name__ == "__main__":
    main()
