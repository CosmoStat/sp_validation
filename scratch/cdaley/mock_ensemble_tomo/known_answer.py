"""Seed-1 known answer from its own alms, plus an independent clean blind twin.

Never reads the input catalogue or any data blind; the mock registry is explicit.
Run on a compute node in the analysis container.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import jv
from scipy.stats import chi2
import yaml

from sp_validation import sacc_io, blinding

ROOT = Path('/project/6001537/cdaley/mock-ensemble-tomo')
CV = ROOT / 'cosmo_val'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
REG = yaml.safe_load((ROOT / 'cat_config.yaml').read_text())
assert Path(REG['paths']['blinds']).resolve() == (ROOT / 'blinds').resolve()
V, U = 'MOCK_00001', 'MOCK_00001_clean'
BLIND = blinding.open_blind('mock-ensemble-tomo', REG)


def load(name, version=U):
    return sacc_io.load(CV / name.format(v=version), blind=REG[version]['blind'])


def test(d, c):
    value = float(d @ np.linalg.solve(c, d))
    return dict(chi2=value, dof=len(d), pte=float(chi2.sf(value, len(d))))


clean = load('{v}_tomo.sacc')
sealed = load('{v}_tomo.sacc', V)
C = np.asarray(clean.covariance.dense)
fine = load('{v}_xi_minsep=0.08_maxsep=300.0_nbins=1000_npatch=1_tomo.sacc')
spectra = np.load(ROOT / 'seeds/seed_00001/maps/spectra.npz')
ell = spectra['ell'][2:]
assert np.array_equal(ell, np.arange(2, 8193))
int_edges = np.geomspace(.08, 300, 1001)
rep_edges = np.geomspace(1, 250, 21)
lo = np.maximum(rep_edges[:-1, None], int_edges[None, :-1])
hi = np.minimum(rep_edges[1:, None], int_edges[None, 1:])
frac = np.maximum(hi**2 - lo**2, 0) / np.diff(int_edges**2)

pairs = [(i, j) for i in range(4) for j in range(i, 4)]
metrics = {}
fig_cl, axes_cl = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
fig_xi, axes_xi = plt.subplots(3, 4, figsize=(12, 8), constrained_layout=True)
all_rows, all_residuals = {'cl': [], 'xi': []}, {'cl': [], 'xi': []}
for k, pair in enumerate(pairs):
    tracers = tuple(sacc_io.source_name(i) for i in pair)
    label = f'{pair[0] + 1} x {pair[1] + 1}'
    own = spectra['realization_ee'][pair][2:]
    ee_rows = np.asarray(clean.indices(sacc_io.CL_EE, tracers), int)
    window = clean.get_bandpower_windows(ee_rows)
    pred = window.weight.T @ np.interp(window.values, spectra['ell'], spectra['realization_ee'][pair], left=0, right=0)
    leff = np.array([clean.data[i].tags['ell'] for i in ee_rows])
    selected = (leff >= 300) & (leff <= 1600)
    rows = ee_rows[selected]
    residual = clean.mean[rows] - pred[selected]
    row = {'EE_300_1600': test(residual, C[np.ix_(rows, rows)])}
    all_rows['cl'].extend(rows)
    all_residuals['cl'].extend(residual)
    sig = np.sqrt(np.diag(C)[ee_rows])
    ax = axes_cl.flat[k]
    ax.axhline(0, color='.5', lw=.8)
    ax.errorbar(leff, (clean.mean[ee_rows] - pred) / sig, yerr=1, fmt='.', color='C0', ms=4)
    ax.axvspan(300, 1600, color='C0', alpha=.08)
    ax.set(title=label, xlabel='ell', ylabel='(EE - own sky) / sigma', ylim=(-5, 5))
    ax = axes_xi.flat[k]
    ax.axhline(0, color='.5', lw=.8)
    joint_rows, joint_res = [], []
    for typ, order, color, legend in ((sacc_io.XI_PLUS, 0, 'C0', 'xi+'), (sacc_io.XI_MINUS, 4, 'C1', 'xi-')):
        view = sacc_io.xi_correlation(fine, pair, grid='integration')
        weights = frac * view.weight[None, :]
        weights /= weights.sum(axis=1)[:, None]
        # Finite direct Bessel sum, no FFTLog or extrapolation of jagged spectra.
        arg = np.outer(view.meanr * np.pi / (180 * 60), ell + .5)
        prediction = weights @ (jv(order, arg) @ ((ell + .5) * own / (2 * np.pi)))
        idx = np.asarray(clean.indices(typ, tracers), int)
        theta = np.array([clean.data[i].tags['theta'] for i in idx])
        residual = clean.mean[idx] - prediction
        sig = np.sqrt(np.diag(C)[idx])
        ax.errorbar(theta, residual / sig, yerr=1, fmt='.', color=color, ms=4, label=legend)
        selected = (theta >= 12) & (theta <= 83)
        joint_rows.extend(idx[selected])
        joint_res.extend(residual[selected])
        row[legend + '_full'] = test(residual, C[np.ix_(idx, idx)])
    row['xi_pm_12_83'] = test(np.array(joint_res), C[np.ix_(joint_rows, joint_rows)])
    all_rows['xi'].extend(joint_rows)
    all_residuals['xi'].extend(joint_res)
    ax.axvspan(12, 83, color='.5', alpha=.1)
    ax.set(title=label, xscale='log', xlabel='theta [arcmin]', ylabel='(xi - own sky) / sigma', ylim=(-5, 5))
    if k == 0:
        ax.legend(fontsize=8)
    for key, typ in [('BB', sacc_io.CL_BB), ('EB', sacc_io.CL_EB)]:
        idx = np.asarray(clean.indices(typ, tracers), int)
        row[key] = test(clean.mean[idx], C[np.ix_(idx, idx)])
    metrics[label] = row
for axes in (axes_cl, axes_xi):
    for ax in axes.flat[10:]:
        ax.set_visible(False)
fig_cl.suptitle('Seed 1: catalogue pseudo-Cl against its own full-sky alms; analytic data covariance')
fig_xi.suptitle('Seed 1: pair-weighted direct J0/J4 sum of own sky; analytic data covariance')
fig_cl.savefig(FIG / 'seed1_ee_known_answer.png', dpi=160)
fig_xi.savefig(FIG / 'seed1_xi_known_answer.png', dpi=160)
plt.close('all')

# Blind twins were measured independently from byte-identical catalogues.
blind_metrics = {}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
for k, (stat, stem) in enumerate([
    ('xi_reporting', '{v}_xi_minsep=1.0_maxsep=250.0_nbins=20_npatch=1_tomo.sacc'),
    ('xi_integration', '{v}_xi_minsep=0.08_maxsep=300.0_nbins=1000_npatch=1_tomo.sacc'),
    ('pseudo_cl', 'pseudo_cl_{v}_powspace_nbins=32_tomo.sacc'),
]):
    raw, blinded = load(stem), load(stem, V)
    expected = blinding.conceal(raw, BLIND).mean - raw.mean
    observed = blinded.mean - raw.mean
    err = observed - expected
    tolerance = 1e-20 if stat == 'pseudo_cl' else 1e-15
    np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=tolerance)
    blind_metrics[stat] = dict(max_abs_shift_error=float(np.max(np.abs(err))),
                               max_abs_shift=float(np.max(np.abs(expected))))
    axes[k].plot(expected, observed, '.', ms=2)
    mn, mx = expected.min(), expected.max()
    axes[k].plot([mn, mx], [mn, mx], color='k', lw=.8)
    axes[k].set(title=stat, xlabel='Designed shift', ylabel='Blinded - clean')
fig.savefig(FIG / 'seed1_blind_known_answer.png', dpi=160)
plt.close(fig)
for dtype in (sacc_io.CL_BB, sacc_io.CL_EB, sacc_io.COSEBI_BB,
              sacc_io.PURE_TYPES['xip_B'], sacc_io.PURE_TYPES['xim_B']):
    idx = clean.indices(dtype)
    error = float(np.max(np.abs(sealed.mean[idx] - clean.mean[idx])))
    blind_metrics[dtype] = dict(max_abs_B_change=error)
    assert error < 1e-15, (dtype, error)

joint = {key: test(np.array(all_residuals[key]), C[np.ix_(rows, rows)]) for key, rows in all_rows.items()}
# A known realisation needn't have high null PTEs. Reject catastrophic E recovery,
# not the B-null behaviour that this ensemble was commissioned to measure.
passed = all(joint[k]['pte'] > 1e-4 for k in joint)
result = dict(per_pair=metrics, joint_known_answer=joint, blind=blind_metrics, passed=passed,
              nulls=json.loads((CV / f'{V}_bmode_summary_tomo.json').read_text()),
              theory='Pair-weighted direct finite J0/J4 sum, ell+0.5, ell=2..8192; no extra pixel window; pseudo-Cl uses stored bandpower windows.')
(ROOT / 'seed1_known_answer.json').write_text(json.dumps(result, indent=2))
lines = ['# Tomographic painted-mock ensemble', '',
         f'Seed 1 known-answer gate: **{"PASS" if passed else "FAIL"}**.', '',
         'All catalogues use exact off-grid alm evaluation, gamma (not reduced shear), and one mock-only blind.',
         'The analytic covariance is the data run\'s unchanged covariance; no mock covariance is computed.', '',
         '## Seed-1 known answer', '',
         '| Pair | EE chi2/dof (300–1600), PTE | xi± chi2/dof (12–83 arcmin), PTE | BB PTE | EB PTE |',
         '|---|---|---|---|---|']
for pair, row in metrics.items():
    fmt = lambda r: f"{r['chi2']:.2f}/{r['dof']}, {r['pte']:.4g}"
    lines.append(f"| {pair} | {fmt(row['EE_300_1600'])} | {fmt(row['xi_pm_12_83'])} | {row['BB']['pte']:.4g} | {row['EB']['pte']:.4g} |")
lines += ['', f'Joint own-sky EE: {joint["cl"]}; joint own-sky xi±: {joint["xi"]}.', '',
          'Known-answer comparisons retain footprint cosmic variance; they are not exact catalogue-to-full-sky identities.',
          'Each pair\'s prediction uses its own realised cross-spectrum.',
          'The reporting-bin xi prediction averages the direct Bessel sum with the integration grid\'s measured pair weights.', '',
          '## Mock blind', '', 'The independently measured clean twin uses identical catalogue bytes.',
          'Blinded minus clean matches the blinding module\'s designed shift for both xi grids and EE.',
          'BB, EB, COSEBIs B and pure-B are unchanged within the tolerances recorded in `seed1_known_answer.json`.', '',
          '## Figures', '', '- `figures/seed1_ee_known_answer.png`', '- `figures/seed1_xi_known_answer.png`', '- `figures/seed1_blind_known_answer.png`', '',
          '## Products', '', '`cosmo_val/MOCK_{seed:05d}*` holds the SACC parts, final tomographic SACC, per-pair NPZs and null summary.',
          '`seeds/seed_{seed:05d}/maps/` holds convergence alms and all auto/cross spectra.',
          'Painted catalogues are temporary Snakemake outputs and are deleted after their statistics and geometry audit complete.',
          '`geometry_audit.json` hashes every retained position, weight and source-bin value against the input catalogue, without reading its shapes.', '',
          '## Execution', '', 'Completion counts and measured cost will be added after the production batch.']
(ROOT / 'RESULT.md').write_text('\n'.join(lines) + '\n')
assert passed, joint
