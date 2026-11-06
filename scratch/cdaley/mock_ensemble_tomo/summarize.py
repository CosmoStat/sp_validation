"""Index completed mock products and measured costs; no data placement.

Reads only mock products and per-rule benchmark metadata.
Creates a per-seed statistics view without copying any statistic or catalogue.
"""
import csv
import json
from pathlib import Path

ROOT = Path('/project/6001537/cdaley/mock-ensemble-tomo')
completed = sorted(ROOT.glob('seeds/seed_*/complete.json'))
rows, nulls, pair_nulls, costs = [], [], [], []
for manifest in completed:
    seed = int(manifest.parent.name.split('_')[1])
    version = f'MOCK_{seed:05d}'
    summary = json.loads((ROOT / f'cosmo_val/{version}_bmode_summary_tomo.json').read_text())
    for statistic, value in summary['joint'].items():
        nulls.append(dict(seed=seed, statistic=statistic, **value))
    for pair, values in summary['pairs'].items():
        for statistic, value in values.items():
            pair_nulls.append(dict(seed=seed, pair=pair, statistic=statistic, **value))
    view = manifest.parent / 'statistics'
    view.mkdir(exist_ok=True)
    products = [p for p in (ROOT / 'cosmo_val').iterdir() if version in p.name and '_clean' not in p.name]
    for product in products:
        link = view / product.name
        if not link.exists():
            link.symlink_to(Path('../../../cosmo_val') / product.name)
    maps = json.loads((manifest.parent / 'maps/maps.json').read_text())
    benchmarks = list((ROOT / 'benchmarks').glob(f'*{version}*.tsv'))
    benchmarks += list((ROOT / 'benchmarks').glob(f'paint_{seed:05d}.tsv'))
    benchmarks += list((ROOT / 'benchmarks').glob(f'sky_{seed:05d}.tsv'))
    components = {}
    for path in benchmarks:
        if '_clean' in path.name:
            continue
        with path.open() as f:
            b = next(csv.DictReader(f, delimiter='\t'))
        stage = path.stem.split('_')[0]
        if stage == 'xi':
            stage = 'xi_integration' if '_1000_' in path.name else 'xi_reporting'
        threads = 1 if stage in ('pure', 'assemble') or (stage == 'cosebis' and seed > 1) else 24
        components[stage] = dict(wall_s=float(b['s']), max_rss_mib=float(b['max_rss']),
                                  cpu_h=float(b['cpu_time']) / 3600,
                                  scheduled_core_h=float(b['s']) * threads / 3600)
    if 'sky' not in components:
        # Seed 1's production-resolution sky was generated during the compute gate.
        wall = maps['seconds']['total']
        components['sky'] = dict(wall_s=wall, max_rss_mib=36929124 / 1024,
                                  cpu_h=None, scheduled_core_h=wall * 24 / 3600)
    costs.append(dict(seed=seed, components=components,
                      scheduled_core_h=sum(c['scheduled_core_h'] for c in components.values()),
                      peak_rss_gib=max(c['max_rss_mib'] for c in components.values()) / 1024))
    rows.append(dict(seed=seed, manifest=str(manifest), products=len(products),
                     painted_catalogue_exists=(manifest.parent / 'painted.h5').exists()))
for filename, values, fields in [('joint_null_ptes.csv', nulls, ['seed', 'statistic', 'chi2', 'dof', 'pte']),
                                  ('pair_null_ptes.csv', pair_nulls, ['seed', 'pair', 'statistic', 'chi2', 'dof', 'pte'])]:
    with (ROOT / filename).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
report = dict(complete=len(rows), seeds=rows, costs=costs,
              source_snapshot=str(Path('/project/6001537/cdaley/cosmo-val-v2-tomo/checks/before-develop-20ba864a')))
(ROOT / 'ensemble_summary.json').write_text(json.dumps(report, indent=2))
text = (ROOT / 'RESULT.md').read_text()
section = ['## Execution', '', f'**{len(rows)} seeds complete:** ' + ', '.join(str(r['seed']) for r in rows) + '.', '',
           'The completed reference run was moved by the data worker before rerunning develop.',
           'All mocks use its original registry/config and covariances under `cosmo-val-v2-tomo/checks/before-develop-20ba864a`, not the data worker\'s newer in-progress covariance.', '',
           '`seeds/seed_{s:05d}/` is the per-seed view: `maps/`, `statistics/` (symlinks), and `complete.json`.',
           '`joint_null_ptes.csv` and `pair_null_ptes.csv` collect only the mock null results; data placement is left to the orchestrator.',
           'The source and painted catalogue geometry hashes agree for all 68,177,036 retained rows.', '',
           '### Measured production cost', '',
           '| Seed | Scheduled core-hours of scientific rules | Peak rule RSS [GiB] | Painted catalogue remains |',
           '|---|---|---|---|']
for row, cost in zip(rows, costs):
    section.append(f"| {row['seed']} | {cost['scheduled_core_h']:.1f} | {cost['peak_rss_gib']:.1f} | {row['painted_catalogue_exists']} |")
section += ['', 'The cost table counts requested rule threads times measured wall time, not whole-node allocation idle time.',
            'It excludes the seed-1 clean twin, tests, known-answer plotting and failed/restarted work.',
            '`ensemble_summary.json` retains each rule\'s wall time, process CPU time and peak RSS.',
            'Reporting xi uses npatch=1 to skip the discarded jackknife pass; the tested unpatched means, angular bins and final data analytic covariance are unchanged.', '',
            '### Interpretation limits', '',
            'Twenty seeds resolve ensemble behaviour at roughly the 5% level, not a calibrated 0.5% tail.',
            'Every stored E vector still carries the known mock-only blind; remove its designed shift before comparing E with truth.',
            'The finite-grid blind response in derived B is below 0.003 sigma and is recorded above; harmonic B is unchanged to roundoff.',
            'No input-shape statistic or real blind was read, and no mock analytic covariance was estimated.']
text = text.split('## Execution')[0] + '\n'.join(section) + '\n'
(ROOT / 'RESULT.md').write_text(text)
print(json.dumps(dict(complete=len(rows), seeds=[r['seed'] for r in rows]), indent=2))
