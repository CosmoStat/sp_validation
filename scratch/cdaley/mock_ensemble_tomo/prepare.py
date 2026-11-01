"""Make a mock-only registry and mirror the bound-9 tomographic data config.

Run in the container on an allocation; reads catalogue metadata, never shapes.
"""
import argparse
import json
import os
from copy import deepcopy
from pathlib import Path

import yaml
from sp_validation.blinding import init

p = argparse.ArgumentParser()
p.add_argument("--seeds", type=int, default=100)
a = p.parse_args()
ROOT = Path('/project/6001537/cdaley/mock-ensemble-tomo')
DATA = Path('/project/6001537/cdaley/cosmo-val-v2-tomo')
REPO = Path('/project/6001537/cdaley/sp/sp_validation-mockens')
source = yaml.safe_load((DATA / 'cat_config.yaml').read_text())['SP_v2.0']
config = yaml.safe_load((DATA / 'nibi_tomo.yaml').read_text())
nz_path = Path(source['shear']['redshift_path'])
entries = {'SP_v2.0': source,
           'nz': {'subdir': str(nz_path.parent), 'dndz': {'path': str(nz_path)}},
           'paths': {'blinds': str(ROOT / 'blinds'), 'output': str(ROOT / 'cosmo_val')}}
versions = [f'MOCK_{s:05d}' for s in range(1, a.seeds + 1)]
versions += ['MOCK_00001_clean']
for version in versions:
    seed = int(version.split('_')[1])
    entry = deepcopy(source)
    entry['subdir'] = str(ROOT / 'seeds' / f'seed_{seed:05d}')
    entry['shear']['path'] = 'painted_clean.h5' if version.endswith('_clean') else 'painted.h5'
    entry['blind'] = 'none' if version.endswith('_clean') else 'mock-ensemble-tomo'
    # The covariance, masks, n(z), bin/weight columns and response are unchanged.
    entries[version] = entry
config['catalog_config'] = str(ROOT / 'cat_config.yaml')
config['container'] = '/project/6001537/cdaley/containers/sp_validation-ddd58fe0.sif'
config['versions'] = versions
config['cosmo_val']['include_rho_tau'] = False
# process_gg always publishes the unpatched means. Avoid a second, discarded
# jackknife pass: final products receive the data's analytic covariance.
config['cosmo_val']['npatch'] = 1
config['fiducial']['npatch'] = 1
config['fiducial']['version'] = config['fiducial']['mock_version'] = versions[0]
config['covariance']['tomographic_reference'] = {v: 'SP_v2.0' for v in versions}
config['covariance']['reference_roots'] = {'SP_v2.0': {'cosmo_val': str(DATA / 'cosmo_val'), 'cosmo_inference': str(DATA / 'cosmo_inference')}}
# This JSON feeds the pseudo-Cl rule, as in the data run (not a covariance here).
planck = REPO / 'papers/cosmo_val/results/cosmology/planck18.json'
(ROOT / 'results/cosmology').mkdir(parents=True, exist_ok=True)
(ROOT / 'results/cosmology/planck18.json').write_bytes(planck.read_bytes())
(ROOT / 'cat_config.yaml').write_text(yaml.safe_dump(entries, sort_keys=False))
(ROOT / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
if not (ROOT / 'blinds/mock-ensemble-tomo.blind.json').exists():
    init('mock-ensemble-tomo', entries)
for name in ('tmp', 'cache', 'logs', 'cosmo_val', 'seeds'):
    (ROOT / name).mkdir(exist_ok=True)
# Record only config / software provenance, not the data's blind record.
import subprocess, hashlib
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
(ROOT / 'provenance.json').write_text(json.dumps(dict(
    git_sha=subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
    source_config_sha256=sha(DATA/'cat_config.yaml'), source_run_sha256=sha(DATA/'nibi_tomo.yaml'),
    nz_sha256=sha(source['shear']['redshift_path']),
    input_catalogue=str(Path(source['subdir']) / source['shear']['path']),
    sky=dict(nside=4096, lmax=8192, dx=200, zmax=3.5, limber=True),
    rotation_seed='1000000 + sky_seed', mock_blind='mock-ensemble-tomo',
), indent=2))
