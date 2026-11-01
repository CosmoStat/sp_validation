"""Record a seed's completed statistic products before deleting its catalogue."""
import datetime
import json
from pathlib import Path

root = Path('/project/6001537/cdaley/mock-ensemble-tomo')
seed = int(snakemake.wildcards.seed)
version = f'MOCK_{seed:05d}'
products = sorted(str(p.relative_to(root)) for p in (root / 'cosmo_val').iterdir()
                  if version in p.name and '_clean' not in p.name)
with open(snakemake.output[0], 'w') as f:
    json.dump(dict(seed=seed, time_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   products=products, summary=str(snakemake.input.summary),
                   spectra=str(snakemake.input.spectra)), f, indent=2)
