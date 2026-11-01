"""Hash all retained positions, weights and bin IDs, without reading shapes."""
import hashlib
import json

import numpy as np
from sp_validation.io import Catalogue
from sp_validation.grammar import materialise

columns = ('RA', 'Dec', 'w_des', 'TOM_BIN_ID')


def geometry(path, select):
    hashes = {name: hashlib.sha256() for name in columns}
    count = 0
    bins = np.zeros(4, dtype=np.int64)
    with Catalogue(path) as cat:
        for part in cat.chunks.values():
            for start in range(0, len(part), 250000):
                rows = materialise(part[start:start + 250000], list(columns))
                if select:
                    rows = rows[np.isin(rows['TOM_BIN_ID'], [1, 2, 3, 4])]
                count += len(rows)
                for b in range(1, 5):
                    bins[b - 1] += np.count_nonzero(rows['TOM_BIN_ID'] == b)
                for name in columns:
                    dtype = '<i8' if name == 'TOM_BIN_ID' else '<f8'
                    hashes[name].update(np.asarray(rows[name], dtype=dtype).tobytes())
    return dict(rows=count, bins=bins.tolist(), sha256={k: h.hexdigest() for k, h in hashes.items()})

source = geometry(snakemake.input.source, True)
mock = geometry(snakemake.input.mock, False)
assert source == mock, (source, mock)
with open(snakemake.output[0], 'w') as f:
    json.dump(dict(source=source, mock=mock, identical=True), f, indent=2)
