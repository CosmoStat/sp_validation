"""Resident synthetic parent benchmark of the production refit-W entry point.

Run on an allocated node in the science container, with TMPDIR on /n17data.
Generation is timed separately; all five branches have distinct shapes,
coordinates, selections and cell migration. No catalogue is saved to disk.
"""

import argparse
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sp_validation import response
from sp_validation.des_jackknife import accumulate


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def parent(rows, groups, chunk_size):
    rng = np.random.default_rng(425)
    gm = SimpleNamespace(
        _step=0.01,
        _des_min_count=3,
        branches={},
        mask_dict={},
        W=(np.geomspace(10, 500, 21), np.geomspace(0.5, 3, 21), None, None),
    )
    group = np.empty(rows, dtype=np.int32)
    for start in range(0, rows, chunk_size):
        end = min(rows, start + chunk_size)
        group[start:end] = np.arange(start, end, dtype=np.int64) * groups // rows
    for j, b in enumerate(response.BRANCHES):
        d = {key: np.empty(rows) for key in ("g1", "g2", "flux", "T")}
        d["flux_err"] = np.broadcast_to(1.0, rows)
        d["Tpsf"] = np.broadcast_to(1.0, rows)
        selected = []
        for start in range(0, rows, chunk_size):
            end = min(rows, start + chunk_size)
            n = end - start
            if b == "ns":
                d["g1"][start:end] = rng.normal(0.02, 0.25, n)
                d["g2"][start:end] = rng.normal(-0.01, 0.25, n)
                d["flux"][start:end] = np.exp(rng.uniform(np.log(8), np.log(600), n))
                d["T"][start:end] = np.exp(rng.uniform(np.log(0.4), np.log(3.2), n))
            else:
                sign = 1 if b.startswith("p") else -1
                for i in (1, 2):
                    shift = sign * 0.008 if b.endswith(str(i)) else 0.0002
                    d[f"g{i}"][start:end] = gm.ns[f"g{i}"][start:end] + shift
                for key in ("flux", "T"):
                    d[key][start:end] = gm.ns[key][start:end] * (
                        1 + rng.normal(0, 0.01, n)
                    )
            snr, size = d["flux"][start:end], d["T"][start:end]
            selected.append(
                start
                + np.flatnonzero((snr > 10) & (snr < 500) & (size > 0.5) & (size < 3))
            )
        gm.mask_dict[b] = np.concatenate(selected)
        setattr(gm, b, d)
        gm.branches[b] = (d["g1"], d["g2"])
        print("generated", b, len(gm.mask_dict[b]), "RSS GiB", rss_gib(), flush=True)
    gm.R_shear = np.empty((2, 2, len(gm.mask_dict["ns"])))
    for start in range(0, gm.R_shear.shape[2], chunk_size):
        end = min(gm.R_shear.shape[2], start + chunk_size)
        gm.R_shear[0, 0, start:end] = rng.normal(0.8, 0.05, end - start)
        gm.R_shear[1, 1, start:end] = rng.normal(0.9, 0.05, end - start)
        gm.R_shear[0, 1, start:end] = 0.02
        gm.R_shear[1, 0, start:end] = -0.01
    return gm, group


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=350_000_000)
    p.add_argument("--groups", type=int, default=10_000)
    p.add_argument("--chunk-size", type=int, default=500_000)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    start = time.perf_counter()
    gm, group = parent(args.rows, args.groups, args.chunk_size)
    generated = time.perf_counter()
    parent_rss = rss_gib()
    acc = accumulate(gm, group, args.chunk_size)
    accumulated = time.perf_counter()
    _, estimates, cov = acc.jackknife()
    end = time.perf_counter()
    assert all(np.isfinite(x).all() for x in estimates.values())
    assert np.isfinite(cov).all()
    result = dict(
        rows=args.rows,
        groups=args.groups,
        cells=400,
        chunk_size=args.chunk_size,
        selected={b: len(idx) for b, idx in gm.mask_dict.items()},
        generation_seconds=generated - start,
        accumulation_seconds=accumulated - generated,
        deletion_seconds=end - accumulated,
        jackknife_seconds=end - generated,
        process_seconds=end - start,
        parent_peak_rss_gib=parent_rss,
        peak_rss_gib=rss_gib(),
        statistic_storage_gib=sum(
            x.nbytes for x in (*acc.sums.values(), acc.rg, acc.training)
        )
        / 1024**3,
    )
    print(json.dumps(result, indent=2), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
