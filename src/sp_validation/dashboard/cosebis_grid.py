"""COSEBIs B-mode PTEs over every reporting-bin scale cut (Paper II Fig. 8).

For each contiguous range of reporting bins ``[i, j]`` on the 20-bin
1′–250′ grid, the COSEBIs are recomputed on the integration-grid ξ± between
those reporting edges, and the B-mode χ² against zero is taken over the first
6 and over all 20 modes with the analytic covariance carried through the same
kernel. Where the 20-mode filters fail to converge (narrow cuts), the n ≤ 6
PTE comes from a 6-mode computation and the n ≤ 20 PTE is NaN.

The product is a cache read by :func:`sp_validation.dashboard.scale_cuts.config_space`;
computing it needs only the run's integration-grid ξ± part and its OneCovariance
covariance, never the catalogue. Usage::

    python -m sp_validation.dashboard.cosebis_grid --run-dir RUN --version V \\
        --out cosebis_grid_V.npz [--nproc N]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SUBSETS = (6, 20)


def _paths(run_dir, version):
    run_dir = Path(run_dir)
    (xi,) = (run_dir / "cosmo_val").glob(f"{version}_xi_minsep=*_npatch=1.sacc")
    (cov,) = (run_dir / "cosmo_inference/data/covariance").glob(
        f"covariance_{version}_g_minsep=*_masked/covariance_{version}_g_minsep=*_masked.dat"
    )
    return xi, cov


def _grid(run_dir):
    """Reporting edges and integration binning from the run's configuration."""
    import yaml

    cfg = yaml.safe_load(
        json.loads((Path(run_dir) / "provenance.json").read_text())["config"]
    )
    cv = cfg["cosmo_val"]
    edges = np.geomspace(cv["theta_min"], cv["theta_max"], cv["nbins"] + 1)
    integ = cv["integration"]
    return (
        edges,
        (integ["min_sep"], integ["max_sep"], integ["nbins"]),
        cv["cosebis"]["nmodes"],
    )


_STATE = {}


def _init(xi_path, cov_path, blind, integ, nmodes):
    from sp_validation import sacc_io
    from sp_validation.b_modes import log_bin_edges

    part = sacc_io.load(xi_path, blind=blind)
    theta, xip, xim = sacc_io.get_xi(part, (0, 0), grid="integration")
    _STATE.update(
        theta=theta,
        xip=xip,
        xim=xim,
        cov=np.loadtxt(cov_path),
        edges=log_bin_edges(*integ),
        nmodes=nmodes,
    )


def _modes(cut, nmodes):
    """B-mode COSEBIs and their covariance for one cut, or None if the filter
    roots fail to converge."""
    from sp_validation.b_modes import cosebis_scan_from_xi

    s = _STATE
    try:
        (r,) = cosebis_scan_from_xi(
            s["theta"],
            s["xip"],
            s["xim"],
            s["cov"],
            *s["edges"],
            nmodes=nmodes,
            scale_cuts=[cut],
        ).values()
    except Exception as e:  # sympy NoConvergence in the filter roots
        print(f"cut {cut}, {nmodes} modes: {type(e).__name__}: {e}", flush=True)
        return None
    return r["Bn"], r["cov"][nmodes:, nmodes:]


def _pair(cut):
    """B PTEs over the first k modes for each k in SUBSETS.

    The log-COSEBIs filter of mode n does not depend on how many modes are
    requested, so a cut whose full mode set fails to converge is retried
    with only the modes a smaller subset needs.
    """
    from scipy import stats

    full = _modes(cut, _STATE["nmodes"])
    out = []
    for k in SUBSETS:
        r = full if full is not None or k >= _STATE["nmodes"] else _modes(cut, k)
        if r is None:
            out.append(np.nan)
            continue
        b, cov_B = r[0][:k], r[1][:k, :k]
        out.append(float(stats.chi2.sf(b @ np.linalg.solve(cov_B, b), k)))
    return out


def compute(run_dir, version, nproc=1):
    from sp_validation.dashboard.products import blind_of

    xi_path, cov_path = _paths(run_dir, version)
    blind = blind_of(xi_path)
    edges, integ, nmodes = _grid(run_dir)
    n = len(edges) - 1
    pairs = [(i, j) for i in range(n) for j in range(i, n)]
    cuts = [(float(edges[i]), float(edges[j + 1])) for i, j in pairs]
    init = (str(xi_path), str(cov_path), blind, integ, nmodes)
    if nproc > 1:
        with ProcessPoolExecutor(nproc, initializer=_init, initargs=init) as pool:
            results = list(pool.map(_pair, cuts))
    else:
        _init(*init)
        results = [_pair(c) for c in cuts]
    pte = {k: np.full((n, n), np.nan) for k in SUBSETS}
    for (i, j), vals in zip(pairs, results):
        for k, v in zip(SUBSETS, vals):
            pte[k][i, j] = v
    return {
        "left_edges": edges[:-1],
        "right_edges": edges[1:],
        **{f"pte_B_n{k}": pte[k] for k in SUBSETS},
        "blind": np.array(blind),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    )
    a = ap.parse_args(argv)
    # One numba thread per worker; the parallelism is over cuts.
    os.environ.setdefault(
        "NUMBA_NUM_THREADS", "1" if a.nproc > 1 else str(os.cpu_count())
    )
    data = compute(a.run_dir, a.version, a.nproc)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, **data)
    tmp.replace(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
