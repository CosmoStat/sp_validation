"""TreeCorr ξ±(θ) sweep over the non-fiducial catalog versions (Design B).

Loops the [non-fiducial version list](sweep_versions.nonfiducial_versions) and
runs the same ``run_2pcf.run_2pcf`` compute the fiducial two_point recipes call,
once per version, writing every version's ξ± text dump into one lc ``{output}``
dir under run_2pcf's native, already-canonical name
``{ver}_xi_minsep={min}_maxsep={max}_nbins={nbins}_npatch={npatch}.txt`` — the
exact pattern ``cosebis_version_comparison._xi_integration`` reconstructs.

Both grids are produced per version: the 1000-bin integration grid feeds
``cosebis_version_comparison`` (``--results-dir``) and the pure E/B sweep's
integral kernels; the 20-bin reporting grid feeds the pure E/B sweep's output
binning. Serial over versions — lc's dask handles cross-output concurrency, and
TreeCorr already uses the recipe's full OpenMP allocation per call.

    apptainer exec ... /usr/local/bin/python run_xi_sweep.py \
        --config .../config.yaml --cat-config .../cat_config.yaml --out <dir>
"""

import argparse
import os
import sys

import yaml

# run_2pcf lives in workflow/scripts; sweep_versions is a sibling here.
# Put workflow/scripts on the path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
_WSCRIPTS = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "workflow", "scripts")
)
for _p in (_HERE, _WSCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_2pcf import run_2pcf  # noqa: E402
from sweep_versions import nonfiducial_versions  # noqa: E402

GRIDS = {
    "reporting": dict(min_sep=1.0, max_sep=250.0, nbins=20, npatch=1),
    "integration": dict(min_sep=0.5, max_sep=300.0, nbins=1000, npatch=1),
}


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="TreeCorr ξ± sweep over the non-fiducial catalog versions."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--cat-config", required=True, help="Absolute path to cat_config.yaml"
    )
    ap.add_argument("--out", required=True, help="Sweep output directory (lc {output})")
    ap.add_argument(
        "--versions",
        nargs="*",
        default=None,
        help="Explicit version keys (default: non-fiducial non-ecut from config)",
    )
    ap.add_argument(
        "--grids",
        nargs="*",
        choices=list(GRIDS),
        default=list(GRIDS),
        help="Which angular grids to measure per version (default: both)",
    )
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    versions = a.versions or nonfiducial_versions(config)
    for ver in versions:
        for grid in a.grids:
            run_2pcf(
                ver=ver,
                cat_config=a.cat_config,
                output_dir=a.out,
                grid=grid,
                **GRIDS[grid],
            )


if __name__ == "__main__":
    _from_cli()
