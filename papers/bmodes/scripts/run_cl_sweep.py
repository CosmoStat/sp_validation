"""Pseudo-Cℓ data-vector + covariance sweep over non-fiducial versions (Design B).

Loops the [non-fiducial version list](sweep_versions.nonfiducial_versions) and
runs the same ``generate_pseudo_cl`` / ``generate_pseudo_cl_cov`` compute the
fiducial cl_bb / covariance recipes call, once per version. Each producer writes
its native ``pseudo_cl_{ver}.fits`` / ``pseudo_cl_cov_{ver}.fits`` into ``--out``;
this driver then renames them to the tagged canonical names
``pseudo_cl_{ver}_blind={blind}_{binning}_nbins={nbins}.fits`` and
``pseudo_cl_cov_{ver}_blind={blind}_{binning}_nbins={nbins}.fits`` — the exact
pattern ``cl_version_comparison._pseudo_cl`` / ``._pseudo_cl_cov`` reconstruct.

Per-version footprint masks resolve internally from cat_config (the version's
``mask:`` entry), so no per-version mask wiring is needed here. Serial over
versions; the estimator uses the recipe's full core allocation per call.

    python run_cl_sweep.py \
        --config .../config.yaml --cat-config .../cat_config.yaml \
        --nside 1024 --npatch 1 --binning powspace --nbins 32 --power 0.5 \
        --blind A --out <dir>
"""

import argparse
import os
import sys

import yaml

# generate_pseudo_cl / generate_pseudo_cl_cov live in workflow/scripts;
# sweep_versions is a sibling here. Put workflow/scripts on the path first.
_HERE = os.path.dirname(os.path.abspath(__file__))
_WSCRIPTS = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "workflow", "scripts")
)
for _p in (_HERE, _WSCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_pseudo_cl import generate_pseudo_cl  # noqa: E402
from generate_pseudo_cl_cov import generate_pseudo_cl_cov  # noqa: E402
from sweep_versions import nonfiducial_versions  # noqa: E402


def _canonical(prefix, ver, blind, binning, nbins):
    return f"{prefix}_{ver}_blind={blind}_{binning}_nbins={nbins}.fits"


def _run_and_tag(producer, prefix, ver, out, blind, binning, nbins, **kw):
    """Run one per-version producer, then rename its native FITS to canonical."""
    producer(
        version=ver, output_dir=out, blind=blind, binning=binning, nbins=nbins, **kw
    )
    native = os.path.join(out, f"{prefix}_{ver}.fits")
    canonical = os.path.join(out, _canonical(prefix, ver, blind, binning, nbins))
    os.replace(native, canonical)
    print(f"[cl_sweep] {ver}: {os.path.basename(canonical)}")


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Pseudo-Cℓ + covariance sweep over the non-fiducial catalog versions."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--cat-config", required=True, help="Absolute path to cat_config.yaml"
    )
    ap.add_argument("--out", required=True, help="Sweep output directory (lc {output})")
    ap.add_argument("--versions", nargs="*", default=None, help="Explicit version keys")
    ap.add_argument("--nside", type=int, default=1024)
    ap.add_argument("--npatch", type=int, default=1)
    ap.add_argument(
        "--binning", choices=["linear", "logspace", "powspace"], default="powspace"
    )
    ap.add_argument("--nbins", type=int, default=32)
    ap.add_argument("--power", type=float, default=0.5)
    ap.add_argument("--blind", choices=["A", "B", "C"], default="A")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    versions = a.versions or nonfiducial_versions(config)
    os.makedirs(a.out, exist_ok=True)
    shared = dict(
        cat_config=a.cat_config,
        nside=a.nside,
        npatch=a.npatch,
        power=a.power,
    )
    for ver in versions:
        _run_and_tag(
            generate_pseudo_cl,
            "pseudo_cl",
            ver,
            a.out,
            a.blind,
            a.binning,
            a.nbins,
            **shared,
        )
        _run_and_tag(
            generate_pseudo_cl_cov,
            "pseudo_cl_cov",
            ver,
            a.out,
            a.blind,
            a.binning,
            a.nbins,
            **shared,
        )


if __name__ == "__main__":
    _from_cli()
