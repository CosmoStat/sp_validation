"""COSEBI B-mode PTE-matrix sweep over non-fiducial versions (Design B, Tier-2).

Loops the [non-fiducial version list](sweep_versions.nonfiducial_versions) and
runs the same gathered-NPZ COSEBI PTE compute the fiducial cosebis_pte_per_cut
recipe calls (``compute_cosebis_pte_single.main``), once per version. The theta
grid, nmodes and per-pair computation are version-independent (read from
``config["fiducial"]``); only the input xi/cov and the output tag change, so each
sweep call is bit-identical to the fiducial call save for those. Per version the
driver reads that version's 1000-bin integration ξ_± from the xi_sweep output dir
and its Gaussian integration covariance from the cov_sweep output dir (both by
absolute path — lc does not wire cross-output deps, so run xi_sweep + cov_sweep
first), emitting the canonical ``cosebis_ptes_{ver}_{blind}.npz`` — the same
gathered layout the fiducial single-output writes, which
config_space_pte_matrices.py adapts via ``_cosebis_matrix_from_npz`` — into
``--out``. Serial over versions (~25 min/version, 206 pairs).

    python run_cosebis_ptes_sweep.py \
        --config .../config.yaml \
        --xi-sweep-dir <two_point/.../xi_sweep> \
        --cov-sweep-dir <covariance/.../cov_sweep> \
        --out <dir> [--blind A] [--versions v1 v2 ...]
"""

import argparse
import os
import sys

import yaml

# compute_cosebis_pte_single, plotting_utils, and sweep_versions are siblings
# here; other compute modules live in workflow/scripts. Put both on the path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_WSCRIPTS = os.path.abspath(
    os.path.join(_HERE, "..", "..", "..", "workflow", "scripts")
)
for _p in (_HERE, _WSCRIPTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from compute_cosebis_pte_single import main as compute_cosebis_pte  # noqa: E402
from sweep_versions import nonfiducial_versions  # noqa: E402


def _xi_integration(xi_sweep_dir, ver):
    return os.path.join(
        xi_sweep_dir, f"{ver}_xi_minsep=0.5_maxsep=300.0_nbins=1000_npatch=1.txt"
    )


def _cov_integration(cov_sweep_dir, ver, blind):
    base = f"covariance_{ver}_{blind}_g_minsep=0.5_maxsep=300.0_nbins=1000_masked"
    return os.path.join(cov_sweep_dir, base, f"{base}_processed.txt")


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(
        description="Gathered COSEBI B-mode PTE-matrix sweep over the non-fiducial catalog versions."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--xi-sweep-dir",
        required=True,
        help="xi_sweep output dir with per-version 1000-bin integration xi_pm .txt",
    )
    ap.add_argument(
        "--cov-sweep-dir",
        required=True,
        help="cov_sweep output dir with per-version {base}/{base}_processed.txt cov",
    )
    ap.add_argument("--out", required=True, help="Sweep output directory (lc {output})")
    ap.add_argument("--blind", default="A", help="Blind tag (paper: A)")
    ap.add_argument("--versions", nargs="*", default=None, help="Explicit version keys")
    a = ap.parse_args(argv)

    with open(a.config) as f:
        config = yaml.safe_load(f)
    versions = a.versions or nonfiducial_versions(config)
    os.makedirs(a.out, exist_ok=True)

    for ver in versions:
        xi_int = _xi_integration(a.xi_sweep_dir, ver)
        cov_int = _cov_integration(a.cov_sweep_dir, ver, a.blind)
        for f in (xi_int, cov_int):
            if not os.path.isfile(f):
                raise FileNotFoundError(f"MISSING upstream input for {ver}: {f}")
        print(f"[cosebis_ptes_sweep] {ver}", flush=True)
        compute_cosebis_pte(config, xi_int, cov_int, a.out, version=ver, blind=a.blind)
        print(
            f"[cosebis_ptes_sweep] {ver} -> "
            f"{os.path.join(a.out, f'cosebis_ptes_{ver}_{a.blind}.npz')}",
            flush=True,
        )
    print(f"[cosebis_ptes_sweep] done -> {a.out}", flush=True)


if __name__ == "__main__":
    _from_cli()
