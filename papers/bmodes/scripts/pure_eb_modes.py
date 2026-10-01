"""Pure E/B modes and their exact covariance for one version.

Reads the fine-grid ξ± (TreeCorr text dump, with its pair counts) and the
Gaussian ξ± covariance on the same grid, and applies
``sp_validation.b_modes.calculate_pure_eb_correlation``: the fixed-operator
pure-E/B estimator averaged into the reporting bins, with covariance
``K C_ξ Kᵀ``. Writes ``<version>_pure_eb.npz``, the input of every downstream
pure-mode plot / PTE.

    python pure_eb_modes.py \
        --xi-integration <xi 1000-bin .txt> \
        --cov-integration <Gaussian ξ± covariance, same grid> \
        --min-sep 1.0 --max-sep 250.0 --nbins 20 --nbins-int 1000 \
        --out <version>_pure_eb.npz
"""

import argparse
import os

import numpy as np

from sp_validation.b_modes import calculate_pure_eb_correlation, log_bin_edges
from sp_validation.sacc_io import PURE_KEYS


def _load_xi(path, nbins):
    """``(meanr, xip, xim, npairs)`` from a TreeCorr text dump.

    TreeCorr's ASCII header is
    r_nom meanr meanlogr xip xim xip_im xim_im sigma_xip sigma_xim weight npairs.
    """
    data = np.loadtxt(path, comments="#", max_rows=nbins)
    return data[:, 1], data[:, 3], data[:, 4], data[:, 10]


def pure_eb_modes(
    xi_integration, cov_integration, min_sep, max_sep, nbins, nbins_int, out_path
):
    results = calculate_pure_eb_correlation(
        *_load_xi(xi_integration, nbins_int),
        np.loadtxt(cov_integration),
        *log_bin_edges(min_sep, max_sep, nbins),
    )
    package = {
        "theta": results["theta"],
        "theta_int": results["theta_int"],
        "xip_total": results["xip"],
        "xim_total": results["xim"],
        **{key: results[key] for key in PURE_KEYS},
        "cov_pure_eb": results["cov"],
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **package)
    print(f"Saved pure E/B to {out_path}")
    return out_path


def _from_snakemake(smk):
    p = smk.params
    pure_eb_modes(
        smk.input["xi_integration"],
        smk.input["cov_integration"],
        p["min_sep"],
        p["max_sep"],
        p["nbins"],
        p["nbins_int"],
        smk.output[0],
    )


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xi-integration", required=True)
    ap.add_argument("--cov-integration", required=True)
    ap.add_argument("--min-sep", type=float, default=1.0)
    ap.add_argument("--max-sep", type=float, default=250.0)
    ap.add_argument("--nbins", type=int, default=20)
    ap.add_argument("--nbins-int", type=int, default=1000)
    ap.add_argument("--out", required=True, help="Output .npz path")
    a = ap.parse_args(argv)
    pure_eb_modes(
        a.xi_integration,
        a.cov_integration,
        a.min_sep,
        a.max_sep,
        a.nbins,
        a.nbins_int,
        a.out,
    )


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821 — injected by Snakemake's script: directive
    except NameError:
        _from_cli()
    else:
        _from_snakemake(snakemake)  # noqa: F821
