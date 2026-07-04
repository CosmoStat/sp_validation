# %%
"""Compute the COSEBIS B-mode PTE matrix over the full scale-cut pair grid.

Reconciliation of the ASTRA ``cosebis_pte_per_cut`` output: the Snakemake DAG
*scattered* one JSON per (version, blind, i_min, i_max) cut via this script and
gathered them downstream. The spec describes the gathered NPZ that scatter never
materialised. Here a single CLI loops the same ``_pte_scale_cut_pairs()`` grid,
runs the *identical* per-pair COSEBI computation (same theta grid, nmodes, NaN-on-
convergence-failure handling), and writes one gathered NPZ — no scatter, no
per-pair deps to wire.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import treecorr
from plotting_utils import compute_chi2_pte

from sp_validation.b_modes import calculate_cosebis


def _pte_scale_cut_pairs():
    """(i_min, i_max) index pairs for the PTE matrix, excluding the polynomial-
    root-unstable subsets. Mirrors _pte_scale_cut_pairs() in claims.smk."""
    unstable = {(9, 10), (10, 11), (11, 12), (13, 14)}
    return [
        (i, j) for i in range(20) for j in range(i + 1, 21) if (i, j) not in unstable
    ]


def _compute_pair(gg, cov_path, nmodes, theta_min, theta_max):
    """COSEBI PTEs for one (theta_min, theta_max) cut.

    Bit-identical to the original per-pair scatter job. Returns a result dict,
    or None on a polynomial-root (sympy NoConvergence) failure.
    """
    try:
        results = calculate_cosebis(
            gg,
            nmodes=nmodes,
            scale_cuts=[(theta_min, theta_max)],
            cov_path=cov_path,
        )
    except Exception as e:
        if "NoConvergence" in str(type(e).__name__) or "NoConvergence" in str(e):
            return None
        raise

    result = results[(theta_min, theta_max)]
    En_full = result["En"]
    Bn_full = result["Bn"]
    cov_full = result["cov"]

    # 6 modes: first 6
    cov_B_6 = cov_full[nmodes : nmodes + 6, nmodes : nmodes + 6]
    cov_E_6 = cov_full[:6, :6]
    chi2_B_6, pte_B_6, _ = compute_chi2_pte(Bn_full[:6], cov_B_6)
    chi2_E_6, pte_E_6, _ = compute_chi2_pte(En_full[:6], cov_E_6)

    # 20 modes: all
    cov_B_20 = cov_full[nmodes:, nmodes:]
    cov_E_20 = cov_full[:nmodes, :nmodes]
    chi2_B_20, pte_B_20, _ = compute_chi2_pte(Bn_full, cov_B_20)
    chi2_E_20, pte_E_20, _ = compute_chi2_pte(En_full, cov_E_20)

    return {
        "En": En_full,
        "Bn": Bn_full,
        "chi2_B_6": chi2_B_6,
        "pte_B_6": pte_B_6,
        "chi2_E_6": chi2_E_6,
        "pte_E_6": pte_E_6,
        "chi2_B_20": chi2_B_20,
        "pte_B_20": pte_B_20,
        "chi2_E_20": chi2_E_20,
        "pte_E_20": pte_E_20,
    }


def main(config, xi_integration, cov_integration, out_dir):
    t_start = time.time()
    fid = config["fiducial"]
    version = fid["version"]
    blind = fid["blind"]
    nmodes = int(fid["nmodes"])  # 20 for full computation

    min_sep_int = fid["min_sep_int"]
    max_sep_int = fid["max_sep_int"]
    nbins_int = fid["nbins_int"]

    gg = treecorr.GGCorrelation(
        min_sep=min_sep_int, max_sep=max_sep_int, nbins=nbins_int, sep_units="arcmin"
    )
    gg.read(xi_integration)

    # Reporting theta grid (nbins+1 = 21 edges): geomspace(1', 250', 21)
    theta_grid = np.geomspace(fid["min_sep"], fid["max_sep"], fid["nbins"] + 1)
    n_grid = fid["nbins"] + 1

    pairs = _pte_scale_cut_pairs()
    n_pairs = len(pairs)

    def _mat():
        return np.full((n_grid, n_grid), np.nan)

    pte_B_6_mat, chi2_B_6_mat = _mat(), _mat()
    pte_E_6_mat, chi2_E_6_mat = _mat(), _mat()
    pte_B_20_mat, chi2_B_20_mat = _mat(), _mat()
    pte_E_20_mat, chi2_E_20_mat = _mat(), _mat()

    pair_arr = np.array(pairs, dtype=int)
    theta_min_arr = np.full(n_pairs, np.nan)
    theta_max_arr = np.full(n_pairs, np.nan)
    En_arr = np.full((n_pairs, nmodes), np.nan)
    Bn_arr = np.full((n_pairs, nmodes), np.nan)
    convergence_failure = np.zeros(n_pairs, dtype=bool)

    for k, (i_min, i_max) in enumerate(pairs):
        theta_min = theta_grid[i_min]
        theta_max = theta_grid[i_max]
        theta_min_arr[k] = theta_min
        theta_max_arr[k] = theta_max

        res = _compute_pair(gg, cov_integration, nmodes, theta_min, theta_max)
        if res is None:
            convergence_failure[k] = True
            print(f"WARNING: polyroots convergence failure for ({i_min}, {i_max}); NaN")
            continue

        En_arr[k, :] = res["En"]
        Bn_arr[k, :] = res["Bn"]
        pte_B_6_mat[i_min, i_max] = res["pte_B_6"]
        chi2_B_6_mat[i_min, i_max] = res["chi2_B_6"]
        pte_E_6_mat[i_min, i_max] = res["pte_E_6"]
        chi2_E_6_mat[i_min, i_max] = res["chi2_E_6"]
        pte_B_20_mat[i_min, i_max] = res["pte_B_20"]
        chi2_B_20_mat[i_min, i_max] = res["chi2_B_20"]
        pte_E_20_mat[i_min, i_max] = res["pte_E_20"]
        chi2_E_20_mat[i_min, i_max] = res["chi2_E_20"]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"cosebis_ptes_{version}_{blind}.npz"
    np.savez(
        npz_path,
        version=version,
        blind=blind,
        nmodes=nmodes,
        mode_subsets=np.array([6, 20]),
        theta_grid=theta_grid,
        pairs=pair_arr,
        theta_min=theta_min_arr,
        theta_max=theta_max_arr,
        En=En_arr,
        Bn=Bn_arr,
        convergence_failure=convergence_failure,
        pte_B_6=pte_B_6_mat,
        chi2_B_6=chi2_B_6_mat,
        pte_E_6=pte_E_6_mat,
        chi2_E_6=chi2_E_6_mat,
        pte_B_20=pte_B_20_mat,
        chi2_B_20=chi2_B_20_mat,
        pte_E_20=pte_E_20_mat,
        chi2_E_20=chi2_E_20_mat,
    )
    print(f"Saved gathered COSEBI PTE matrices to {npz_path}")
    print(f"[TIMING] Total ({n_pairs} pairs): {time.time() - t_start:.2f}s")


def _from_cli(argv=None):
    import yaml

    ap = argparse.ArgumentParser(
        description="Gathered COSEBI B-mode PTE matrix over the scale-cut pair grid (fiducial version + blind)."
    )
    ap.add_argument(
        "--config", required=True, help="Absolute path to bmodes config.yaml"
    )
    ap.add_argument(
        "--xi-integration",
        required=True,
        help="Fiducial 1000-bin integration-grid TreeCorr xi_pm .txt dump",
    )
    ap.add_argument(
        "--cov-integration",
        required=True,
        help="Fiducial 1000-bin Gaussian covariance (processed .txt)",
    )
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        config = yaml.safe_load(f)
    main(config, a.xi_integration, a.cov_integration, a.out)


if __name__ == "__main__":
    _from_cli()
