"""Calculate PTE matrices for pure E/B-mode scale-cut robustness.

CLI refactor of the former Snakemake ``script:`` rule. Reads the gathered
pure-E/B ``semianalytic.npz`` (data vectors + MC covariance), evaluates the
ξ_+^B / ξ_-^B / joint ξ_tot^B χ² PTE matrices over the scale-cut grid via
``sp_validation.b_modes.calculate_eb_statistics`` (Hartlap-corrected inverse
MC covariance), and writes the PTE matrices to
``{out}/{version}_{blind}_pure_eb_ptes.npz``.

    python calculate_pure_eb_ptes.py \
        --version SP_v1.4.6.3_leak_corr --blind A \
        --pure-eb-data <..._pure_eb_semianalytic.npz> \
        --cov-integration <cov ..._processed.txt> \
        --npatch 1 --n-samples 2000 --out <output_dir>
"""

import argparse
import os

import numpy as np

from sp_validation.b_modes import calculate_eb_statistics


def calculate_ptes(
    version,
    blind,
    pure_eb_data,
    cov_integration,
    npatch,
    n_samples,
    output_dir,
):
    dataset = np.load(pure_eb_data)

    theta = dataset["theta"]

    results = {
        "theta": theta,
        # The MC draws are the realisations behind this covariance.
        "n_eff": int(n_samples),
        "xip_E": dataset["xip_E"],
        "xim_E": dataset["xim_E"],
        "xip_B": dataset["xip_B"],
        "xim_B": dataset["xim_B"],
        "xip_amb": dataset["xip_amb"],
        "xim_amb": dataset["xim_amb"],
        "cov": dataset["cov_pure_eb"],
    }

    print(f"Calculating PTE matrices for {version}...")
    results = calculate_eb_statistics(results)

    pte_matrices = results["pte_matrices"]
    output_data = {
        "theta": theta,
        "pte_xip_B": pte_matrices["xip_B"],
        "pte_xim_B": pte_matrices["xim_B"],
        "pte_combined": pte_matrices["combined"],
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{version}_{blind}_pure_eb_ptes.npz")
    np.savez(out_path, **output_data)
    print(f"Saved PTE matrices to {out_path}")
    return out_path


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--blind", default="A")
    ap.add_argument("--pure-eb-data", required=True, help="Gathered semianalytic .npz")
    ap.add_argument(
        "--cov-integration",
        default=None,
        help="Integration-grid covariance _processed.txt (optional)",
    )
    ap.add_argument("--npatch", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)
    calculate_ptes(
        version=a.version,
        blind=a.blind,
        pure_eb_data=a.pure_eb_data,
        cov_integration=a.cov_integration,
        npatch=a.npatch,
        n_samples=a.n_samples,
        output_dir=a.out,
    )


if __name__ == "__main__":
    _from_cli()
