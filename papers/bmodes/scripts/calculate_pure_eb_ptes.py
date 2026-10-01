"""Calculate PTE matrices for pure E/B-mode scale-cut robustness.

CLI refactor of the former Snakemake ``script:`` rule. Reads the pure-E/B
``_pure_eb.npz`` (data vectors + analytic covariance), evaluates the
ξ_+^B / ξ_-^B / joint ξ_tot^B χ² PTE matrices over the scale-cut grid via
``sp_validation.b_modes.calculate_eb_statistics``, and writes the PTE matrices
to ``{out}/{version}_pure_eb_ptes.npz``.

    python calculate_pure_eb_ptes.py \
        --version SP_v1.4.6.3_leak_corr \
        --pure-eb-data <..._pure_eb.npz> --out <output_dir>
"""

import argparse
import os

import numpy as np

from sp_validation.b_modes import calculate_eb_statistics


def calculate_ptes(
    version,
    pure_eb_data,
    output_dir,
):
    dataset = np.load(pure_eb_data)

    theta = dataset["theta"]

    results = {
        "theta": theta,
        # The covariance is analytic: no Hartlap factor.
        "npatch": None,
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
    out_path = os.path.join(output_dir, f"{version}_pure_eb_ptes.npz")
    np.savez(out_path, **output_data)
    print(f"Saved PTE matrices to {out_path}")
    return out_path


def _from_cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", required=True)
    ap.add_argument("--pure-eb-data", required=True, help="Pure E/B .npz")
    ap.add_argument("--out", required=True, help="Output directory (lc {output})")
    a = ap.parse_args(argv)
    calculate_ptes(
        version=a.version,
        pure_eb_data=a.pure_eb_data,
        output_dir=a.out,
    )


if __name__ == "__main__":
    _from_cli()
