"""Gather MC sample chunks and compute final pure E/B covariance."""

import os
import sys
from pathlib import Path

import numpy as np

# Unbuffered output for Snakemake log streaming
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)


def _load_snakemake():
    if hasattr(sys, "ps1"):
        from snakemake_helpers import snakemake_interactive

        return snakemake_interactive(
            "results/paper_plots/intermediate/SP_v1.4.6_leak_corr_A_pure_eb_semianalytic.npz",
            str(Path.cwd()),
        )
    from snakemake.script import snakemake

    return snakemake


snakemake = _load_snakemake()
params = snakemake.params


def _load_xi(path, nbins):
    """Load 2PCF from treecorr output file."""
    # Treecorr files have header comments (##) and column names (#), then data,
    # then ## cov followed by covariance matrix
    data = np.loadtxt(path, comments="#", max_rows=nbins)
    return {
        "meanr": data[:, 1],
        "xip": data[:, 3],
        "xim": data[:, 4],
    }


def main():
    from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes

    numeric_params = {
        "min_sep": float(params["min_sep"]),
        "max_sep": float(params["max_sep"]),
        "nbins": int(params["nbins"]),
        "min_sep_int": float(params["min_sep_int"]),
        "max_sep_int": float(params["max_sep_int"]),
        "nbins_int": int(params["nbins_int"]),
        "npatch": int(params["npatch"]),
    }

    blind = params.get("blind", "A")
    print(f"Gathering pure E/B for blind {blind}")

    # Load precomputed correlation functions from Snakemake inputs
    gg = _load_xi(snakemake.input["xi_reporting"], numeric_params["nbins"])
    gg_int = _load_xi(snakemake.input["xi_integration"], numeric_params["nbins_int"])

    # Compute data vectors from correlation functions
    eb_results = get_pure_EB_modes(
        theta=gg["meanr"],
        xip=gg["xip"],
        xim=gg["xim"],
        theta_int=gg_int["meanr"],
        xip_int=gg_int["xip"],
        xim_int=gg_int["xim"],
        tmin=numeric_params["min_sep"],
        tmax=numeric_params["max_sep"],
    )
    xip_E, xim_E, xip_B, xim_B, xip_amb, xim_amb = eb_results

    # Load and concatenate all chunks
    chunk_files = sorted(snakemake.input["chunks"])
    all_samples = []
    for chunk_file in chunk_files:
        data = np.load(chunk_file)
        all_samples.append(data["eb_samples"])
        print(f"Loaded {len(data['eb_samples'])} samples from {chunk_file}")

    eb_samples = np.vstack(all_samples)
    print(f"Total samples: {len(eb_samples)}")

    # Compute covariance from all samples
    cov_pure_eb = np.cov(eb_samples.T)

    # Package results
    package = {
        "theta": gg["meanr"],
        "theta_int": gg_int["meanr"],
        "xip_total": gg["xip"],
        "xim_total": gg["xim"],
        "xip_E": xip_E,
        "xim_E": xim_E,
        "xip_B": xip_B,
        "xim_B": xim_B,
        "xip_amb": xip_amb,
        "xim_amb": xim_amb,
        "cov_pure_eb": cov_pure_eb,
    }

    np.savez(snakemake.output[0], **package)
    print(f"Saved to {snakemake.output[0]}")


if __name__ == "__main__":
    main()
