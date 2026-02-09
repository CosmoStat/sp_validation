"""Precompute per-blind Pure E/B covariance.

Data vectors (xip_E, xim_E, xip_B, xim_B, etc.) are shared across blinds and
loaded from the base pure_eb file. Only the covariance is recomputed per-blind
using MC propagation through the pure E/B transform.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pyccl as ccl
import treecorr
import tqdm
from scipy import sparse

from cosmo_numba.B_modes.schneider2022 import get_pure_EB_modes
from sp_validation.cosmology import get_theo_xi
from sp_validation.cosmo_val import CosmologyValidation

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


def _build_cosmology(cosmo_params):
    """Build CCL cosmology from PLANCK18 params dict."""
    return ccl.Cosmology(
        Omega_c=cosmo_params["Omega_m"] - cosmo_params["Omega_b"],
        Omega_b=cosmo_params["Omega_b"],
        h=cosmo_params["h"],
        sigma8=cosmo_params["sigma_8"],
        n_s=cosmo_params["n_s"],
    )


def compute_pure_eb_covariance(
    theta_rep, theta_int, cov_int, z_dist, cosmo, n_samples, min_sep, max_sep
):
    """Compute pure E/B covariance via MC propagation.

    This is the covariance-only part extracted from calculate_pure_eb_correlation.

    Parameters
    ----------
    theta_rep : array
        Reporting binning theta values (arcmin)
    theta_int : array
        Integration binning theta values (arcmin)
    cov_int : array
        Integration covariance matrix (2*nbins_int x 2*nbins_int)
    z_dist : array
        Redshift distribution (N x 2: z, n(z))
    cosmo : pyccl.Cosmology
        Cosmology for theoretical xi predictions
    n_samples : int
        Number of MC samples
    min_sep, max_sep : float
        Scale cuts (arcmin)

    Returns
    -------
    cov : array
        Pure E/B covariance matrix (6*nbins x 6*nbins)
    """
    nbins_int = len(theta_int)
    nbins_rep = len(theta_rep)

    # Build binning matrix (integration -> reporting)
    # Assume log-spaced bins - compute edges from theta values
    log_theta = np.log10(theta_rep)
    d_log = log_theta[1] - log_theta[0]
    reporting_bin_edges = 10 ** np.concatenate([
        log_theta - d_log/2,
        [log_theta[-1] + d_log/2]
    ])

    bin_indices = np.digitize(theta_int, reporting_bin_edges) - 1
    valid_mask = (bin_indices >= 0) & (bin_indices < nbins_rep)
    row_indices = bin_indices[valid_mask]
    col_indices = np.where(valid_mask)[0]

    binning_matrix = sparse.csr_matrix(
        (np.ones(len(row_indices)), (row_indices, col_indices)),
        shape=(nbins_rep, nbins_int)
    )
    row_sums = np.array(binning_matrix.sum(axis=1)).flatten()
    binning_matrix = sparse.diags(1/row_sums) @ binning_matrix

    # Generate theoretical xi+/xi- predictions
    mean_int = np.concatenate(get_theo_xi(
        theta=theta_int, z=z_dist[:, 0], nz=z_dist[:, 1],
        backend="ccl", cosmo=cosmo
    ))

    # Sample from integration covariance
    # Use fixed seed so all blinds share the same MC noise, enabling comparison
    rng = np.random.default_rng(seed=42)
    samples_int = rng.multivariate_normal(
        mean_int, cov_int, size=n_samples
    )
    samples_int_xip = samples_int[:, :nbins_int]
    samples_int_xim = samples_int[:, nbins_int:]
    samples_rep_xip = (binning_matrix @ samples_int_xip.T).T
    samples_rep_xim = (binning_matrix @ samples_int_xim.T).T

    # Transform each sample through pure E/B
    transformed_samples = [
        np.concatenate(get_pure_EB_modes(
            theta=theta_rep, theta_int=theta_int,
            xip=samples_rep_xip[i], xim=samples_rep_xim[i],
            xip_int=samples_int_xip[i], xim_int=samples_int_xim[i],
            tmin=min_sep, tmax=max_sep
        ))
        for i in tqdm.tqdm(range(n_samples), desc="MC samples")
    ]

    eb_samples = np.array(transformed_samples)
    return np.cov(eb_samples.T)


def main():
    cosmo_cov = _build_cosmology(params["cosmo_params"])

    # Extract blind from output path
    output_path = Path(snakemake.output[0])
    blind = output_path.stem.split("_")[-4]  # e.g., SP_v1.4.6_leak_corr_A_pure_eb_semianalytic → 'A'
    print(f"Computing per-blind covariance for blind {blind}")

    # Load data vectors from base file
    base_data = np.load(snakemake.input["base_pure_eb"])
    print(f"Loaded data vectors from {snakemake.input['base_pure_eb']}")

    # Load integration 2PCF for binning info
    gg_int = treecorr.GGCorrelation(
        min_sep=float(params["min_sep_int"]),
        max_sep=float(params["max_sep_int"]),
        nbins=int(params["nbins_int"]),
        sep_units="arcmin"
    )
    gg_int.read(snakemake.input["xi_integration"])
    theta_int = gg_int.meanr

    # Load per-blind covariance
    cov_int = np.loadtxt(snakemake.input["cov_integration"])
    print(f"Loaded per-blind covariance from {snakemake.input['cov_integration']}")

    # Get per-blind z_dist via CosmologyValidation
    cv = CosmologyValidation(
        versions=[params["version"]],
        catalog_config="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/cat_config.yaml",
        output_dir="/n17data/cdaley/unions/pure_eb/code/sp_validation/notebooks/cosmo_val/output",
    )
    cv.blind = blind
    z, nz = cv.get_redshift(params["version"])
    z_dist = np.column_stack([z, nz])
    print(f"Loaded n(z) for blind {blind}")

    # Compute covariance
    cov_pure_eb = compute_pure_eb_covariance(
        theta_rep=base_data["theta"],
        theta_int=theta_int,
        cov_int=cov_int,
        z_dist=z_dist,
        cosmo=cosmo_cov,
        n_samples=int(params["n_samples"]),
        min_sep=float(params["min_sep"]),
        max_sep=float(params["max_sep"]),
    )

    # Package results - data vectors from base, covariance per-blind
    package = {
        "theta": base_data["theta"],
        "theta_int": base_data["theta_int"],
        "xip_total": base_data["xip_total"],
        "xim_total": base_data["xim_total"],
        "xip_E": base_data["xip_E"],
        "xim_E": base_data["xim_E"],
        "xip_B": base_data["xip_B"],
        "xim_B": base_data["xim_B"],
        "xip_amb": base_data["xip_amb"],
        "xim_amb": base_data["xim_amb"],
        "cov_pure_eb": cov_pure_eb,
    }

    np.savez(snakemake.output[0], **package)
    print(f"Saved to {snakemake.output[0]}")


if __name__ == "__main__":
    main()
