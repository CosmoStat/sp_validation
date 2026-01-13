"""Generate pseudo-Cls and covariances for harmonic-space analysis.

Supports two binning modes:
- Linear binning with configurable ell_step (for COSEBIS comparison)
- Power-space binning with configurable n_ell_bins (for standard analysis)

Designed to be called by Snakemake or standalone.
"""

import os
import sys

from astropy.io import fits
import numpy as np

from sp_validation.cosmo_val import CosmologyValidation


def generate_pseudo_cl(
    version: str,
    output_cl: str,
    output_cov: str,
    cat_config: str,
    nside: int = 1024,
    npatch: int = 1,
    blind: str = None,
    cosmo_params: dict = None,
    binning: str = "linear",
    ell_step: int = None,
    n_ell_bins: int = None,
    power: float = 0.5,
):
    """Generate pseudo-Cl and covariance.

    Parameters
    ----------
    version : str
        Catalog version (e.g., "SP_v1.4.6_leak_corr")
    output_cl : str
        Output path for pseudo-Cl FITS file
    output_cov : str
        Output path for covariance FITS file
    cat_config : str
        Path to catalog configuration YAML
    nside : int
        HEALPix nside for map-based estimation
    npatch : int
        Number of jackknife patches
    blind : str, optional
        Blind identifier (A, B, or C) to override n(z) path
    cosmo_params : dict, optional
        Cosmological parameters for covariance calculation.
        Keys: Omega_m, sigma_8, n_s, h, Omega_b.
        If None, uses Planck 2018 defaults.
    binning : str
        Binning mode: "linear" or "powspace"
    ell_step : int, optional
        Width of ell bins for linear binning (required if binning="linear")
    n_ell_bins : int, optional
        Number of ell bins for powspace binning (required if binning="powspace")
    power : float
        Power for powspace binning (0.5 = sqrt spacing)
    """
    # Derive output_dir from whichever output is provided
    output_path = output_cl if output_cl is not None else output_cov
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    blind_str = f" blind={blind}" if blind else ""
    if binning == "linear":
        bin_str = f"ell_step={ell_step}"
    else:
        bin_str = f"n_ell_bins={n_ell_bins}, power={power}"

    print(f"\n{'='*60}")
    print(f"Generating pseudo-Cl for {version}{blind_str}")
    print(f"Binning: {binning} ({bin_str})")
    if cosmo_params:
        print(f"Cosmology: Ω_m={cosmo_params.get('Omega_m')}, σ_8={cosmo_params.get('sigma_8')}")
    else:
        print("Cosmology: Planck 2018 defaults")
    print(f"{'='*60}\n")

    # Remap config cosmology param names to get_cosmo() expected names
    # Config uses: sigma_8, n_s  →  get_cosmo expects: sig8, ns
    if cosmo_params:
        cosmo_params = {
            "Omega_m": cosmo_params.get("Omega_m"),
            "Omega_b": cosmo_params.get("Omega_b"),
            "h": cosmo_params.get("h"),
            "sig8": cosmo_params.get("sigma_8"),
            "ns": cosmo_params.get("n_s"),
        }

    # Build CV kwargs based on binning mode
    cv_kwargs = dict(
        versions=[version],
        catalog_config=cat_config,
        output_dir=output_dir,
        binning=binning,
        nside=nside,
        cell_method="catalog",
        nrandom_cell=100,
        blind=blind,
        cosmo_params=cosmo_params,
        # Standard params (not used for pseudo-Cl but required)
        npatch=npatch,
        theta_min=1.0,
        theta_max=250.0,
        nbins=20,
    )
    if binning == "linear":
        cv_kwargs["ell_step"] = ell_step
    else:
        cv_kwargs["n_ell_bins"] = n_ell_bins
        cv_kwargs["power"] = power

    cv = CosmologyValidation(**cv_kwargs)

    # Calculate pseudo-Cls
    cv.calculate_pseudo_cl()

    # Rename to final output (if output_cl is specified)
    src_cl = os.path.join(output_dir, f"pseudo_cl_{version}.fits")
    if os.path.exists(src_cl):
        with fits.open(src_cl) as hdul:
            data = hdul["PSEUDO_CELL"].data
            n_ell = len(data["ELL"])
            print(f"Generated pseudo-Cl with {n_ell} ell bins")
            print(f"ell range: [{data['ELL'].min():.1f}, {data['ELL'].max():.1f}]")
        if output_cl is not None and src_cl != output_cl:
            os.rename(src_cl, output_cl)
            print(f"Saved to: {output_cl}")
        elif output_cl is None:
            os.remove(src_cl)
            print("Removed intermediate pseudo-Cl (only covariance requested)")

    # Calculate covariance
    print(f"\nCalculating covariance...")
    cv.calculate_pseudo_cl_eb_cov()

    src_cov = os.path.join(output_dir, f"pseudo_cl_cov_{version}.fits")
    if os.path.exists(src_cov) and src_cov != output_cov:
        os.rename(src_cov, output_cov)
        print(f"Saved covariance to: {output_cov}")


if __name__ == "__main__":
    # Check if running under Snakemake
    try:
        snakemake  # noqa: F821
    except NameError:
        # Standalone usage
        print("Usage: Run via Snakemake rule 'fine_pseudo_cl'")
        sys.exit(1)

    generate_pseudo_cl(
        version=snakemake.params.version,  # noqa: F821
        output_cl=getattr(snakemake.output, "pseudo_cl", None),  # noqa: F821
        output_cov=snakemake.output.pseudo_cl_cov,  # noqa: F821
        cat_config=snakemake.params.cat_config,  # noqa: F821
        nside=snakemake.params.nside,  # noqa: F821
        npatch=snakemake.params.npatch,  # noqa: F821
        blind=snakemake.params.get("blind", None),  # noqa: F821
        cosmo_params=snakemake.params.get("cosmo_params", None),  # noqa: F821
        binning=snakemake.params.get("binning", "linear"),  # noqa: F821
        ell_step=snakemake.params.get("ell_step", None),  # noqa: F821
        n_ell_bins=snakemake.params.get("n_ell_bins", None),  # noqa: F821
        power=snakemake.params.get("power", 0.5),  # noqa: F821
    )
