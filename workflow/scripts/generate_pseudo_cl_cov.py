"""Generate pseudo-Cl covariances (no data vector).

Supports two binning modes:
- Linear binning with configurable nbins for COSEBIS
- Power-space binning with configurable nbins for standard C_ell analysis

See generate_pseudo_cl.py for data vector generation.
"""

import os
import sys

from astropy.io import fits

from sp_validation.cosmo_val import CosmologyValidation


def generate_pseudo_cl_cov(
    version: str,
    output_cov: str,
    cat_config: str,
    nside: int = 1024,
    npatch: int = 1,
    blind: str = None,
    cosmo_params: dict = None,
    binning: str = "powspace",
    nbins: int = 32,
    power: float = 0.5,
):
    """Generate pseudo-Cl covariance.

    Parameters
    ----------
    version : str
        Catalog version (e.g., "SP_v1.4.6_leak_corr")
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
        Cosmological parameters. Keys: Omega_m, sigma_8, n_s, h, Omega_b.
        If None, uses Planck 2018 defaults.
    binning : str
        Binning mode: "linear" or "powspace"
    nbins : int
        Number of ell bins
    power : float
        Power for powspace binning (0.5 = sqrt spacing)
    """
    output_dir = os.path.dirname(output_cov)
    os.makedirs(output_dir, exist_ok=True)

    blind_str = f" blind={blind}" if blind else ""
    if binning == "linear":
        ell_step = max(1, (2048 - 2) // nbins)
        bin_str = f"nbins={nbins} (ell_step={ell_step})"
    elif binning == "logspace":
        bin_str = f"nbins={nbins} (geomspace)"
    else:
        bin_str = f"nbins={nbins}, power={power}"

    print(f"\n{'=' * 60}")
    print(f"Generating pseudo-Cl covariance for {version}{blind_str}")
    print(f"Binning: {binning} ({bin_str})")
    if cosmo_params:
        print(
            f"Cosmology: Om={cosmo_params.get('Omega_m')}, s8={cosmo_params.get('sigma_8')}"
        )
    else:
        print("Cosmology: Planck 2018 defaults")
    print(f"{'=' * 60}\n")

    # Remap config cosmology param names to get_cosmo() expected names
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
        npatch=npatch,
        theta_min=1.0,
        theta_max=250.0,
        nbins=20,
    )
    if binning == "linear":
        cv_kwargs["ell_step"] = ell_step
    else:
        cv_kwargs["n_ell_bins"] = nbins
        cv_kwargs["power"] = power

    cv = CosmologyValidation(**cv_kwargs)

    # Calculate covariance only
    print("Calculating covariance...")
    cv.calculate_pseudo_cl_eb_cov()

    # Rename to final output
    src_cov = os.path.join(output_dir, f"pseudo_cl_cov_{version}.fits")
    if os.path.exists(src_cov):
        with fits.open(src_cov) as hdul:
            # CV outputs covariance blocks as COVAR_XX_YY extensions
            cov = hdul["COVAR_BB_BB"].data
            n_ell = int(len(cov) ** 0.5)
            print(f"Generated covariance matrix: {n_ell}x{n_ell}")
        if src_cov != output_cov:
            os.rename(src_cov, output_cov)
            print(f"Saved to: {output_cov}")


if __name__ == "__main__":
    try:
        snakemake  # noqa: F821
    except NameError:
        print("Usage: Run via Snakemake rule 'pseudo_cl_cov'")
        sys.exit(1)

    generate_pseudo_cl_cov(
        version=snakemake.params.version,  # noqa: F821
        output_cov=snakemake.output.pseudo_cl_cov,  # noqa: F821
        cat_config=snakemake.params.cat_config,  # noqa: F821
        nside=snakemake.params.nside,  # noqa: F821
        npatch=snakemake.params.npatch,  # noqa: F821
        blind=snakemake.params.get("blind", None),  # noqa: F821
        cosmo_params=snakemake.params.get("cosmo_params", None),  # noqa: F821
        binning=snakemake.params.binning,  # noqa: F821
        nbins=snakemake.params.nbins,  # noqa: F821
        power=snakemake.params.get("power", 0.5),  # noqa: F821
    )
