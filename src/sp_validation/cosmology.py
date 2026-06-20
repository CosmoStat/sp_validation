"""COSMOLOGY.

:Name: cosmology.py

:Description: This file contains methods for science
              validation of a weak-lensing shape catalogue.
              Depends on a cosmological model.

:Author: Axel Guinot, Martin Kilbinger, Cail Daley

"""

import numpy as np
import pyccl as ccl
import camb

from astropy import cosmology
from astropy.cosmology import Planck18


# =============================================================================
# Fiducial Cosmology: astropy Planck18
# =============================================================================
# Source: Planck 2018 Paper VI, Table 2 (TT,TE,EE+lowE+lensing+BAO)
# Reference: Planck Collaboration 2020, A&A, 641, A6
#
# Note on sigma_8 / A_s consistency:
# CAMB with A_s=2.1e-9 and m_nu=0.06 eV derives sigma_8 ~ 0.806, not 0.8102.
# This ~0.5% difference arises from Planck's MCMC marginalization details.
# Policy: Use sigma_8=0.8102 for codes taking sigma_8 directly (CosmoCov, CCL);
#         use A_s=2.1e-9 for CAMB-based predictions.
# =============================================================================
PLANCK18 = {
    "Omega_m": Planck18.Om0,              # 0.30966
    "Omega_b": Planck18.Ob0,              # 0.04897
    "h": Planck18.h,                      # 0.6766
    "n_s": Planck18.meta["n"],            # 0.9665
    "sigma_8": Planck18.meta["sigma8"],   # 0.8102
    "A_s": 2.1e-9,                        # ln(10^10 A_s) = 3.047
    "m_nu": 0.06,                         # eV, sum of neutrino masses
    "w0": -1.0,
    "wa": 0.0,
}


def _ccl_to_camb(cosmo):
    """Convert CCL cosmology object to CAMB parameter format.

    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object

    Returns
    -------
    dict
        CAMB parameters dictionary with As properly set
    """

    h = cosmo["h"]
    camb_params = {
        "H0": h * 100,
        "ombh2": cosmo["Omega_b"] * h**2,
        "omch2": cosmo["Omega_c"] * h**2,
        "ns": cosmo["n_s"],
    }

    # Handle normalization: prefer As, but convert sigma8 to As if needed
    As_val = cosmo.__getitem__("A_s")
    sigma8_val = cosmo.__getitem__("sigma8")

    if As_val is not None and not np.isnan(As_val):
        # Use As directly
        camb_params["As"] = As_val
    elif sigma8_val is not None:
        # Convert sigma8 to As using iterative CAMB calculation
        # see https://cosmocoffee.info/viewtopic.php?t=475
        As_fiducial = 2e-9  # Standard fiducial value

        # Step 1: Calculate current sigma8 with fiducial As
        temp_params = camb_params.copy()
        temp_params["As"] = As_fiducial

        pars = camb.set_params(**temp_params)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        sigma8_current = results.get_sigma8_0()

        # Step 2: Scale As to match target sigma8
        # As scales as sigma8^2
        As_scaled = As_fiducial * (sigma8_val / sigma8_current) ** 2
        camb_params["As"] = As_scaled

        # Step 3: Verify the result
        temp_params["As"] = As_scaled
        pars = camb.set_params(**temp_params)
        pars.set_matter_power(redshifts=[0.0], kmax=2.0)
        results = camb.get_results(pars)
        sigma8_final = results.get_sigma8_0()

        # Check accuracy (warn if >1% difference)
        relative_error = abs(sigma8_final - sigma8_val) / sigma8_val
        if relative_error > 0.01:
            print(
                f"Warning: CAMB sigma8 conversion accuracy: target={sigma8_val:.4f}, "
                f"achieved={sigma8_final:.4f}, error={relative_error:.1%}"
            )
    else:
        # No normalization specified, use CAMB default
        pass

    # Add dark energy parameters if they exist
    for camb_key, cosmo_key in [("w", "w0"), ("wa", "wa")]:
        if hasattr(cosmo._params, cosmo_key):
            camb_params[camb_key] = cosmo[cosmo_key]

    return camb_params


def _camb_to_ccl(camb_params):
    """Convert CAMB parameter format to CCL parameter dictionary.

    Parameters
    ----------
    camb_params : dict
        CAMB parameters dictionary with required keys:
        H0, ombh2, omch2, ns, and either As or sigma8

    Returns
    -------
    dict
        CCL parameters dictionary
    """
    h = camb_params["H0"] / 100.0
    ccl_params = {
        "Omega_c": camb_params["omch2"] / h**2,
        "Omega_b": camb_params["ombh2"] / h**2,
        "h": h,
        "n_s": camb_params["ns"],
        **{
            k: camb_params[v]
            for k, v in [("w0", "w"), ("wa", "wa")]
            if v in camb_params
        },
    }

    # CCL accepts either A_s or sigma8 directly
    if "As" in camb_params:
        ccl_params["A_s"] = camb_params["As"]
    elif "sigma8" in camb_params:
        ccl_params["sigma8"] = camb_params["sigma8"]
    else:
        raise ValueError("Must provide either 'As' or 'sigma8' in camb_params")

    return ccl_params


def _cosmocov_to_ccl(cosmocov_params):
    """Convert CosmoCov parameter format to CCL parameter dictionary.

    Parameters
    ----------
    cosmocov_params : dict
        CosmoCov parameters dictionary with required keys:
        Omega_m, omb, h0, sigma_8, n_spec

    Returns
    -------
    dict
        CCL parameters dictionary
    """
    required_params = ["Omega_m", "omb", "h0", "n_spec", "sigma_8"]
    missing_params = [p for p in required_params if p not in cosmocov_params]
    if missing_params:
        raise KeyError(f"Missing required cosmological parameters: {missing_params}")

    ccl_params = {
        "Omega_c": cosmocov_params["Omega_m"] - cosmocov_params["omb"],
        "Omega_b": cosmocov_params["omb"],
        "h": cosmocov_params["h0"],
        "sigma8": cosmocov_params["sigma_8"],
        "n_s": cosmocov_params["n_spec"],
        **{k: cosmocov_params[k] for k in ["w0", "wa"] if k in cosmocov_params},
    }

    return ccl_params


def get_cosmo(
    Omega_b=None,
    Omega_m=None,
    h=None,
    sig8=None,
    ns=None,
    w0=None,
    wa=None,
    mnu=None,
    transfer_function="boltzmann_camb",
    matter_power_spectrum="halofit",
    cosmocov_params=None,
    camb_params=None,
    extra_params=None,
):
    """Get CCL cosmology object with user-specified parameters.

    Defaults to astropy Planck18 cosmology (Table 2: TT,TE,EE+lowE+lensing+BAO).
    Can also use CosmoCov or CAMB parameter formats.

    Parameters
    ----------
    Omega_m : float, default=None
        Matter density parameter (defaults to Planck18: 0.30966)
    Omega_b : float, default=None
        Baryon density parameter (defaults to Planck18: 0.04897)
    h : float, default=None
        Reduced Hubble constant (defaults to Planck18: 0.6766)
    sig8 : float, default=None
        RMS matter fluctuation amplitude at 8 Mpc/h (defaults to Planck18: 0.8102)
    ns : float, default=None
        Scalar spectral index (defaults to Planck18: 0.9665)
    w0 : float, default=None
        Dark energy equation of state parameter (defaults to -1.0)
    wa : float, default=None
        Dark energy equation of state evolution parameter (defaults to 0.0)
    mnu : float, default=None
        Total neutrino mass in eV (defaults to 0.06 eV)
    transfer_function : str, default='boltzmann_camb'
        Transfer function to use
    matter_power_spectrum : str, default='halofit'
        Matter power spectrum to use
    cosmocov_params : dict, optional
        Parameters in CosmoCov format (Omega_m, omb, h0, sigma_8, n_spec)
        If provided, entries override above parameters. Mutually exclusive with
        camb_params.
    camb_params : dict, optional
        Parameters in CAMB format (H0, ombh2, omch2, ns, sigma8)
        If provided, entries override above parameters. Mutually exclusive with
        cosmocov_params.
    extra_params : dict, optional
        Additional parameters to pass to CCL (e.g., for CAMB non-linear settings)

    Returns
    -------
    Cosmology
        pyccl cosmology object
    """
    # Check for parameter format conflicts
    if cosmocov_params is not None and camb_params is not None:
        raise ValueError(
            "Cannot provide both cosmocov_params and camb_params. Choose one format."
        )

    # Convert parameters to CCL format
    if cosmocov_params is not None:
        print("Using CosmoCov parameters to create CCL cosmology.")
        ccl_params = _cosmocov_to_ccl(cosmocov_params)
    elif camb_params is not None:
        print("Using CAMB parameters to create CCL cosmology.")
        ccl_params = _camb_to_ccl(camb_params)
    else:
        ccl_params = {}

    # Planck 2018 defaults from astropy (see PLANCK18 dict at module level)
    planck_defaults = {
        "Omega_m": PLANCK18["Omega_m"],
        "Omega_b": PLANCK18["Omega_b"],
        "h": PLANCK18["h"],
        "sig8": PLANCK18["sigma_8"],
        "ns": PLANCK18["n_s"],
        "w0": PLANCK18["w0"],
        "wa": PLANCK18["wa"],
        "mnu": PLANCK18["m_nu"],
    }

    mnu = ccl_params.get("mnu", mnu or planck_defaults["mnu"])
    h = ccl_params.get("h", h or planck_defaults["h"])
    pars = camb.CAMBparams()

    pars.set_cosmology(
        mnu=mnu,
        H0=h * 100,
    )

    Omega_nu = pars.omeganu

    combined_params = {
        "Omega_c": ccl_params.get(
            "Omega_c",
            (Omega_m or planck_defaults["Omega_m"])
            - (Omega_b or planck_defaults["Omega_b"])
            - Omega_nu,
        ),
        "Omega_b": ccl_params.get("Omega_b", Omega_b or planck_defaults["Omega_b"]),
        "h": h,
        "sigma8": ccl_params.get("sigma8", sig8 or planck_defaults["sig8"]),
        "n_s": ccl_params.get("n_s", ns or planck_defaults["ns"]),
        "w0": ccl_params.get("w0", w0 or planck_defaults["w0"]),
        "wa": ccl_params.get("wa", wa or planck_defaults["wa"]),
        "m_nu": mnu,
        "extra_parameters": extra_params
    }

    return ccl.Cosmology(
        **combined_params,
        transfer_function=transfer_function,
        matter_power_spectrum=matter_power_spectrum,
    )


def get_theo_c_ell(
    ell,
    z,
    nz,
    backend="ccl",
    cosmo=None,
    Omega_b=None,
    Omega_m=None,
    h=None,
    sig8=None,
    ns=None,
    w0=None,
    wa=None,
):
    """Calculate theoretical angular power spectrum C_ell for weak lensing.

    Parameters
    ----------
    ell : array
        Multipole moments (e.g., np.arange(2, 2000))
    z : array
        Redshifts for n(z) distribution
    nz : array
        n(z) redshift distribution
    backend : str, default="ccl"
        Backend to use: "ccl" or "camb"
    cosmo : ccl.Cosmology, optional
        CCL cosmology object. If None, will create using individual parameters.
    Omega_b : float, optional
        Baryon density parameter (defaults to Planck 2018)
    Omega_m : float, optional
        Matter density parameter (defaults to Planck 2018)
    h : float, optional
        Reduced Hubble constant (defaults to Planck 2018)
    sig8 : float, optional
        RMS matter fluctuation amplitude at 8 Mpc/h (defaults to Planck 2018)
    ns : float, optional
        Scalar spectral index (defaults to Planck 2018)
    w0 : float, optional
        Dark energy equation of state parameter (defaults to -1.0)
    wa : float, optional
        Dark energy equation of state evolution parameter (defaults to 0.0)

    Returns
    -------
    cl : array
        Angular power spectrum
    """
    if cosmo is None:
        cosmo = get_cosmo(
            Omega_b=Omega_b,
            Omega_m=Omega_m,
            h=h,
            sig8=sig8,
            ns=ns,
            w0=w0,
            wa=wa,
        )

    if backend == "ccl":
        # Create lensing tracer
        lens = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))

        # Calculate power spectrum
        cl = ccl.angular_cl(cosmo, lens, lens, ell)

    elif backend == "camb":
        # Convert CCL cosmology to CAMB parameters
        import camb

        camb_kwargs = _ccl_to_camb(cosmo)

        # Set up CAMB parameters
        pars = camb.set_params(
            **camb_kwargs,
            WantTransfer=True,
            NonLinear=camb.model.NonLinear_both,
        )

        # Adjust for neutrino contribution
        if "mnu" in camb_kwargs and camb_kwargs["mnu"] > 0:
            omch2_adj = camb_kwargs["omch2"] - pars.omeganu * (pars.H0 / 100) ** 2
            pars.set_cosmology(omch2=omch2_adj)

        # Set up lensing source window
        pars.min_l = ell.min()
        pars.set_for_lmax(ell.max())
        pars.SourceWindows = [
            camb.sources.SplinedSourceWindow(z=z, W=nz, source_type="lensing")
        ]

        # Calculate power spectrum
        results = camb.get_results(pars)
        theory_cls = results.get_source_cls_dict(lmax=ell.max(), raw_cl=True)
        cl_full = theory_cls["W1xW1"]

        # Interpolate to match input ell array
        # CAMB returns C_ell for ell = 0, 1, 2, ..., lmax
        ell_camb = np.arange(len(cl_full))
        cl = np.interp(ell, ell_camb, cl_full)

    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'ccl' or 'camb'")

    return cl


def c_ell_to_xi(cosmo, theta, ell, cl):
    """Convert angular power spectrum to correlation functions using CCL.

    Parameters
    ----------
    cosmo : ccl.Cosmology
        CCL cosmology object (used for correlation function calculation)
    theta : array
        Angular separations in arcminutes
    ell : array
        Multipole moments
    cl : array
        Angular power spectrum

    Returns
    -------
    xip, xim : arrays
        xi+ and xi- correlation functions
    """
    theta_deg = theta / 60.0  # arcmin to degrees

    xip = ccl.correlation(
        cosmo, ell=ell, C_ell=cl, theta=theta_deg, type="GG+", method="Bessel"
    )
    xim = ccl.correlation(
        cosmo, ell=ell, C_ell=cl, theta=theta_deg, type="GG-", method="Bessel"
    )

    return xip, xim


def get_theo_xi(
    theta,
    z,
    nz,
    Omega_m=None,
    h=None,
    Omega_b=None,
    sig8=None,
    ns=None,
    ell_min=10,
    ell_max=20000,
    n_ell=500,
    backend="ccl",
    cosmo=None,
    **cosmo_kwargs,
):
    """Calculate theoretical xi+/xi- using individual parameters.

    Parameters
    ----------
    theta : array
        Angular separations in arcminutes
    z : array
        Redshift array
    nz : array
        n(z) redshift distribution
    Omega_m : float, default=None
        Matter density parameter (defaults to Planck 2018)
    h : float, default=None
        Reduced Hubble constant (defaults to Planck 2018)
    Omega_b : float, default=None
        Baryon density parameter (defaults to Planck 2018)
    sig8 : float, default=None
        RMS matter fluctuation amplitude at 8 Mpc/h (defaults to Planck 2018)
    ns : float, default=None
        Scalar spectral index (defaults to Planck 2018)
    ell_min : int, default=0
        Minimum ell for power spectrum calculation
    ell_max : int, default=20000
        Maximum ell for power spectrum calculation
    n_ell : int, default=500
        Number of ell bins
    backend : str, default="ccl"
        Backend to use: "ccl" or "camb"
    **cosmo_kwargs
        Additional arguments passed to backend

    Returns
    -------
    xip, xim : arrays
        Theoretical xi+ and xi- correlation functions
    """
    # Create ell array for C_ell calculation
    ell = np.geomspace(ell_min, ell_max, n_ell)

    # Use provided cosmology or create from parameters
    if cosmo is None:
        cosmo = get_cosmo(
            Omega_m=Omega_m, Omega_b=Omega_b, h=h, sig8=sig8, ns=ns, **cosmo_kwargs
        )

    # Calculate C_ell
    cl = get_theo_c_ell(ell, z, nz, backend=backend, cosmo=cosmo)

    # Convert to xi
    return c_ell_to_xi(cosmo, theta, ell, cl)
