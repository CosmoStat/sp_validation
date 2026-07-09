import sys
from multiprocessing import Pool

import astropy.constants as const
import astropy.units as u
import camb
import numpy as np
import scipy.integrate as integrate
from cs_util.cosmo import PLANCK18
from scipy import interpolate
from scipy.special import j0, jn

######################################################################################################

######################################################################################################

def process_theta(theta, nz_file, output_root):
    """Compute shear correlation functions for a single angular scale.

    For a given angular separation, this function computes the weak-lensing
    correlation functions xi+ and xi- over a range of maximum wavenumbers
    (kmax). The calculation includes nonlinear matter power spectra from
    CAMB and optionally intrinsic-alignment contributions. Results are
    appended to output text files.

    Parameters
    ----------
    theta : float
        Angular separation in arcminutes.
    nz_file : str
        Path to the source redshift distribution file. The file must contain
        two columns giving redshift and n(z).
    output_root : str
        Prefix of the output files. Results are written to
        ``{output_root}_xip.txt`` and ``{output_root}_xim.txt``.

    Returns
    -------
    float
        The input angular separation, returned for bookkeeping when running
        in parallel.
    """

    def Hz(z):
        """Return the Hubble expansion rate.

        Computes the Hubble parameter assuming a flat LCDM cosmology.

        Parameters
        ----------
        z : float or ndarray
            Redshift.

        Returns
        -------
        float or ndarray
            Hubble parameter in km s^-1 Mpc^-1.
        """
        return H0 * np.sqrt(Omega_m*(1+z)**3 + (1-Omega_m))

    def rz_interp(want_z):
        """Create an interpolation between redshift and comoving distance.

        Computes the line-of-sight comoving distance by numerical integration
        and returns an interpolation function in either direction.

        Parameters
        ----------
        want_z : bool
            If True, return an interpolator mapping comoving distance to
            redshift. Otherwise return an interpolator mapping redshift to
            comoving distance.

        Returns
        -------
        scipy.interpolate.interp1d
            Interpolation function relating redshift and comoving distance.
        """
        hz_integrand = lambda zz: c/Hz(zz)
        rz_ref = np.array([integrate.quad(hz_integrand, 0, z)[0] for z in zs])

        if want_z == True:
            return interpolate.interp1d(rz_ref, zs, bounds_error=False, fill_value="extrapolate")
        else:
            return interpolate.interp1d(zs, rz_ref, bounds_error=False, fill_value="extrapolate")


    def W_gg(z, rz):
        """Compute the lensing efficiency kernel.

        Evaluates the lensing kernel for the supplied source redshift
        distribution.

        Parameters
        ----------
        z : float
            Lens redshift.
        rz : callable
            Function returning comoving distance as a function of redshift.

        Returns
        -------
        float
            Weak-lensing efficiency kernel evaluated at z.
        """
        z_integrate = np.linspace(z, zmax, n)
        r_zmin = rz(z)
        nz_int = som_nz_interp(z_integrate) * (1 - r_zmin / rz(z_integrate))
        prefactor = 3 * H0**2 * Omega_m * (1+z) * r_zmin / (2 * c**2)

        return prefactor * integrate.simpson(nz_int, x=z_integrate)


    def C_ell(ell, kmax, want_IA):
        """Compute the angular power spectrum.

        Calculates the Limber-approximated cosmic shear power spectrum,
        optionally including intrinsic-alignment (GI and II) contributions.

        Parameters
        ----------
        ell : float
            Angular multipole.
        kmax : float
            Maximum wavenumber used to truncate the Limber integral.
        want_IA : bool
            If True, include intrinsic-alignment contributions.

        Returns
        -------
        float
            Total cosmic shear angular power spectrum at the specified multipole.
        """
        z_min = rz_interp_wantz((ell + 0.5) / kmax)
        z_valid = zs[zs >= z_min]

        if len(z_valid) == 0:
            return 0.0

        rzs = rz_interp_noz(z_valid)
        W_ggs =  W_gg_interp(z_valid)
        Pks = pkz_nl_interp((z_valid, (ell + 0.5) / rzs))
        Hzs = Hz(z_valid)

        gg_integrand = c * W_ggs**2 * Pks / (Hzs * rzs**2)
        C_ell_gg = integrate.simpson(gg_integrand, x=z_valid)

        if want_IA == True:

            Dzs = pkz_lin_interp((z_valid, (ell + 0.5) / rzs)) / pkz_lin_interp((0, (ell + 0.5) / rzs))
            P_ia = -A_IA * c1 * Omega_m / Dzs
            W_ias = Hzs * som_nz_interp(z_valid) / c

            gI_integrand = c * W_ggs * Pks * W_ias * P_ia / (Hzs * rzs**2)
            II_integrand = c * Pks * W_ias**2 * P_ia**2 / (Hzs * rzs**2)

            C_ell_gI = integrate.simpson(gI_integrand, x=z_valid)
            C_ell_II = integrate.simpson(II_integrand, x=z_valid)

            return C_ell_gg + C_ell_gI + C_ell_II

        return C_ell_gg

    def xi(theta_rad, kmax, want_IA):
        """Compute the shear correlation functions.

        Evaluates the real-space shear correlation functions xi+ and xi-
        by Hankel-transforming the convergence power spectrum.

        Parameters
        ----------
        theta_rad : float
            Angular separation in radians.
        kmax : float
            Maximum wavenumber used in the Limber integration.
        want_IA : bool
            If True, include intrinsic-alignment contributions.

        Returns
        -------
        tuple of float
            The pair (xi_plus, xi_minus).
        """
        C_ell_vals = np.array([C_ell(ell, kmax, want_IA) for ell in ells])

        xip_integrand = ells * C_ell_vals * j0(ells * theta_rad)
        xim_integrand = ells * C_ell_vals * jn(4, ells * theta_rad)

        return integrate.simpson(xip_integrand, x=ells)/(2 * np.pi), integrate.simpson(xim_integrand, x=ells)/(2 * np.pi)
    ###########################################################################################

    c = const.c.to('km/s')
    H0 = PLANCK18['h'] * 100
    Omega_m =  PLANCK18['Omega_m']

    A_IA = 0.83
    c1 = 5e-14 * (u.Mpc**3.0) / u.solMass

    zmin = 1e-5
    zmax = 4
    n = 500
    zs = np.linspace(zmin,zmax,n)
    ells = np.linspace(2,1e5,int(1e5-1))

    kmaxs = np.logspace(-4,2,200)
    theta_rad = theta * (np.pi / (180 * 60))

    ombh2 = PLANCK18['Omega_b'] * PLANCK18['h']**2
    omch2 = (PLANCK18['Omega_m'] - PLANCK18['Omega_b']) * PLANCK18['h']**2
    pars = camb.set_params(
        H0=H0, ombh2=ombh2, omch2=omch2, mnu=PLANCK18['m_nu'], As=PLANCK18['As'], ns=PLANCK18['n_s'],
        halofit_version='mead2020_feedback', lmax=3000, WantTransfer=True)

    nz_z, som_nz = np.loadtxt(f'{nz_file}', unpack=True)
    som_nz_interp = interpolate.interp1d(nz_z,som_nz, bounds_error=False, fill_value=None)

    pars.set_matter_power(redshifts = np.linspace(zmin,zmax, 150), kmax=200)
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    k_nonlin, z_nonlin, pk_nonlin = results.get_nonlinear_matter_power_spectrum(hubble_units=False,
                                                                                k_hunit=False)

    pkz_nl_interp = interpolate.RegularGridInterpolator((z_nonlin, k_nonlin), pk_nonlin,
                                    bounds_error=False, fill_value=None)

    k_lin, z_lin, pk_lin = results.get_linear_matter_power_spectrum(hubble_units=False, k_hunit=False)

    pkz_lin_interp = interpolate.RegularGridInterpolator((z_lin, k_lin), pk_lin,
                                    bounds_error=False, fill_value=None)

    rz_interp_wantz = rz_interp(True)
    rz_interp_noz = rz_interp(False)
    W_gg_vals = np.array([W_gg(z, rz_interp_noz) for z in zs])
    W_gg_interp = interpolate.interp1d(zs, W_gg_vals, bounds_error=False, fill_value="extrapolate")

    ###########################################################################################
    xis = np.array([xi(theta_rad, kmax, True) for kmax in kmaxs])
    xip = xis[:,0]
    xim = xis[:,1]

    # Write results immediately to avoid thread conflicts
    with open(f'{output_root}_xip.txt', "a") as f:
        new_arr = np.concatenate(([theta], xip))
        np.savetxt(f, new_arr, fmt='%.8e')

    with open(f'{output_root}_xim.txt', "a") as f:
        new_arr = np.concatenate(([theta], xim))
        np.savetxt(f, new_arr, fmt='%.8e')

    return theta

   ###########################################################################################

if __name__ == "__main__":
    """Run the shear-correlation calculation in parallel.

    The script expects three command-line arguments:

    1. Block index specifying which subset of angular scales to process.
    2. Path to the source redshift distribution file.
    3. Output file prefix.

    The 50 angular scales between 1 and 20 arcmin are divided into
    blocks of 10 values. Each block is processed in parallel using
    multiprocessing, with one worker per angular scale. Each worker
    computes xi+ and xi- over the predefined range of kmax values and
    appends the results to the output files.
    """
    i = int(sys.argv[1])
    nz_file = sys.argv[2]
    output_root = sys.argv[3]

    thetas = np.linspace(1,20,50)
    theta_block = thetas[i*10:(i+1)*10]

    # Run in parallel to speed up calculations for multiple angular scales
    with Pool(processes=10) as pool:
        pool.starmap(process_theta, [(theta, nz_file, output_root) for theta in theta_block])
