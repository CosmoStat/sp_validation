"""Validate COSEBIS integration: harmonic vs config-space using theory.

Uses CCL theory C_ℓ and ξ±(θ) which are mathematically related by exact Hankel
transforms. If the COSEBIS code is correct, E_n computed from either method
should agree to numerical precision.

This is a clean validation that doesn't depend on mock data consistency.

Author: Claude Code
"""

import numpy as np
import pyccl as ccl
from cosmo_numba.B_modes.cosebis import COSEBIS


def setup_cosmology_and_tracer():
    """Setup CCL cosmology and weak lensing tracer."""
    cosmo = ccl.Cosmology(
        Omega_c=0.27,
        Omega_b=0.045,
        h=0.67,
        sigma8=0.8,
        n_s=0.96,
        transfer_function="boltzmann_camb",
        matter_power_spectrum="halofit",
    )

    # Simple Gaussian n(z) centered at z=1.0
    z = np.linspace(0.01, 3.0, 300)
    nz = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
    nz /= np.trapezoid(nz, z)

    tracer = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))

    return cosmo, tracer


def get_theory_cls(ell, cosmo, tracer):
    """Compute theory E-mode C_ℓ."""
    cl_ee = ccl.angular_cl(cosmo, tracer, tracer, ell)
    cl_bb = np.zeros_like(cl_ee)
    return cl_ee, cl_bb


def get_theory_xipm(cosmo, ell, cl_ee, theta_arcmin):
    """Compute ξ±(θ) from C_ℓ using CCL's built-in correlation function."""
    theta_rad = np.deg2rad(theta_arcmin / 60)

    xip = ccl.correlation(cosmo, ell=ell, C_ell=cl_ee, theta=theta_rad, type="GG+")
    xim = ccl.correlation(cosmo, ell=ell, C_ell=cl_ee, theta=theta_rad, type="GG-")

    return xip, xim


def main():
    print("=" * 60)
    print("COSEBIS Theory Validation: Harmonic vs Config-space")
    print("=" * 60)

    # Parameters matching real data analysis
    theta_min = 1.0  # arcmin
    theta_max = 250.0  # arcmin
    nmodes = 6

    # Fine grids for integration
    # Use integer ell for CCL compatibility, but float64 for numba
    ell = np.unique(np.geomspace(2, 30000, 2000).astype(int)).astype(np.float64)
    n_theta = 1000
    theta_integration = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)

    print("\nParameters:")
    print(f"  theta range: [{theta_min}, {theta_max}] arcmin")
    print(f"  COSEBIS modes: 1-{nmodes}")
    print(f"  ℓ grid: {len(ell)} points in [{ell.min()}, {ell.max()}]")
    print(f"  θ grid: {n_theta} points")

    # Setup cosmology
    print("\nSetting up CCL cosmology...")
    cosmo, tracer = setup_cosmology_and_tracer()

    # Generate theory power spectrum
    print("Generating theory C_ℓ...")
    cl_ee, cl_bb = get_theory_cls(ell, cosmo, tracer)
    print(f"  C_ℓ(ℓ=100): {cl_ee[np.argmin(np.abs(ell - 100))]:.3e}")
    print(f"  C_ℓ(ℓ=1000): {cl_ee[np.argmin(np.abs(ell - 1000))]:.3e}")

    # Generate theory ξ±(θ) from C_ℓ using CCL's FFTLog
    print("\nComputing theory ξ±(θ) from C_ℓ using CCL...")
    xip, xim = get_theory_xipm(cosmo, ell, cl_ee, theta_integration)
    print(f"  ξ+(θ=10'): {xip[np.argmin(np.abs(theta_integration - 10))]:.3e}")
    print(f"  ξ-(θ=10'): {xim[np.argmin(np.abs(theta_integration - 10))]:.3e}")

    # Compute COSEBIS from both methods
    print("\nComputing COSEBIS from harmonic-space (C_ℓ)...")
    cosebis = COSEBIS(theta_min, theta_max, nmodes)

    # Fine θ grid for W_n(ℓ) filter functions
    # Note: Using 20000 points for better high-ℓ accuracy (FFT-log issue)
    theta_grid = np.logspace(np.log10(theta_min), np.log10(theta_max), 20000)
    ce_harm, cb_harm = cosebis.cosebis_from_Cell(
        ell=ell, Cell_E=cl_ee, Cell_B=cl_bb, theta=theta_grid, cache=True
    )

    print("\nComputing COSEBIS from config-space (ξ±)...")
    # For config-space, need dtheta for integration
    dtheta = np.gradient(theta_integration)

    ce_config, cb_config = cosebis.cosebis_from_xipm(
        theta=theta_integration, dtheta=dtheta, xi_plus=xip, xi_minus=xim, cache=True
    )

    # Compare
    print("\n" + "=" * 60)
    print("Results: E_n comparison (harmonic vs config-space)")
    print("=" * 60)
    print(f"{'Mode':<6} {'Harmonic':>14} {'Config':>14} {'Ratio':>10} {'Diff (σ)':>10}")
    print("-" * 60)

    for n in range(nmodes):
        ratio = ce_harm[n] / ce_config[n] if ce_config[n] != 0 else np.nan
        # For theory, expect exact agreement, so σ is not meaningful
        # Just report absolute difference
        diff = ce_harm[n] - ce_config[n]
        print(
            f"E_{n + 1:<4} {ce_harm[n]:>14.6e} {ce_config[n]:>14.6e} {ratio:>10.4f} {diff:>10.2e}"
        )

    print("\n" + "=" * 60)
    print("Results: B_n (should be ~0)")
    print("=" * 60)
    print(f"{'Mode':<6} {'Harmonic':>14} {'Config':>14}")
    print("-" * 60)

    for n in range(nmodes):
        print(f"B_{n + 1:<4} {cb_harm[n]:>14.6e} {cb_config[n]:>14.6e}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ratios = ce_harm / ce_config
    print(f"Mean E_n ratio (harm/config): {np.mean(ratios):.4f}")
    print(f"Std E_n ratio: {np.std(ratios):.4f}")
    print(f"Max B_n (harmonic): {np.max(np.abs(cb_harm)):.2e}")
    print(f"Max B_n (config): {np.max(np.abs(cb_config)):.2e}")

    # Verdict
    tolerance = 0.05  # 5% agreement expected for well-resolved integration
    if np.all(np.abs(ratios - 1) < tolerance):
        print(
            f"\n✓ PASS: Harmonic and config-space agree within {tolerance * 100:.0f}%"
        )
    else:
        print(f"\n✗ FAIL: Discrepancy exceeds {tolerance * 100:.0f}%")
        print("  This could indicate:")
        print("  1. Insufficient integration resolution")
        print("  2. Different ℓ/θ range coverage")
        print("  3. Bug in one of the integration methods")

    return {
        "ce_harm": ce_harm,
        "cb_harm": cb_harm,
        "ce_config": ce_config,
        "cb_config": cb_config,
        "ratios": ratios,
    }


if __name__ == "__main__":
    results = main()
