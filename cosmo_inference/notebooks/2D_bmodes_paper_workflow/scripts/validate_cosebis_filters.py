"""Validate COSEBIS filter functions W_n(ℓ) against direct integration.

According to Schneider et al. 2010 Eq. 6:
W_n(ℓ) = ∫ dθ θ T_+(θ) J_0(ℓθ)

The cosmo_numba code computes W_n via FFT-log. This script validates
that computation against direct numerical integration.

Author: Claude Code
"""

import numpy as np
from scipy import special
from scipy import integrate

from cosmo_numba.B_modes.cosebis import COSEBIS


def direct_Wn_integration(ell_val, theta_arcmin, Tp):
    """Compute W_n(ℓ) via direct numerical integration of Eq. 6."""
    theta_rad = np.deg2rad(theta_arcmin / 60)

    # W_n(ℓ) = ∫ dθ θ T_+(θ) J_0(ℓθ)
    x = ell_val * theta_rad
    j0 = special.j0(x)
    integrand = theta_rad * Tp * j0

    # Simple trapezoidal integration
    Wn = np.trapezoid(integrand, theta_rad)
    return Wn


def main():
    print("=" * 60)
    print("COSEBIS Filter Function Validation")
    print("=" * 60)

    theta_min = 1.0  # arcmin
    theta_max = 250.0  # arcmin
    nmodes = 3

    # Fine θ grid
    n_theta = 10000
    theta = np.linspace(theta_min, theta_max, n_theta)

    print(f"\nParameters:")
    print(f"  theta range: [{theta_min}, {theta_max}] arcmin")
    print(f"  theta grid: {n_theta} points")
    print(f"  COSEBIS modes: 1-{nmodes}")

    # Initialize COSEBIS
    cosebis = COSEBIS(theta_min, theta_max, nmodes)

    # Get T+ filter functions
    print("\nComputing T+(θ) filters...")
    Tp = cosebis.get_Tp_log(theta)
    print(f"  T_1(θ=10'): {Tp[0, np.argmin(np.abs(theta - 10))]:.4f}")
    print(f"  T_2(θ=10'): {Tp[1, np.argmin(np.abs(theta - 10))]:.4f}")

    # Get W_n(ℓ) from FFT-log (stored in cosmo_numba)
    print("\nComputing W_n(ℓ) via FFT-log...")
    theta_fftlog = np.logspace(np.log10(theta_min), np.log10(theta_max), 5000)
    Wn_fftlog = cosebis.get_Wn_log(theta_fftlog)

    # Test at a few ℓ values
    ell_test = np.array([50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])

    print("\n" + "=" * 60)
    print("Comparison: W_n from FFT-log vs direct integration")
    print("=" * 60)

    for n in range(nmodes):
        print(f"\n--- Mode n={n+1} ---")
        print(f"{'ell':>8} {'FFT-log':>14} {'Direct':>14} {'Ratio':>10}")
        print("-" * 50)

        for ell in ell_test:
            # FFT-log value
            Wn_fft = Wn_fftlog[n].eval(np.array([ell]))[0]

            # Direct integration
            Wn_direct = direct_Wn_integration(ell, theta, Tp[n])

            ratio = Wn_fft / Wn_direct if Wn_direct != 0 else np.nan
            print(f"{ell:>8.0f} {Wn_fft:>14.6e} {Wn_direct:>14.6e} {ratio:>10.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    print("""
If FFT-log and direct integration agree:
  → The filter functions are correct
  → The issue is elsewhere (integration formula, units, etc.)

If they disagree:
  → The FFT-log computation has a bug
  → Could be normalization, units, or algorithm issue
""")


if __name__ == "__main__":
    main()
