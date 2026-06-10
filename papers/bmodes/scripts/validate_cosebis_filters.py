"""Validate COSEBIS filter functions W_n(ℓ) against direct integration.

The cosmo_numba code computes W_n(ℓ) via FFT-log (Hankel transform of T_+^log).
This script validates that against direct numerical integration:

  W_n(ℓ) = ∫ dθ θ T_+^log(θ) J_0(ℓθ)

where θ is in radians and T_+^log is the log-basis filter function.

Tests both full [1,250]' and fiducial [12,83]' scale cuts,
for modes n = 1 through 20, across a dense ℓ grid.

Author: Claude Code
"""

import numpy as np
from scipy import special
from scipy import integrate
import matplotlib.pyplot as plt

from cosmo_numba.B_modes.cosebis import COSEBIS


def direct_Wn_integration(ell_val, theta_arcmin, Tp):
    """Compute W_n(ℓ) via direct numerical integration.

    W_n(ℓ) = ∫ dθ θ T_+^log(θ) J_0(ℓθ)

    Uses trapezoidal rule on a fine theta grid.
    """
    theta_rad = np.deg2rad(theta_arcmin / 60)
    j0 = special.j0(ell_val * theta_rad)
    integrand = theta_rad * Tp * j0
    return np.trapezoid(integrand, theta_rad)


def direct_Wn_quad(ell_val, theta_arcmin, Tp):
    """Compute W_n(ℓ) via adaptive quadrature (gold standard).

    Interpolates T_+^log and uses scipy.integrate.quad.
    """
    from scipy.interpolate import CubicSpline
    theta_rad = np.deg2rad(theta_arcmin / 60)
    cs = CubicSpline(theta_rad, Tp)

    def integrand(t):
        return t * cs(t) * special.j0(ell_val * t)

    result, _ = integrate.quad(
        integrand, theta_rad[0], theta_rad[-1],
        limit=500, epsrel=1e-10, epsabs=0,
    )
    return result


def validate_scale_cut(theta_min, theta_max, nmodes, ell_test, n_theta=200_000):
    """Run validation for one scale cut.

    Returns dict with keys 'fftlog', 'direct', 'ratio' — each (nmodes, n_ell).
    """
    print(f"\n{'='*70}")
    print(f"Scale cut: [{theta_min}, {theta_max}] arcmin, modes 1-{nmodes}")
    print(f"{'='*70}")

    cosebis = COSEBIS(theta_min, theta_max, nmodes)

    # Fine log-spaced θ grid for direct integration
    theta = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)

    print("Computing T_+^log filters...")
    Tp = cosebis.get_Tp_log(theta)

    # FFT-log W_n(ℓ) — pass ell directly, let it use 100k internal grid
    print("Computing W_n(ℓ) via FFT-log (100k grid + padding)...")
    Wn_fftlog = cosebis.get_Wn_log(ell_test)

    # Direct integration at each ℓ
    print(f"Computing W_n(ℓ) via direct integration ({n_theta} θ points)...")
    n_ell = len(ell_test)
    Wn_direct = np.zeros((nmodes, n_ell))
    for n in range(nmodes):
        for i, ell in enumerate(ell_test):
            Wn_direct[n, i] = direct_Wn_integration(ell, theta, Tp[n])

    # Ratio
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(
            np.abs(Wn_direct) > 1e-30,
            Wn_fftlog / Wn_direct,
            np.nan,
        )

    return {
        'fftlog': Wn_fftlog,
        'direct': Wn_direct,
        'ratio': ratio,
        'Tp': Tp,
        'theta': theta,
    }


def print_table(ell_test, result, nmodes, indices=None, spot_modes=None):
    """Print comparison table for selected modes at selected ℓ indices."""
    if spot_modes is None:
        spot_modes = list(range(min(nmodes, 6)))
    if indices is None:
        indices = list(range(len(ell_test)))

    for n in spot_modes:
        print(f"\n--- Mode n={n+1} ---")
        print(f"{'ell':>8} {'FFT-log':>14} {'Direct':>14} {'Ratio':>10}")
        print("-" * 50)

        for i in indices:
            r = result['ratio'][n, i]
            print(
                f"{ell_test[i]:>8.0f} "
                f"{result['fftlog'][n, i]:>14.6e} "
                f"{result['direct'][n, i]:>14.6e} "
                f"{r:>10.4f}"
            )


def quad_spot_check(theta_min, theta_max, nmodes, ell_spot):
    """Spot-check a few (mode, ℓ) values with adaptive quadrature."""
    print(f"\n{'='*70}")
    print(f"Adaptive quadrature spot check: [{theta_min}, {theta_max}]'")
    print(f"{'='*70}")

    cosebis = COSEBIS(theta_min, theta_max, nmodes)
    theta_fine = np.logspace(np.log10(theta_min), np.log10(theta_max), 500_000)
    Tp = cosebis.get_Tp_log(theta_fine)

    Wn_fftlog = cosebis.get_Wn_log(np.array(ell_spot, dtype=float))

    print(f"{'mode':>6} {'ell':>8} {'FFT-log':>14} {'quad':>14} {'trapz(500k)':>14} {'FFT/quad':>10} {'trapz/quad':>10}")
    print("-" * 80)

    for n in range(min(nmodes, 5)):
        for j, ell in enumerate(ell_spot):
            wn_fft = Wn_fftlog[n, j]
            wn_quad = direct_Wn_quad(ell, theta_fine, Tp[n])
            wn_trapz = direct_Wn_integration(ell, theta_fine, Tp[n])
            r_fft = wn_fft / wn_quad if abs(wn_quad) > 1e-30 else np.nan
            r_trapz = wn_trapz / wn_quad if abs(wn_quad) > 1e-30 else np.nan
            print(
                f"{n+1:>6d} {ell:>8.0f} {wn_fft:>14.6e} {wn_quad:>14.6e} "
                f"{wn_trapz:>14.6e} {r_fft:>10.6f} {r_trapz:>10.6f}"
            )


def plot_results(ell_test, results, output_path):
    r"""Plot W_n ratio (FFT-log / direct) for both scale cuts."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for ax, (label, result) in zip(axes, results.items()):
        nmodes = result['ratio'].shape[0]
        cmap = plt.get_cmap('viridis', nmodes)

        for n in range(nmodes):
            valid = np.isfinite(result['ratio'][n])
            ax.plot(
                ell_test[valid], result['ratio'][n, valid],
                color=cmap(n), alpha=0.7, lw=1.2,
                label=f"n={n+1}" if n < 10 else None,
            )

        ax.axhline(1.0, color='k', ls='--', lw=0.8)
        ax.axhspan(0.99, 1.01, color='green', alpha=0.1)
        ax.set_ylabel("FFT-log / Direct")
        ax.set_title(rf"$W_n(\ell)$ accuracy: {label}")
        ax.set_ylim(0.9, 1.1)
        ax.legend(ncol=5, fontsize=7, loc='lower left')

    axes[1].set_xlabel(r"$\ell$")
    axes[1].set_xscale('log')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved: {output_path}")
    plt.close(fig)


def main():
    print("COSEBIS W_n(ℓ) Filter Validation")
    print("FFT-log (cosmo_numba, post-Jan 23 rewrite) vs direct integration")

    nmodes = 20

    # Dense ℓ grid covering the pipeline range
    ell_test = np.logspace(np.log10(12), np.log10(3000), 200)

    spot_idx = [0, 25, 50, 75, 100, 125, 150, 175, 199]

    # --- Full scale cut [1, 250]' ---
    result_full = validate_scale_cut(1.0, 250.0, nmodes, ell_test)
    print_table(ell_test, result_full, nmodes, indices=spot_idx)

    # --- Fiducial scale cut [12, 83]' ---
    result_fid = validate_scale_cut(12.0, 83.0, nmodes, ell_test)
    print_table(ell_test, result_fid, nmodes, indices=spot_idx)

    # --- Summary statistics ---
    for label, result in [("Full [1,250]'", result_full), ("Fiducial [12,83]'", result_fid)]:
        print(f"\n{'='*70}")
        print(f"Summary: {label}")
        print(f"{'='*70}")
        for n in range(nmodes):
            valid = np.isfinite(result['ratio'][n])
            if valid.any():
                r = result['ratio'][n, valid]
                print(
                    f"  n={n+1:>2d}: ratio range [{r.min():.4f}, {r.max():.4f}], "
                    f"median={np.median(r):.4f}, |1-ratio| max={np.max(np.abs(1-r)):.4f}"
                )

    # --- Adaptive quadrature spot checks ---
    ell_spot = [50, 200, 500, 1000, 2000]
    quad_spot_check(1.0, 250.0, 5, ell_spot)
    quad_spot_check(12.0, 83.0, 5, ell_spot)

    # --- Plot ---
    results = {
        "[1, 250]' (full)": result_full,
        "[12, 83]' (fiducial)": result_fid,
    }
    output_path = "workflow/scripts/validate_cosebis_filters_results.png"
    plot_results(ell_test, results, output_path)

    # --- Verdict ---
    all_ratios = np.concatenate([
        result_full['ratio'][np.isfinite(result_full['ratio'])],
        result_fid['ratio'][np.isfinite(result_fid['ratio'])],
    ])
    max_err = np.max(np.abs(1 - all_ratios))
    print(f"\n{'='*70}")
    print(f"VERDICT")
    print(f"{'='*70}")
    print(f"Maximum |1 - ratio| across all modes, scale cuts, ℓ: {max_err:.6f}")
    if max_err < 0.01:
        print("PASS: FFT-log W_n(ℓ) accurate to <1%")
        print("  → Harmonic/config disagreement must come from elsewhere")
        print("    (bandpower sampling, mask, noise bias)")
    elif max_err < 0.05:
        print("MARGINAL: FFT-log W_n(ℓ) accurate to <5% but >1%")
        print("  → May contribute to harmonic/config disagreement")
    else:
        print("FAIL: FFT-log W_n(ℓ) inaccurate (>5% error)")
        print("  → File upstream issue on cosmo_numba")


if __name__ == "__main__":
    main()
