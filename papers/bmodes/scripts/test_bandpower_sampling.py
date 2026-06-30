"""Test whether 32-bin bandpower sampling causes harmonic/config COSEBIS disagreement.

Controlled experiment: compute COSEBIS from the SAME smooth theory C(ℓ)
via three paths:

1. Dense ℓ:     W_n(ℓ) × C(ℓ) integrated over 10,000 ℓ points  (ground truth)
2. 32-bin:      W_n(ℓ) × C(ℓ) integrated over the exact 32 powspace bins
3. Config-space: theory ξ±(θ) → cosebis_from_xipm()               (independent check)

If (1) ≈ (3) but (2) ≠ (1), bandpower sampling is the cause.
If (1) ≈ (2), something else is going on.

Uses a simple power-law C(ℓ) = A × ℓ^α  to avoid external dependencies.
Also tests with a more realistic shape: C(ℓ) = A × ℓ^α × exp(-ℓ/ℓ_0).

Author: Claude Code
"""

import numpy as np
from cosmo_numba.B_modes.cosebis import COSEBIS
from scipy import special

# The exact 32 powspace ℓ values from the pipeline
ELL_32 = np.array(
    [
        12.5,
        24.0,
        38.5,
        56.5,
        78.0,
        103.0,
        131.5,
        163.5,
        199.0,
        238.0,
        281.0,
        327.5,
        377.0,
        430.0,
        487.0,
        547.5,
        611.0,
        678.0,
        749.0,
        823.5,
        901.0,
        982.0,
        1067.0,
        1155.5,
        1247.5,
        1343.0,
        1441.5,
        1544.0,
        1650.0,
        1759.5,
        1872.5,
        1988.5,
    ]
)


def theory_cl(ell, model="power_law"):
    """Smooth theory C(ℓ) for testing."""
    if model == "power_law":
        # Simple power law, roughly cosmic-shear-like
        return 1e-7 * (ell / 100.0) ** (-1.5)
    elif model == "realistic":
        # Power law with exponential cutoff (more structure)
        return 1e-7 * (ell / 100.0) ** (-1.2) * np.exp(-ell / 1500.0)
    else:
        raise ValueError(f"Unknown model: {model}")


def config_space_cosebis(theta_min, theta_max, nmodes, cl_model, n_theta=5_000):
    """Compute COSEBIS via config-space path: C(ℓ) → ξ±(θ) → cosebis_from_xipm.

    ξ+(θ) = (1/2π) ∫ dℓ ℓ C(ℓ) J_0(ℓθ)
    ξ-(θ) = (1/2π) ∫ dℓ ℓ C(ℓ) J_4(ℓθ)
    """
    # Dense ℓ grid for Hankel transform
    n_ell = 20_000
    ell_dense = np.logspace(np.log10(1), np.log10(1e5), n_ell)
    cl = theory_cl(ell_dense, model=cl_model)

    # θ grid (log-spaced in arcmin, clipped to avoid float precision issues)
    theta_arcmin = np.logspace(np.log10(theta_min), np.log10(theta_max), n_theta)
    theta_arcmin = np.clip(theta_arcmin, theta_min, theta_max)
    theta_rad = np.deg2rad(theta_arcmin / 60.0)

    # Vectorized Hankel transform: (n_theta, n_ell) outer product
    # Process in chunks to avoid memory blow-up
    chunk = 500
    xip = np.zeros(n_theta)
    xim = np.zeros(n_theta)
    weight = ell_dense * cl / (2 * np.pi)  # (n_ell,)

    for start in range(0, n_theta, chunk):
        end = min(start + chunk, n_theta)
        t_chunk = theta_rad[start:end, None]  # (chunk, 1)
        x = ell_dense[None, :] * t_chunk  # (chunk, n_ell)
        xip[start:end] = np.trapezoid(weight * special.j0(x), ell_dense, axis=1)
        xim[start:end] = np.trapezoid(weight * special.jv(4, x), ell_dense, axis=1)

    # COSEBIS from ξ±
    cosebis = COSEBIS(theta_min, theta_max, nmodes)
    ce, cb = cosebis.cosebis_from_xipm(theta_arcmin, xip, xim)
    return ce, cb


def harmonic_cosebis(ell, cl_E, theta_min, theta_max, nmodes):
    """Compute COSEBIS via harmonic path: C(ℓ) → cosebis_from_Cell."""
    cosebis = COSEBIS(theta_min, theta_max, nmodes)
    cl_B = np.zeros_like(cl_E)
    ce, cb = cosebis.cosebis_from_Cell(ell, cl_E, cl_B)
    return ce, cb


def make_powspace_bins(ell_min, ell_max, nbins, power=0.5):
    """Generate powspace bin centres matching NaMaster convention."""
    edges = np.linspace(ell_min**power, ell_max**power, nbins + 1) ** (1.0 / power)
    return 0.5 * (edges[:-1] + edges[1:])


def main():
    nmodes = 20
    bin_counts = [32, 64, 100, 200, 500]

    for cl_model in ["power_law", "realistic"]:
        for theta_min, theta_max, label in [
            (1.0, 250.0, "full [1,250]'"),
            (12.0, 83.0, "fiducial [12,83]'"),
        ]:
            print(f"\n{'=' * 75}")
            print(f"C(ℓ) model: {cl_model}, scale cut: {label}")
            print(f"{'=' * 75}")

            # --- Ground truth: dense ℓ ---
            ell_dense = np.logspace(np.log10(2), np.log10(5000), 10_000)
            cl_dense = theory_cl(ell_dense, model=cl_model)
            ce_dense, _ = harmonic_cosebis(
                ell_dense,
                cl_dense,
                theta_min,
                theta_max,
                nmodes,
            )

            # --- Config-space reference ---
            print("Computing config-space reference (Hankel + xipm)...")
            ce_config, _ = config_space_cosebis(
                theta_min,
                theta_max,
                nmodes,
                cl_model,
            )

            # --- Binned paths ---
            ce_binned = {}
            for nb in bin_counts:
                ell_b = make_powspace_bins(8, 2100, nb)
                cl_b = theory_cl(ell_b, model=cl_model)
                ce_b, _ = harmonic_cosebis(ell_b, cl_b, theta_min, theta_max, nmodes)
                ce_binned[nb] = ce_b

            # --- Table: ratio to dense for each bin count ---
            hdr = f"{'mode':>5}"
            for nb in bin_counts:
                hdr += f" {nb:>8d}-bin"
            hdr += "     dense/cfg"
            print(f"\n{hdr}")
            print("-" * (14 + 13 * len(bin_counts)))

            for n in range(nmodes):
                line = f"  n={n + 1:>2d}"
                for nb in bin_counts:
                    r = (
                        ce_binned[nb][n] / ce_dense[n]
                        if abs(ce_dense[n]) > 1e-30
                        else np.nan
                    )
                    line += f" {r:>12.4f}"
                r_cfg = (
                    ce_dense[n] / ce_config[n] if abs(ce_config[n]) > 1e-30 else np.nan
                )
                line += f" {r_cfg:>12.4f}"
                print(line)

            # Summary: max error over modes 1-6 (where signal is meaningful)
            n_summary = min(6, nmodes)
            print(f"\n  Max |1 - ratio| for modes 1-{n_summary} (ratio to dense):")
            for nb in bin_counts:
                ratios = ce_binned[nb][:n_summary] / ce_dense[:n_summary]
                valid = np.isfinite(ratios)
                if valid.any():
                    err = np.max(np.abs(1 - ratios[valid]))
                    print(f"    {nb:>4d} bins: {err:.4f}  ({err * 100:.1f}%)")
            ratios_cfg = ce_dense[:n_summary] / ce_config[:n_summary]
            valid = np.isfinite(ratios_cfg)
            if valid.any():
                err = np.max(np.abs(1 - ratios_cfg[valid]))
                print(f"    dense/config: {err:.4f}  ({err * 100:.1f}%)")


if __name__ == "__main__":
    main()
