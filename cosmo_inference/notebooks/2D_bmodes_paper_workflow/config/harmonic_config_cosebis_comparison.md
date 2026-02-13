# Harmonic vs Configuration-Space COSEBIS Cross-Validation

Cross-validate COSEBIS E_n and B_n modes computed from two independent paths.

## Purpose

Validate consistency between harmonic-space and configuration-space COSEBIS estimates. Both paths should yield the same E_n and B_n modes since they measure the same underlying shear field. Disagreement would indicate a problem in one of the estimation pipelines.

## Methods

### Path 1: Harmonic (pseudo-C_ell -> COSEBIS)

1. Pseudo-C_ell from NaMaster (powspace binning, `cl.cosebis_nbins` bins — default 96)
2. `COSEBIS.cosebis_from_Cell(ell, Cell_E, Cell_B, theta)` transforms to E_n, B_n
3. Covariance propagated from C_ell covariance via linear transform T: `Cov_COSEBIS = T @ Cov_Cell @ T^T`

### Path 2: Configuration (xi_pm -> COSEBIS)

1. Fine-binned 2PCF from TreeCorr (integration grid: min_sep_int to max_sep_int, nbins_int bins)
2. `sp_validation.b_modes.calculate_cosebis()` integrates xi_pm with COSEBIS filter functions
3. Covariance from `COSEBIS.cosebis_covariance_from_xipm_covariance()` applied to theoretical xi_pm covariance

## Angular Range

Parameterized by `{angular_range}` wildcard:
- **full**: `cosebis.theta_min` to `cosebis.theta_max` (1–250 arcmin)
- **fiducial**: `fiducial.fiducial_min_scale` to `fiducial.fiducial_max_scale` (12–83 arcmin)

The rule runs once per angular range, producing separate evidence and figures.

## Reliable Modes

At 96-bin powspace, modes n = 1–8 are quantitatively reliable from the harmonic path (validated on GLASS mocks: E_n/config within 2% for modes 1–5, <1% at the dense-ell harmonic ceiling for modes 1–7). Higher modes (n > 8) are shown grayed out for completeness but excluded from PTE calculations. The root cause for mode 9+ is W_n(ell) numerical precision at ~10^-14 amplitudes, not binning. The reliable mode count depends on `cl.cosebis_nbins` (6 at 32 bins, 8 at 96+ bins).

## B-mode PTEs

Chi-squared PTEs are computed for B-modes using reliable modes only (modes 1–8 at 96-bin), from both paths independently:
- **Harmonic-space**: propagated from pseudo-C_ell Gaussian covariance via linear transform
- **Config-space**: from CosmoCov theoretical xi_pm covariance

At fiducial scale cuts, both methods find B-modes consistent with zero. At full range, both fail due to the known small-scale contamination.

## Config References

| Parameter | Config Key |
|-----------|------------|
| n_modes | `fiducial.nmodes` |
| theta_min (full) | `cosebis.theta_min` |
| theta_max (full) | `cosebis.theta_max` |
| theta_min (fiducial) | `fiducial.fiducial_min_scale` |
| theta_max (fiducial) | `fiducial.fiducial_max_scale` |
| powspace_nbins | `cl.cosebis_nbins` (default 96; separate from `cl.n_ell_bins` used for BB PTEs) |

## Figures

Per angular range:
1. **Data vector** (`figure.png`): Fiducial catalog. E-modes (top) and B-modes (bottom, B_n/sigma_n). Config vs harmonic overlaid. Unreliable modes grayed.
2. **Version comparison** (`figure_versions.png`): All leak-corrected versions, B_n/sigma_n.
3. **Paper figure**: `harmonic_config_cosebis_{angular_range}.png`

## Known Limitations

- Cross-covariance between harmonic and configuration-space methods is unknown, so no formal chi2/PTE is computed on the difference between the two paths.
- At 96 bins, modes n > 8 remain unreliable due to W_n(ell) numerical precision limits (COSEBIS amplitudes ~10^-14 at high modes), not binning. This is a fundamental ceiling validated on GLASS mocks with dense integer-ell sampling.

## Depends on

- cosebis (COSEBIS methodology)
- cl (harmonic-space pseudo-Cl estimation)
