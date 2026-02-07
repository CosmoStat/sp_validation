# Harmonic vs Configuration-Space COSEBIS Cross-Validation

Cross-validate COSEBIS E_n and B_n modes computed from two independent paths.

## Purpose

Validate consistency between harmonic-space and configuration-space COSEBIS estimates. Both paths should yield the same E_n and B_n modes since they measure the same underlying shear field. Disagreement would indicate a problem in one of the estimation pipelines.

## Methods

### Path 1: Harmonic (pseudo-C_ell -> COSEBIS)

1. Pseudo-C_ell from NaMaster (powspace binning, 32 bins)
2. `COSEBIS.cosebis_from_Cell(ell, Cell_E, Cell_B, theta)` transforms to E_n, B_n
3. Covariance propagated from C_ell covariance via linear transform T: `Cov_COSEBIS = T @ Cov_Cell @ T^T`

### Path 2: Configuration (xi_pm -> COSEBIS)

1. Fine-binned 2PCF from TreeCorr (integration grid: min_sep_int to max_sep_int, nbins_int bins)
2. `sp_validation.b_modes.calculate_cosebis()` integrates xi_pm with COSEBIS filter functions
3. Covariance from `COSEBIS.cosebis_covariance_from_xipm_covariance()` applied to theoretical xi_pm covariance

## Config References

| Parameter | Config Key |
|-----------|------------|
| n_modes | `fiducial.nmodes` |
| theta_min | `cosebis.theta_min` |
| theta_max | `cosebis.theta_max` |
| powspace_nbins | `cl.n_ell_bins` |

## Statistical Test

Chi-squared statistic on the residual (harmonic - config) using config-space covariance:

    chi2 = (E_harm - E_config)^T Cov_config_E^{-1} (E_harm - E_config)

PTE from chi2 distribution with n_modes degrees of freedom. Computed separately for E-modes and B-modes.

## Known Limitations

The harmonic path uses FFT-log W_n(ell) filter functions from cosmo_numba, which have known accuracy limitations at high ell. Standard powspace nbins=32 binning mitigates but may not fully resolve this. Results should be interpreted as a cross-check, not a precision test.

## Depends on

- cosebis (COSEBIS methodology)
- cl (harmonic-space pseudo-Cl estimation)
