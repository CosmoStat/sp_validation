# Harmonic vs Configuration-Space COSEBIS Cross-Validation

Cross-validate COSEBIS E_n and B_n modes computed from two independent paths.

## Purpose

Validate consistency between harmonic-space and configuration-space COSEBIS estimates. Both paths should yield the same E_n and B_n modes since they measure the same underlying shear field. Disagreement would indicate a problem in one of the estimation pipelines.

This is a visual cross-check. No chi2/PTE is computed on the difference because the cross-covariance between methods is unknown.

## Methods

### Path 1: Harmonic (pseudo-C_ell -> COSEBIS)

1. Pseudo-C_ell from NaMaster (powspace binning, 32 bins)
2. `COSEBIS.cosebis_from_Cell(ell, Cell_E, Cell_B, theta)` transforms to E_n, B_n
3. Covariance propagated from C_ell covariance via linear transform T: `Cov_COSEBIS = T @ Cov_Cell @ T^T`

### Path 2: Configuration (xi_pm -> COSEBIS)

1. Fine-binned 2PCF from TreeCorr (integration grid: min_sep_int to max_sep_int, nbins_int bins)
2. `sp_validation.b_modes.calculate_cosebis()` integrates xi_pm with COSEBIS filter functions
3. Covariance from `COSEBIS.cosebis_covariance_from_xipm_covariance()` applied to theoretical xi_pm covariance

## Scale Cuts

Both full range and fiducial scale cuts are shown:

| Scale Cut | Range |
|-----------|-------|
| Full | `cosebis.theta_min` to `cosebis.theta_max` (1-250 arcmin) |
| Fiducial | `fiducial.fiducial_min_scale` to `fiducial.fiducial_max_scale` (12-83 arcmin) |

## Config References

| Parameter | Config Key |
|-----------|------------|
| n_modes | `fiducial.nmodes` |
| theta_min (full) | `cosebis.theta_min` |
| theta_max (full) | `cosebis.theta_max` |
| theta_min (fiducial) | `fiducial.fiducial_min_scale` |
| theta_max (fiducial) | `fiducial.fiducial_max_scale` |
| powspace_nbins | `cl.n_ell_bins` |

## Figures

1. **Data vector** (fiducial catalog): E-modes in raw E_n units, B-modes in B_n/sigma_n. Both scale cuts overlaid.
2. **Version comparison**: All catalog versions, B-modes in B_n/sigma_n. Config-space (filled markers) vs harmonic-space (open markers).

## Known Limitations

The harmonic path uses FFT-log W_n(ell) filter functions from cosmo_numba, which have known accuracy limitations at high ell. Standard powspace nbins=32 binning mitigates but may not fully resolve this. Results should be interpreted as a cross-check, not a precision test.

## Depends on

- cosebis (COSEBIS methodology)
- cl (harmonic-space pseudo-Cl estimation)
