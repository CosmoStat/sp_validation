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

Full range only: `cosebis.theta_min` to `cosebis.theta_max` (1-250 arcmin). The fiducial scale cut is not used — integration over the narrower range is less stable for higher modes and complicates the narrative.

## Reliable Modes

At 96-bin powspace, modes n = 1-8 are quantitatively reliable from the harmonic path (validated on GLASS mocks: E_n/config within 2% for modes 1-5, <1% at the dense-ell harmonic ceiling for modes 1-7). Higher modes (n > 8) are shown grayed out for completeness but excluded from PTE calculations. The root cause for mode 9+ is W_n(ell) numerical precision at ~10^-14 amplitudes, not binning. The reliable mode count depends on `cl.cosebis_nbins` (6 at 32 bins, 8 at 96+ bins).

## Harmonic B-mode PTEs

Chi-squared PTEs are computed for harmonic-space B-modes using reliable modes only (modes 1-8 at 96-bin, modes 1-6 at 32-bin), with the propagated harmonic covariance. These complement the config-space COSEBIS PTEs and the raw pseudo-C_ell PTEs. The expectation is that COSEBIS B-modes from harmonic space are less extreme than raw pseudo-C_ell B-modes, because constant offsets (e.g., from PSF leakage) get absorbed primarily by combinations of high-n modes.

## Config References

| Parameter | Config Key |
|-----------|------------|
| n_modes | `fiducial.nmodes` |
| theta_min | `cosebis.theta_min` |
| theta_max | `cosebis.theta_max` |
| powspace_nbins | `cl.cosebis_nbins` (default 96; separate from `cl.n_ell_bins` used for BB PTEs) |

## Figures

1. **Data vector** (`figure.png`): Fiducial catalog only. E-modes (top, raw E_n units) and B-modes (bottom, B_n/sigma_n). Config vs harmonic overlaid. Unreliable modes grayed.
2. **Version comparison** (`figure_versions.png`): All leak-corrected versions, single panel, B-modes in B_n/sigma_n. Config = filled markers, harmonic = open markers. Unreliable modes grayed.

## Known Limitations

- Cross-covariance between harmonic and configuration-space methods is unknown, so no formal chi2/PTE is computed on the difference between the two paths.
- At 96 bins, modes n > 8 remain unreliable due to W_n(ell) numerical precision limits (COSEBIS amplitudes ~10^-14 at high modes), not binning. This is a fundamental ceiling validated on GLASS mocks with dense integer-ell sampling.

## Depends on

- cosebis (COSEBIS methodology)
- cl (harmonic-space pseudo-Cl estimation)
