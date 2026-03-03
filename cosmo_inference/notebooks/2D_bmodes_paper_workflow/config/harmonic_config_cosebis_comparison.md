# Harmonic vs Configuration-Space COSEBIS Comparison

Cross-validation of COSEBIS computed from harmonic-space C_ℓ vs configuration-space ξ±.

## Purpose

Verify consistency between two independent paths to COSEBIS:
1. **Harmonic**: Integrate finely-binned pseudo-C_ℓ over ℓ using T_n(ℓ) kernels
2. **Configuration**: Integrate ξ± over θ using W_n(θ) log filters

Agreement validates both the pseudo-C_ℓ estimation and the COSEBIS integration machinery.

## Method

For each catalog version:
1. Load pseudo-C_ℓ (EE, BB) from NaMaster output
2. Compute COSEBIS E_n, B_n from C_ℓ via `cosebis_from_Cell()`
3. Compute COSEBIS E_n, B_n from ξ± via `calculate_cosebis()`
4. Propagate covariance through the C_ℓ → COSEBIS transform
5. Compare residuals, compute χ² and PTE

## Config References

| Parameter | Config Key | Description |
|-----------|------------|-------------|
| versions | `versions` | Catalog versions to compare |
| nmodes_long | `cosebis.nmodes` | Full mode count (20) |
| nmodes_short | `cosebis.mode_subsets[0]` | Short mode count (6) for focused PTE |
| theta_min | `cosebis.theta_min` | Angular range minimum (arcmin) |
| theta_max | `cosebis.theta_max` | Angular range maximum (arcmin) |
| min_sep_int | `fiducial.min_sep_int` | Integration grid min (arcmin) |
| max_sep_int | `fiducial.max_sep_int` | Integration grid max (arcmin) |
| nbins_int | `fiducial.nbins_int` | Integration grid bins |
| npatch | `fiducial.npatch` | Jackknife patches |

## Requirements

For accurate harmonic→COSEBIS conversion, pseudo-C_ℓ should be finely binned:
- `binning='linear'` with `ell_step=1` or `ell_step=2`
- Standard sqrt-binned C_ℓ (n_ell_bins=32) loses information at high ℓ

## Outputs

- Comparison figure: E_n and B_n from both methods, residuals, χ²/PTE
- Statistics file: Per-version χ² and PTE for modes 1-6 and 1-20
