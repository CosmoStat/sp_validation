# COSEBIS Specification

Complete Orthogonal Sets of E/B-Integrals — a model-independent B-mode null test.

## Purpose

COSEBIS compress 2PCF information into discrete modes with clean E/B separation. Unlike band powers, they're complete and orthogonal over a finite angular range. B-modes should be consistent with zero for a pure lensing signal.

## Config References

| Parameter | Config Key | Description |
|-----------|------------|-------------|
| n_modes | `cosebis.nmodes` | Number of COSEBIS modes to compute |
| θ_min | `cosebis.theta_min` | Minimum angular scale (arcmin) |
| θ_max | `cosebis.theta_max` | Maximum angular scale (arcmin) |
| mode_subsets | `cosebis.mode_subsets` | Mode counts for PTE evaluation |
| scale_cut | `fiducial.fiducial_min_scale` to `fiducial.fiducial_max_scale` | Fiducial scale range |
| pte_range | `statistics.pte_healthy_range` | Healthy PTE bounds |

## Integration Grid

High-resolution 2PCF for accurate mode integration:

| Parameter | Config Key |
|-----------|------------|
| min_sep | `fiducial.min_sep_int` |
| max_sep | `fiducial.max_sep_int` |
| nbins | `fiducial.nbins_int` |

## Versions

Compare across catalog versions from `config.versions`:
- Fiducial: `fiducial.version`
- All versions tested for consistency

## Blind Handling

COSEBIS B_n data vectors are identical across blinds A, B, C. Covariances vary with blind via n(z)-dependent theoretical predictions.

**Report minimum PTE across blinds** — the most conservative value.

## Analysis Decisions

- **No Hartlap correction**: Using theoretical (CosmoCov) covariance, not jackknife
- **Conservative scale cuts**: Require full bin containment within θ range
- **PTE thresholds**: Values outside healthy range flag potential systematics

## Outputs

- B-mode plot with fiducial scale cuts
- B-mode plot without scale cuts (full range)
- PTE values for each mode subset
