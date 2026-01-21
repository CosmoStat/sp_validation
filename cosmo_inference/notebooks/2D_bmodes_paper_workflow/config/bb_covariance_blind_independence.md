# BB Covariance Blind Independence

Method: [Covariance](covariance.md), [Pure E/B](pure_eb.md), [COSEBIS](cosebis.md), [Cl](cl.md)

## Claim

BB covariances computed via **analytic propagation** are blind-independent, while BB covariances computed via **MC sampling** inherit noise that varies across blinds.

## Rationale

B-mode covariance theoretically depends only on survey geometry and noise properties, not on the cosmological signal (which is expected to be zero). However, the method of covariance propagation matters:

- **Analytic propagation** (COSEBIS): `Cov_out = T @ Cov_in @ T.T` preserves the blind-independence of noise covariance perfectly.
- **MC propagation** (Pure E/B): Drawing samples from `N(μ, Σ)` then estimating covariance empirically introduces finite-sample variance that depends on the input distribution, which varies per blind.

## Evidence

Diagonal ratios relative to blind A for both B-modes and E-modes:
- Pure E/B: `cov(ξ+^B)`, `cov(ξ-^B)` vs `cov(ξ+^E)`, `cov(ξ-^E)`
- COSEBIS: `cov(B_n)` vs `cov(E_n)`
- Harmonic: `cov(C_ℓ^BB)` vs `cov(C_ℓ^EE)`

**Metrics:**
1. Diagonal ratio plot — ratio of B/A and C/A diagonals, report min/max
2. Comparison of BB vs EE deviations across methods

**Expected results:**
| Method | BB deviation | EE deviation | Notes |
|--------|--------------|--------------|-------|
| COSEBIS (analytic) | <0.01% | ~10% | Analytic propagation |
| Pure E/B (MC) | ~10-13% | ~10-13% | MC noise dominates |
| Harmonic (MC) | ~8% | ~10% | MC noise dominates |

**Pass criteria:**
1. COSEBIS B_n covariance blind-independent (<0.1%)
2. E-mode covariances vary ~10% (sample variance from cosmological signal)

## Config References

| Parameter | Config Key |
|-----------|------------|
| Version | `fiducial.version` |
| Scale range | `fiducial.min_sep` to `fiducial.max_sep` |
| Bins | `fiducial.nbins` |
| COSEBIS nmodes | `fiducial.nmodes` |
| COSEBIS θ range | `cosebis.theta_min` to `cosebis.theta_max` |
| ℓ range | `cl.ell_min` to `cl.ell_max` |
| n_ell_bins | `cl.n_ell_bins` |

## Outputs

- `evidence.json` — ratios, deviations, comparison statistics
- `figure.png` — multi-panel ratio plot comparing BB vs EE stability
