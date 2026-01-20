# Covariance

Semi-analytical covariance estimation using CosmoCov.

## Purpose

Provides theoretical covariance matrices for ξ± correlation functions. Avoids jackknife noise by propagating analytical Gaussian covariance through the analysis pipeline.

## Method

1. **CosmoCov** computes Gaussian-only covariance for integration binning
2. **Sampling** draws realizations from integration covariance
3. **Propagation** bins samples to reporting scale, propagates through E/B decomposition

Gaussian-only approximation: non-Gaussian contributions computationally prohibitive at 1000 bins.

## Config References

| Parameter | Config Key | Description |
|-----------|------------|-------------|
| Samples | `covariance.n_samples` | MC samples for propagation (default 2000) |
| Ω_m | `covariance.cosmology.Omega_m` | Matter density |
| σ_8 | `covariance.cosmology.sigma_8` | Amplitude of fluctuations |
| n_s | `covariance.cosmology.n_s` | Spectral index |
| h | `covariance.cosmology.h` | Hubble parameter |
| Ω_b | `covariance.cosmology.Omega_b` | Baryon density |
| Mask Cls | `covariance.mask_cls_files` | Per-version mask power spectra |
| Use masked | `covariance.default_masked` | Whether to use masked covariance |

## Survey Properties

Extracted from catalog config (`cat_config.yaml`) per version:
- Area (deg²)
- n_e (gal/arcmin²)
- σ_e (shape noise)

## Binning

| Scale | min | max | bins |
|-------|-----|-----|------|
| Integration | 0.5' | 500' | 1000 |
| Reporting | 1' | 250' | 20 |

## File Naming

```
covariance_{version}_{blind}_{gaussian}_minsep={min}_maxsep={max}_nbins={n}{mask_suffix}_processed.txt
```

- `version`: SP_v1.4.6_leak_corr, etc.
- `blind`: A, B, or C
- `gaussian`: g (Gaussian-only) or ng (non-Gaussian)
- `mask_suffix`: empty or `_masked`

## Multi-Blind Strategy

All B-mode claims compute statistics for each blind (A, B, C) and report the **minimum PTE** — the most conservative value. This ensures robustness to n(z) variations between blinds.

Requires covariances for all three blinds at both:
- **Reporting binning** (20 bins) — for PTE calculation
- **Integration binning** (1000 bins) — for COSEBIS/pure E/B propagation

### Covariance Sensitivity

Covariance diagonals shift by ~10% between blinds due to n(z) dependence of the lensing kernel. See [Covariance Blind Consistency](covariance_blind_consistency.md) for validation.

### PTE Stability

Despite ~10% covariance shifts, PTEs remain stable:

| Statistic | PTE variation (absolute) | Notes |
|-----------|-------------------------|-------|
| COSEBIS B_n | <0.001 | Effectively zero; B-mode signal identical across blinds |
| Pure-mode ξ±^B | ~0.06 | 6 percentage points; covariance-driven, not signal |

The COSEBIS stability confirms that B-mode signal is blind-invariant. Pure-mode variation is larger because per-blind covariances affect the chi-squared normalization, not the data vector.

## Covariance Usage Policy

Official results use specific covariance sources for consistency and correctness:

| Quantity | Source | Gaussian | Binning | Notes |
|----------|--------|----------|---------|-------|
| Total ξ± errors | CosmoCov | ng | 20-bin (reporting) | Per-blind, masked |
| Pure E/B mode errors | MC propagation | g | 1000→20 bin | Conservative: underestimates uncertainty |
| COSEBIS errors | MC propagation | g | 1000→scales | Per scale cut |

**Not used:**
- TreeCorr jackknife covariance — too noisy for official results
- NPZ `cov_xip_xim` field — deprecated (was jackknife)

### Why Gaussian for E/B propagation?

Non-Gaussian covariance at 1000-bin integration scale is computationally prohibitive. Using Gaussian-only **underestimates** the E/B covariance, which **overestimates** B-mode significance. This is conservative for a null test: if B-modes pass with underestimated errors, they would also pass with correct errors.

## Related Specs

- [Pure E/B](pure_eb.md) — uses semi-analytic covariance propagation
- [COSEBIS](cosebis.md) — covariance propagation for bandpowers
- [Covariance Blind Consistency](covariance_blind_consistency.md) — validates diagonal agreement across blinds
