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

| Parameter | Source | Description |
|-----------|--------|-------------|
| Samples | `config.yaml: covariance.n_samples` | MC samples for propagation (default 2000) |
| Use masked | `config.yaml: covariance.default_masked` | Whether to use masked covariance |
| Cosmology | `Snakefile: PLANCK18` | astropy Planck18 via cs_util.cosmo |
| Mask Cls | `covariance.smk: MASK_CLS_FILES` | Per-version mask power spectra |

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

## Blind Handling

B-mode claims use the fiducial blind from `config["fiducial"]["blind"]`. Covariances are computed for the fiducial blind only.

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

### Footprint Masks

Two footprint masks at nside=4096, generated from the comprehensive catalog with
only spatially-structured cuts (no galaxy selection cuts):

| Mask | Area | Used by |
|------|------|---------|
| Standard footprint | 2894 deg² | v1.4.5, v1.4.6, v1.4.11.3 (and ecut variants) |
| Star-halo footprint | 2517 deg² | v1.4.8 |

Each version gets its own covariance from its own survey properties (A, n_e, sigma_e).
`resolve_covariance_version()` is the identity function — no cross-version covariance sharing.
`MASK_CLS_FILES` (covariance.smk) maps to two mask power spectrum files based on whether
the version is in `STARHALO_VERSIONS`.

## Related Specs

- [Pure E/B](pure_eb.md) — uses semi-analytic covariance propagation
- [COSEBIS](cosebis.md) — covariance propagation for bandpowers
- [Covariance Blind Consistency](covariance_blind_consistency.md) — validates diagonal agreement across blinds
