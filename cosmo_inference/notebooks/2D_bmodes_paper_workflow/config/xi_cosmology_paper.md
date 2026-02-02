# ξ Cosmology Paper B-mode Reporting

Spec for B-mode validation reporting in the configuration-space cosmology paper (Goh et al., unions_2d_shear_xi).

Depends on: [pure_eb_data_vector](pure_eb_data_vector.md), [cosebis_version_comparison](cosebis_version_comparison.md), [covariance_blind_consistency](covariance_blind_consistency.md)

## Scope

This spec defines what B-mode evidence appears in Paper II (Goh et al.). Paper III (Daley et al.) contains the full version comparison and methodological details.

## Reporting Choices

### Catalog Version

Report only the **fiducial catalog** (with leakage correction, see `fiducial.version` in config). Version comparisons belong in Paper III.

### COSEBIS Mode Count

Use **n=6** modes, not n=20. Rationale:
- Fewer modes = more conservative test (less prone to noise fluctuations)
- n=6 captures the dominant B-mode signal at small scales
- Consistent with scale cuts that exclude small and large angular scales

### Statistics to Report

Report both full-range and fiducial scale cut PTEs for both statistics. All PTEs are the **minimum across blinds** (conservative).

| Statistic | Macro | Value |
|-----------|-------|-------|
| COSEBIS full PTE | `\cosebisfullPte` | 1.52e-06 |
| COSEBIS fiducial PTE | `\cosebisfiducialPte` | 0.29 |
| Joint ξ±^B full PTE | `\ebfullPte` | 0.048 |
| Joint ξ±^B fiducial PTE | `\ebfiducialPte` | 0.28 |

The joint test combines ξ+^B and ξ-^B using the full cross-covariance matrix.

### Scale Cuts

| Correlation | Min (arcmin) | Max (arcmin) |
|-------------|--------------|--------------|
| ξ+^B | 6 | 85 |
| ξ-^B | 15 | 85 |
| COSEBIS | 6 | 85 |

## Text Requirements

The E/B mode section should:
1. State that significant B-modes appear at extreme scales (motivating scale cuts)
2. Report passing PTEs at fiducial scale cuts
3. Reference Paper III for methodology and version comparison
4. Use auto-generated macros (never hardcode values)

## Blinding and Covariance

The cosmological parameters (Ωm, σ8) are blinded with three independent blinds (A, B, C). Blinding affects the theoretical ξ± predictions, which propagate into the CosmoCov semi-analytical covariance matrices.

### Covariance Variation Between Blinds

From `covariance_blind_consistency`:
- ξ+ covariance diagonals vary by up to **9%** between blinds
- ξ- covariance diagonals vary by up to **8.6%** between blinds
- All blinds pass the 10% consistency threshold

### PTE Variation Between Blinds

The covariance variations propagate into PTE estimates differently for the two statistics:

**Pure E/B (fiducial scale cuts):**

| Blind | ξ+^B PTE | ξ-^B PTE | Joint PTE |
|-------|----------|----------|-----------|
| A | 0.483 | 0.097 | 0.284 |
| B | 0.489 | 0.098 | 0.311 |
| C | 0.502 | 0.083 | 0.288 |
| **Δ (max − min)** | 0.02 | 0.015 | 0.027 |

**COSEBIS (fiducial scale cuts, n=6):**

| Blind | PTE |
|-------|-----|
| A | 0.2927 |
| B | 0.2927 |
| C | 0.2927 |
| **Δ (max − min)** | <0.0001 |

The near-identical COSEBIS PTEs reflect the compressed information in mode space — the integration over angular scales averages out the blind-dependent covariance variations. Pure E/B PTEs show more sensitivity (ΔPTE ≈ 0.03) because the test operates directly on angular bins where covariance differences are localized.

### Reporting Strategy

Report the **minimum PTE across blinds** as the conservative estimate. This ensures reported PTEs remain valid regardless of which blind is eventually unblinded.

**Additional macros for blinding discussion:**

| Macro | Value | Description |
|-------|-------|-------------|
| `\covXipMaxDev` | 9.08% | Max ξ+ covariance deviation between blinds |
| `\covXimMaxDev` | 8.59% | Max ξ- covariance deviation between blinds |
| `\ebJointPteDelta` | 0.027 | Joint PTE variation across blinds (max − min) |

## Config References

| Parameter | Config Key |
|-----------|------------|
| Fiducial version | `fiducial.version` |
| COSEBIS modes | `cosebis.mode_subsets` (use n=6 subset) |
| Scale cuts | `fiducial.fiducial_min_scale`, `fiducial.fiducial_max_scale` |
