# ξ Cosmology Paper B-mode Reporting

Spec for B-mode validation reporting in the configuration-space cosmology paper (Goh et al., unions_2d_shear_xi).

Depends on: [pure_eb_data_vector](pure_eb_data_vector.md), [cosebis_version_comparison](cosebis_version_comparison.md), [covariance_blind_consistency](covariance_blind_consistency.md)

## Scope

This spec defines what B-mode evidence appears in the config-space paper (Goh et al.). The B-modes paper (Daley et al.) contains the full version comparison and methodological details.

## Reporting Choices

### Catalog Version

Report only the **fiducial catalog** (with leakage correction, see `fiducial.version` in config). Version comparisons belong in the B-modes paper.

### COSEBIS Mode Count

Use **n=6** modes, not n=20. Rationale:
- Fewer modes = more conservative test (less prone to noise fluctuations)
- n=6 captures the dominant B-mode signal at small scales
- Consistent with scale cuts that exclude small and large angular scales

### Statistics to Report

Report both full-range and fiducial scale cut PTEs for both statistics. All PTEs are the **minimum across blinds** (conservative).

| Statistic | Macro |
|-----------|-------|
| COSEBIS full PTE | `\cosebisfullPte` |
| COSEBIS fiducial PTE | `\cosebisfiducialPte` |
| Joint ξ±^B full PTE | `\ebfullPte` |
| Joint ξ±^B fiducial PTE | `\ebfiducialPte` |

Values read from `evidence.json` at build time; see `generate_paper_macros.py`.

The joint test combines ξ+^B and ξ-^B using the full cross-covariance matrix.

### Scale Cuts

Scale cuts from config (`fiducial.fiducial_xip_scale_cut`, `fiducial.fiducial_xim_scale_cut`):

| Correlation | Min (arcmin) | Max (arcmin) |
|-------------|--------------|--------------|
| ξ+^B | 12 | 83 |
| ξ-^B | 12 | 83 |
| COSEBIS | 12 | 83 |

## Text Requirements

The E/B mode section should:
1. State that significant B-modes appear at extreme scales (motivating scale cuts)
2. Report passing PTEs at fiducial scale cuts
3. Reference the B-modes paper for methodology and version comparison
4. Use auto-generated macros (never hardcode values)

## Blinding and Covariance

The cosmological parameters (Ωm, σ8) are blinded with three independent blinds (A, B, C). Blinding affects the theoretical ξ± predictions, which propagate into the CosmoCov semi-analytical covariance matrices.

### Covariance Variation Between Blinds

From `covariance_blind_consistency`:
- ξ+ covariance diagonals vary by up to **9%** between blinds
- ξ- covariance diagonals vary by up to **8.6%** between blinds
- All blinds pass the 10% consistency threshold

### PTE Variation Between Blinds

The covariance variations propagate into PTE estimates differently for the two statistics. Example values (illustrative, see evidence.json for current):

**Pure E/B (fiducial scale cuts):**

| Blind | ξ+^B PTE | ξ-^B PTE | Joint PTE |
|-------|----------|----------|-----------|
| A | 0.48 | 0.10 | 0.28 |
| B | 0.49 | 0.10 | 0.31 |
| C | 0.50 | 0.08 | 0.29 |
| **Δ** | ~0.02 | ~0.02 | ~0.03 |

**COSEBIS (fiducial scale cuts, n=6):**

| Blind | PTE |
|-------|-----|
| A, B, C | ~0.29 |
| **Δ** | <0.001 |

The near-identical COSEBIS PTEs reflect the compressed information in mode space — the integration over angular scales averages out the blind-dependent covariance variations. Pure E/B PTEs show more sensitivity (ΔPTE ≈ 0.03) because the test operates directly on angular bins where covariance differences are localized.

### Reporting Strategy

Report the **minimum PTE across blinds** as the conservative estimate. This ensures reported PTEs remain valid regardless of which blind is eventually unblinded.

**Additional macros for blinding discussion:**

| Macro | Description |
|-------|-------------|
| `\covXipMaxDev` | Max ξ+ covariance deviation between blinds |
| `\covXimMaxDev` | Max ξ- covariance deviation between blinds |
| `\ebJointPteDelta` | Joint PTE variation across blinds (max − min) |

Values read from `covariance_blind_consistency/evidence.json` at build time.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Fiducial version | `fiducial.version` |
| COSEBIS modes | `cosebis.mode_subsets` (use n=6 subset) |
| Scale cuts | `fiducial.fiducial_min_scale`, `fiducial.fiducial_max_scale` |
