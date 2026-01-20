# Pure E/B Mode Decomposition

Schneider et al. decomposition of shear correlation functions into pure E-mode, B-mode, and ambiguous components.

## Purpose

Traditional ξ± mix E and B modes. Pure-mode decomposition cleanly separates:
- **E-modes**: Cosmological lensing signal
- **B-modes**: Should be zero for pure lensing; non-zero indicates systematics
- **Ambiguous modes**: Modes that cannot be uniquely assigned to E or B

## Config References

| Parameter | Config Key | Description |
|-----------|------------|-------------|
| ξ+ scale cut | `fiducial.fiducial_xip_scale_cut` | Angular range for ξ+ |
| ξ- scale cut | `fiducial.fiducial_xim_scale_cut` | Angular range for ξ- |
| version | `fiducial.version` | Catalog version |

## Method

Following Schneider et al. (2002), decompose:
- ξ+(θ) → ξ+^E(θ) + ξ+^B(θ) + ξ+^amb(θ)
- ξ-(θ) → ξ-^E(θ) + ξ-^B(θ) + ξ-^amb(θ)

Uses semi-analytical covariance propagation through the decomposition.

## Data Products

Per-blind precomputed decomposition stored in:
`results/paper_plots/intermediate/{version}_{blind}_pure_eb_semianalytic.npz`

where `{blind}` is A, B, or C. Each NPZ contains:
- `theta`: Angular bins
- `xip_E`, `xim_E`: Pure E-mode components (identical across blinds)
- `xip_B`, `xim_B`: Pure B-mode components (identical across blinds)
- `xip_amb`, `xim_amb`: Ambiguous components (identical across blinds)
- `cov_pure_eb`: Full covariance matrix for decomposed modes (blind-specific via MC propagation)

## Plotting Conventions

- Total ξ± shown with filled markers
- E-modes in teal (secondary, lower alpha)
- B-modes in crimson (primary, unfilled markers, full opacity)
- Ambiguous in purple (secondary, lower alpha)
- Fiducial scale range highlighted
