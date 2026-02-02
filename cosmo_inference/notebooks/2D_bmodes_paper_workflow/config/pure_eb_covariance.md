# Pure E/B Covariance

Depends: [Pure E/B](pure_eb.md), [Covariance](covariance.md), [2D Plots](2d_plots.md), [Pure E/B Data Vector](pure_eb_data_vector.md), [Pure E/B Version Comparison](pure_eb_version_comparison.md)
Method: [Pure E/B](pure_eb.md), [Covariance](covariance.md)
Plotting: [2D Plots](2d_plots.md)

## Claim

Pure E/B covariance blocks are well-conditioned; ill-conditioning is localized to ambiguous modes. The 6-block correlation structure validates the semi-analytic covariance propagation for B-mode tests.

## Evidence

Block-wise condition numbers for the 120×120 pure E/B covariance (6 blocks of 20 bins each for ξ+/ξ- × E/B/amb):

| Block | Description | Condition Number |
|-------|-------------|------------------|
| ξ_E | ξ+^E and ξ-^E combined | ~10^5 (well-conditioned) |
| ξ_B | ξ+^B and ξ-^B combined | ~10^5 (well-conditioned) |
| ξ_amb | ξ+^amb and ξ-^amb combined | ~10^15 (ill-conditioned) |

**Key metrics:**
- Full matrix positive definite
- E and B blocks stable for PTE calculation
- Ill-conditioning confined to ambiguous modes (expected)

## Config References

| Parameter | Config Key |
|-----------|------------|
| Version | `fiducial.version` |
| Blind | `fiducial.blind` |
| Integration bins | `fiducial.nbins_int` |
| Reporting bins | `fiducial.nbins` |

## Outputs

- `pure_eb_covariance.png` — 6-block correlation matrix heatmap (vlag diverging colormap)
- `evidence.json` — condition numbers, eigenvalue bounds, positive definiteness

## Visualization

Correlation matrix with:
- vlag diverging colormap (−1 to +1)
- Block boundaries marked
- Labels for E/B/amb blocks
