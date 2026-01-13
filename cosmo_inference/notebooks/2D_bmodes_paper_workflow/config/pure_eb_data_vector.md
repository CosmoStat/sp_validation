# Pure E/B Data Vector

Depends: [Pure E/B](pure_eb.md), [Pure E/B Covariance](pure_eb_covariance.md), [1D Plots](1d_plots.md)
Method: [Pure E/B](pure_eb.md)
Covariance: [Pure E/B Covariance](pure_eb_covariance.md)
Plotting: [1D Plots](1d_plots.md)

## Claim

B-mode signals in UNIONS cosmic shear are consistent with zero at fiducial scale cuts, validating the absence of significant systematic contamination in the shear measurement pipeline.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Fiducial version | `fiducial.version` |
| xi+ scale cut | `fiducial.fiducial_xip_scale_cut` |
| xi- scale cut | `fiducial.fiducial_xim_scale_cut` |
| PTE range | `statistics.pte_healthy_range` |

## Evidence

PTE values for B-mode null tests computed for each blind (A, B, C) at two scale ranges:

1. **Fiducial scale cuts**: Angular range used for cosmological inference
2. **Full theta range**: All measured angular bins (no cuts)

For each range, report:
- xi+^B PTE
- xi-^B PTE
- Joint [xi+^B, xi-^B] PTE using full cross-covariance between components

**Report minimum PTE across blinds** — the most conservative value. Fiducial PTEs should fall within healthy range for null hypothesis consistency. Full-range PTEs may show tension at scales excluded from analysis.

| Metric | Description |
|--------|-------------|
| `fiducial.pte_xip_B_{blind}` | Per-blind xi+ B-mode PTE at fiducial cuts |
| `fiducial.pte_xim_B_{blind}` | Per-blind xi- B-mode PTE at fiducial cuts |
| `fiducial.pte_joint_{blind}` | Per-blind joint PTE at fiducial cuts |
| `fiducial.pte_*_min` | Minimum across blinds (conservative) |
| `full.pte_xip_B_{blind}` | Per-blind xi+ B-mode PTE, full range |
| `full.pte_xim_B_{blind}` | Per-blind xi- B-mode PTE, full range |
| `full.pte_joint_{blind}` | Per-blind joint PTE, full range |
| `full.pte_*_min` | Minimum across blinds (conservative) |

## Outputs

- `figure.png` -- Pure E/B decomposition showing xi+^B and xi-^B consistent with zero

Figure shows:
- 2x2 layout: top row (xi^tot, xi^E), bottom row (xi^amb, xi^B)
- Each panel shows xi+ (filled markers) and xi- (open markers)
- Color coding: total (black), E-modes (teal), ambiguous (purple), B-modes (crimson)
- Excluded scale regions shaded gray (outside fiducial range)
