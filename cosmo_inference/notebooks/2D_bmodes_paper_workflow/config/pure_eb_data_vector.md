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

PTE values for B-mode null tests at two scale ranges, using fiducial blind (A):

1. **Fiducial scale cuts**: Angular range used for cosmological inference
2. **Full theta range**: All measured angular bins (no cuts)

For each range, report:
- xi+^B PTE
- xi-^B PTE
- Joint [xi+^B, xi-^B] PTE using full cross-covariance between components

Fiducial PTEs should fall within healthy range for null hypothesis consistency. Full-range PTEs may show tension at scales excluded from analysis.

| Metric | Description |
|--------|-------------|
| `fiducial.pte_xip_B` | xi+ B-mode PTE at fiducial cuts |
| `fiducial.pte_xim_B` | xi- B-mode PTE at fiducial cuts |
| `fiducial.pte_joint` | Joint PTE at fiducial cuts |
| `full.pte_xip_B` | xi+ B-mode PTE, full range |
| `full.pte_xim_B` | xi- B-mode PTE, full range |
| `full.pte_joint` | Joint PTE, full range |

## Outputs

- `figure.png` -- Pure E/B decomposition showing xi+^B and xi-^B consistent with zero

Figure shows:
- 2x2 layout: top row (xi^tot, xi^E), bottom row (xi^amb, xi^B)
- Each panel shows xi+ (filled markers) and xi- (open markers)
- Color coding: total (black), E-modes (teal), ambiguous (purple), B-modes (crimson)
- Excluded scale regions shaded gray (outside fiducial range)
