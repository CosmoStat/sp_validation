# COSEBIS Version Comparison

Depends: [COSEBIS](cosebis.md), [1D Plots](1d_plots.md)
Method: [COSEBIS](cosebis.md)
Plotting: [1D Plots](1d_plots.md)

## Claim

B-mode COSEBIS magnitudes are small relative to measurement uncertainty across catalog versions. Visualization shows $B_n / \sigma_n$ (in units of standard deviation from zero) at fiducial and full angular ranges.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Versions | `versions` |
| Fiducial | `fiducial.version` |
| Scale cut | `fiducial.fiducial_min_scale` to `fiducial.fiducial_max_scale` |

## Evidence

This claim produces visualizations only. Statistical evidence (PTEs) is in [COSEBIS PTE Matrix](cosebis_pte_matrix.md).

| Metric | Description |
|--------|-------------|
| `scale_cuts` | Angular ranges shown |
| `versions_plotted` | Catalog versions included |

## Outputs

- `figure_stacked.png` — Two-panel figure showing $B_n / \sigma_n$
  - Top: Full range (no scale cuts)
  - Bottom: Fiducial scale cut
  - Error bars are unity by construction (normalized)
