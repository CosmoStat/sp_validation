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

This claim produces visualizations only. Statistical evidence (PTEs) is in [Config-Space PTE Matrices](config_space_pte_matrices.md).

| Metric | Description |
|--------|-------------|
| `scale_cuts` | Angular ranges shown |
| `versions_plotted` | Catalog versions included |

## Outputs

Main figure shows leak_corr catalog versions from `config.versions` for catalog evolution comparison.

- `figure_stacked.png` — Two-panel figure showing $B_n / \sigma_n$ (catalog evolution)
  - Top: Full range (no scale cuts)
  - Bottom: Fiducial scale cut
  - Error bars are unity by construction (normalized)
  - Legend labels from `config.version_labels`
