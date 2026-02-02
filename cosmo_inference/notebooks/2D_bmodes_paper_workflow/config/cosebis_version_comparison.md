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

Main figure shows leak_corr catalog versions only (v1.4.5, v1.4.6, v1.4.8) for catalog evolution comparison. Second figure shows correction impact (v1.4.6 leak_corr vs uncorrected).

- `figure_stacked.png` — Two-panel figure showing $B_n / \sigma_n$ (catalog evolution)
  - Top: Full range (no scale cuts)
  - Bottom: Fiducial scale cut
  - Error bars are unity by construction (normalized)
  - Legend: Initial (v1.4.5), Fiducial (v1.4.6), Masked (v1.4.8)

- `figure_correction.png` — Leakage correction impact comparison
  - Same layout as main figure
  - Shows v1.4.6 leak_corr vs v1.4.6 uncorrected
  - Demonstrates impact of leakage correction on COSEBIS B-modes
