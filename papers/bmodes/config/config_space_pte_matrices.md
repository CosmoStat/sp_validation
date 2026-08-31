# Configuration-Space PTE Matrices

Depends: [Pure E/B](pure_eb.md), [COSEBIS](cosebis.md), [2D Plots](2d_plots.md)
Method: [Pure E/B](pure_eb.md), [COSEBIS](cosebis.md)
Plotting: [2D Plots](2d_plots.md)

## Claim

Fiducial angular scale cuts are justified by PTE heatmaps across all (theta_min, theta_max) combinations. The main text presents a 1x3 composite for the fiducial version showing xi+^B, xi-^B, and COSEBIS B_n. The appendix presents an Nx3 composite showing all catalog versions (from `config.versions`) with each of the three B-mode statistics. Rows represent versions (labeled via `config.plotting.version_labels`), columns represent statistics.

Scale cuts from `fiducial.fiducial_xip_scale_cut` and `fiducial.fiducial_xim_scale_cut`. COSEBIS uses the same unified range.

## Blind Handling

Uses fiducial blind from `config["fiducial"]["blind"]`. Data vectors (ξ+^B, ξ-^B, COSEBIS B_n) are identical across blinds; covariances vary with blind via n(z)-dependent theoretical predictions.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Versions | `versions` |
| Fiducial version | `fiducial.version` |
| xi+ scale cut | `fiducial.fiducial_xip_scale_cut` |
| xi- scale cut | `fiducial.fiducial_xim_scale_cut` |
| COSEBIS scale cut | `fiducial.fiducial_min_scale`, `fiducial.fiducial_max_scale` |

## Evidence

Per-version statistics for each statistic:

| Metric | Description |
|--------|-------------|
| `{version}.role` | "fiducial" or "appendix" |
| `{version}.xip_stats.pte_at_fiducial` | xi+^B PTE at fiducial scale cut |
| `{version}.xip_stats.pte_at_full_range` | xi+^B PTE at full theta range |
| `{version}.xim_stats.pte_at_fiducial` | xi-^B PTE at fiducial scale cut |
| `{version}.xim_stats.pte_at_full_range` | xi-^B PTE at full theta range |
| `{version}.cosebis_stats.pte_at_fiducial` | COSEBIS B_n PTE at fiducial |
| `{version}.cosebis_stats.pte_at_full_range` | COSEBIS B_n PTE at full theta range |

## Outputs

- `figure_fiducial.png` — 1x3 composite for fiducial version (main text)
  - Single row: xi+^B, xi-^B, COSEBIS B_n
  - Y-axis label on leftmost panel
  - Single shared colorbar on right

- `figure_appendix.png` — Nx3 composite for all versions (appendix)
  - Rows: All versions from `config.versions` with labels from `config.plotting.version_labels`
  - Columns: xi+^B, xi-^B, COSEBIS B_n
  - Y-axis label on each row's leftmost panel
  - Version labels on right side of each row
  - Single shared colorbar on right

Each panel uses a discrete PTE colormap (`make_pte_colormap` from `plotting_utils.py`) with solid blue below 0.05, solid red above 0.95, and a gradient between. No contour overlays. Fiducial scale cut marked with a plain black-edged rectangle (no hatching).
