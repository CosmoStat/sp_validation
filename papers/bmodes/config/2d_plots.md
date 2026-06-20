# 2D Plotting Specification

Shared styling for PTE heatmaps, covariance matrices, and 2D diagnostic plots.

## Purpose

Visualize parameter-space coverage (PTE grids) and correlation structure (covariance matrices). Emphasis on identifying outliers and structure.

## A&A Figure Dimensions

Standard widths (inches):
- **Single-column**: 3.54 (88mm)
- **Double-column**: 7.24 (180mm)

Square heatmaps use single-column width (3.54 × 3.54). Larger matrices may use double-column.

## Config References

| Element | Config Key |
|---------|------------|
| PTE thresholds | `statistics.pte_healthy_range` |

## PTE Heatmaps

Grid search results showing PTE as function of two parameters.

### Colormap

`vlag` — diverging, emphasizes extremes (low/high PTE).

### Contours

Mark statistical significance at `statistics.pte_healthy_range` bounds — the 2σ equivalent thresholds.

### Fiducial Marker

Hatched rectangle highlighting the chosen fiducial configuration.

## Axis Label Convention for Scale-Cut Heatmaps

When plotting PTE matrices as function of scale cuts, axis labels must show the **included range boundary**:

- **Lower-cut axis (x-axis)**: Show the **lower edge** of the included bin — the minimum scale actually included in the analysis
- **Upper-cut axis (y-axis)**: Show the **upper edge** of the included bin — the maximum scale actually included in the analysis

This convention ensures the tick labels directly answer "what range is included in this analysis?"

### Implementation

**Config space** (angular scales): `theta_grid` is bin edges (21 values for 20 bins).
- x-axis (θ_min): use `theta_grid[i]` (lower edge)
- y-axis (θ_max): use `theta_grid[i+1]` (upper edge)

**Harmonic space** (multipoles): `ell` from FITS is bin centers. Compute edges from binning config:
```python
def compute_ell_edges(lmin, lmax, n_bins, power=0.5):
    start = np.power(lmin, power)
    end = np.power(lmax, power)
    return np.power(np.linspace(start, end, n_bins + 1), 1 / power)
```
- x-axis (ℓ_min): use `ell_edges[i]` (lower edge)
- y-axis (ℓ_max): use `ell_edges[i+1]` (upper edge)

## Covariance Matrices

Visualize E/B mode covariance structure.

### Colormap

`icefire` — symmetric diverging for positive/negative correlations.

### Structure

Show block separation between E-modes, B-modes, and ambiguous modes. Highlight cross-correlations between mode types.
