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

## Covariance Matrices

Visualize E/B mode covariance structure.

### Colormap

`icefire` — symmetric diverging for positive/negative correlations.

### Structure

Show block separation between E-modes, B-modes, and ambiguous modes. Highlight cross-correlations between mode types.
