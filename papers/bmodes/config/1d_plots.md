# 1D Plotting Specification

Shared styling for version comparisons, B-mode error bar plots, and diagnostic figures.

## Purpose

Consistent visual language across all 1D comparison plots. Fiducial version emphasized, comparison versions subdued. Clear error visualization with minimal clutter.

## A&A Figure Dimensions

Standard widths (inches):
- **Single-column**: 3.54 (88mm)
- **Double-column**: 7.24 (180mm)

All figures use one of these two widths. Height varies by content.

## Config References

| Element | Config Key |
|---------|------------|
| Versions | `versions` |
| Fiducial | `fiducial.version` |
| Palette | `plotting.palette` |
| Fiducial alpha | `plotting.version_alpha.fiducial` |
| Comparison alpha | `plotting.version_alpha.comparison` |
| Marker style | `plotting.markers.style` |
| Line width | `plotting.markers.linewidth` |
| Cap size | `plotting.markers.capsize` |
| X-offsets | `plotting.x_offsets` |

## Visual Design

### Version Styling

Fiducial version at full opacity, comparison versions subdued to create visual hierarchy.

### Markers

Unfilled circles with moderate line weight — clean, distinguishable at small sizes. Error caps sized for visibility without dominating.

### Axes

- **Zero line**: Black reference for null hypothesis
- **Grid**: Subtle guides on both axes
- **Mode highlight**: Background shading when relevant

### Legend

Bottom center, outside plot area. Single column for clarity. Framed for readability against varied backgrounds.
