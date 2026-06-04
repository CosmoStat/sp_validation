# Harmonic-Space PTE Matrices

Depends: [Harmonic-Space Power Spectra](cl.md), [2D Plots](2d_plots.md)
Method: [Harmonic-Space Power Spectra](cl.md)
Plotting: [2D Plots](2d_plots.md)

## Claim

Harmonic-space B-mode PTEs are consistent with noise at fiducial multipole range for the fiducial catalog version. PTE heatmaps across all (ell_min, ell_max) combinations show where B-modes become significant. The appendix presents all catalog versions (from `config.versions`) for comparison.

Fiducial scale cuts from `cl.fiducial_ell_min` and `cl.fiducial_ell_max`. Full B-mode test range spans all multipole bins present in the input pseudo-Cℓ file.

## Blind Handling

Uses fiducial blind from `config["fiducial"]["blind"]`. The C_ℓ^BB data vector is identical across blinds; covariances vary with blind via n(z)-dependent theoretical predictions.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Versions | `versions` |
| Fiducial version | `fiducial.version` |
| Fiducial multipole range | `cl.fiducial_ell_min`, `cl.fiducial_ell_max` |

Note: Number of multipole bins and full multipole range are determined by the input pseudo-Cℓ file, not config.

## Evidence

Per-version statistics:

| Metric | Description |
|--------|-------------|
| `{version}.role` | "fiducial" or "appendix" |
| `{version}.pte_at_fiducial` | C_l^BB PTE at fiducial multipole range |
| `{version}.pte_at_full_range` | C_l^BB PTE at full multipole range |
| `{version}.n_evaluated` | Number of (ell_min, ell_max) pairs |
| `{version}.ell_range` | [ell_min, ell_max] of full range |
| `{version}.fiducial_ell_range` | [ell_min, ell_max] of fiducial range |
| `{version}.n_ell_bins` | Number of multipole bins |

## Outputs

- `figure_fiducial.png` — PTE heatmap for fiducial version (main text)
- `figure_appendix.png` — N-panel composite for all versions from `config.versions` (appendix)

Heatmaps use a discrete PTE colormap (`make_pte_colormap` from `plotting_utils.py`) with solid blue below 0.05, solid red above 0.95, and a gradient between. No contour overlays. Fiducial multipole range marked with a plain black-edged rectangle (no hatching).
