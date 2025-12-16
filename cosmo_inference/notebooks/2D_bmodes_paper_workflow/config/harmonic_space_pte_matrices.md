# Harmonic-Space PTE Matrices

Depends: [Harmonic-Space Power Spectra](cl.md), [2D Plots](2d_plots.md)
Method: [Harmonic-Space Power Spectra](cl.md)
Plotting: [2D Plots](2d_plots.md)

## Claim

Harmonic-space B-mode PTEs are consistent with noise at fiducial multipole range for the fiducial catalog version. PTE heatmaps across all (ell_min, ell_max) combinations show where B-modes become significant. The appendix presents all three catalog versions (v1.4.5, v1.4.6, v1.4.8) for comparison.

Fiducial scale cuts: ell_min=300, ell_max=1600 (from Paper II, Goh et al.). Full B-mode test range: 50 < ell < 2000.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Versions | `versions` |
| Fiducial version | `fiducial.version` |
| Full multipole range | `cl.ell_min`, `cl.ell_max` |
| Fiducial multipole range | `cl.fiducial_ell_min`, `cl.fiducial_ell_max` |
| Number of bins | `cl.n_ell_bins` |

## Evidence

Per-version statistics:

| Metric | Description |
|--------|-------------|
| `{version}.role` | "fiducial" or "appendix" |
| `{version}.pte_at_fiducial` | C_l^BB PTE at fiducial multipole range [300, 1600] |
| `{version}.pte_at_full_range` | C_l^BB PTE at full multipole range |
| `{version}.n_evaluated` | Number of (ell_min, ell_max) pairs |
| `{version}.ell_range` | [ell_min, ell_max] of full range |
| `{version}.fiducial_ell_range` | [ell_min, ell_max] of fiducial range |
| `{version}.n_ell_bins` | Number of multipole bins |

## Outputs

- `figure_fiducial.png` — PTE heatmap for fiducial version (main text)
- `figure_appendix.png` — 3-panel composite for all versions (v1.4.5, v1.4.6, v1.4.8) (appendix)

Heatmaps use vlag colormap [0, 1] with contours at 0.05/0.95. Fiducial multipole range marked with hatched rectangle.
