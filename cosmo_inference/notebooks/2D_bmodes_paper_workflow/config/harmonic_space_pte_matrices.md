# Harmonic-Space PTE Matrices

Depends: [Harmonic-Space Power Spectra](cl.md), [2D Plots](2d_plots.md)
Method: [Harmonic-Space Power Spectra](cl.md)
Plotting: [2D Plots](2d_plots.md)

## Claim

Harmonic-space B-mode PTEs are consistent with noise at fiducial multipole range for the fiducial catalog version. PTE heatmaps across all (ell_min, ell_max) combinations show where B-modes become significant. The appendix presents all three catalog versions (v1.4.5, v1.4.6, v1.4.8) for comparison.

Fiducial scale cuts: ell_min=300, ell_max=1600 (from Paper II, Goh et al.). Full B-mode test range: 50 < ell < 2000.

## Blind Handling

The C_ℓ^BB data vector is computed from catalog-level shear maps and is identical across blinds A, B, C.

Per-blind 32-bin covariances are computed using blind-specific n(z) distributions. The reported PTE at fiducial is the minimum across blinds A, B, C — if any blind shows significance, we report it. This conservative approach is consistent with configuration-space statistics.

Evidence includes both the minimum PTE (`pte_at_fiducial`) and per-blind PTEs (`pte_at_fiducial_A`, `pte_at_fiducial_B`, `pte_at_fiducial_C`).

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
| `{version}.pte_at_fiducial` | Minimum C_l^BB PTE across blinds at fiducial multipole range [300, 1600] |
| `{version}.pte_at_fiducial_A` | C_l^BB PTE for blind A at fiducial multipole range |
| `{version}.pte_at_fiducial_B` | C_l^BB PTE for blind B at fiducial multipole range |
| `{version}.pte_at_fiducial_C` | C_l^BB PTE for blind C at fiducial multipole range |
| `{version}.pte_at_full_range` | Minimum C_l^BB PTE across blinds at full multipole range |
| `{version}.pte_at_full_range_A` | C_l^BB PTE for blind A at full multipole range |
| `{version}.pte_at_full_range_B` | C_l^BB PTE for blind B at full multipole range |
| `{version}.pte_at_full_range_C` | C_l^BB PTE for blind C at full multipole range |
| `{version}.n_evaluated` | Number of (ell_min, ell_max) pairs |
| `{version}.ell_range` | [ell_min, ell_max] of full range |
| `{version}.fiducial_ell_range` | [ell_min, ell_max] of fiducial range |
| `{version}.n_ell_bins` | Number of multipole bins |

## Outputs

Main and appendix figures show leak_corr versions. Second appendix figure shows uncorrected version for comparison.

- `figure_fiducial.png` — PTE heatmap for fiducial version (main text)
- `figure_appendix.png` — 3-panel composite for all leak_corr versions (v1.4.5, v1.4.6, v1.4.8) (appendix)
- `figure_appendix_uncorrected.png` — 3-panel composite for uncorrected versions (appendix)

Heatmaps use vlag colormap [0, 1] with contours at 0.05/0.95. Fiducial multipole range marked with hatched rectangle.
