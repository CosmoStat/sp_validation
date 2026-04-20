# Harmonic-Space Version Comparison

Depends: [Harmonic-Space Power Spectra](cl.md), [2D Plots](2d_plots.md)
Method: [Harmonic-Space Power Spectra](cl.md)
Plotting: [2D Plots](2d_plots.md)

## Claim

B-mode power spectra C_ell^BB are consistent with zero for all leak-corrected versions.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Versions | `versions` |
| ell bins | `cl.n_ell_bins` |
| PTE healthy range | `statistics.pte_healthy_range` |

## Evidence

Per-version B-mode PTEs:

| Metric | Description |
|--------|-------------|
| `{version}.pte_bb` | B-mode power spectrum PTE |
| `{version}.pte_eb` | E-B cross PTE |
| `{version}.chi2_bb` | B-mode chi-squared |
| `{version}.dof` | Degrees of freedom |

## Outputs

- `figure.png` — Two-panel plot (BB top, EB bottom) with all versions overlaid
- Version comparison uses distinct colors/markers
- sqrt(ell) x-axis scaling
