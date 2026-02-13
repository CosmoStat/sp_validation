# Harmonic-Space Fiducial

Depends: [Harmonic-Space Power Spectra](cl.md), [2D Plots](2d_plots.md)
Method: [Harmonic-Space Power Spectra](cl.md)

## Claim

The fiducial catalog shows B-mode power spectra consistent with zero across all ell bins, validating the absence of significant systematic contamination in harmonic space. PTEs are reported both for the full ell range and with scale cuts (ell_min=300, ell_max=1600) from the harmonic space paper.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Fiducial version | `fiducial.version` |
| ell bins | `cl.n_ell_bins` |
| ell range | `cl.ell_min`, `cl.ell_max` |
| Scale cuts | From harmonic space paper: ell_min=300, ell_max=1600 |

## Evidence

Power spectra and PTEs for BB and EB components, with and without scale cuts:

| Metric | Description |
|--------|-------------|
| `pte_bb_full` | B-mode PTE, full ell range |
| `pte_eb_full` | E-B cross PTE, full ell range |
| `pte_bb_cut` | B-mode PTE with scale cuts |
| `pte_eb_cut` | E-B cross PTE with scale cuts |
| `ell_min_cut` | Lower scale cut (300) |
| `ell_max_cut` | Upper scale cut (1600) |
| `n_ell_bins` | Number of ell bins |

## Outputs

Produces 9 figures: 1 paper figure + 4 per-version leak-corrected + 4 per-version uncorrected.

**Paper figure (leak-corrected, fiducial version):**
- `figure.png` — Two-panel figure (BB top, EB bottom), no title
- BB: filled circles, EB: unfilled squares
- Data normalized by errors (C_ell / sigma)
- Scale cuts marked with shaded excluded regions
- sqrt(ell) x-axis scaling matches bandpower binning

**Per-version figures (leak-corrected, with title):**
- `figure_v{X.Y.Z}.png` — One per leak-corrected version, with version title

**Per-version figures (uncorrected, with title):**
- `figure_v{X.Y.Z}_uncorrected.png` — One per uncorrected version, with version title
- Labeled "(uncorrected)" in legend
- For validation/comparison purposes, not included in paper
