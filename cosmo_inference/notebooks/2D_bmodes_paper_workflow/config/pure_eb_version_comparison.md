# Pure E/B Version Comparison

Depends: [Pure E/B](pure_eb.md), [1D Plots](1d_plots.md)
Method: [Pure E/B](pure_eb.md)
Plotting: [1D Plots](1d_plots.md)

## Claim

B-mode correlation functions $\xi_{\pm}^B$ are consistent with zero across catalog versions. Total correlation functions $\xi_{\pm}$ show cosmological signal stability. Data outside fiducial scale cuts displayed greyed out.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Versions | `versions` |
| Fiducial | `fiducial.version` |
| Scale cut (xi+) | `fiducial.fiducial_xip_scale_cut` |
| Scale cut (xi-) | `fiducial.fiducial_xim_scale_cut` |

## Evidence

This claim produces visualizations only. Statistical evidence (PTEs) is in [Config Space PTE Matrices](config_space_pte_matrices.md).

| Metric | Description |
|--------|-------------|
| `scale_cuts` | Angular ranges (fiducial cuts shown; excluded data greyed) |
| `versions_plotted` | Catalog versions included |

## Outputs

Main figure shows leak_corr catalog versions from `config.versions` for catalog evolution comparison.

- `figure.png` — Four-panel figure with asymmetric row heights (catalog evolution)
  - Top row (2/3 height): $\xi_+$ (left), $\xi_-$ (right) as $\theta \xi \times 10^4$
  - Bottom row (1/3 height): $\xi_+^B / \sigma$ (left), $\xi_-^B / \sigma$ (right)
  - Data outside fiducial scale cuts shown greyed out
  - B-mode error bars are unity by construction (normalized)
  - Legend labels from `config.version_labels`
