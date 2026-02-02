# COSEBIs Data Vector

Depends: [COSEBIS](cosebis.md), [1D Plots](1d_plots.md)
Method: [COSEBIS](cosebis.md)
Plotting: [1D Plots](1d_plots.md)

## Claim

COSEBIS B-modes at fiducial version (`fiducial.version`) are consistent with zero across the full angular range and at fiducial scale cuts.

## Blind Handling

Uses fiducial blind from `config["fiducial"]["blind"]`. COSEBIS B_n data vectors are identical across blinds; covariances vary with blind via n(z)-dependent theoretical predictions. Statistical evidence (PTEs) is in [Config-Space PTE Matrices](config_space_pte_matrices.md).

## Config References

| Parameter | Config Key |
|-----------|------------|
| Fiducial version | `fiducial.version` |
| Fiducial scale cut | `fiducial.fiducial_min_scale` to `fiducial.fiducial_max_scale` |
| Number of modes | `fiducial.nmodes` |

## Evidence

This claim produces visualizations only. Statistical evidence (PTEs) is in [Config-Space PTE Matrices](config_space_pte_matrices.md).

| Metric | Description |
|--------|-------------|
| `fiducial_scale_cut` | Angular range for fiducial |
| `full_scale_cut` | Full angular range |
| `nmodes` | Number of COSEBIS modes |

## Outputs

- `figure.png` — Single-panel figure showing $B_n / \sigma_n$ for fiducial version
  - Both scale cuts (fiducial and full) overplotted with different colors
  - Error bars are unity by construction (normalized)
  - Paper figure for main text B-mode validation

## Notes

This is the paper-ready data vector figure. For multi-version comparison, see [COSEBIS Version Comparison](cosebis_version_comparison.md).
