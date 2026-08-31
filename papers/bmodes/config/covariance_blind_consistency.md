# Covariance Blind Consistency

Method: [Covariance](covariance.md)

## Claim

Reporting covariance diagonals are consistent across blinds A, B, and C. Differences arise only from n(z) variations between blinds; survey geometry and cosmology are identical.

## Evidence

Diagonal ratios relative to blind A:
- `diag(Cov_B) / diag(Cov_A)`
- `diag(Cov_C) / diag(Cov_A)`

Computed separately for ξ+ and ξ- blocks.

**Metrics:**
- Max absolute deviation from unity
- Mean absolute deviation
- Pass/fail at 1% and 10% thresholds

## Config References

| Parameter | Config Key |
|-----------|------------|
| Version | `fiducial.version` |
| Scale range | `fiducial.min_sep` to `fiducial.max_sep` |
| Bins | `fiducial.nbins` |

## Outputs

- `evidence.json` — ratios, deviations, pass/fail flags
- `figure.png` — two-panel (ξ+, ξ-) ratio plot with ±1%, ±10% bands
