# BB Covariance Blind Independence

Method: [Covariance](covariance.md), [Pure E/B](pure_eb.md), [COSEBIS](cosebis.md), [Cl](cl.md)

## Claim

BB covariances computed via **analytic propagation** are independent of the A/B/C n(z) realisation, while BB covariances computed via **MC sampling** inherit noise that varies across realisations.

Each realisation is its own catalogue entry, identical to the mock version's but for its n(z): `fiducial.nz_realisations` maps the labels A/B/C to those entries (`SP_v1.4.6.3_{A,B,C}_leak_corr`).

## Evidence

Diagonal ratios relative to realisation A for both B-modes and E-modes:
- Pure E/B: `cov(ξ+^B)`, `cov(ξ-^B)` vs `cov(ξ+^E)`, `cov(ξ-^E)`
- COSEBIS: `cov(B_n)` vs `cov(E_n)`
- Harmonic: `cov(C_ℓ^BB)` vs `cov(C_ℓ^EE)`

**Metrics:**
1. Diagonal ratio plot — ratio of B/A and C/A diagonals, report min/max
2. Comparison of BB vs EE deviations across methods

## Config References

| Parameter | Config Key |
|-----------|------------|
| Version | `fiducial.version` |
| n(z) realisations | `fiducial.nz_realisations` |
| Scale range | `fiducial.min_sep` to `fiducial.max_sep` |
| Bins | `fiducial.nbins` |
| COSEBIS nmodes | `fiducial.nmodes` |
| COSEBIS θ range | `cosebis.theta_min` to `cosebis.theta_max` |
| n_ell_bins | `cl.n_ell_bins` |

## Outputs

- `evidence.json` — ratios, deviations, comparison statistics
- `figure.png` — multi-panel ratio plot comparing BB vs EE stability
