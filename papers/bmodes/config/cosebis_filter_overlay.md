# COSEBIS Filter Overlay

## Claim

The COSEBIS harmonic-space filter functions W_n(ell) become increasingly oscillatory
for higher modes n, making them impossible to resolve with 32 coarse bandpower bins.

## Method

Overlay the continuous W_n(ell) filter functions (modes n = 1 through 6) on the
measured 32-bin BB bandpower spectrum from the fiducial catalog. Show both the full
[1', 250'] and fiducial [12', 83'] angular scale cuts.

W_n(ell) computed via cosmo_numba's FFT-log Hankel transform on a dense ell grid
(2000 points). Each filter peak-normalized for visual comparison.

## Config References

| Parameter | Config Key |
|-----------|------------|
| Full theta range | `cosebis.theta_min`, `cosebis.theta_max` |
| Fiducial theta range | `fiducial.fiducial_min_scale`, `fiducial.fiducial_max_scale` |
| Number of ell bins | `cl.n_ell_bins` |

## Evidence

| Metric | Description |
|--------|-------------|
| `nmodes_shown` | Number of COSEBIS modes displayed |
| `n_bandpower_bins` | Number of bandpower bins in data |
| `ell_range` | Effective ell range of the bandpowers |

## Outputs

- `figure.png` — Two-panel overlay: W_n filters + BB data for full and fiducial scale cuts
