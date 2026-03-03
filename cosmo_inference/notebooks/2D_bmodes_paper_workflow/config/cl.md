# Harmonic-Space Power Spectra

Pseudo-Cl estimation of cosmic shear E/B-mode power spectra using NaMaster.

## Method

Angular power spectra C_ell are the harmonic-space analog of configuration-space correlation functions. For cosmic shear:

- **C_ell^EE**: E-mode auto-power spectrum (cosmological signal)
- **C_ell^BB**: B-mode auto-power spectrum (null test for systematics)
- **C_ell^EB**: E-B cross-power spectrum (should be zero)

Pseudo-Cl estimation accounts for:
1. Survey mask geometry via mode-coupling matrices
2. Bandpower binning (sqrt(ell)-linear spacing)
3. Noise bias subtraction

## Config References

| Parameter | Config Key |
|-----------|------------|
| ell bins | `cl.n_ell_bins` |
| ell range | `cl.ell_min`, `cl.ell_max` |
| Fiducial version | `fiducial.version` |

## Data Source

Pseudo-Cl files computed by Sasha Guerrini using NaMaster:
- `pseudo_cl_{version}.fits` — Power spectrum estimates
- `pseudo_cl_cov_{version}.fits` — Bandpower covariance matrix

Location: `/home/guerrini/sp_validation/notebooks/cosmo_val/output/`

## Comparison to Configuration-Space

Harmonic-space methods integrate over angular scales, averaging out localized B-mode contamination. This explains why:
- v1.4.8 passes harmonic tests but fails configuration-space tests
- PTE healthy fractions are higher in harmonic space (~96%) than configuration space (~67-73%)

The complementary sensitivity makes both approaches valuable for systematic validation.
