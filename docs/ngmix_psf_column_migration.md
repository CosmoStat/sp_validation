# ShapePipe-v2 column-grammar migration (shapepipe → sp_validation)

**Status:** code staged on `migrate/ngmix-psf-column-names` (draft PR
[#201](https://github.com/CosmoStat/sp_validation/pull/201)). The renames, the
σ→T units change and the `spread_model` removal are complete and the test suite
is green against synthetic catalogues carrying the new columns. The one item
that genuinely needs a *regenerated* catalogue — re-validating the `*_PSF_ORIG`
value change — is deferred until [CosmoStat/shapepipe#761](https://github.com/CosmoStat/shapepipe/pull/761)
lands in #741 and a v2 catalogue is produced.

shapepipe#761 turns the shape-measurement output into **one column grammar for
the whole catalogue**: every estimator names its outputs
`ESTIMATOR_COMPONENT[_ERR][_OBJECT]_SHEAR` (uppercase), stores a single size
`T = 2σ²` sourced from `cs_util.size`, and splits ellipticity into named scalar
components. The galaxy is the implicit default object and carries **no token**
(`NGMIX_G1_NOSHEAR`, never `..._GAL_...`). This document is the authoritative
old→new map for the sp_validation consumer side.

## What is *not* a value change (safe renames)

Both shape estimators now store native **`g`** (reduced shear). HSM always stored
galsim's `observed_shape.g1/.g2` under the `E*` names — name and value disagreed;
#761 renamed the columns to match the value (`HSM_G1/G2_*`), moving **zero
numbers**. So there is **no e→g conversion anywhere in this migration** — the ρ/τ
leakage code (`shear_psf_leakage`) reads `g` straight into treecorr, which is what
it was always implicitly validated against. `cs_util.shape` exists for whichever
repo needs a genuine e↔g conversion; sp_validation does not.

## Column map

`{shear} ∈ {NOSHEAR, 1P, 1M, 2P, 2M}` (uppercase). Moments branch prefixes `NGMIXm_`.

### ngmix — galaxy (implicit object, no token)

| Old | New | Note |
|---|---|---|
| `NGMIX_ELL_{shear}` (2-vec) | `NGMIX_G1_{shear}`, `NGMIX_G2_{shear}` | split into named scalars |
| `NGMIX_ELL_ERR_{shear}` (2-vec) | `NGMIX_G1_ERR_{shear}`, `NGMIX_G2_ERR_{shear}` | split; value-safe |
| `NGMIX_T_{shear}` / `_T_ERR_` | unchanged | no `_GAL` token added |
| `NGMIX_FLUX_{shear}` / `_ERR` | unchanged | |
| `NGMIX_FLAGS_{shear}`, `NGMIX_SNR_{shear}` | unchanged | |
| `NGMIX_MCAL_FLAGS`, `NGMIX_N_EPOCH` | unchanged | OBJECT/SHEAR-less metadata |

### ngmix — reconvolved PSF (metacal kernel; value-safe rename)

| Old | New | Note |
|---|---|---|
| `NGMIX_Tpsf_{shear}` | `NGMIX_T_PSF_RECONV_{shear}` | same value (the `T/Tpsf` size-ratio cut) |

### ngmix — original PSF (⚠ **value change** — shapepipe#749 fix)

`*_PSF_ORIG` now carries a *true* fit to the original image PSF, no longer the
reconvolved-kernel alias the old `ELL_PSFo`/`T_PSFo` columns silently held. A
mechanical rename is correct for the *names*; the *numbers* move, so any size-ratio
or leakage cut fed by these must be re-validated against a regenerated catalogue.

| Old | New |
|---|---|
| `NGMIX_ELL_PSFo_{shear}_0` (or `[:, 0]`) | `NGMIX_G1_PSF_ORIG_{shear}` |
| `NGMIX_ELL_PSFo_{shear}_1` (or `[:, 1]`) | `NGMIX_G2_PSF_ORIG_{shear}` |
| `NGMIX_T_PSFo_{shear}` | `NGMIX_T_PSF_ORIG_{shear}` |

### HSM — star / PSF validation catalogue

`SIGMA_*_HSM` → `HSM_T_*` is a **units change**: the new column stores the area
`T = 2σ²` directly (the producer applied `sigma_to_T`). Downstream must stop
squaring — the stored value is already `T`.

| Old | New | Note |
|---|---|---|
| `E1_PSF_HSM` / `E2_PSF_HSM` | `HSM_G1_PSF` / `HSM_G2_PSF` | native `g`, pure rename |
| `E1_STAR_HSM` / `E2_STAR_HSM` | `HSM_G1_STAR` / `HSM_G2_STAR` | native `g`, pure rename |
| `SIGMA_PSF_HSM` | `HSM_T_PSF` | **units:** now `T = 2σ²`, an area |
| `SIGMA_STAR_HSM` | `HSM_T_STAR` | **units:** now `T = 2σ²`, an area |
| `FLAG_PSF_HSM` | `HSM_FLAG_PSF` | singular `FLAG` for HSM |
| `FLAG_STAR_HSM` | `HSM_FLAG_STAR` | singular `FLAG` for HSM |

The `piff_T` / DES block in `cat_config.yaml` uses Piff columns, **not** this
grammar — left untouched.

### `spread_model` — removed

shapepipe no longer writes `SPREAD_MODEL` / `SPREADERR_MODEL` / `SPREAD_CLASS`
(Axel's review: these must never be cut on). The `do_spread_model` branch,
`classification_galaxy_base`'s parameter, and `catalog.match_spread_class` are
removed; star/galaxy classification uses the size-based path.

## Size arithmetic routes through `cs_util.size`

`T = 2σ²` is the only stored size. Anything needing σ / FWHM / r50 calls
`cs_util.size` (`T_to_fwhm`, `T_to_r50`, …) — never inline `SIGMA ** 2` or
`sqrt(T)`. The pre-migration `papers/catalog/2025_12_*` hand-rolled
`T = SIGMA_*_HSM ** 2` (a factor-of-2 error even then, since `SIGMA_*_HSM` never
truly held σ) now reads `HSM_T_*` directly.

## `square_size` is retired

The σ→T change makes the old squaring dead: `HSM_T_*` (and DES's `piff_T`) already
hold `T`, so nothing squares. The per-dataset `square_size:` flags in
`cat_config.yaml` are dropped, the `not_square_size` list in `rho_tau.py` is
removed, and both param builders (`rho_tau.get_params_rho_tau`,
`cosmo_val/compute_theory_cov.py`) set `square_size = False`. The `square_size`
argument to `shear_psf_leakage`'s `build_cat_to_compute_{rho,tau}` is now always
`False`; passing a `T`-column with no squaring is correct independently of that
repo's own migration.

## Consumer sites in sp_validation

- **Core** — `src/sp_validation/{galaxy.py, calibration.py, catalog.py}`.
- **Configs** — `cosmo_val/cat_config.yaml` (HSM blocks; DES/piff spared);
  `config/calibration/mask_v1.X.*.yaml` ×10 (`NGMIX_ELL_PSFo_NOSHEAR_0/_1`).
- **rho/τ + covariance** — `src/sp_validation/rho_tau.py`,
  `src/sp_validation/cosmo_val/psf_systematics.py`, `cosmo_val/compute_theory_cov.py`,
  `src/sp_validation/glass_mock.py`.
- **Scripts** — `scripts/calibration/{extract_info,params,calibrate_comprehensive_cat}.py`,
  `scripts/apply_alpha_snr_size_bin.py`, `scripts/examples/demo_*.py`.
- **Papers** — `papers/catalog/2025_12_*`, `papers/catalog/{hist_mag,2025_09_19_alpha_leakage_correction}.py`,
  `papers/harmonic/2025_09_11_psf_leakage_cell.py`.
- **Tests** — `src/sp_validation/tests/{test_calibration,test_cosmo_val}.py`
  (synthetic catalogues + configs updated in lock-step; this is the migration's
  internal-consistency check).

## Deferred / coordinated

- **`*_PSF_ORIG` value re-validation** — blocked on a #761-regenerated catalogue.
  The rename is staged; the "does the size-ratio / leakage cut still behave"
  check waits for real v2 data.
- **`NGMIX_MOM_FAIL`** (read at `galaxy.py`) was not found in the #761 producer
  code; left as-is pending confirmation against a regenerated header.
- **galsim family** (`GALSIM_*` reads in `calibration.py`, `galaxy.py`) is renamed
  on the producer side too but is out of this constitution's ngmix+HSM scope and
  is left untouched here — flagged for a follow-up if the galsim path is live.
- **`shear_psf_leakage`** ρ/τ internals are Sacha Guerrini's separate PR; this
  migration only sets the `square_size=False` interface and passes `T`-columns.

— Claude on behalf of Cail
