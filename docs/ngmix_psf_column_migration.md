# ShapePipe-v2 column-grammar migration (shapepipe → sp_validation)

**Status:** code-complete on `migrate/ngmix-psf-column-names` (draft PR
[#201](https://github.com/CosmoStat/sp_validation/pull/201)). Every shape-column
read in the live package, configs, calibration scripts, paper figures, and
notebooks uses the ShapePipe-v2 grammar; the σ→T units change and the
`spread_model` removal are in; the dead `galsim` estimator path is removed; and
the suite is green against synthetic catalogues carrying the new columns.

**No code work is left waiting on a regenerated catalogue.** The one *value*
change — `*_PSF_ORIG` now holds a true original-PSF fit (shapepipe#749) rather
than the reconvolved-kernel alias the old columns silently held — is a straight
column rename at the code level and is already in place; the code does not care
that the numbers moved. All a v2 catalogue enables is a *look-at-the-numbers*
sanity check (do the α-leakage / size-ratio cuts still behave), which is analysis,
not code, and does not gate the PR. **The real merge gate is cutover timing:**
merging this branch makes `develop` *require* v2 columns and stop reading today's
catalogues, so #201 should land together with — or just after —
[shapepipe#761](https://github.com/CosmoStat/shapepipe/pull/761)→#741 and the
first v2 catalogue.

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
| `NGMIX_MOM_FAIL` | `NGMIX_MCAL_TYPES_FAIL` | **renamed + semantics change** (see below) |

`NGMIX_MOM_FAIL` → `NGMIX_MCAL_TYPES_FAIL` is more than a rename: in ngmix v1 the
column counted moments-initial-guess failures from `get_guess`, which no longer
exists in v2, so the producer reused the slot for a *failed-metacal-types* count.
sp_validation only ever cuts on it as `== 0` (keep objects with no failure), and
that cut stays correct — but the underlying failure mode changed, so if the
post-cut galaxy count looks off against a regenerated v2 catalogue, this is the
first line to check. The cut lives in `galaxy.classification_galaxy_ngmix`; the
column is also carried through `params.add_cols_pre_cal` and every
`config/calibration/mask_v1.X.*.yaml`.

### ngmix — reconvolved PSF (metacal kernel; value-safe rename)

| Old | New | Note |
|---|---|---|
| `NGMIX_Tpsf_{shear}` | `NGMIX_T_PSF_RECONV_{shear}` | same value (the `T/Tpsf` size-ratio cut) |

### ngmix — original PSF (value change — shapepipe#749 fix — *not a code blocker*)

`*_PSF_ORIG` now carries a *true* fit to the original image PSF, no longer the
reconvolved-kernel alias the old `ELL_PSFo`/`T_PSFo` columns silently held. The
rename is a straight column rename in sp_validation and is correct as-is — the
code does not care that the numbers moved. The only thing a regenerated catalogue
buys is a *look-at-the-numbers* check that the α-leakage / size-ratio cuts still
behave; that is analysis, not code, and it does not gate this PR (see Status).

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
removed, and the `square_size` key is gone from both param builders
(`rho_tau.get_params_rho_tau`, `cosmo_val/compute_theory_cov.py`). The parameter
is also removed from `shear_psf_leakage`'s `build_cat_to_compute_{rho,tau}` and
`CovTauTh` ([PR #27](https://github.com/CosmoStat/shear_psf_leakage/pull/27)), so
sp_validation no longer *passes* it — the two migrations are coordinated. (Passing
`square_size=False` into the post-#27 leakage handlers would raise `TypeError`;
dropping the argument here is what keeps the container green once #27 lands.)

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
- **Notebooks** — `cosmo_inference/notebooks/cfis_analysis.ipynb`
  (`E1/E2_PSF_HSM` → `HSM_G1/G2_PSF`); a sweep of all tracked notebooks found no
  other old tokens.
- **Tests** — `src/sp_validation/tests/{test_calibration,test_cosmo_val}.py`
  (synthetic catalogues + configs updated in lock-step; this is the migration's
  internal-consistency check).

## galsim estimator path — removed as dead code

**ShapePipe cannot produce galsim shapes.** There is no galsim shape-measurement
runner in the pipeline — `ngmix_runner` measures shapes and writes `NGMIX_*`, but
no galsim analogue exists. The only trace of galsim-as-shapes is a dormant
serialization hook in `make_cat` (`SHAPE_MEASUREMENT_TYPE=galsim` reads shapes from
an optional 5th input catalogue that nothing upstream fills), and every production
config sets `SHAPE_MEASUREMENT_TYPE=ngmix` with galsim commented out. shapepipe#761
did rename the `GALSIM_*` family onto the grammar (`GALSIM_GAL_ELL_*` /
`GALSIM_*_SIGMA_*` → scalar `GALSIM_G1/G2_*`, `GALSIM_T*`), but that is cheap
serialization symmetry, not a live capability.

On the consumer side the galsim reader was **dead and already broken**: `shape` is
hardcoded to `"ngmix"`, `extract_info.py` raises for any other value, nothing
outside `scratch/` instantiates `metacal(prefix="GALSIM")` or calls
`classification_galaxy_galsim`, and the shared
`col_1p = f"{prefix}_T_PSF_RECONV_1P"` read in `metacal._read_data` never matched
the galsim producer output (`GALSIM_T_PSF_*`, not `..._T_PSF_RECONV_*`). Carrying
an untestable, already-broken path onto the new grammar is a worse end state than
deleting it, so this branch **removes** it:

- `calibration.metacal._read_data_galsim`, the `prefix == "GALSIM"` dispatch
  branch (now `else: raise` — unknown prefixes fail loudly), and the two galsim
  ellipticity-sign flips in `_shear_response` / `_selection_response`;
- `galaxy.classification_galaxy_galsim`;
- the `sh == "galsim"` branch in `catalog.get_snr` (now `else: raise`);
- the unused `shape_method` argument on `get_calibrated_quantities` /
  `get_calibrated_m_c`, and the `galsim` mentions in `params.py` / `extract_info.py`.

ngmix is the sole estimator sp_validation supports; the `prefix` parameter stays
(it names the column family and could serve a future `NGMIXm` moments family).
To revive galsim shapes, the capability must return **end-to-end**: a galsim
shape-measurement runner in ShapePipe (so the columns are producible at all), then
a migrated-and-tested consumer here — restore commit `e0eaa9f`, rename onto the
grammar, and add coverage — not the dead stub that was removed.

## Coordinated repos

- **`shear_psf_leakage`** ([PR #27](https://github.com/CosmoStat/shear_psf_leakage/pull/27),
  "Adopt ShapePipe-v2 HSM column grammar, retire square_size") is the sibling
  consumer migration. It lands on the same HSM grammar
  (`HSM_G1/G2_{PSF,STAR}`, `HSM_T_*`, `HSM_FLAG_*`) and removed the `square_size`
  parameter from `build_cat_to_compute_{rho,tau}` and `CovTauTh`; this PR drops the
  matching argument, so the two land together (see "`square_size` is retired").
- **shapepipe#761** (producer) still renames the `GALSIM_*` family onto the grammar
  for columns nothing can create. If the goal is to simplify the grammar,
  retiring that serialization is a producer-side follow-up worth raising there.

— Claude on behalf of Cail
