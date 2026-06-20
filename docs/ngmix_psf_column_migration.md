# ngmix PSF / shape column migration (shapepipe → sp_validation)

**Status:** planning — blocked on [CosmoStat/shapepipe#761](https://github.com/CosmoStat/shapepipe/pull/761)
(which lands in #741, the ngmix v2.0 PR).

shapepipe#761 makes the ngmix shape-measurement output columns a single source of
truth: it restores an explicit fit to the *original* PSF (stored alongside the
metacal-*reconvolved* one), splits the 2-D ellipticity into scalar `G1`/`G2`, and
renames every ngmix column to a consistent `ESTIMATOR_COMPONENT_OBJECT` grammar.
This document records the downstream migration `sp_validation` needs once #761
(and #741) land.

## Column map

`{shear} ∈ {NOSHEAR, 1P, 1M, 2P, 2M}`. The moments branch prefixes `NGMIXm_`.

| Old (current ngmix_v2 catalogue) | New (#761) | Note |
|---|---|---|
| `NGMIX_ELL_{shear}` (2-vector) | `NGMIX_G1_GAL_{shear}`, `NGMIX_G2_GAL_{shear}` | ⚠ **dim change**: 2-D vector → two scalars |
| `NGMIX_ELL_ERR_{shear}` | `NGMIX_G1_ERR_GAL_{shear}`, `NGMIX_G2_ERR_GAL_{shear}` | ⚠ dim change |
| `NGMIX_T_{shear}` | `NGMIX_T_GAL_{shear}` | gains `_GAL` |
| `NGMIX_T_ERR_{shear}` | `NGMIX_T_ERR_GAL_{shear}` | gains `_GAL` |
| `NGMIX_FLUX_{shear}` / `_ERR` | `NGMIX_FLUX_GAL_{shear}` / `_ERR_GAL` | gains `_GAL` |
| `NGMIX_FLAGS_{shear}` | `NGMIX_FLAGS_GAL_{shear}` | gains `_GAL` |
| `NGMIX_Tpsf_{shear}` | `NGMIX_T_PSF_RECONV_{shear}` | value identical (still the reconvolved-kernel size — the `Tgal/Tpsf` cut) |
| `NGMIX_ELL_PSFo_{shear}_0/_1` | `NGMIX_G1_PSF_ORIG_{shear}`, `NGMIX_G2_PSF_ORIG_{shear}` | ⚠⚠ **value changed** + dim split |
| `NGMIX_T_PSFo_{shear}` | `NGMIX_T_PSF_ORIG_{shear}` | ⚠⚠ **value changed** |
| — | `NGMIX_G1/G2_PSF_RECONV_{shear}` + PSF `_ERR` columns | new |
| `r50`, `r50psf` (+ `_err`) | — | removed (never shipped; use `cs_util.size.T_to_r50(T)`) |
| `NGMIX_MCAL_FLAGS`, `NGMIX_N_EPOCH`, `NGMIX_MOM_FAIL` | unchanged | OBJECT/SHEAR-less metadata |

### Two changes a mechanical `sed` would silently corrupt
1. **`*_PSF_ORIG` now carries a *true* original-PSF fit**, not the reconvolved-kernel
   alias it used to (the shapepipe#749 fix). Any PSF-leakage / α-size analysis reading
   these columns **changes value** — re-validate, don't assume.
2. **`NGMIX_ELL_*` → `G1`/`G2` is a dimensionality change**, not a rename: code that
   indexes `ELL[..., 0/1]` must switch to the two scalar columns.

## Consumer sites in sp_validation (@ develop)

**PSF-leakage / calibration** — the `*_PSFo` value change matters here, re-validate:
- [ ] `config/calibration/mask_v1.X.{2..11}.yaml` (×10) — `NGMIX_ELL_PSFo_NOSHEAR_0/_1` → `NGMIX_G1/G2_PSF_ORIG_NOSHEAR`
- [ ] `src/sp_validation/calibration.py`
- [ ] `src/sp_validation/cat.py`
- [ ] `src/sp_validation/galaxy.py`

**Size cuts / masking** — `Tpsf` rename, value-safe:
- [ ] `cosmo_inference/scripts/masking.py` — `NGMIX_Tpsf_NOSHEAR` → `NGMIX_T_PSF_RECONV_NOSHEAR`
- [ ] `scripts/apply_alpha_snr_size_bin.py`

**Scripts / paper figures** — verify each:
- [ ] `scripts/calibration/{calibrate_comprehensive_cat,extract_info,params}.py`
- [ ] `scripts/examples/demo_{calibrate_minimal_cat,comprehensive_to_minimal_cat}.py`
- [ ] `papers/catalog/{2025_09_19_alpha_leakage_correction,hist_mag}.py`
- [ ] `scratch/kilbinger/plot_binned_quantities.py`

**Unchanged — no action:** `NGMIX_MOM_FAIL`, `NGMIX_MCAL_FLAGS`, `NGMIX_N_EPOCH`.

## Sequencing
1. shapepipe#761 merges into #741 (ngmix v2.0).
2. #741 merges to shapepipe `develop`; a catalogue is regenerated with the new columns.
3. Apply this migration and validate against the regenerated catalogue — in particular,
   confirm the `*_PSF_ORIG` value change behaves as intended for α-leakage, rather than
   assuming a name-only swap.

— Claude on behalf of Cail
