# Draft PR: Catalog Path Cleanup & Guardrails

## Summary

- Normalised every `subdir` entry in `notebooks/cosmo_val/cat_config.yaml` so the loader relies on explicit `/n17data/...` locations rather than the old `data_dir` fallback. Shared resources (redshift distributions, common star/psf FITS) now point at their canonical mounts.  
- Updated the Pseudo-`C_\ell` code path in `src/sp_validation/cosmo_val.py` to read redshift files directly from the catalog configuration, removing the `data_base_dir` assumption.  
- Added `src/sp_validation/tests/test_catalog_paths.py`, a lightweight validator that resolves each shear/star/psf reference and records a small allow-list for catalogues whose source files are currently unavailable.  
- Kept the additive-bias regression test focused on the catalogues that still have data on disk; the missing releases stay excluded so CI remains green.

## Outstanding Data Gaps

All of the catalogues below are missing their **shear** FITS on the shared mounts (star/psf often missing as well):

1. `SP_v1.0`, `SP_v1.1`, and `SP_matched_MP_v1.0`  
2. The `SP_v1.4` family: `SP_v1.4`, `SP_v1.4_conv`, `SP_v1.4_noalpha`  
3. Every `SP_v1.4-P1+3*` derivative (same missing 2022-era inputs as item 2)

These entries remain in `cat_config.yaml` for provenance, are listed in the validator allow-list, and are intentionally omitted from the additive-bias parameter set. The follow-up PR should either restore their data to `/n17data/...`, retarget them at new releases (e.g. the 2024 v1.4.1 products), or mark them as deprecated.

## Proposed Next Steps

1. **Decide The Fate Of The Legacy Catalogues**  
   - Recover the missing FITS/parquet files, or formally deprecate the versions above (update docs/tests accordingly).

2. **Re-home The v1.4 Variants (if data exist)**  
   - Point each affected block at the correct `/n17data/UNIONS/WL/v1.4.x/<variant>/` directory.  
   - Prefer catalogue-local star/psf fits when the directory is more than a symlink; fall back to the shared root otherwise.  
   - Remove the entries from `EXPECTED_MISSING` and re-enable them in regression tests once their files resolve.

3. **Document The Conventions**  
   - Add a short developer note describing the absolute-path policy, preferred star/psf strategy, and how to run the new path validator before pushing catalog edits.  
   - Optionally list the currently offline datasets so collaborators know what needs recovering.

## Validation Checklist

- `app python -m pytest src/sp_validation/tests/test_catalog_paths.py`  
- `app python -m pytest src/sp_validation/tests/test_cosmo_val.py -k additive_bias`  
- Manual `ls` spot-check of any retargeted catalogue directories to ensure we pick up the intended local star/psf files rather than parent symlinks.
