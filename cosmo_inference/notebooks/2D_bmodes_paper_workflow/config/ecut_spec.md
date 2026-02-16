This is your spec for a Ralph loop, a meditative iteration toward a desired state.

## Desired State

Three version comparison figures (pure E/B, COSEBIS, harmonic) showing B-modes for 4
leak-corrected catalog versions side by side:

- v1.4.6 (all galaxies)
- v1.4.6 with |e| < 0.7
- v1.4.11.3 (all galaxies)
- v1.4.11.3 with |e| < 0.7

Each ecut version has its own covariance computed from the filtered catalog's n_eff and
sigma_e (with area inherited from the parent version). The full pipeline runs through
existing rules — ecut versions are just new entries in `config["versions"]` and
`cat_config.yaml`.

**Done when:**
- `snakemake ecut_version_comparisons --dry-run` resolves cleanly (convenience target)
- Filtered catalogs exist at `results/ecut/SP_v1.4.{6,11.3}_ecut07.fits`
- `cat_config.yaml` has correct (non-zero) n_e and sigma_e for both ecut versions
- Existing version comparison rules accept a `{comparison}` wildcard in the output path
  (e.g., `results/claims/{comparison}_pure_eb_version_comparison/`), with an input
  function mapping `paper` → current versions, `ecut` → ecut versions
- `snakemake paper_pure_eb_version_comparison --dry-run` still works (backward compat via convenience target or alias)

**Why:** Calum reports that |e| < 0.7 significantly reduces B-modes in v1.4.11 and v1.4.6,
but without trustable error bars. We need the full pipeline's covariances to quantify this.
DES-Y3 used e < 0.8 to remove stars.

## Context

### How versions flow through the pipeline

Everything is driven by `config["versions"]` in `workflow/config/config.yaml`. Adding a
version there (plus its `cat_config.yaml` entry) makes it flow through all existing rules:
`xi`, `covariance`, `pure_eb_data_vector`, `cosebis_data_vector`, `cl_data_vector`.
The 2PCF and covariance are independent and can run in parallel.

Key resolution functions in `workflow/Snakefile`:
- `get_shear_catalog()` — resolves catalog path from cat_config, strips `_leak_corr`
- `build_redshift_path()` — v1.4.11.x uses v1.4.6's n(z)
- `resolve_covariance_version()` — identity function (each version gets its own covariance)
- Wildcard constraint: `version=r"SP_v[\d.]+(_w_iv)?(_leak_corr)?"` — needs `_ecut\d+`

Version comparison rules in `workflow/rules/claims.smk` (lines 131, 271, 386) use
`VERSIONS_LEAK_CORR` for inputs and derive version lists from config in the scripts.
These should be parameterized to accept a version list via `snakemake.params`, so the
same rules serve both paper and ecut comparisons.

### Covariance depends on survey properties

`get_cat_params()` in `workflow/rules/covariance.smk` (line 4) reads `cov_th.{A, n_e, sigma_e}`
from `cat_config.yaml`. These flow into CosmoCov INI files. Ecut versions need their own
values — same area as parent, but recomputed n_e and sigma_e.

### Catalog filtering

A preprocessing rule writes filtered FITS files (`results/ecut/`). Filter:
`sqrt(e1_leak_corrected² + e2_leak_corrected²) < 0.7`. Only `_leak_corr` versions — the
uncorrected columns can't be consistently filtered to guarantee the same rows.

### Caveats

- **n(z)**: Ellipticity cut preferentially removes faint/noisy objects. We reuse parent
  n(z) — acceptable for B-mode null test, not for cosmological inference.
- **x_offsets**: Paper comparison has 4 versions with specific spacing. Ecut comparison
  also has 4, so same offsets work.

### Key files

| What | Where |
|------|-------|
| Workflow config | `workflow/config/config.yaml` (search `ecut`) |
| Catalog config | `code/sp_validation/notebooks/cosmo_val/cat_config.yaml` (search `ecut07`) |
| Pipeline orchestration | `workflow/Snakefile` (wildcard constraints, version resolution functions) |
| Version comparison rules | `workflow/rules/claims.smk` lines 131, 271, 386 |
| Version comparison scripts | `workflow/scripts/{pure_eb,cosebis,cl}_version_comparison.py` |
| Covariance params | `workflow/rules/covariance.smk` line 4 (`get_cat_params`) |

## Skills

`/snakemake` for DAG operations and job submission.
