# cosmo_val/ — cosmology-validation code + config

The side-by-side home for cosmology-validation code and its configuration:
rho/tau statistics, E-/B-mode decomposition, PSF-leakage diagnostics, and the
catalogue-comparison plots that vet a calibrated shear catalogue. Promoted out
of the old `notebooks/` so the code and the config it reads live together.

What belongs here:

- **Validation drivers and config** — the runners and the `*.yaml` catalogue
  configuration they consume, kept next to each other.

What does *not* belong here:

- **Reusable library code** lives in `src/sp_validation/`; import it from here
  rather than copying it.
- **Cosmology inference** (CosmoSIS / CosmoCov) is its sibling, `cosmo_inference/`.
- **The orchestrated, multi-run analysis** lives in `workflow/`, which wraps
  these diagnostics as Snakemake rules and writes products to `results/`.
