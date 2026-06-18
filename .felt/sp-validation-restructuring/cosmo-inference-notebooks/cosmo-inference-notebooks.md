---
id: 01KVD3ZZBZW9S2FDJVR1C5KVKM
name: Migrate cosmo_inference notebooks (on Candide)
status: open
tags:
    - sp-validation
    - reorg
    - todo
created-at: 2026-06-18T10:21:27.296014865Z
updated-at: 2026-06-18T10:21:50.849714039Z
outcome: 'TODO: make cosmo_inference notebook-free like cosmo_val — paper plots → papers/, working notebooks → scratch/scripts, reusable bits → src/. Do it on Candide where the stack runs so each migration can be executed/verified.'
---

When [[cosmo-val-notebooks]] made `cosmo_val/` notebook-free, `cosmo_inference/`
was waved through as "untouched" — but that was an unexamined default, not a
decision (Cail flagged it). It holds **16 notebooks** that need the same
treatment, and the parallel is glaring: the harmonic paper plots already moved
to `papers/harmonic/`, but the configuration-space cosmic-shear plots sitting
next to them did not.

Deferred deliberately to **Candide** (now back up): unlike the cosmo_val pass,
these should be *carefully* migrated with the scientific stack available, so each
moved/converted notebook can actually be run and verified rather than transformed
blind. No rush; correctness over speed.

**Disposition analysis (current `cosmo_inference/` notebooks):**

1. **2D cosmic-shear paper figures → `papers/` (new `cosmic_shear_2d/`, parallel to `papers/harmonic/`).**
   These are publication plots for the configuration-space (ξ±) cosmic-shear paper:
   - `notebooks/2D_cosmic_shear_configuration_plots/` — `S8_om_sigma8_whisker`, `best_fit_xipm`, `contours`, `get_chi2`, `get_chi2_glass_mock`, `get_prior_psf_leakage`, `glass_mock_hist`, `masking`, `nonlin_k_analysis`
   - `notebooks/2D_cosmic_shear_consistency/` — already part `.py` (`check_consistency`, `get_param_values`, `compare_harmonic_scale_cut`) + notebooks `2025_11_13_plot_contours`, `2026_02_11_test_sampling_glass_mocks`
   - `notebooks/2D_cosmic_shear_unblinding/` — already `.py` (`unblinding_party_plots`, `utils`)
   Papers/ keeps notebooks (final-figure assembly), so these can stay notebooks once relocated.

2. **Loose working notebooks → `scratch/` (or percent-light scripts).** Exploratory / chain-retrieval:
   `cfis_analysis`, `cfis_mcmc`, top-level `get_chi2` + `get_chi2_cell`, `notebooks/get_prior_psf_leakage`.
   Note redundancy: three `get_chi2*` notebooks — consolidate.

3. **Reusable bits → `src/sp_validation/`** where any genuinely general helper turns up (same rule as the cosmo_val pass; `cosmo_inference/scripts/` already holds the real pipeline scripts).

Ownership: this is largely Sacha's inference domain — coordinate / let him drive the
judgment calls, especially which `cosmic_shear_2d` notebooks are live paper figures
vs. stale. The PR #197 description should stop calling cosmo_inference "untouched"
and point here instead.
