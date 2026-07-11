---
id: 01KX4MBNPK94DX92BZGCEJR5R3
name: Current data-product map (origin/develop, pre-SACC)
tags:
    - finding
    - sacc
created-at: 2026-07-10T01:45:41.587031688+02:00
updated-at: 2026-07-10T01:45:41.587031688+02:00
outcome: Where every statistic is computed/written today; feeds PR 2-4 design. SACC surface is greenfield.
---

Distilled from an Explore-agent survey of origin/develop (2026-07-10), for the [[sp_validation/smokescreen-sacc-prs]] PRs. File:line anchors are the agent's; re-verify when implementing.

## Writers today (what sacc_io replaces)

| Product | Entry | Format today |
|---|---|---|
| ξ± | `cosmo_val/real_space.py` `RealSpaceMixin.calculate_2pcf` | TreeCorr .txt + optional FITS (BIN1/BIN2/ANGBIN/VALUE/ANG) |
| pure E/B | `cosmo_val/pure_eb.py` → `b_modes.calculate_pure_eb_correlation` + `save_pure_eb_results` | .npz (theta, xip_E/B, xim_E/B, amb, cov) |
| COSEBIs | `cosmo_val/cosebis.py` → `b_modes.calculate_cosebis` + `save_cosebis_results` | .npz (En, Bn, cov, chi2/PTE; multi-scale-cut dict) |
| pseudo-Cℓ | `cosmo_val/pseudo_cl.py` mixin over `pseudo_cl.py` primitives (NaMaster) | FITS `pseudo_cl_{ver}_blind={A,B,C}_{binning}_nbins={n}.fits`; separate `pseudo_cl_cov_…` rule (iNKA + optional OneCovariance) |
| ρ/τ | `cosmo_val/psf_systematics.py` → `rho_tau.py` (shear_psf_leakage) | `rho_stats_{base}.fits`, `tau_stats_{base}.fits` + `.npy` covs (th/jk/sim) |
| n(z) | read-only via `core.py get_redshift` from cat-config `nz.path` | FITS or ASCII; no writer in-package |

## 2pt-FITS assembly (what PR 3 must reproduce byte-for-byte)

`cosmo_inference/scripts/cosmosis_fitting.py`: `nz_to_fits` (NZDATA HDU), `_create_2pt_hdu` (XI_PLUS/XI_MINUS, QUANT=G+R), `covdat_to_fits` (blocked cov with STRT_i offset headers; xi+ | xi− | optionally TAU_0_PLUS/TAU_2_PLUS blocks). Output `cosmosis_{root}[_cell].fits`. **`tests/test_cosmosis_fitting.py` already exercises this assembly with synthetic deterministic inputs** — the natural scaffold for the PR-3 byte-compare; a *real* current output for the compare must be located on candide (papers/ or cosmo_inference data/ trees) at PR-3 time.

## Orchestration & tests

- Snakemake: `workflow/rules/twopoint.smk` — rules `xi`, `xi_highres` (10000-bin fine grid for COSEBIs), `rho_tau_stats`, `pseudo_cl`, `pseudo_cl_cov`; scripts in `workflow/scripts/` wrap the mixins (PR-4 touch points).
- `CosmologyValidation` (`cosmo_val/core.py`): versions + cat-config YAML; `_output_path`; **a `blind` A/B/C label already threads through pseudo-Cℓ naming — vestigial catalogue-level blinding, superseded by data-vector blinding; PR 6 should reconcile.**
- Tests import the full stack (container/venv); fast/slow markers; `test_glass_mock.py` + `glass_mock.py` generate synthetic catalogues → basis for PR-6 mocks.

## Facts that shape the design

- **SACC is greenfield** — one stray `.sacc` string in `test_config_paths_exist.py`, nothing else.
- Everything is single-bin (non-tomographic) today; tomography is the round's target → sacc_io is tomography-native from day one (`source_i` tracers), single-bin as the n=1 case.
- OneCovariance glue half-exists: `statistics.py cov_from_one_covariance` reshapes its flat output (col 9/10 gauss+ng/gauss). PR 3 adds the n(z)→OneCovariance input side. Sacha's `scratch/guerrini/one_covariance.py` overlaps — reserved, don't touch, don't import.
- Fine vs coarse ξ± in ONE sacc file needs a mechanism (likely sacc data-point tags); the PR-2 round-trip tests must verify whatever is chosen survives write→read.

## Real 2pt-FITS references for the PR-3 byte-compare (located 2026-07-10)

- Real data: `cosmo_inference/data/SP_v1.4.6_leak_corr_A_minsep=1.0_maxsep=250.0_nbins=20_npatch=1/cosmosis_SP_v1.4.6_leak_corr_A_….fits` (+ non-leak-corr sibling; also SP_v1.4.5 variants).
- Mocks: hundreds of `cosmo_inference/data/glass_mock_NNNNN/cosmosis_glass_mock_NNNNN.fits`.
