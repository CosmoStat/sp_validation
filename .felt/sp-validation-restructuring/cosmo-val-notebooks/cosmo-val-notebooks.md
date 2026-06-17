---
id: 01KVAS2NAWA972AQCJ5WH8MMTG
name: 'Convert cosmo_val notebooks: lib-extract + scratch'
tags:
    - sp-validation
    - reorg
created-at: 2026-06-17T12:32:12.124718213Z
updated-at: 2026-06-17T12:37:18.735012128Z
outcome: |-
    5 cosmo_val notebooks resolved by the rule "reasonably-reusable code -> library,
    the rest -> scratch scripts." Three generic helpers lifted into
    src/sp_validation/basic.py (chi2_and_pte, corr_from_cov, cov_from_one_covariance);
    all five notebooks converted to jupytext percent-light .py under scratch/guerrini/
    (magics guarded, paths preserved verbatim). The namaster covariance toolkit moved
    with them and a latent broken import (utils_cosmo_val -> rho_tau) was repaired.
---

The cosmo_val notebooks were relocated wholesale during phase-2 (the module was
promoted from `notebooks/cosmo_val/` to top-level `cosmo_val/`, notebooks riding
along untouched) — never actually converted. Cail flagged this: analysis modules
should be notebook-free; reusable code should land in the library, the rest in
`scratch/`. The house style for "notebook-free but cell-runnable" is jupytext
percent-light `.py` with guarded magics (`run_cosmo_val.py` is the template).

**Move-map (intentional, for the PR #197 report):**

Lifted to library — `src/sp_validation/basic.py`:
- `chi2_and_pte(data_vector, cov, verbose=False)` — χ²/reduced-χ²/PTE (was a local def in compute_pte_cell).
- `corr_from_cov(cov)` — correlation from covariance (was a local def in one_covariance).
- `cov_from_one_covariance(cov_one_cov, gaussian=True)` — parse OneCovariance `covariance_list` output (was `get_cov_from_one_cov` in one_covariance).

Converted to percent-light scripts under `scratch/guerrini/`:
- `compute_pte_cell.py`, `one_covariance.py` (import the lifted helpers + `SquareRootScale` from `rho_tau`)
- `plot_comparison.py`, `get_prior_leakage.py`
- `exploration.py` + `namaster_utils.py` (the `investigation namaster/` bundle)

Deleted: `cosmo_val/*.ipynb` (5) and the `cosmo_val/investigation namaster/` directory.

**Flagged for Sacha:** `namaster_utils.py` is a cohesive ~200-line NaMaster
covariance toolkit that overlaps `cosmo_val/harmonic_covariance_gaussian_sims.py`.
It's a real library candidate, but promoting it properly means consolidating the
two and adding tests — an API decision in Sacha's domain — so it stays in scratch
for now rather than forking the covariance API behind his back. The repair of its
broken `utils_cosmo_val` import is a latent stale reference the phase-2 move missed
(nothing in the package imported it, so the guard net never caught it).
