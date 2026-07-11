---
id: 01KX4M42SJY25YJ6JM2H0DPR3Z
name: 'Smokescreen 1.5.6 + firecrown: verified API facts'
tags:
    - finding
    - sacc
    - blinding
created-at: 2026-07-10T01:41:32.850257146+02:00
updated-at: 2026-07-10T01:41:32.850257146+02:00
outcome: Smokescreen requires py3.12 + extract-blind-merge architecture; firecrown not pip-resolvable (numcosmo_py) → --no-deps + explicit closure; PRD's (S8,Ωm) box needs our own seed→shift translation layer
---

Verified facts about Smokescreen 1.5.6 (PyPI) and firecrown v1.15.1, read from source and PyPI metadata on 2026-07-10. These ground PRs 1, 5, and 6 of [[sp_validation/smokescreen-sacc-prs]].

## Packaging (PR 1)

- **Smokescreen 1.5.6** is on PyPI (name `Smokescreen`), `requires-python >=3.12` — this is where the PRD's py3.12 floor comes from. Deps: astropy≥5.2, cryptography, jsonargparse[signatures]≥4, numpy≥2.2, sacc≥0.12, scipy≥1.9, pytest (sloppy upstream runtime dep, harmless). It imports firecrown at module level (`smokescreen/datavector.py:31`) but does **not declare it** — DESC assumes conda-forge.
- **firecrown is not on PyPI**; conda-forge only. Git tag v1.15.1 pip-installs from source (setuptools-scm), but its declared deps include `numcosmo_py`, which doesn't exist on PyPI → **uv/pip resolution is impossible with deps**. Since ≤v1.12 this has been so. Consequence: firecrown installs `--no-deps` from the git tag, with its actually-needed runtime closure declared explicitly by us (pyccl, sacc, pydantic, …— empirically determined; see env recipe in the PR-1 body). Smokescreen 1.5.6 supports firecrown ≥1.15 via explicit version-gated imports (`datavector.py:34`).
- The container base (shapepipe:develop) is already `python:3.12-slim-bookworm`, so `requires-python = ">=3.12"` aligns pyproject with the actual runtime; no base-image change needed.

## ConcealDataVector mechanics (PR 6)

`ConcealDataVector(cosmo, likelihood, shifts_dict, sacc_data, systm_dict=None, seed="2112", shift_distr="flat")`:

- `likelihood` is a **path to a module file** (or module) exposing firecrown's `build_likelihood(build_parameters)` convention; it receives `NamedParameters({'sacc_data': sacc_data})`. The sampling likelihood stays CosmoSIS; only `compute_theory_vector` is used.
- **`_verify_sacc_consistency` requires `sacc_data.mean` to equal `likelihood.get_data_vector()` element-for-element** (`datavector.py:209-266`) — the SACC file handed to Smokescreen must contain *exactly* the rows the likelihood models, nothing else. A master file carrying COSEBIs + ρ/τ cannot be passed directly. → **extract → blind → merge**: extract the to-be-blinded block (ξ± / pseudo-Cℓ) with its covariance sub-block into a temporary SACC, blind it, merge values back into the master, then re-derive COSEBIs/pure-E/B from blinded ξ±. Same seed ⇒ same hidden cosmology across per-block passes, so fine-grid ξ±, coarse-grid ξ±, and Cℓ blind consistently in separate passes.
- **Seed leak**: `save_concealed_datavector` stamps `metadata['seed_smokescreen'] = seed` (`datavector.py:580`), plus `concealed=True`, creator, timestamp. We strip the seed key (the PRD's "strip it").
- Blinding factor: `calculate_concealing_factor(factor_type="add")` → f = t(hidden) − t(fid); `apply_concealing_to_likelihood_datavec()` adds it to `likelihood.get_data_vector()`.

## Shift semantics — the (S8, Ωm) box needs our own translation layer (PR 6)

`draw_flat_or_deterministic_param_shifts(cosmo, shifts_dict, seed)` (`param_shifts.py:56`):

- Keys are validated against **CCL parameter names** (`cosmo._params`) — `S8` and `Omega_m` are not CCL params, so the PRD's "uniform in (S8, Ωm)" **cannot be expressed directly**.
- Tuple values are **absolute (lo, hi) bounds** (uniform draw of the new value, not a delta); single values are **absolute replacements** despite the docstring claiming FIDUCIAL+SHIFT (`_create_concealed_cosmo` does `dict[k] = shifts[k]`).
- Draw uses global `np.random.seed(string_to_seed(seed))` in `cosmo.to_dict()` key order — deterministic given seed.

Design: our blinding step draws (ΔS8, ΔΩm) itself from a seeded RNG, converts to absolute `{sigma8: S8/√(Ωm/0.3), Omega_c: Ωm − Ωb − Ων}`, and passes those as **deterministic** values to Smokescreen. Seed→shift stays a pure function (reproducible, hash-commitment-friendly); the box is exactly the PRD's.

## Encryption / plaintext deletion (PR 6)

`smokescreen.encryption.encrypt_file(path, save_file=True, keep_original=False)` — Fernet key, writes `<stem>.encrpt` + `<stem>.key` side by side, deletes plaintext when `keep_original=False`. Key lives on the same disk (PRD: "protection is social discipline either way"). `decrypt_file` restores.

## pip-installability: the full story (empirical, 2026-07-10)

Resolution alone wasn't enough — two import-time walls surfaced only when actually running:

- **numpy < 2.5 is required**: firecrown 1.15.1's `DataVector` subclasses `npt.NDArray`, which numpy 2.5 turned into a non-subclassable typing alias → TypeError at import. Bisected to exactly 2.5.0. `[blinding]` carries `numpy>=2.2,<2.5`; 2.4.x verified ABI-clean against the whole compiled stack + fast suite (111 passed; 2 failures are env-shaped: hardcoded `python3.12 -m snakemake`, and candide-path expectations in a fresh worktree).
- **Unpatched firecrown imports NumCosmo** through two non-cosmic-shear paths (eager LSST-bin re-export in `generators/__init__`; cluster likelihoods → lsstdesc-crow → `Ncm.IntegralND` subclass at module load). `scripts/patch_firecrown.py` (in PR #243) fixes both + ships a loud `numcosmo_py` shim; exact-string surgery, version-checked against the v1.15.1 pin, idempotent, self-verifying. Deceptive-green guard: CI's image build now smoke-tests `import sacc; import firecrown.likelihood; import smokescreen`.
- firecrown's real pip closure beyond the base scientific stack is just `lsstdesc-crow` (pydantic/pyyaml/rich/typer arrive with the base stack); the uv-overrides route installs it automatically as a firecrown dep.

**Correction to a subagent claim**: Smokescreen 1.5.6 DOES ship `smokescreen.encryption` (`encrypt_file`/`decrypt_file`, verified in installed site-packages) — an env-report claimed otherwise; the source read above stands.

**Watch item for PR 5** (RESOLVED — see PR 5 landed facts below): firecrown's environment.yml pins `sacc>=2.1,<2.2`; we run sacc 2.4 (PRD + Smokescreen require it). Toy pipeline fine; validate firecrown's SACC internals on real tomographic data in PR 5.

**Shared venv state**: `/automnt/n17data/cdaley/unions/code/sp_validation-worktrees/venv` = `[test,glass,blinding]` pins + numpy 2.4.3 + `patch_firecrown.py` applied. Toy probes in `sp_validation-worktrees/env-probe/` all pass on it (finite 60-elt theory vector; blinded ≠ original; cov untouched; seed visible in metadata pre-strip).

## PR 5 landed facts (2026-07-10, empirical — likelihood + review probes)

- **`ConcealDataVector` consumes `sp_validation.blinding_likelihood` as-is** — driven end-to-end in review: `_load_likelihood` (ModuleType path) → `_verify_sacc_consistency` passes → `calculate_concealing_factor(factor_type="add")` → `apply_concealing_to_likelihood_datavec()`; the applied shift equals the theory difference exactly.
- **Smokescreen never passes `theory_config` through `build_parameters`** — `_load_likelihood` sends only `{'sacc_data': ...}`. The module's `TheoryConfig()` *defaults are the blinding fiducial*; overrides are reachable only from direct callers (tests, CLI). PR 6 sets the fiducial by changing the defaults (one config surface), not by threading overrides through Smokescreen.
- **Watch item resolved**: sacc 2.4 works through firecrown 1.15.1's full `read` + `compute_theory_vector` on tomographic-shaped `sacc_io` files despite firecrown's `sacc<2.2` environment.yml pin — no issue surfaced anywhere in PR 5's 19 tests or the review probes.
- **CAMB↔CCL floor** (single-bin synthetic fixture, 12 θ ∈ [5,250]′, HMCode2020+feedback, σ8-matched): ξ+ 0.21% / ξ− 0.10%; tolerances 0.5%/1.0%. σ8-matching is load-bearing — nominal A_s leaves σ8 3% off and blows the comparison to ~9–10%. The `halofit_version` string must match on both stacks (mead2020 vs mead2020_feedback differ by several % at k≳1).
- **CCLFactory routing**: `camb_extra_params` requires `creation_mode=PURE_CCL_MODE` (+ default `BOLTZMANN_CAMB` transfer); in DEFAULT mode firecrown/CCL already default to halofit-nonlinear. `firecrown.modeling_tools` is the non-deprecated import path for `CCLFactory`/`PoweSpecAmplitudeParameter` (member `.AS` — misspelling "Powe" is upstream's).

## PR 6 landed fact: Smokescreen's systematics overlay (found as a real bug, 2026-07-10)

`calculate_concealing_factor` builds BOTH theory vectors via `modify_default_params(firecrown_defaults, cosmo.to_dict(), systm_dict)` — and `ccl.Cosmology.to_dict()` carries no systematics, so **any firecrown default not overridden via `systm_dict` silently rides along** (observed: `ia_bias=0.5` while our fiducial sets 0.0 — the coarse Smokescreen factor diverged ~7% at low θ from the direct-CCL fine/Cℓ paths of the same blind; round-trip tests cannot see it because the same factor cancels). Consequence: **every non-cosmological parameter of the blinding fiducial must be pinned through `ConcealDataVector(systm_dict=…)`** — PR 6 passes `theory.ia_params()` and regression-guards Smokescreen's realized factor against the direct path (<1e-8; fails ~7% if the pin drops).
