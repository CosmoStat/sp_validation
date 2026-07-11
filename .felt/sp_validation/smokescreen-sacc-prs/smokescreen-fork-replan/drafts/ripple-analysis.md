# Ripple analysis: how the fork ruling changes PRD #241

The ruling forks Smokescreen to `UNIONS-WL/Smokescreen`, gives it a `theory_fn` protocol with a shipped default CCL backend, and sidelines firecrown to an inherited, lazy-imported, unsupported path (never installed by us). This document states, per PRD §7 row and per existing draft PR, what survives, what changes, what is deleted, and what is new. Fork changes are prefixed **PR 0.1**; sp_validation-side rows keep the PRD's numbering.

Draft-PR numbers below (#243, #245, #249, #250, #251, #253, #255) refer to the *existing draft PRs* against the old PRD; they are PR references, not the issue numbers the board will assign. New board issues are unnumbered here (nothing is posted yet) — see the issue-tree draft. **Draft-PR numbers are not sequential with PRD rows:** row 4 (migration) = #251, row 5 (cross-check) = #250 — the two are inverted. This matters for `Closes #N` wiring; read the number, not the position.

**This document is retrospective — it is allowed to describe the delta.** But it directly seeds the per-PR PRDs, which must assume *nothing* about the journey. So each PR section below separates the **end-state spec** (what the PRD author lifts) from the **delta rationale** (why it changed — context only, never transcribed into a PRD). Delta language ("becomes", "dissolves", "no longer a workaround") lives only in the rationale blocks.

## Top line

- **New work, first in the stack:** PR 0.1 — fork Smokescreen to UNIONS-WL, give it a theory-callable protocol plus a shipped default CCL backend, and a fixed pure draw. Firecrown is not deleted: it stays as an inherited, lazy-imported, unsupported path — never imported at module level, never declared, not installed, tested, or maintained by us (no CI job, no patch script, no numpy ceiling). sp_validation never installs it.
- **The blinding dependency simplifies, but not by "transitive inheritance."** Stock Smokescreen `import`s both `pyccl` (`datavector.py:29`) and `firecrown` (`datavector.py:31`) at module level yet declares *neither* in `Requires-Dist` (verified: `smokescreen-1.5.6` METADATA declares astropy, cryptography, jsonargparse, numpy≥2.2, sacc≥0.12, scipy — no pyccl, no firecrown; DESC assumes conda-forge). So neither dep "arrives transitively" today. **PR 0.1's packaging job is to fix this**: the fork *declares and uses* `pyccl` (its default CCL backend imports it) and makes the firecrown import lazy so a clean install never needs firecrown. After PR 0.1, `cryptography` and `sacc>=0.12` arrive from the fork's `Requires-Dist` (both already declared) and `pyccl` arrives because the fork now declares it — but only because PR 0.1 rewrites the packaging. This is a **precondition, not an inheritance**: the "do not re-pin" instruction in the sp_validation deps PR is only sound *after* PR 0.1 has been verified to preserve cryptography/sacc and add pyccl in the fork's `pyproject`.
- **Everything the SACC layout contract touches is untouched by the ruling** — PRs 2, 3, 4, 7 do not care where theory comes from.
- **The firecrown likelihood (row 5) is deleted;** its CAMB↔CCL cross-check is demoted to a test inside the blinding PR (row 6), not a standalone PR.
- **The blinding PR (row 6) is reworked** onto the fork's protocol: our draw becomes a fixed CCL-native draw in the fork, the direct-CCL fine/Cℓ paths become theory backends, and the `systm_dict` IA-pin bug vanishes with firecrown.

## Ordering constraint (load-bearing)

Stock Smokescreen — even freshly forked, before PR 0.1's backend refactor — still `import pyccl` (`datavector.py:29`) and `import firecrown` (`datavector.py:31`) at module level and `ImportError`s on any clean install lacking them. A fork install is therefore **unusable until PR 0.1 makes the firecrown import lazy and declares pyccl.** Consequence for the board: the sp_validation deps PR (row 1) cannot pin a fork SHA prior to PR 0.1's firecrown-lazy-import + packaging commits, or it transitively drags firecrown back in. "sp_validation never installs firecrown" is the *end state after PR 0.1*, not a property of the mere fork.

## Row-by-row

| PRD §7 | Old draft PR | Under the fork | Change class |
|---|---|---|---|
| — | — | **PR 0.1** fork + `theory_fn` protocol + default CCL backend + lazy firecrown (`UNIONS-WL/Smokescreen`) | **NEW** |
| 1 — dependencies | #243 | Drop firecrown, `--no-deps` closure, `patch_firecrown.py`, numpy<2.5 ceiling entirely. Raise Python floor `>=3.11` → `>=3.12` (from Smokescreen itself). Blinding deps go CORE: pinned `git+https` fork URL + `pyccl` (already core) + `cryptography`; sacc is core. No `[blinding]` extra. | **SIMPLIFIED** |
| 2 — `sacc_io` writers | #245 | Untouched. SACC layout contract is theory-agnostic. | **AS-IS** |
| 3 — converters | #249 | Untouched. | **AS-IS** |
| 4 — cosmo_val migration | #251 | Untouched. | **AS-IS** |
| 5 — firecrown likelihood + CAMB↔CCL | #250 | Firecrown likelihood **deleted**. CAMB↔CCL cross-check demoted to a test folded into the blinding PR (row 6) — no standalone PR. | **RETIRED (deliverable struck; cross-check folded into row 6)** |
| 6 — Smokescreen blinding | #253 | Reworked onto the fork protocol: our draw → fork's fixed CCL-native draw; three shift paths → three sp_validation `theory_fn` backends (overriding the fork's default CCL backend); `systm_dict` IA-pin bug gone (our path is a plain CCL callable, never firecrown's defaults overlay); CAMB↔CCL cross-check folded in as a test. Custody scheme unchanged. | **REWORKED** |
| 7 — native SACC likelihood | #255 | Untouched (CosmoSIS-side, never involved firecrown). | **AS-IS** |

## What each PR becomes

### PR 0.1 — Fork and adapt Smokescreen (NEW, first)
Repo: `UNIONS-WL/Smokescreen`. This is the highest-risk PR in the stack and is **decomposed now**, not conditionally: it bundles a protocol-plus-default-backend-plus-draw refactor, firecrown lazy-import sidelining with fork-test migration, and packaging/CI/docs — the packaging part is the board's critical-path unblock for sp_validation. The decomposition below is the plan, filed as sub-issues under `ISSUE-FORK`.

**End-state spec (what the fork exposes):**
- **Theory-backend protocol + default CCL backend.** `theory_fn(cosmo_params) -> np.ndarray` aligned to the SACC rows is the theory entry point; any callable satisfies it (e.g. a ΔΣ emulator). The fork ships a **built-in default CCL backend** so a standard cosmic-shear SACC file blinds out of the box, and power users override with their own callable. `import smokescreen` imports no theory backend — pyccl is imported inside the default-backend module at construction, firecrown only along its lazy path.
- **Firecrown lazy, not deleted.** The firecrown integration is retained from upstream but made optional and lazy: no module-level firecrown import, and it is not installed, tested, supported, or maintained by us — no firecrown-defaults overlay on the default path, no patch script, no numpy ceiling. Docs mark it inherited-and-unsupported.
- **Fixed pure draw.** The shift draw uses a local `default_rng` (order-independent, no global seed) and speaks CCL-native parameter names (e.g. `sigma8`, `Omega_c`). No injectable `draw_fn` abstraction — arbitrary-parameter-space indirection is deferred. The uniform amplitude envelope is calibrated to an equivalent S8 amplitude on the sp_validation side.
- **Declared packaging.** The fork's `pyproject` declares `pyccl` (used by the default CCL backend) alongside the inherited `cryptography`/`sacc>=0.12`. Install identity is a pinned git tag/SHA — no rename, no PyPI publish (see below).
- **Fork CI** runs a backend smoke-test (`import smokescreen` + a synthetic-callable blind) as the required gate.
- **Fork docs** document the pip-only install path: `pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>` — no conda, no PyPI — and note the firecrown path is inherited-and-unsupported. README/docs updated to match.

**Decomposition (two sub-PRs under ISSUE-FORK):**
1. **Protocol + default CCL backend + draw refactor + firecrown sidelining + fork-test migration** — `datavector.py` / `param_shifts.py` / a new backend module: introduce the `theory_fn` protocol and the default CCL backend replacing the module-level firecrown import (`:31`) and the `ParamsMap`-through-likelihood path in `calculate_concealing_factor`; make the firecrown import lazy (retain the path, sideline it); fix the draw (local `default_rng`, order-independent, CCL-native names); migrate DESC's test suite (which assumes module-level firecrown) onto the synthetic/CCL backends. The draw fix folds in here — with the injectable abstraction dropped it is a small change coupled to the same `datavector` rewrite, not a standalone concern.
2. **Packaging + install identity + CI + docs** — settle install identity as a pinned `git+https` tag/SHA (no rename, no PyPI); declare `pyccl`; re-point fork CI; write the pip-only install docs (firecrown noted inherited-and-unsupported). **This sub-PR unblocks sp_validation's deps PR and should not wait on review of the large protocol refactor** — it is why the decomposition exists.

*Delta rationale (context only, not for the PRD):* today `datavector.py:31` imports firecrown at module level and `calculate_concealing_factor` pushes a firecrown `ParamsMap` through the likelihood; the protocol makes the paper's "only needs a module conforming to the firecrown interface" claim true in its general reading. The global-seed/`cosmo.to_dict()`-order draw is why PR #253 drew its own hidden cosmology as a workaround — that workaround dissolves into the fork. The drafts previously coined "unions-smokescreen"; that name presumed a PyPI publish the ruling did not mandate — the ruling settles identity as a pinned git tag/SHA. Long-term, Cail will talk to Arthur (upstream maintainer) about PRing changes back, so the fork diff is kept small and clean to preserve upstream-PR-ability — which is why firecrown is sidelined lazily rather than amputated: a smaller diff that does not antagonize the maintainer by ripping out his integration.

### PR 1 (#243) — Dependencies (SIMPLIFIED)
**End-state spec:**
- Blinding deps are **core**, not an extra: the pinned `git+https://github.com/UNIONS-WL/Smokescreen@<tag>` fork URL, `pyccl` (already core), and `cryptography`. `sacc>=0.12` is promoted to a top-level dep (it is not one today). There is no `[blinding]` extra. Do not re-pin `cryptography`/`sacc` in sp_validation *provided PR 0.1's fork `pyproject` is verified to keep them in `Requires-Dist`* — they then arrive from the fork.
- Python floor raised `>=3.11` → `>=3.12` (Smokescreen's own `requires-python`). **Verify against the actual Dockerfile `FROM` line** — the api-facts fiber records the base as `python:3.12-slim-bookworm` (shapepipe:develop), so the floor aligns with the existing base and no base-image change is needed; confirm the tag before asserting it in the PR.
- No numpy ceiling anywhere; the compiled stack is unconstrained.

*Delta rationale (context only):* deleted from the old PR — firecrown git-tag pin, `--no-deps` install + explicit runtime closure, `scripts/patch_firecrown.py` (deleted outright, not migrated — sp_validation never installs firecrown, and the fork's inherited firecrown path is lazy and needs no patching), the `numpy<2.5` ceiling and its CI guard, the firecrown/numcosmo import smoke-test, and the `[blinding]` extra itself (blinding deps promoted to core).

**Container ripple.** The install source moves from `smokescreen 1.5.6` (PyPI) to the pinned fork URL — a Dockerfile + `uv.lock` change. The deploy workflow (`.github/workflows/deploy-image.yml`) builds and runs the test suite inside the freshly built image before publish, so the fork install is exercised there — the container/`uv.lock` change lands *with* this PR. **This PR cannot pin a fork SHA earlier than PR 0.1's firecrown-lazy-import + packaging commits** (see ordering constraint above).

### PR 2 (#245) — `sacc_io` writers (AS-IS)
No change. The two-file layout amendment (`{version}.sacc` + `{version}_xi_fine.sacc`), custom ρ/τ/pure-EB types, insertion-order covariance assembly all stand.

### PR 3 (#249) — Converters (AS-IS)
No change. SACC→2pt-FITS byte-compare and SACC↔OneCovariance glue are unaffected by the theory backend.

### PR 4 (#251) — cosmo_val migration (AS-IS)
No change. Writers-to-SACC and Snakemake part-assembly are format work.

### PR 5 (#250) — firecrown likelihood + CAMB↔CCL (RETIRED as a PR)
This row is retired as a standalone PR. The minimal firecrown likelihood is deleted outright — nothing in the required path builds one. The CAMB↔CCL cross-check it carried never needed firecrown and moves into the blinding PR (row 6) as a test — see that section for its spec. There is no PR #250 successor in the plan.

*Delta rationale (context only):* the cross-check was demoted from its own PRD to a test folded into blinding because it is a theory-consistency guard on the same CCL backend the blinding PR introduces, not an independent deliverable. `prd-camb-ccl-crosscheck.md` is retired.

### PR 6 (#253) — Smokescreen blinding (REWORKED)
**End-state spec (kept from the landed design):**
- Custody = §4(b): OS-entropy seed → blind → publish sha256(seed) + config digest → encrypt seed+truths (`smokescreen.encryption`, no firecrown dep) → strip `seed_smokescreen`, stamp `concealed=True` + blind label. Unblind verifies hash before subtracting.
- Derived stats re-derived, never shifted (COSEBIs, pure-E/B re-run through pipeline estimators on blinded ξ±); covariances never change.
- Two-tracer direct paths; commitment binds (seed, config); pure-EB seam follows the pipeline's edge-based bounds.

**End-state spec (on the fork protocol):**
- The amplitude-shift envelope is expressed through the fork's fixed CCL-native draw (order-independent, local `default_rng`), calibrated on the sp_validation side to an equivalent S8 amplitude.
- Three theory backends through one `ConcealDataVector` call: coarse ξ±, fine-grid ξ±, pseudo-Cℓ (W @ ΔCℓ_EE), each a `theory_fn` callable supplied by this PR (overriding the fork's built-in default CCL backend, which does not know the master layout or IA config).
- Cross-backend consistency: the same hidden cosmology through two theory callables agrees to machine precision (~1e-10).
- **CAMB↔CCL cross-check (folded-in test):** a theory-consistency test — same cosmology, same n(z), ξ± on our θ grid within tolerance. It must settle the σ8-for-CCL vs A_s-for-CAMB amplitude convention (σ8-matching is load-bearing: nominal A_s leaves σ8 ~3% off and blows the comparison to ~9–10% — verified, api-facts PR-5 facts), and `halofit_version` strings must match on both stacks (mead2020 vs mead2020_feedback differ several % at k≳1). Observed floor on the single-bin synthetic fixture: ξ+ 0.21% / ξ− 0.10% against 0.5%/1.0% tolerances.
- Acceptance: B-mode estimators unchanged under blinding on mocks — unchanged to the estimator's numerical floor. **TODO for the blinding-PRD author (do not lift this line verbatim):** the ΔBₙ/Bₙ figure is a hole the PRD must fill — cite it from the run that produces it, or state the criterion qualitatively; never carry an unsourced number. The acceptance criterion is not "implementable cold" until this is resolved.
- Acceptance: the fork blinds a real sp_validation `sacc_io` file end-to-end (integration line).

*Delta rationale (context only):* our `draw_hidden_cosmology(seed, config)` is no longer a private sp_validation workaround around Smokescreen's replace-not-add global-seed semantics. The direct-CCL fine/Cℓ code stops being a *bypass* of the likelihood and becomes a first-class backend — the "fine grid can't pass through firecrown's ConstGaussian densification" constraint dissolves. The `systm_dict` IA-pin bug (firecrown's `modify_default_params` overlaying un-overridden defaults — IA amplitude riding at 0.5 vs our 0.0, a ~7% low-θ divergence per api-facts) is gone from our path: firecrown is off the default flow (lazy, unsupported, never on our blinding path), and our `theory_fn` backends are plain CCL callables that compute exactly our fiducial; the regression guard is unneeded. The CAMB↔CCL cross-check folds in here because it guards the same CCL backend this PR introduces.

### PR 7 (#255) — Native SACC likelihood (AS-IS)
No change. CosmoSIS-side `SaccLikeUnions(SaccClLikelihood)` shim, χ² equality with the PR-3 converter path. Never involved firecrown.

## Net delta

- **+1 repo** in the effort: `UNIONS-WL/Smokescreen`. Adding it to the board README and getting a `UNIONS-WL/Smokescreen` issue onto `UNIONS-WL/projects/1` is itself work — filed as an explicit precursor (see issue-tree).
- **+1 PR** (0.1) on the fork, **decomposed into two sub-PRs** (protocol+draw+firecrown-sidelining+fork-tests / packaging+CI+docs).
- **-1 deliverable:** the firecrown likelihood.
- **-1 PR:** the standalone CAMB↔CCL cross-check PR (#250) is retired; the cross-check survives as a test folded into the blinding PR.
- **1 PR reworked** (6), **1 PR simplified** (1); **4 PRs unchanged** (2, 3, 4, 7).
- **Deleted from sp_validation:** the firecrown likelihood module, the numpy ceiling, the firecrown `--no-deps` closure, `patch_firecrown.py`, the `systm_dict` IA-pin and its regression guard, the `[blinding]` extra (blinding deps promoted to core).
- **Sidelined in the fork (not deleted):** firecrown becomes lazy-imported, inherited, and unsupported — no module-level import, no CI job, no numpy ceiling, no patch script. **Added to the fork:** a built-in default CCL backend (pyccl declared and used).
- **New in sp_validation deps (core):** the pinned `git+https` fork URL, `cryptography`, and `sacc>=0.12` (promoted to top-level), alongside the pre-existing `pyccl`.
