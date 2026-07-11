# Grounding for the Smokescreen-fork PRD replan (2026-07-10)

Audience: workflow agents drafting the replan documents. Everything here is settled with Cail today; do not relitigate it. Verify code claims against source where paths are given.

## The ruling

1. **Fork Smokescreen to the `UNIONS-WL` GitHub org** (community-facing, not personal). No PR against DESC upstream; PRs go against our fork. Rationale: keep Smokescreen as the community-trusted, community-maintained blinding object (grad students, CosmoStat group, cosmo pipe users — not just sp_validation), while removing what we can't live with.
2. **Drop firecrown entirely** from the required path — it is deleted from the PRD, not demoted. No `--no-deps` install, no `patch_firecrown.py`, no numpy<2.5 ceiling. Firecrown may remain as an optional lazy-import adapter in the fork, but we never install it.
3. **PR 0.1 (new, first): make Smokescreen backend-independent** in the fork:
   - Theory backend protocol: `theory_fn(cosmo_params) -> np.ndarray` aligned to the SACC rows. Firecrown becomes one optional adapter behind lazy imports. This makes the Smokescreen paper's "only needs a module conforming to the firecrown interface" claim actually true in its general reading (today `datavector.py:31` imports firecrown at module level and `calculate_concealing_factor` pushes firecrown `ParamsMap` through the likelihood).
   - Injectable, pure shift draw: replace the global `np.random.seed` + `cosmo.to_dict()`-order-dependent draw in `param_shifts.py` with a local `default_rng`, order-independent, injectable `draw_fn(seed) -> cosmology params` that supports arbitrary parameter spaces — so the PRD's uniform (S8, Ωm) box is expressible natively (S8/Ωm are not CCL params; that inexpressibility is why PR #253 drew its own hidden cosmology as a workaround — that workaround dissolves into the fork).
   - Motivating probe requirement: blinding must work for ANY data vector, e.g. a ΔΣ vector whose theory comes from an emulator — any callable satisfies the protocol.
4. **Blinding stays an optional extra dependency-wise** but the extra shrinks to roughly `smokescreen-fork + pyccl + cryptography` (sacc is core).

## What the fork inherits from the PR #253 design (keep)

- Custody scheme unchanged: OS-entropy seed → blind → publish sha256 commitment (+ config digest) → encrypt seed + truths (`smokescreen.encryption`, no firecrown dep) → strip `seed_smokescreen`, stamp `concealed=True` + blind label. Unblind verifies hash before subtracting.
- Derived statistics re-derived, never shifted (COSEBIs, pure-E/B re-run through pipeline estimators on blinded ξ±); covariances never change.
- Cross-path/backend consistency tests (previously Smokescreen-vs-direct at 1e-10) become backend-vs-backend tests.
- The three shift paths (coarse ξ± / fine ξ± / pseudo-Cℓ = W @ ΔCℓ_EE) unify: all through `ConcealDataVector`, three theory callables. The fine-grid and pseudo-Cℓ direct-CCL code from PR #253 becomes backends instead of bypasses.

## State of the world

- PRD #241 text: `prd241.md` (this directory); comments: `prd241-comments.md`.
- Seven draft PRs exist implementing the OLD PRD: #243 deps, #245 sacc_io, #249 converters, #250 firecrown likelihood + CAMB↔CCL cross-check, #251 cosmo_val migration, #253 blinding, #255 (see repo). PRs 2/3/4 (#245/#249/#251) are expected untouched by the replan — the SACC layout contract doesn't care where theory comes from. #243 simplifies (drop firecrown/patch/numpy ceiling; keep py3.12 floor — that comes from Smokescreen itself). #250's firecrown likelihood is deleted; its CAMB↔CCL cross-check survives (never needed firecrown). #253 restructures around backends and loses `draw_hidden_cosmology` as a private function (moves into the fork). The `systm_dict` IA-pin (firecrown defaults overlay bug) vanishes with firecrown.
- Verified Smokescreen/firecrown facts (source-read + empirical): fiber `sp_validation/smokescreen-sacc-prs/smokescreen-api-facts` — file `.felt/sp_validation/smokescreen-sacc-prs/smokescreen-api-facts/smokescreen-api-facts.md`.
- PR-6 blinding architecture as landed: `.felt/sp_validation/smokescreen-sacc-prs/blinding-design/blinding-design.md`.
- SACC layout contract: `.felt/sp_validation/smokescreen-sacc-prs/sacc-layout-contract/sacc-layout-contract.md`.
- Smokescreen 1.5.6 source: installed in `/automnt/n17data/cdaley/unions/code/sp_validation-worktrees/venv` site-packages (read it there), and on PyPI/GitHub (DESC `LSSTDESC/Smokescreen`).

## Pass-2 amendments (Cail, 2026-07-10 evening — these OVERRIDE anything above or in the pass-1 drafts that conflicts)

1. **Firecrown has ZERO presence — not even an optional adapter.** The fork deletes the firecrown path outright: no lazy adapter, no `patch_firecrown.py` anywhere (fork or sp_validation), no guarded firecrown CI job, no numpy<2.5 ceiling anywhere. Any pass-1 text about a "retained-but-guarded adapter" is void. Tone rule for anything that will ever be GitHub-facing: neutral about firecrown, no editorializing.
2. **Fork docs are a first-class deliverable:** document the pip-only install path — `pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>` — no conda, no PyPI publish (for now). README/docs updated accordingly.
3. **Draw: native pyccl parameters, no injectable-draw_fn abstraction.** Fix the real bugs (local `default_rng`, order-independent draw, no global `np.random.seed`) but the shift specification speaks CCL-native parameter names (e.g. sigma8, Omega_c) with an envelope we calibrate to equivalent S8 amplitude on the sp_validation side. Arbitrary-parameter-space indirection is explicitly deferred ("maybe later something more fancy"). This shrinks fork sub-PR (a); the drafting agents may re-decide the sub-PR decomposition (e.g. fold the draw bugfix into the protocol PR) if that is now cleaner.
4. **No blinding extra — blinding deps go CORE.** With firecrown gone the extra is just the fork + cryptography (pyccl is already core; sacc becomes core in PR 2). Delete the `[blinding]` extra; core deps include the pinned `git+https` fork URL. Keep the py3.12 floor.
5. **Install identity: pinned git tag/SHA, settled — no rename, no PyPI.** Long-term Cail will talk to Arthur (upstream maintainer) about migrating changes back / PRing upstream; keep the fork diff small and clean to preserve upstream-PR-ability.
6. **CAMB↔CCL cross-check is demoted from its own PR to a test**: fold it into the blinding PR's test suite (or an issue), not a standalone PRD. `prd-camb-ccl-crosscheck.md` is retired as a PRD; its verified technical content (σ8-matching load-bearing, halofit_version string match, observed 0.21%/0.10% floor, 0.5%/1.0% tolerances) moves into the blinding PRD's test section.
7. **Delete the cross-repo sacc-pin compatibility section** — it was a firecrown-only concern; Smokescreen itself requires sacc>=0.12 and we run 2.4. (An end-to-end "fork blinds a real sp_validation SACC file" integration test is still welcome as a blinding-PR acceptance line.)
8. PRs 2/3/4/7 confirmed untouched — do not edit their PRDs except where they reference the deleted extra or firecrown.

## Pass-3 amendments (Cail, 2026-07-10 late — these OVERRIDE pass-2 items 1 and the "fork imports no theory backend" ruling where they conflict)

1. **Firecrown code is NOT deleted from the fork.** It stays as an *optional, lazy-imported* path — never imported at module level, never declared as a dependency, **not installed, not tested, not supported, not maintained by us** (no CI job, no patch script, no numpy ceiling in our packaging — those were only ever consequences of installing it). Docs say plainly: the firecrown path is inherited from upstream and unsupported in this fork. Rationale: smaller, merge-able-upstream diff; don't antagonize the upstream maintainer by amputating his integration. sp_validation still never installs firecrown (unchanged).
2. **The fork SHIPS a default CCL theory backend.** pyccl is a declared dependency *and used*: a built-in backend so a casual user can blind a standard cosmic-shear SACC file out of the box without writing a theory callable; power users override with their own `theory_fn` (e.g. ΔΣ emulator). Pass-2's "fork imports no theory backend at all" is void; lazy-import discipline applies to firecrown only. The "maximally minimal import" acceptance criteria (greps asserting no pyccl import) must be replaced accordingly.
3. Everything else from pass 2 stands (native-CCL draw semantics, no extra / deps core, pinned git identity, pip-only docs, crosscheck folded into blinding, PRs 2/3/4/7 untouched).

## Deliverables Cail wants (markdown drafts only — NOTHING is posted to GitHub)

1. **Ripple analysis**: how the ruling changes PRD #241 — row by row, PR by PR (which PRs survive as-is, which change how, what's deleted, what's new).
2. **Issue-tree proposal**: a new umbrella issue for the Smokescreen/blinding stream; sub-issue "fork and adapt Smokescreen"; PRs attached per issue (board model: issues nest, PRs ride Development links, same-repo `Closes #N`). Name each issue, its repo (sp_validation vs the UNIONS-WL/Smokescreen fork — note cross-repo nesting is allowed), and which PRs attach.
3. **One PRD per planned PR**, including PR 0.1 (fork + backend independence). Style contract: concise, top-down, high quality, assumes NOTHING about the journey (no references to old PR numbers as context, no "previously we…"; a fresh senior engineer must be able to implement from it alone). State the desired end state, the interfaces, the acceptance criteria.

Quality bar: several adversarial review rounds; final gate per document is a low-effort fable review for concision/top-down-ness.
