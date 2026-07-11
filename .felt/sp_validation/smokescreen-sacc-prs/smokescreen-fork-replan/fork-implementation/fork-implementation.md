---
id: 01KX6YH5ZYRQRN1Y7CZA7VR9DE
name: Execute the Smokescreen-fork replan (fork, 2 fork PRs, sp_validation PR rework, board wiring)
status: active
tags:
    - constitution
    - sp-validation
    - sacc
    - blinding
created-at: 2026-07-10T23:21:56.734989469+02:00
updated-at: 2026-07-11T11:58:35.83088803+02:00
outcome: 'In progress: fork PRs #4 (protocol, reviewed+fixed, 64/64) & #5 (packaging, stacked) open; #241 rewritten state-shaped; #243 reworked and CI-green (0 confirmed review findings); #253 reworked (7 confirmed findings being fixed; CI red pending stack onto #243''s branch for the fork pin).'
shuttle:
    kind: oneshot
    host: candide
    project_dir: /automnt/n17data/cdaley/unions/code/sp_validation
    agent: claude-fable
    runtime:
        dispatched_at: "2026-07-10T21:24:03.631335Z"
        session_uuid: 27f88a84-70fa-4418-8d60-074954811780
    effort: low
---

Executes the Smokescreen-fork replan ruled with Cail on 2026-07-10 (parent fiber [[sp_validation/smokescreen-sacc-prs/smokescreen-fork-replan]]). The plan is fully drafted and human-approved: the parent's `grounding.md` carries the rulings (pass-2 and pass-3 amendment sections override where they conflict with earlier text), and `drafts/` carries the reviewed specs — `ripple-analysis.md` + `issue-tree.md` (the plan) and eight live PRDs (the per-PR contracts; two more carry RETIRED banners and are historical). Those documents are the contract for this work — cite them, don't re-derive them. The upstream context: sp_validation's blinding stack (PRD sp_validation#241, seven existing draft PRs) currently depends on firecrown, which is uninstallable-by-pip pain; the replan forks DESC's Smokescreen to `UNIONS-WL/Smokescreen`, makes theory pluggable (shipped CCL default backend + bring-your-own `theory_fn`; firecrown retained lazy/undeclared/unsupported), and strips firecrown from sp_validation entirely.

## Desired State

**Source of truth: the live body of sp_validation#241** (rewritten in place 2026-07-11; Cail edits it directly — re-read it each session and reconcile before building). Two later rulings supersede the drafts where they conflict: (a) **one SACC file per catalogue version** — fine ξ± rides in `{version}.sacc` as `grid='fine'` tagged points with a dense per-pair covariance block (see the re-ruled [[sp_validation/smokescreen-sacc-prs/sacc-layout-contract]]); PR #245 already implements this. (b) **Blinding is per intermediate product, at birth** — a `blind-init` step draws the seed once per catalogue version and publishes `hash(seed)`; each blindable part (coarse ξ±, fine ξ±, pseudo-Cℓ) is blinded the moment it is computed; only blinded intermediates persist on disk; COSEBIs/pure-E/B are derived from the already-blinded fine ξ± (born blinded); terminal assembly asserts the commitment hash is identical across parts. `prd-smokescreen-blinding.md`'s terminal-master-blind framing is superseded accordingly; its backends, custody primitives, and CAMB↔CCL cross-check stand.

**GitHub structure (per `drafts/issue-tree.md`):**
- `LSSTDESC/Smokescreen` is forked to `UNIONS-WL/Smokescreen` (org fork, `gh repo fork --org UNIONS-WL`). If org-level fork creation is denied by permissions, that is a genuine block: stop and lead the outcome with "Blocked:".
- The issue tree from `drafts/issue-tree.md` exists on GitHub: umbrella + sub-issues in their stated repos, nested as sub-issues, on board `UNIONS-WL/projects/1` with Status + Work area set per the board model in `/automnt/n17data/cdaley/unions/CLAUDE.md`. A comment on sp_validation#241 states the plan amendment concisely (top-down, neutral tone — see tone rule below), citing the new issues.
- Every PR is **draft**, carries `Closes #N` to its same-repo issue, and nothing is merged — merging is Cail's gesture.

**Fork PRs (against `UNIONS-WL/Smokescreen`, contracts in `drafts/prd-fork-protocol.md` and `drafts/prd-fork-packaging-ci.md`):**
- PROTO: theory-callable protocol (`theory_fn`), shipped default CCL backend (pyccl imported by the backend module), firecrown moved behind a lazy import (retained, undeclared, unsupported — never imported at module level), draw fixed (local `default_rng`, order-independent, CCL-native parameter names), DESC test suite migrated to the non-firecrown default. `import smokescreen` succeeds in a clean env with no firecrown.
- PKG: packaging (pyccl declared; cryptography/sacc retained; no firecrown in `Requires-Dist`), pip-only install docs (`pip install git+https://github.com/UNIONS-WL/Smokescreen@<tag>` — no conda, no PyPI), CI green on the non-firecrown path.
- Keep the diff small and upstream-PR-able: Cail may later PR this against DESC upstream. GitHub-facing text stays **neutral about firecrown** — no editorializing (collaboration relations).

**sp_validation PRs (contracts in `drafts/prd-dependencies.md`, `drafts/prd-smokescreen-blinding.md`, plus `ripple-analysis.md` rows):**
- The deps PR (#243) is reworked per prd-dependencies: firecrown/`--no-deps` closure/`patch_firecrown.py`/numpy<2.5 ceiling all gone; blinding deps CORE (no `[blinding]` extra) — fork pinned by git tag/SHA (only after the fork's lazy-import + packaging commits exist), pyccl, cryptography, sacc promoted to top-level; py3.12 floor (a raise from 3.11); container/`uv.lock` ripple included.
- The blinding PR (#253) is reworked per prd-smokescreen-blinding: all three shift paths (coarse ξ±, fine ξ±, pseudo-Cℓ) through the fork protocol; the CAMB↔CCL cross-check folded in as tests (σ8-matched, halofit strings matched); `_verify_sacc_consistency` semantics per the protocol PRD (length guard); custody scheme unchanged from the landed design.
- #250's firecrown likelihood deliverable is struck per the ripple analysis (the PR is retitled/reworked or closed-with-comment, whichever the issue tree specifies).
- PRs #245, #249, #251, #255 are NOT touched — Cail is reviewing them as-is.

**Quality bar:** each PR's test suite passes in CI; scientific-stack tests run in the container per repo conventions; a fresh-eyes adversarial review of each substantive diff against its PRD before the session that produced it closes. Never run heavy compute on the candide login node — allocate SLURM first and pass the `srun --jobid` prefix to any delegated runner (see `/automnt/n17data/cdaley/unions/CLAUDE.md`).

**Not doing:** merging anything; posting opinions about firecrown/DESC; touching PRs #245/#249/#251/#255; PyPI publishing; upstream (LSSTDESC) PRs.

## Status

Not started. First moves: fork the repo, post the issue tree, then PROTO — the deps PR rework is gated on the fork's packaging commits existing.

## PROTO implementation (Smokescreen#2, branch feat/theory-backend-protocol)

**Done — draft PR https://github.com/UNIONS-WL/Smokescreen/pull/4 (Closes #2), branch feat/theory-backend-protocol, full suite 57 passed / 0 failed / 0 skipped.** Commits: 9d02f52 (protocol refactor), 76ded3f (inherited firecrown module), d00c7a7 (test migration), 8b80bf2 (backend-convention docstrings).

Implemented per `drafts/prd-fork-protocol.md`. Landed shape: `ConcealDataVector(fiducial_params, shifts_dict, sacc_data, *, seed, theory_fn=None, shift_distr)`; both theory vectors from `theory_fn`; default backend `smokescreen/backends/ccl.py::build_ccl_theory_fn` (the sole `import pyccl` in src, so `import smokescreen` imports no backend); pure `param_shifts.draw_param_shifts` (sorted keys, one `default_rng` scalar draw per key, deltas, SHA-256 string→seed); length guard only, no `_verify_sacc_consistency`. Inherited firecrown path kept as `firecrown_datavector.FirecrownConcealDataVector` (lazy, unsupported, not wired in).

**Ambiguity resolutions (for the PR reviewer to check):**
- *"firecrown retained lazily in-tree"*: **team-lead ruled my first take (delete the path, keep only examples/) NOT accepted** — the retention requirement is for upstream relations (a later DESC PR must not read as amputating firecrown). Resolved per ruling: the pre-refactor firecrown-likelihood class is preserved as `FirecrownConcealDataVector` in a new `src/smokescreen/firecrown_datavector.py`, all firecrown/pyccl imports function-local (via `_import_firecrown()` + local imports), module docstring marks it inherited-upstream/unsupported, not wired into the default flow (no main module imports it), no fork test exercises it. `examples/` also untouched. Grep confirms zero module-scope `firecrown`/`numcosmo` in `src/`; `import smokescreen` does not import the module; importing the module with no firecrown installed succeeds and imports no firecrown. Commit 76ded3f.
- *Default backend transfer function*: used pyccl-native `eisenstein_hu` + `halofit` (set inside the backend, overridable by cosmo_params) so the shipped backend runs against a bare pyccl install — the venv has no CAMB/CLASS. A caller's own `theory_fn` may route through a Boltzmann code.
- *Seed normalization*: added `param_shifts._normalize_seed` (SHA-256 → 64-bit int) per the PRD's explicit spec rather than reusing `utils.string_to_seed` (MD5 % 1e8) — they differ, and the PRD names SHA-256. `string_to_seed` left in utils untouched (used nowhere on the new path; kept for any inherited caller).
- *ξ± angle units*: the SACC `theta` tag is assumed to be **arcmin** and converted to degrees for `ccl.correlation`. Standard-cosmic-shear SACCs carry arcmin; a backend fed degrees would be off. Recorded because the default backend's row→theory alignment is its own contract, unchecked by Smokescreen.
- *No `_verify_sacc_consistency`*: dropped, not re-expressed (deliberate, non-behavior-preserving per PRD §Consistency check). Sole guard is the construction-time length check.
