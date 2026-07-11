---
id: 01KX4JPKG1K7EAKFTATKBNKJ5R
name: 'Implement PRD #241: SACC + Smokescreen draft PRs'
status: active
tags:
    - constitution
    - sp-validation
    - sacc
    - blinding
created-at: 2026-07-10T01:16:42.625888497+02:00
updated-at: 2026-07-10T17:33:51.121781533+02:00
outcome: 'Done: all seven PRD-#241 draft PRs open, tested, adversarially reviewed, CI green (#243 #245 #249 #250 #251 #253 #255), sub-issues board-wired, summary comment maps PR↔row. Team review is the next act; follow-ups (upstream CSL bugs, PSF variant, tomographic) flagged in #255 and Status.'
shuttle:
    kind: oneshot
    host: candide
    project_dir: /automnt/n17data/cdaley/unions/code/sp_validation
    agent: claude-fable
    effort: low
    runtime:
        dispatched_at: "2026-07-10T21:55:49.888417Z"
        session_uuid: c3320e08-977e-4fdd-b31a-0bc6d7f10388
        handed_off_at: "2026-07-10T13:45:40.670475074Z"
---

[PRD #241](https://github.com/CosmoStat/sp_validation/issues/241) (sub-issue of umbrella #232) specifies sp_validation's adoption of SACC as the standard data-product format and Smokescreen for data-vector blinding, decomposed into seven PRs. The PRD is agreed in principle and its open questions (secret custody model, shift-envelope size, fiducial model config) change config values, not code — so implementation proceeds now, on the PRD's stated defaults, as **draft PRs** the team reviews against the PRD. Cail has abundant compute this week; the intent is fully tested, robust implementations, not sketches. The PRD text is the contract for *what*; this constitution carries the *how*.

## Desired State

Seven **draft** PRs open on CosmoStat/sp_validation, one per row of the PRD's §7 table, each:

- based on the repo's integration branch (fetch and confirm the branch-vs-origin state before building; never build on a stale base),
- attached to the board correctly: one sub-issue per PR under #241 (same repo, Work area = Inference Pipeline, Kind = Task, Status = In Progress), PR body carrying `Closes #<that sub-issue>`,
- **fully tested**: unit tests for every writer/converter, the PR-3 converter byte-compared against a real current 2pt-FITS output, the PR-6 acceptance test proving B-mode estimators (COSEBIs Bₙ, pure-mode ξ_B) unchanged under blinding on mocks, and the suite green in CI,
- marked draft, no review requests, no @-mentions; PR bodies written in the worker's voice, signed "— Claude (<model>) on behalf of Cail.",
- **adversarially reviewed before opening**: a fresh-context reviewer attacks the diff (ordering/permutation, silent-empty selections, fitness-for-purpose driven end-to-end); this gate has caught a HIGH silent-misalignment bug in two of the first three reviewed PRs (PR 2: reader-side sorting; PR 3: tomographic truncation) — findings fixed and noted in the PR body before the draft opens.

Specifically NOT doing: merging anything; resolving the PRD's open questions (implement §4(b) hash-commitment custody and the ±0.075 S8 / ±0.1 Ωm envelope as defaults, clearly marked configurable); touching Sacha's reserved zones ([[sp_validation/reserved-sacha]]: `scratch/guerrini/` and the namaster_utils→source migration — mechanical import fixes at the boundary are fine, flagged in the PR body).

Done when all seven draft PRs are open, tested, and cross-linked from #241 with a summary comment mapping PR ↔ PRD row.

## Sequencing & environment

PR 1 is **done** ([#243](https://github.com/CosmoStat/sp_validation/pull/243), branch `feat/sacc-1-deps`): py3.12 floor; `sacc>=2.4,<3` core; `[blinding]` extra (firecrown git v1.15.1 + smokescreen==1.5.6 + pyccl==3.3.4, exact pins for seed-reproducibility); `uv-overrides.txt` drops firecrown's pip-unresolvable/unused connector deps (numcosmo-py, cosmosis, cobaya) — firecrown is NOT on PyPI, this override mechanism is the only clean pip path (see [[sp_validation/smokescreen-sacc-prs/smokescreen-api-facts]]).

**Branch convention: the PRs stack** (branch `feat/sacc-N-<slug>`, PR base = the branch it builds on): 2 ← `feat/sacc-1-deps`; 3, 4, 5 ← `feat/sacc-2-sacc-io` (all three import `sacc_io` or build files with it); 6 ← the PR-5 branch (needs the firecrown likelihood); 7 last. GitHub retargets stacked PRs automatically when a base merges — CI builds the Docker image and runs the fast suite in it on every push, so a PR whose tests import sacc/firecrown must carry PR 1's deps in its image. GitHub retargets stacked PRs automatically when the base merges. 5 needs the firecrown likelihood before 6; 7 comes last.

**Test environment**: shared uv py3.12 venv at `/automnt/n17data/cdaley/unions/code/sp_validation-worktrees/venv` (full `[test,glass,blinding]` stack, exact PR-1 pins), editable-installed against whichever worktree is under test (re-point with `uv pip install --python <venv>/bin/python --no-deps -e <worktree>`). Worktrees live under `sp_validation-worktrees/`, one per PR branch; the main checkout sits on an unrelated image-sims branch — never build there. Working toy scripts for the firecrown likelihood and a full Smokescreen blind round-trip live in `sp_validation-worktrees/env-probe/` — they seed PR 5/6 implementations.

**Heavy runs go through SLURM — never bare on the login node.** Anything that JIT-compiles numba, runs CCL/CAMB theory, or drives the slow suite (including the container launcher below, and including *subagents and review workflows told to run code*) gets an allocation first (`salloc -p comp -c N --no-shell` → `srun --jobid=<id> …`, or `sbatch`). This was violated on 2026-07-10 (multi-core pytest at 1300% on the login node, caught by Cail); the on-run model in the global CLAUDE.md applies to test runs too. When dispatching agents that may execute code, put the srun prefix INTO their instructions.

**cosmo_numba tests (COSEBIs/pure-EB re-derivation, PR-6 acceptance) need a hybrid launcher**: cosmo_numba + NumbaQuadpack live only in the apptainer container (compiled against glibc ≥2.29 — a copied `.so` cannot load on the host), while the blinding stack lives only in the venv. The verified union: run the venv's python *inside* the container with the two container-installed packages copied to `sp_validation-worktrees/extra-site/` — `apptainer exec --bind /home,/automnt,/n17data /n17data/cdaley/containers/containers/ env PYTHONPATH=<worktree>/src:…/extra-site …/venv/bin/python -m pytest …`. Such tests gate on `pytest.importorskip("cosmo_numba")` + `slow`, so they skip in CI (image lacks cosmo_numba) and in the plain venv, and run under the launcher.

Facts already verified (don't re-derive): θ convention is arcminutes (firecrown hard-codes `/60`); n(z) normalization is irrelevant to all consumers (CosmoSIS, OneCovariance, CCL all normalize internally); SACC officially defines `galaxy_shear_cosebi_ee/bb`; OneCovariance has no SACC I/O; Smokescreen writes the seed into blinded-file metadata (strip it); CosmoSIS's `sacc_like` ξ± path is prototype-grade — validate, don't trust.

## Status

**The desired state is realized — all seven draft PRs are open, tested, adversarially reviewed, and CI-green**: [#243](https://github.com/CosmoStat/sp_validation/pull/243) (PR 1, deps), [#245](https://github.com/CosmoStat/sp_validation/pull/245) (PR 2, `sacc_io`), [#249](https://github.com/CosmoStat/sp_validation/pull/249) (PR 3, converter, closes #246), [#251](https://github.com/CosmoStat/sp_validation/pull/251) (PR 4, migration, closes #247), [#250](https://github.com/CosmoStat/sp_validation/pull/250) (PR 5, likelihood, closes #248), [#253](https://github.com/CosmoStat/sp_validation/pull/253) (PR 6, blinding, closes #252), [#255](https://github.com/CosmoStat/sp_validation/pull/255) (PR 7, native sacc_like, closes #254). The [#241 summary comment](https://github.com/CosmoStat/sp_validation/issues/241#issuecomment-4936723538) maps PR ↔ PRD row with reviewer notes (stacking, the two-file layout amendment, defaults-not-decisions). CI observed green on every tip (#255: run 29103598394 `completed success`, observed 2026-07-10).

PR 7 landed this session (rulings + as-landed record in [[sp_validation/smokescreen-sacc-prs/pr7-sacc-like-design]]): CSL's `sacc_like` ξ± path was confirmed prototype-grade with concrete mechanisms — no arcmin→rad conversion (2pt_like converts explicitly; sacc_like evaluates the radians theory spline at arcmin tags → theory silently ~0), ordering assumed never enforced, latent `keep_tracers` NameError. Adoption is a subclass shim (`sacc_like_unions.py`, radian-θ copy swapped in around theory extraction only + loud ordering guard), `inference_prep` revived onto the assembled `{version}.sacc` per PR 4's dormant-note contract, and equality with the PR-3 converter path observed to machine zero (synthetic χ²=1.18829133107 and real SP_v1.4.6_leak_corr ξ-only χ²=420310.753739, Δχ²=0 both; tripwire test fails the day upstream fixes units → retire the shim). Adversarial review (5 lenses, 3 refuters/finding) confirmed 2 MEDIUMs, fixed pre-draft; an independent fix-review cleared them on all attack vectors.

**Where this picks up next** (human's desk, not a worker's): the team reviews the seven drafts against the PRD; the PRD's open questions (custody, envelope, fiducial config) are config-value discussions on top of implemented defaults. Follow-ups flagged but out of scope: upstream CSL issues (unit gap, `sacc_like.py` L106 NameError) — recommended in #255's body, Cail's call whether/how to file at joezuntz/cosmosis-standard-library; PSF/xi_sys pipeline variant + glass-mock prep still on `cosmosis_fitting.py`; tomographic support (converter fail-fast + shim guard) when the tomographic round needs it; PR 4's `dd474b3` "UNVERIFIED WIP" message still wants a squash-merge or reword.

Durable environment lessons (beyond the env section above): the venv-vs-CI import matrix bites — the venv carries optional deps (cosmosis) the CI image lacks, and `test_imports.py` sweeps every package module, so optional-dep imports must be lazy (caught as a red CI run on #255, fixed in `151d997`); the sandbox container at `/n17data/cdaley/containers/containers/` is a stale shapepipe base (numpy yes, sacc/current cs_util no) — fine for bare-import checks, useless for the suite until rebuilt; CSL shallow clone lives at `sp_validation-worktrees/csl` (pin 4fd2f1c), cosmosis 3.25.2 is installed in the shared venv, and the new tests want `CSL_DIR` in the environment.
