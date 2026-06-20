---
id: 01KTCHX03Y3PVZYS5QHFWZRF5N
name: SP Validation Restructuring
status: open
tags:
    - constitution
    - sp-validation
    - reorg
    - shuttle
created-at: 0001-01-01T00:00:00Z
outcome: 'Phase 2 — the top-level moves — complete on cleanup/restructuring (PR #197), guard net green throughout (full suite 86 passed, 0 skipped; move-map guard active with 5 registered moves). The tree has the target shape: workflow/ + papers/{bmodes,catalog,harmonic} (bmodes split), cosmo_val/ promoted from notebooks/ with all tracked references swept and on-disk outputs moved along (candide-absolute paths through the pure_eb symlink stay live), tracked scratch/ with conventions README, one top-level results/ (contents gitignored), root output/ ignored, dead hand-listed notebook block dropped from .gitignore. CI hermeticity restored: the guard suite now passes inside the Docker image (no .git directory, no cluster mounts) — git-querying guards fall back to a tree walk of the image''s tracked-only content, and the workflow dry-run guard skips off-cluster like its data-bound siblings. Cleanup begun: defunct/ deleted; nbstripout + 2 MB large-file pre-commit hooks added (CONTRIBUTING documents activation). Remaining: fold glass_mock core into src/ (own pass), curate notebooks/ to official demos (reduction-notebook triage with Martin), branch/milestone tidy (#188/#189) with Cail.'
horizon: now
shuttle:
    kind: oneshot
    interactive: true
    host: candide
    project_dir: /automnt/n17data/cdaley/unions/pure_eb/code/sp_validation
    agent: claude-opus
---

## Driver session - interactive; first action: land Sacha's #192

This fiber is now an **interactive oneshot driver shuttle** for the sp_validation
reorganization, dispatched on the sp_validation city (candide). Attach and drive the
milestone together. The settled plan below - target layout, division of labor with
Sacha, the modular workflow, cleanup, sequencing - is the Desired State.

**First action: land Sacha's foundation merge, PR #192** (`sachaguer:develop` -> `develop`,
"Merge Sacha's fork with fiducial sp_validation"). It is basically ready and is milestone
item 1 - the clean base the restructuring builds on. On dispatch, the worker:

1. Verifies #192 is genuinely merge-ready: pull `develop`, fetch the PR, confirm it merges
   cleanly with tests/CI green, and surface anything that is not.
2. Leaves the actual merge to Cail (or merges on his explicit go-ahead) - landing a
   collaborator's PR is his gesture; the worker preps and verifies.
3. With #192 in, proceeds into the restructuring (item 2 / PR #188) per the sequencing
   below: delete `defunct/` + fix `.gitignore`, promote `cosmo_val/`, collect `papers/`,
   the path-translation sweep, `scratch/` + nbstripout.

Interactive (attach and talk to it) and oneshot. Keep this plan current as the reorg lands.


We're cleaning up `sp_validation` before the next analysis cycle. This fiber holds the
settled reorg plan, refined across three Cail passes and a Cail+Sacha pass (2026-06-02),
grounded in a full map of what's on disk. It ships as a GitHub milestone; this work is the
restructuring proposal PR within it.

## Guiding principle

Make the things a person actually *runs* siblings at the top level, with shared library
code underneath. The asymmetry that keeps causing confusion: `cosmo_val` is buried in
`notebooks/` while `cosmo_inference/` is top-level. Fix that. Be aggressive on deletion —
it all stays in git history.

## Target top-level layout

```
sp_validation/
├── src/sp_validation/    library — + glass_mock core folded in
├── cosmo_val/            PROMOTED from notebooks/cosmo_val/ — validation code + config
├── cosmo_inference/      ~as-is — inference code + config (cosmosis/cosmocov, pipeline.sh)
├── workflow/             ALL analysis — modular Snakemake, multi-person → results/
├── papers/               final-figure assembly only (PDF, color, layout)
│   ├── catalog/             ← notebooks/cosmo_val/catalog_paper_plot/
│   ├── cosmic_shear_2d/     ← 2D_cosmic_shear_paper_plots/   (config space)
│   ├── harmonic/            ← 2D_harmonic_space_..._plots/   (harmonic space)
│   └── bmodes/              ← 2D_bmodes_paper_workflow/      (analysis → workflow/)
├── scripts/             only REAL reduction scripts (catalog builders, masking)
├── scratch/             TRACKED, per-person — ad hoc + personal custom workflows
├── notebooks/           CURATED — official demo / tutorial notebooks only
├── results/             analysis products + diagnostic plots; CONTENTS gitignored
├── docs/  tests/  config/
```

`cosmo_val` and `cosmo_inference` sit side by side as the code+config homes (someone who
only does validation opens `cosmo_val/`); `workflow/` orchestrates the analysis across
them.

## Division of labor (settled with Sacha)

The boundary is **the inputs to a paper figure**: everything up to that point is analysis;
the figure itself is presentation.

- **`workflow/` — all analysis.** Generic, reusable, modular, organized for multiple people.
  Produces analysis products *and diagnostic plots* (sp_validation makes many — fine, they
  go to `results/`). This is where the bulk of the work lives.
- **`papers/<paper>/` — final-figure assembly only.** Building the figure PDF, tweaking
  colors/layout, recombining data for presentation. Tied to one paper; **may never touch
  Snakemake.** Not everyone uses it.
- **`scratch/<person>/` — personal & ad hoc.** Experimentation, and personal *custom
  workflows* for one-off analysis. Tracked (sharing scratch is valuable).

## Why the workflow is modular, not monolithic

Nothing in real science is computed once — the catalog changed ~20× in the first release
suite, and every paper varies the data vector, covariance, and inference. So the workflow
is **parameterized, not a fixed run**: the shared thing is the rules, the config changes
every time. Snakemake's `module` directive imports rule *definitions* under your own config
and an output `prefix`, and lets you override individual rules:

```python
module analysis:
    snakefile: "../../workflow/Snakefile"
    config:    config              # this run's catalog, cuts, blind
    prefix:    "results/bmodes"    # products land here — no clobbering
use rule * from analysis
# swap is per-rule: redefine just the data-vector rule with the same output to override
```

Single top-level `results/`; each run/paper namespaces under `results/<name>/` via the
`prefix`. **DAG dry-run** (`snakemake --dry-run`) is the backpressure starter — the safety
net for the module work; the broader test suite fills in during implementation.

## Cleanup

- **`defunct/`** — already quarantined (Oct 2024). Delete.
- **`notebooks/`** — curated to *official* demo/tutorial notebooks only; the rest deleted or
  moved to scratch. Salvage `run_cosmo_val.py` + `cat_config.yaml` → `cosmo_val/`;
  `catalog_paper_plot/` → `papers/catalog/`; Martin's real reduction notebooks → `scripts/`.
- **`scratch/`** is tracked, not gitignored. Bloat guard is tooling: `nbstripout` (strips
  notebook outputs on commit — git is 319 MB today, heavy from *committed notebook outputs*,
  and the `.gitignore` is scarred with hand-listed notebook paths) + a pre-commit size hook.
- **Paths stay messy (revised 2026-06-05).** The original plan was a mechanical sweep rewriting
  ~117 hardcoded absolute paths (`/home/…`, `/n17data/…`, `/automnt/…`, several `/home/guerrini/…`)
  to a single repo-relative `results/`. **Cail's call: that's not realistic — they'll stay messy.**
  We do *not* rewrite them to repo-relative. The realistic guard is a **Candide-local existence
  checker** (guard ③): load every config, assert every referenced path exists *on the machine this
  runs on*. Not portable, not CI — local back-pressure that goes red when a move orphans a reference.
- **gitignore fix** — root `results/` and `output/` aren't currently ignored; ignore
  `results/` contents (keep the dir), drop the hand-listed notebook block.

## The milestone

Shipped as a GitHub milestone, a suite of PRs in sequence:

1. **Foundation** — Sacha merges his pending local code into `develop`. His to make.
2. **Restructuring** — PR #188. The proposal is *not* committed to the repo: it lives in
   the PR description and here in the fiber (`report.html`, `proposal.md`; rendered PDF
   for PR attachment). Decided 2026-06-03 that a rendered report shouldn't sit in the code
   tree. With nothing committed, the branch == `develop` and GitHub auto-closed the PR.
3. **Glass mocks → tomography** — stub.
4. **Input pipeline → tomography** — stub.

## Implementation sequencing (within PR 2 when it goes live)

1. Delete `defunct/` + audited old notebooks; fix `.gitignore`.
2. Promote `cosmo_val/`, collect `papers/`, curate `notebooks/`, fold glass_mock into `src/`.
3. Path-translation sweep (paper plots + scripts) → single `results/`.
4. Add `scratch/`, nbstripout + size hook, README discipline; DAG-dry-run test.
5. (incremental) build out modular `workflow/`; wire the first paper as a `module` composition.

## Resolved decisions

- harmonic vs cosmic_shear are **separate** papers (harmonic vs config space), separate dirs.
- outputs go to a **single top-level `results/`**, namespaced per run via module `prefix`.
- Martin's reduction notebooks: which are real scripts — review with Martin.
- `2D_bmodes_paper_workflow/` is not split mid-paper: its analysis folds into `workflow/`,
  its final figures into `papers/bmodes/`, sequenced after Sacha's foundation merge.

### Settled 2026-06-05 (interactive session with Cail)

- **Paths stay messy → existence checker, not translation.** See revised Cleanup bullet above.
- **Back-pressure suite finalized** to ①②③⑤⑥; DAG guard simplified to "dry-run passes"; see the
  suite section. Build dispatched as the [[back-pressure-suite]] Codex shuttle.
- **Branch topology — DEFERRED until the back-pressure tests land** (doing it now would break the
  Codex worktree/branch off `cleanup/restructuring`). Target end-state Cail wants: **one clean
  branch = Sacha's foundation merge + restructuring on top**, no superseded clutter, *no work
  lost* (names don't matter). Mechanics to settle then: `cleanup/restructuring` and
  `restructuring/proposal` are duplicate names for one line of work (PR #188's head is
  `restructuring/proposal`); collapse to one. Close #189 (foundation stub, superseded by folding
  the foundation into #188); keep #190/#191 as tomography stubs.
- **`uv.lock` — leave gitignored for now.** Cail lukewarm ("not sure it's necessary"); container
  is canonical. One-line flip to track later if we want pinned-dep reproducibility.

## Cail's direction (2026-06-05 night) + back-pressure invariants — HOLD until pure_eb tree clears

Cail opened the operator and gave the call that **overrides the worker's "wait for Sacha's #192 to merge to develop" sequencing**:

- **Don't wait for Sacha.** His foundation PR (#192) isn't merged yet, and that's fine — **fold his branch into our cleanup branch** and build the restructuring on top. Base the restructuring branch on `develop` + #192's head so we get his foundation without racing his merge gesture.
- **Restore the closed restructuring draft PR #188** (`restructuring/proposal → develop`, CLOSED/draft — "closed by us somehow", Cail was confused by it). Reopen it.
- **Clean up the milestone** "Restructuring & tomography prep" (open:4 closed:1 — #188 is the closed one; #189 foundation-merge, #190/#191 tomography drafts, #192 Sacha). Reopen #188; tidy sequencing. *Surface the milestone re-ordering for Cail, don't auto-decide which drafts stay.*

### Back-pressure FIRST — establish the invariant suite GREEN before moving a single file

Cail's framing: "what things do we want to be varying and still maintain as we make these changes that keep us on the right track? Could be simple things like imports." This is **characterization / golden-master testing before a refactor**: write guards that are green on the *pre-reorg* tree (develop + #192 folded in), commit them, then do the reorg in small reviewable `git mv` steps keeping every guard green. The suite is the back-pressure that resists silent drift. The reorg is "done right" **iff** the suite stays green.

### The suite — settled with Cail (2026-06-05 interactive session)

Refined live from the original 7-item sketch. **Final set to build (cheap → strong):**

1. **① Imports resolve (Cail's #1) — GREEN, done.** `tests/test_imports.py` walks the *package* AND the standalone `scripts/` dir, importing every module (scripts live outside the package; AST-resolved, not executed). First thing to break on a bad move. Already caught a dead script (`plot_leakage.py` imports a never-existent `sp_validation.correlation`). This is the *model* for the rest.
2. **② Snakemake dry-run passes — build.** Deliberately **not** DAG-identity/golden (Cail: "we don't care about the weather; it's okay if the rules change — making a *superior* restructure that breaks the DAG is unlikely"). The guard is simply: **`snakemake -n` on the bmodes workflow exits 0.** Probe first — Cail isn't sure bmodes still runs, and dry-run may be slow. If it's broken or too slow, **report it as signal, don't paper over it** (skip/xfail with a clear reason and surface it).
3. **③ Config-path existence — Candide-local — build.** The reframe of the dead path-translation idea: load every config (`config.yaml`, `cat_config.yaml`, mask yamls, cosmosis `.ini`), extract path-shaped values, assert they exist *on this machine*. Skip cleanly when not on Candide.
4. **⑤ Symlink integrity — build.** Every tracked symlink resolves to an existing target. Cheap; this tree has real symlinks that moves silently break.
5. **⑥ Dangling-reference grep — build (harness now).** After a move, grep the tree for old *internal* dir names → zero hits. Parameterized by a move-map that's empty pre-reorg (so trivially green now); the harness lands so it activates the moment moves begin. This is the *internal-reference* cousin of the absolute-path mess we're punting on.

**Dropped, with reasons:**
- **Entry-point / console-script resolution** — N/A: `pyproject.toml` declares no `[project.scripts]`; folded into ①.
- **"No work lost" content manifest** — too strong (Cail: "lots of deletes, we're going to be brutal").
- **Package installs** — CI already builds the image and runs pytest, so it's covered.
- **Output-schema / value invariance** — needs real cluster data, its reference run is flagged unreliable, and it's "testing specific code," which Cail wants deferred until that code changes.

**Build vehicle:** dispatched as an autonomous **Codex shuttle** ([[back-pressure-suite]]) — mechanical-with-investigation, well-specified, and offloading it keeps the interactive session free for design. Mirror `tests/test_imports.py` for style; run everything in the container.

**Collision hold:** sp_validation is nested at `pure_eb/code/sp_validation`, so pure_eb-tree agents share its working tree. As of this writing the active sharer is `science/pure_eb/citation-audit-skill/skill-rework` (candide). **Do not start the reorg while another agent has the pure_eb tree dirty** — the operator is monitoring and will engage this driver (or re-dispatch) once the tree clears. At push time, `git -C <sp_validation> status` must be clean of other agents' edits before moving files.

## ENGAGE — 2026-06-05 ~02:45 CEST — AUTONOMOUS, gate OPEN (operator dispatch)

The operator verified the push-gate OPEN and is dispatching this **autonomously** (Cail asleep; he said "forge ahead" + "when it seems like we can start pushing, let's do it"). Ground truth at dispatch: candide has NO worker on the pure_eb/sp_validation tree; no git merge in progress (MERGE_HEAD absent); debug's federated-identity deploy is settled (parked, main untouched); PR #196 (docs) landed on develop. Do the work unattended; leave a report + a clean green-guarded checkpoint for Cail's morning. Realize the plan in "Cail's direction + back-pressure invariants" above.

### Scope (in order) — back-pressure FIRST, stage the move, don't force it

**Phase 0 — clean base.** The tree has ~19 leftover untracked files (sphinx `.rst` under docs/source, a notebooks dir) from the docs PRs. Decide each: gitignore the generated artifacts or commit if intended; land develop clean. Re-verify #192's CURRENT state (`gh pr view 192 --repo CosmoStat/sp_validation`) — it may have merged while we waited. If merged, base on develop; if still open, fold sachaguer's branch into a fresh `cleanup/restructuring` branch (Cail's call: don't wait for his merge).

**Phase 1 — back-pressure suite GREEN (the first things to add).** On the clean base, build the invariant suite from the section above and get it GREEN, committed to the cleanup branch: (1) imports+scripts resolve (walk package AND scripts/); (2) snakemake DAG-identity golden (`--dry-run` rule list + job counts frozen); (3) config-path resolution; (4) output-schema invariance on the smallest catalog→pseudo-Cl run (columns+dtypes+shape; values if cheap); (5) no-dangling-path grep; (6) git-mv provenance. Then **restore the closed draft PR #188** and **propose** a milestone cleanup for "Restructuring & tomography prep" (list a recommendation; don't auto-decide which drafts stay).

**Phase 2 — BEGIN the reorg only behind green guards; DON'T force completion.** With the suite green, start the top-level layout move (workflow/paper/scratch per #188) in small `git mv` steps, re-running the suite after each. Do NOT attempt to finish the 120-file move unattended — stop at a clean, suite-green checkpoint and write the report. The full move + merges stay Cail's gesture.

**Deliverables:** a `cleanup/restructuring` branch (back-pressure suite green); #188 restored; milestone-cleanup recommendation; `report.html` in the fiber dir; outcome + felt history. Propose-never-commit: no merges to develop, no force-pushes to others' branches.

**Exit:** autonomous — at the green-guarded checkpoint, close for Cail's review.
