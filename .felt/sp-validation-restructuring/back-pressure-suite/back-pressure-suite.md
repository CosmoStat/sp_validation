---
id: 01KTCKDA2N1VEPZJRBF04ZZ6VW
name: Back-pressure test suite for the reorg
status: active
tags:
    - constitution
    - sp-validation
    - reorg
    - shuttle
created-at: 2026-06-05T21:15:56.629448647+02:00
outcome: 'Back-pressure guards are green with one intentional legacy import xfail: 45 passed, 1 skipped, 1 xfailed. Config paths, tracked symlinks, and B-modes workflow dry-run now pass; remaining xfail is plot_leakage.py importing a historical LF leakage helper that never existed in this tree.'
shuttle:
    enabled: true
    kind: oneshot
    host: candide
    project_dir: /automnt/n17data/cdaley/unions/pure_eb/code/sp_validation
    agent: codex
---

## Desired State

Four **back-pressure guards** built and **green on the pre-reorg tree** (branch
`cleanup/restructuring`), committed locally. They mirror the existing guard ①
(`src/sp_validation/tests/test_imports.py` — read it first; it is the style model).
The point of the suite: capture invariants that must stay true as the restructuring
moves files around, so a botched `git mv` goes red. Build them while everything is
green, *before* any file moves.

Build these four (the numbering matches the parent constitution
[[sp-validation-restructuring]] — ① is already done):

- **② `snakemake -n` passes.** A test that runs `snakemake --dry-run` on the bmodes
  workflow and asserts it **exits 0**. This is *not* a DAG-identity/golden test — we do
  **not** freeze the rule graph (Cail: "we don't care about the weather; it's fine if
  the rules change"). The invariant is just: the workflow still dry-runs clean.
  - Workflow: `cosmo_inference/notebooks/2D_bmodes_paper_workflow/`, Snakefile +
    `rules/*.smk`, config at `config/config.yaml`.
  - **Probe first.** Cail is not sure bmodes still dry-runs, and it may be slow. Run the
    dry-run by hand in the container before writing the test. If it passes quickly →
    write the test plainly. If it is **broken** or **too slow to run in CI** → do **not**
    paper over it: write the test but `pytest.mark.skip`/`xfail` it with a clear reason
    string naming exactly what's wrong (e.g. the missing input, the error, the runtime),
    and record the finding in this fiber's outcome + a `felt history` note so we see it.
    A red/slow dry-run is *signal*, not a failure to hide.

- **③ Config-path existence — Candide-local.** A test that loads each config, extracts
  every path-shaped value, and asserts it **exists on this machine**. Skip cleanly (don't
  fail) when not running on Candide / when the mount is absent, so it's honest off-host.
  - Configs to cover: `cosmo_inference/notebooks/2D_bmodes_paper_workflow/config/config.yaml`,
    `notebooks/cosmo_val/cat_config.yaml`, `config/calibration/*.yaml`, and the cosmosis
    `*.ini` (there are ~57 — sample/scan them for `path`/`file`/`dir`-shaped keys and
    absolute-path values; you decide a sensible extraction heuristic). Use your judgment on
    what counts as a "path" (absolute paths, values under keys like `*_path`/`*_file`/`dir`,
    things that look like filesystem paths). Report how many paths checked / how many missing.

- **⑤ Symlink integrity.** A test that every **tracked symlink** resolves to an existing
  target. `git ls-files -s | awk '$1==120000'` lists symlinks; assert each target exists.
  Five lines; this tree has real symlinks that file-moves silently break.

- **⑥ Dangling-reference grep — harness.** A test parameterized by a **move-map** of
  `(old_internal_dir → new_location)` pairs. The map is **empty pre-reorg**, so the test is
  trivially green now; structure it so that adding a pair later makes it grep the tree for
  the old name and assert **zero hits** (excluding `.git/`, `.felt/`, `.snakemake/`,
  results, and the test file itself). This is the internal-reference cousin of the absolute-
  path mess we are deliberately *not* sweeping. Land the harness; it activates when moves begin.

**Done = all four exist, run green (or are honestly skipped/xfailed with a stated reason),
committed locally to `cleanup/restructuring`, with the dry-run probe's findings recorded.**

## Context

This is item-by-item back-pressure for the larger sp_validation restructuring (parent:
[[sp-validation-restructuring]]). Guard ① (`tests/test_imports.py`, 42 passed / 1 xfailed)
is the proven pattern — same dir, same pytest style, same "honest baseline with strict xfail
for known-broken" discipline. Match it.

**Run everything in the container.** `app` is an interactive bash function that won't exist
in this session; use the raw invocation:

```
apptainer exec --bind /home,/scratch,/automnt,/n17data,/n23data1,/n09data \
  /n17data/cdaley/containers/containers/ <command>
```

Container Python is **3.12** (`python3.12`); the full scientific stack (snakemake, pyyaml,
pytest, treecorr, …) is there. Run the suite with
`apptainer exec … python3.12 -m pytest src/sp_validation/tests/test_<name>.py -v`.

**Scope — additive only.**
- ONLY create new test files under `src/sp_validation/tests/`. If the Candide-local checker
  genuinely needs a helper, keep it as a pytest fixture/module under that dir — do not scatter
  scripts elsewhere. Do not modify existing code. Do not move or rename anything.

**⚠️ Git safety — the working tree is SHARED with a live interactive Claude session (the
driver of the parent constitution).** This is non-negotiable:
- You are on branch `cleanup/restructuring`. Commit your new test files there, locally.
- **NEVER** run `git add -A`, `git add .`, `git add -u`, `git commit -a`, `git reset --hard`,
  `git checkout -- .`, `git stash`, or `git clean`. Stage ONLY the specific new files you
  create, by explicit path (`git add src/sp_validation/tests/test_foo.py`).
- There are unrelated uncommitted/untracked files in the tree (other sessions' `.felt/`
  directories, an untracked `cosmo_inference/notebooks/2D_cosmic_shear_paper_plots/`). **Leave
  them completely alone** — do not add, move, clean, or touch them.
- **Do NOT push.** Commit locally only; the driver session + Cail review and push together.

**Exit (autonomous oneshot).** When the four guards are built and green (or honestly
skipped/xfailed with reasons): rewrite `outcome` to a one/two-line CLI headline (what's green,
what the dry-run probe found, what's skipped and why), append a `felt history` event, commit
locally, set `status: closed`, then `kill $PPID`. Do not self-`tempered`.
