---
name: SP Validation Restructuring
status: open
created-at: 0001-01-01T00:00:00Z
outcome: |-
    Reorg plan, settled. Core move: promote the things you run (cosmo_val, cosmo_inference)
    to top-level siblings over shared src/; curate notebooks/ to official demos; add a
    TRACKED scratch/ with nbstripout + size hook. Division of labor (settled with Sacha):
    WORKFLOW holds ALL analysis — generic, modular (Snakemake `module`), multi-person, makes
    diagnostic plots too → single results/. PAPER dir holds only final-figure assembly (PDF,
    color, layout, recombining for presentation) and may never touch Snakemake. SCRATCH is
    per-person ad hoc, can hold custom workflows. The line: everything up to the inputs of a
    paper figure is analysis. Shipped as a GitHub MILESTONE: (1) Sacha foundation merge, (2)
    the restructuring proposal (PR #188 — proposal lives in the PR description + the fiber,
    NOT committed to the repo; auto-closed by GitHub once its branch went empty),
    (3) glass mocks → tomography, (4) input pipeline → tomography.
---

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
- **Path translation** — collecting paper dirs breaks ~35 hardcoded absolute paths (mostly
  `cosmosis_fitting.py`, `eb_plots.py`, `contour_plots.py`, glass-mock scripts; several at
  `/home/guerrini/…`). Mechanical sweep, covering `scripts/` too → single repo-relative
  `results/`.
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
