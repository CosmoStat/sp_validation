# Restructuring `sp_validation`

*A proposal — draft, no implementation yet. The rich version of this document is
[`proposal.html`](./proposal.html).*

One organizing principle — **the things you run live at the top** — a clean three-way
split between analysis, papers, and scratch, and a modular workflow built for more than one
person.

---

## The shape

Today `cosmo_val` is buried inside `notebooks/` while `cosmo_inference/` is top-level, so
you constantly hunt for where each one lives. The fix: the things a person actually runs
sit side by side at the top, sharing library code in `src/` underneath.

```
sp_validation/
├── src/sp_validation/   library code (+ glass_mock core)
├── cosmo_val/           validation: code + config        (promoted from notebooks/)
├── cosmo_inference/     inference: code + config         (cosmosis / cosmocov)
├── workflow/            ALL analysis — modular Snakemake, multi-person → results/
├── papers/             final-figure assembly only (PDF, colour, layout)
├── scripts/            real reduction scripts (catalog builders, masking)
├── scratch/            per-person — ad hoc work + personal workflows (tracked)
├── notebooks/          curated to official demos / tutorials
├── results/            analysis products + diagnostic plots (contents gitignored)
└── docs/  tests/  config/
```

---

## Division of labor

The boundary is **the inputs to a paper figure**: everything up to that point is analysis;
the figure itself is presentation.

- **`workflow/` — all analysis.** Generic, reusable, modular, organized for multiple people.
  Produces analysis products *and diagnostic plots* (sp_validation makes many — they go to
  `results/`). The bulk of the work lives here.
- **`papers/<paper>/` — final-figure assembly only.** The figure PDF, colours, layout,
  recombining data for presentation. Tied to one paper, and may never touch Snakemake.
- **`scratch/<person>/` — personal and ad hoc.** Experiments and one-off custom workflows.
  Tracked, because seeing each other's scratch is useful.

---

## How the workflow scales — modular, not monolithic

Nothing in this analysis is computed once: the catalog changed ~20× in the first release
suite, and every paper varies the data vector, covariance, and inference. So the workflow
is *parameterized* — the rules are shared, the config changes each time. Snakemake's
`module` directive imports the rules under your own config and an output `prefix`, and lets
you override any single rule:

```python
module analysis:
    snakefile: "../../workflow/Snakefile"
    config:    config              # this run's catalog, cuts, blind
    prefix:    "results/bmodes"    # products land here — no clobbering

use rule * from analysis
# swap is per-rule: redefine just the data-vector rule to override it
```

One top-level `results/`; each run namespaces under `results/<name>/` via the prefix, so
people don't clobber each other. A `--dry-run` on each composition is the safety net that
lets the structure grow without silent breakage.

---

## Cleanup

- **Delete** `defunct/` (quarantined since 2024) and the exploratory 2021–22 notebooks — it
  all stays in git history.
- **Curate** `notebooks/` to official demos and tutorials; personal scratchy ones move to
  `scratch/`.
- **Discipline** via tooling, not bans: `nbstripout` strips notebook outputs on commit (the
  repo's weight today is committed notebook outputs), plus a pre-commit size hook.
- **Path translation** — collecting the paper dirs breaks ~35 hardcoded absolute paths; a
  mechanical sweep rewrites them (scripts included) to the single repo-relative `results/`.

---

## The milestone

A suite of PRs, in sequence:

1. **Foundation** — merge pending local code into `develop`. *(Sacha)*
2. **Restructuring** — this proposal. *(this PR — draft, no implementation yet)*
3. **Glass mocks → tomography.**
4. **Input pipeline → tomography.**

This PR is the proposal only. Implementation follows once the foundation merge lands and
the shape is agreed.

---

*— Claude on behalf of Cail*
