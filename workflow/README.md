# workflow/ — shared analysis

The modular Snakemake workflow that produces every analysis product and
diagnostic plot. Generic and reusable: the rules live here once, organized for
several people to run them side by side without stepping on each other.

What belongs here:

- **Reusable rules and their scripts** — `Snakefile`, `rules/`, `scripts/`,
  and the shared helpers in `common.py`. Anything generic enough that a second
  run or a second person would want it.
- **Everything up to a paper figure's inputs.** The boundary is the inputs to a
  figure: catalogues, correlation functions, covariances, chains, and the
  diagnostic plots that vet them — all analysis, all here. Outputs go to the
  top-level `results/`, never alongside the rules.

What does *not* belong here:

- **Final-figure assembly** (PDF, colour, layout for one paper) lives in
  `papers/<paper>/`.
- **One-off, exploratory work** lives in `scratch/<you>/`; promote it here only
  once it is generic and worth sharing.

Runs stay modular, not monolithic: a paper or run composes these rules with
Snakemake's `module` directive under its own config and an output `prefix`, so
each namespaces cleanly under `results/<name>/`.
