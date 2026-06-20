# papers/ — final-figure assembly

Per-paper directories (`papers/<paper>/`) for turning analysis products into the
figures that go into a manuscript. Presentation, not analysis: PDF output,
colour, layout, panel composition — all tied to one paper.

What belongs here:

- **Final-figure assembly** that reads finished products from `results/` and
  emits publication-ready figures. A paper may compose the shared `workflow/`
  rules (with its own config and output `prefix`), or it may never touch
  Snakemake at all.
- **Paper-specific config and presentation code** that only makes sense for this
  one manuscript.

What does *not* belong here:

- **The analysis itself** — producing catalogues, correlation functions,
  covariances, chains, and diagnostic plots — lives in `workflow/`. The boundary
  is the inputs to a figure: everything up to that point is analysis.
- **Exploratory or one-off work** lives in `scratch/<you>/`.
