# results/ — analysis products and diagnostic plots

The single top-level sink for everything the analysis produces: catalogues,
correlation functions, covariances, chains, and the diagnostic plots that vet
them. Contents are gitignored; the directory itself is kept.

How it fills up:

- **`workflow/` writes here.** Each run composes the shared rules under its own
  config and an output `prefix`, so products namespace under `results/<name>/`
  without clobbering another run's.
- **`papers/` reads from here.** Final-figure assembly takes finished products
  out of `results/` — it does not write analysis products back in.

Keep it disposable: nothing here is committed, so anything you cannot regenerate
from `workflow/` belongs in off-repo storage instead.
