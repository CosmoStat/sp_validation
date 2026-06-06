---
id: 01KTCHX02W2H6V4A8S86GDT112
name: 'Docs: real narrative pages, not template stubs'
status: closed
tags:
    - constitution
    - sp_validation
    - docs
created-at: 2026-06-04T16:51:19.13286682+02:00
closed-at: 2026-06-04T21:48:33.143370641+02:00
outcome: 'SHIPPED & LIVE. PR #196 merged to develop (574f4bf); deploy-docs published to gh-pages and cosmostat.github.io/sp_validation now serves the real pages — verified live: landing page shows the writing-pass content (''Four main tasks make up a run''), zero template boilerplate, all four nav sections, about-page philosophizing gone. Delivered: four real narrative pages (writing-pass-polished), existing guides woven into a regrouped toctree, citing stub removed, a new deploy-docs.yml CI (build-on-PR + deploy-on-develop + downloadable docs-html artifact), and docstring RST fixes (78b4d17). Desired state realized. Interactive: session alive for Cail; fiber his to temper/close on the kanban.'
shuttle:
    enabled: true
    kind: oneshot
    interactive: true
    host: candide
    project_dir: /automnt/n17data/cdaley/unions/pure_eb/code/sp_validation
    agent: claude-opus
tempered: true
---

# Docs: real narrative pages, not template stubs

The `sp_validation` developer docs deploy to [cosmostat.github.io/sp_validation](https://cosmostat.github.io/sp_validation/) but read as blank. Every hand-written page is still the unmodified Python-package-template placeholder. The Sphinx *machinery* is fine — this is a **content** job, not a plumbing one.

## Desired State

The deployed docs introduce the package and orient a developer, instead of showing template boilerplate. Concretely, the four stub pages under `docs/source/` carry real, grounded prose:

- **`index.rst`** — a true landing page: what `sp_validation` is (validation of weak-lensing galaxy/star-shape catalogues produced by ShapePipe), the four tasks it performs (shear validation → post-processing → cosmology validation → cosmology inference), and how to navigate the docs. Strip the template `.. note::`, the Admonitions / Code-Blocks demo sections, everything that literally says "you should put some introductory information here."
- **`about.rst`** — the package in its CosmoStat / UNIONS context; the problem it solves; authors and contributors (from the README); contact.
- **`installation.rst`** — the *real* install story: container / Apptainer first (the recommended path — `apptainer build --sandbox sp_validation docker://ghcr.io/cosmostat/sp_validation:develop`), then dev install via `uv pip install -e '.[develop]'`. The package is **not on PyPI** — remove the template's `pip install sp_validation` and PyPI-URL instructions; don't ship a false claim.
- **`quickstart.rst`** — a real first-run walkthrough grounded in the actual entry points: the validation run, `notebooks/params.py` configuration, the `cosmo_inference/pipeline.sh` flags (`--pcf` / `--covmat` / `--inference` / `--mcmc_process`). Not the template's bare "import sp_validation" stub.

The existing *real* docs — `run_validation.md`, `post_processing.md`, `Leakage_object_Tutorial.md` — are woven into the `toc.rst` toctree coherently (they exist, but the narrative around them is missing), and the build no longer warns about stray `.md` files outside any toctree.

The build is **verified clean**, not assumed: build the docs the way CI does and inspect the rendered HTML. Either inside the container (`ghcr.io/cosmostat/sp_validation:develop`, which carries the stack autodoc imports) or via `uv pip install -e '.[docs]'`, run `sphinx-apidoc -t docs/_templates -feTMo docs/source src/sp_validation` then `sphinx-build -b html docs/source docs/_build`, and open / inspect `docs/_build/index.html`. No autodoc import failures leaving API pages empty; the landing page renders real content.

Work on a branch off `develop` (the prior docs PRs used `chore/…`) and open a PR — **don't merge**. Docs redeploy when Cail merges to `develop`; PR-per-change is this repo's grammar.

### Explicitly not

- **Not re-doing the CI / deploy / Sphinx stack** — that landed in [[docs-deploy-modernized]] and works. Don't go hunting for broken wiring: the apidoc, autodoc-with-source-links, theme, and gh-pages deploy all function (verified — the live nav is fully populated and API module pages carry documented members).
- **Not a theme / design overhaul** — keep the `sphinxawesome_theme`.
- **Don't invent science.** Every statement about what the package does is grounded in the README, `CLAUDE.md`, or the code. Where the right *framing* or emphasis needs Cail's taste (what to foreground, scientific nuance, how far the quickstart should walk), stage it and ask in the interactive phase rather than guessing.

## Context

**The reframe.** Cail's read was "we made a PR to hook up the docs but they're super blank — not hooked up correctly." Ground truth (verified 2026-06-04): the wiring is *fully functional*. CI builds inside the published container, `sphinx-apidoc` + autodoc generate populated API pages (e.g. `sp_validation.cosmology`, `.b_modes` render documented members with `[source]` links), the awesome theme renders, gh-pages deploys, and the sidebar nav lists every module and guide. What reads as "blank" is that `index`, `about`, `installation`, `quickstart` are still the **python-package-template placeholders** — so the landing page and Getting Started section have no real content. The job is to write that content, not to fix plumbing.

**Source material is already written** — most of this is porting / adapting, not inventing:

- `README.md` — strong prose overview, the four-task summary, authors / contributors, the container-install walkthrough.
- `CLAUDE.md` — architecture and per-module map (`b_modes`, `calibration`, `cat`, `cosmo_val`, `cosmology`, `rho_tau`, …), the `cosmo_inference/pipeline.sh` flags, container usage.
- The existing real docs (`run_validation.md`, `post_processing.md`, `Leakage_object_Tutorial.md`) — content to weave in, not duplicate.

**Build environment (candide).** This dispatches on candide in the repo (`/automnt/n17data/cdaley/unions/pure_eb/code/sp_validation`). The container is the reliable autodoc env; the `.[docs]` install also works. [[docs-deploy-modernized]] records the `conf.py` PEP-621 metadata traps already fixed, and notes the *remaining* warnings are exactly these content issues (markdown header levels; two stray `.md` not in any toctree).

**Prior art.** [[docs-deploy-modernized]] is the CI / stack PR Cail is thinking of — it hooked up the deploy. [[front-page-makeover]] cleaned the *README* badges (the GitHub front page), which is distinct from the Sphinx landing page.

**Interactive handoff.** Do a strong first pass solo — real `index` / `about` / `installation` / `quickstart` grounded in the material above, build locally, confirm the rendered landing page shows real content — then leave the session alive at a checkpoint for Cail to refine the scientific framing, emphasis, and quickstart narrative before the PR is finalized.
