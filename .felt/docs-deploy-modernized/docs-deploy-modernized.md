---
name: 'Docs deploy modernized: gh-pages CI + Sphinx 8'
tags:
    - docs
    - ci
created-at: 2026-06-04T00:47:23.328929872+02:00
outcome: Replaced the dead master/py3.8/conda cd-build.yml with a deploy-docs job in deploy-image.yml that builds inside the freshly-published container image and pushes API docs to gh-pages on develop. Modernized the docs stack and conf.py for Sphinx 8+/9.
---

The old `cd-build.yml` was a relic of the pre-uv era: triggered on `master`, set up py3.8 via conda, ran `setup.py test`, and deployed Sphinx API docs to gh-pages. Fully dead since the move to `develop`/uv and the container CI. Removed it; the image build + tests now live in `deploy-image.yml` (see [[docker-build-uv-venv-base]]).

**New CI shape.** Docs are a `deploy-docs` job in `deploy-image.yml`, `needs: build-and-push-image`, gated `if: github.ref == 'refs/heads/develop'`. It runs *inside* the just-published `ghcr.io/cosmostat/sp_validation:develop` image (which already carries the full scientific stack autodoc needs to import), `uv pip install '.[docs]'`, then `sphinx-apidoc … src/sp_validation` + `sphinx-build`, and deploys with `peaceiris/actions-gh-pages@v4`. Building in the image is the key move — it avoids recompiling pymaster/pyccl on a bare runner.

**Stack haussmannized.** The `docs` extra dropped exact py3.8-era pins (sphinx 4.3.1, myst 0.16, nbsphinx, jupyter) for `>=` floors: `sphinx>=8`, `myst-parser>=4`, `numpydoc>=1.8`, `sphinxcontrib-bibtex>=2.6`, `sphinxawesome-theme>=5.3`. nbsphinx/nbsphinx-link/jupyter removed entirely — there are no `.ipynb` in the docs, the notebook-autodiscovery code in conf.py was long commented out.

**conf.py PEP 621 metadata trap.** Two `conf.py` lookups broke under modern packaging because `[project]` metadata no longer populates the legacy fields: `mdata['Home-page']` (removed; was only used by dead nbsphinx badge code) and `mdata['Author']` (PEP 621 `authors=[{name=…}]` lands in `Author-email` as `"Name <email>, …"`, not `Author`). Fixed author to `mdata.get('Author') or re.sub(r'\s*<[^>]*>','', mdata.get('Author-email','CosmoStat'))` → "Martin Kilbinger, Cail Daley, Sacha Guerrini". Also migrated removed-config: `autodoc_default_flags`→`autodoc_default_options`, `html_use_smartypants`→`smartquotes`, and the sphinxawesome 3.x theme options (`nav_include_hidden`/`show_nav`/`breadcrumbs_*`) → 5.x/6.x (`show_prev_next`/`show_scrolltop`/`globaltoc_includehidden`), dropping the bare `'sphinxawesome_theme'` extension entry (5.x loads via `html_theme` only).

**Verified** by an ephemeral-uv smoke build (sphinx 9.1.0, myst 5.1.0, theme 6.0.2): full pipeline builds clean except `sphinx-apidoc`+autodoc of the real modules, which needs the container env the CI job provides. The remaining warnings are pre-existing content issues (markdown header levels, two stray `.md` not in any toctree).

**One-time repo setting:** GitHub Pages must be enabled pointing at the `gh-pages` branch; the first develop run creates that branch.
