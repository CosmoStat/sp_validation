---
id: 01KTCHX03YS1ANPKJ5WTRGTZ3E
name: 'Front-page makeover: badges, CI, contributor docs'
tags:
    - ci
    - docs
created-at: 2026-06-04T02:53:52.875583891+02:00
outcome: Replaced 13 mostly-dead README badges (pointed at impossible user martin.kilbinger/...) with 8 honest ones on CosmoStat/sp_validation; turned the empty 'CI' into a real pytest-in-container gate; fixed a testpaths bug and the contributor docs.
---

Cleanup of the `sp_validation` repo front page, committed as `2db5cbc` on branch
`chore/front-page-makeover` (off the post-merge develop, `6788cfc`). Builds on the
CI/docs modernization in [[docs-deploy-modernized]].

**Badges (13 → 8).** Every old badge pointed at `martin.kilbinger/sp_validation` —
not a valid GitHub username (dots are illegal), so all dynamic ones 404'd red. The
table was the stock CosmoStat template (pypi, codecov, CodeFactor, wemake, pyup),
but the repo isn't on PyPI/codecov/CodeFactor and pyup.io is defunct. New flat row
points at `CosmoStat/sp_validation`: docs · CI · container · python 3.11+ · license ·
ruff · contribute · conduct. `wemake` → `ruff` (the linter actually enforced).

**The testpaths bug (reusable gotcha).** `pyproject.toml` had `testpaths =
["sp_validation"]`, but the package is `src/sp_validation`. That path doesn't exist,
so pytest warns and **falls back to recursive search from cwd**, which sweeps up
`cosmo_inference/.../mock_cosebis_bias_test.py` (matches `*_test.py`) — it runs
`plt.style.use(...)` at import and **errors during collection**. So bare `pytest`
was broken. Fixed → `testpaths = ["src/sp_validation/tests"]`; bare `pytest` now
collects 36 cleanly. Full suite: **36 passed in ~177s** in the container (needs the
full scientific stack; the old CLAUDE.md claim of "~18s" was stale).

**Real CI.** There was no working test CI: `ci-build.yml` was a py3.8/conda/
`setup.py test` relic triggered on a nonexistent `master`; the only live workflow
just did `import sp_validation`. Now the image workflow runs the full suite *inside
the freshly-built image before pushing* (failing test blocks publication), with the
`[test]` extra baked into the dev image (`Dockerfile` `-e .` → `-e '.[test]'`). The
Dockerfile already `COPY . /sp_validation` + editable-installs, so the image carries
the test files — no extra setup needed. Deleted `ci-build.yml` and `.pyup.yml` (the
latter referenced three requirements files that don't exist). Ruff `target-version`
py38 → py311.

**Authorship reconciled, not invented.** Old README listed Authors = Kilbinger,
Guinot. But `pyproject.toml` *and* `CONTRIBUTORS.md` say authors = Kilbinger, Daley,
Guerrini — set by Martin himself in commit `5684e08`. Aligned the README to that
(Guinot → contributor). Lucie's surname is **Baumont** (per her own commits
`lucie.baumont@cea.fr`), not "Beaumont"; fixed in CONTRIBUTORS.md along with the
`Ayçoberry` mojibake.

**GitHub Pages is live** — `cosmostat.github.io/sp_validation` returns 200, so the
docs badge link resolves.

**Cluster-data tests, now CI-aware.** The first CI run surfaced that three
`TestCosmologyValidation` tests (`test_additive_bias_base_columns`,
`test_additive_bias_leak_corrected_columns`, `test_catalog_paths_exist`) load real
UNIONS catalogues off `/n17data` — they pass on-cluster and in the container, but
can't on a bare runner. The new test-before-publish gate correctly *blocked the
push* on that first run. Fixed by guarding them with
`skipif(not Path("/n17data").exists())` so they run where the data lives and skip
off-cluster. CI now lands **33 passed, 3 skipped**.

**Status:** pushed to `chore/front-page-makeover`, PR
[#195](https://github.com/CosmoStat/sp_validation/pull/195) open into develop, CI
green. Awaiting merge.

Minor heads-up for later: the docker `actions/*` in `deploy-image.yml` still run on
Node 20, which GitHub force-migrates to Node 24 in June 2026 — non-blocking, just a
future bump.
