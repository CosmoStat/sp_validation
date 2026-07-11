---
## comment by cailmdaley
All seven draft PRs are now open — the PRD's implementation series is complete and ready for team review. The map:

| PRD §7 row | PR | Sub-issue | Content |
|---|---|---|---|
| 1 — dependencies | #243 | — | py3.12 floor; `sacc` core dep; `[blinding]` extra (firecrown 1.15.1 + smokescreen 1.5.6 + pyccl 3.3.4, exact pins for seed-reproducibility) |
| 2 — `sacc_io` writers | #245 | — | writers/readers for ξ±, pseudo-Cℓ, COSEBIs, pure-E/B, ρ/τ, n(z), covariance assembly; round-trip + ordering-contract tests |
| 3 — converters | #249 | #246 | SACC→2pt-FITS byte-compared against the current `cosmosis_fitting.py` output (synthetic + real SP_v1.4.6_leak_corr / glass_mock); SACC↔OneCovariance glue |
| 4 — cosmo_val migration | #251 | #247 | cosmo_val + Snakemake produce per-statistic SACC parts, assembled into the terminal `{version}.sacc` (+ `{version}_xi_fine.sacc`) |
| 5 — firecrown likelihood | #250 | #248 | firecrown likelihood over the analysis SACC + CAMB↔CCL cross-check test |
| 6 — Smokescreen blinding | #253 | #252 | blinding step with §4(b) hash-commitment custody; acceptance observed: B-mode estimators unchanged on mocks (ΔBₙ/Bₙ ~ 6.5e-6 under a ~20% E-shift) |
| 7 — native SACC likelihood | #255 | #254 | CosmoSIS `sacc_like` adopted via a unit-fixing shim, χ² equality with the PR-3 converter path observed to machine zero (synthetic + real data) |

Notes for reviewers:

- **The PRs stack** (each based on the branch it builds on: 2←1, 3/4/5←2, 6←5, 7←4+3). GitHub retargets automatically as the stack merges from the bottom; until then a PR's diff can include its parents' files — each body says which files are its own.
- **One PRD amendment**, flagged on the layout: two SACC files per catalogue version (`{version}.sacc` analysis vector + `{version}_xi_fine.sacc` COSEBIs/pure-EB integration input) — sacc 2.4's covariance model makes a 10000-bin fine grid + full analysis covariance in one file physically impractical (details in #245).
- **Open questions from the PRD** (custody model, shift envelope, fiducial config) are implemented on the stated defaults, clearly marked configurable — they gate discussion, not code.
- Every PR was adversarially reviewed before opening (fresh-context multi-lens review, findings verified by independent refuters); confirmed findings are documented in each PR body.

— Claude (Fable) on behalf of Cail.
