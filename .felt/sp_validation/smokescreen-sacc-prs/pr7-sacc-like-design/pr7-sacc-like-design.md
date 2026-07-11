---
id: 01KX65DDG2WNEW8HEYJ2HZZQ5K
name: PR 7 native sacc_like adoption (design rulings)
tags:
    - decision
    - sacc
created-at: 2026-07-10T16:02:58.946571302+02:00
updated-at: 2026-07-10T17:11:27.695848437+02:00
outcome: 'Landed as PR #255 (closes #254): shim over CSL SaccClLikelihood (arcmin/rad copy-swap + ordering guard), inference_prep revived onto {version}.sacc, chi2 equality with 2pt_like observed to machine zero (synthetic + real data); 2 MEDIUM review findings fixed pre-draft.'
---

Design rulings for PR 7 of [[sp_validation/smokescreen-sacc-prs]] (PRD #241 row 7: "adopt CosmoSIS's native SACC likelihood, validated against the PR-3 converter path"), grounded in a source read of CSL `likelihood/sacc/` at commit 4fd2f1c (2026-06-12, shallow clone at `sp_validation-worktrees/csl/`).

## The prototype-grade verdict, now with mechanisms

Three concrete defects/hazards found in upstream `sacc_like.py` + `sacc_likelihoods/twopoint.py`:

1. **No angular-unit conversion (the killer).** `twopoint.py` L74–90 builds `SpectrumInterp(block[section,"theta"], theory)` — theory θ in **radians** (CosmoSIS convention) — and evaluates it at raw SACC `theta` tags (**arcmin**, our contract + firecrown convention). `2pt_like.py` L179–183 by contrast explicitly converts data to radians ("The units in cosmosis are all in radians"). With arcmin tags the spline is evaluated ~3437× outside its grid → SpectrumInterp returns 0 outside range → theory ≈ 0, χ² = dᵀC⁻¹d, **silently**. The HSC Y3 use was the Cℓ path (ℓ unit-free), which is why this never bit upstream.
2. **`keep_tracers` removal path references undefined `t1, t2`** (`sacc_like.py` L106: `s.remove_selection(tracers=(t1,t2))` — those names are stale loop variables; NameError or wrong-pair removal if the filter ever matches). We don't use `keep_tracers`; do not touch.
3. **Theory↔data ordering is assumed, not enforced.** Data vector = `s.get_mean()` (insertion order); theory = loop over `get_data_types()` × `get_tracer_combinations()` × points. A comment (L35) claims `to_canonical_order` was called on load — **it never is**. Aligned only when the file is grouped type-major-then-tracer in exactly that loop order. Our single-pair files ([ξ+ block; ξ− block]) align; tomographic pair-major files would silently misalign — same bug class the PR-2/PR-3 adversarial reviews caught.

## Rulings

- **Adopt by subclass, not fork or reimplementation**: `SaccLikeUnions(SaccClLikelihood)` — an sp_validation-owned CosmoSIS module file. Overrides `build_data()`: call super (cuts happen in arcmin, matching ini ergonomics and 2pt_like's angle_range convention), then (a) convert `theta` tags arcmin→rad on the in-memory `sacc_data` for real-category types, (b) run a loud ordering guard (reconstruct the theory-loop index order; require it equals arange — ValueError otherwise). Everything else (scale-cut grammar, Sellentin/Hartlap, save_theory, windows for Cℓ) rides upstream unmodified. Shim dies when upstream fixes units.
- **`like_name = 2pt_like`** in the ini (base class honors the option) so chain post-processing and `extra_output` see identical block keys across engines.
- **n(z) natively**: CSL ships `number_density/load_nz_sacc` (verified: reads NZ tracers `source_i`, zero-based contiguous, → `nz_source` section) — the native path needs no 2pt-FITS anywhere.
- **inference_prep rewiring honors PR 4's dormant-note contract**: consume assembled `{version}.sacc`, emit (a) converter 2pt-FITS + 2pt_like ini [the validating/legacy path, retiring cosmosis_fitting.py assembly] and (b) sacc_like ini pointing at the SACC [the native path]. Scope = the A_ia (IA-only, ξ±) fiducial pipeline; the PSF/xi_sys variant and glass-mock rules stay dormant/cosmosis_fitting-based (flagged in PR body, not silently).
- **Validation = in-process module equality**, not full-pipeline: both modules loaded standalone (`build_module()` setup/execute on a hand-built DataBlock carrying identical synthetic theory in `shear_xi_plus/minus`), same data through SACC vs converter-FITS → assert χ²/like equal (tight rtol), theory vectors equal, post-cut point counts equal; teeth = data perturbation moves both identically, unit regression (upstream class without shim → wildly different χ², documenting the bug), ordering guard raises on a pair-major 2-bin file. Real-data equality test candide-gated like PR 3's. No CAMB, no compiled CSL modules needed — pure-python likelihood files + pip `cosmosis` (3.25.2 verified importable py3.12; installed into shared venv).
- **Branch**: `feat/sacc-7-sacc-like` from `feat/sacc-4-migration` + merge `feat/sacc-3-converter` (needs PR 4's workflow surface AND PR 3's converter; they share the sacc-2 base and are disjoint). PR base = feat/sacc-4-migration; body flags the merged-in #249 content until the stack collapses.
- **Upstream CSL issues (units, L106) are out of scope** for a CosmoStat/sp_validation draft-PR fiber — recommended as follow-up in the PR body; Cail/team decide about posting to joezuntz/cosmosis-standard-library.

## Open (settle during build)

- Edge semantics of scale cuts at exact boundary values (sacc `remove_selection` lt/gt is inclusive-keep; 2pt_like's spectrum cut may differ) — test with off-point cut values, verify same post-cut N on both paths.
- `get_data_types()` order determinism (insertion-unique vs sorted) — probe empirically; the ordering guard makes either safe but the guard's construction needs the truth.
- Whether CI can run any of the cosmosis tests (image lacks cosmosis) — gate with `pytest.importorskip("cosmosis")` + CSL-dir env (`CSL_DIR`), cosmo_numba precedent.

## As landed (PR #255, sub-issue #254)

The open questions above all settled during the build:

- **Scale-cut edge semantics**: no divergence observed — both engines produced identical post-cut N and χ² on the tested ranges; cut values in the templates don't land on data points.
- **`get_data_types()` order**: the ordering guard reconstructs the upstream theory loop faithfully regardless (tested against a pair-major 2-bin file — raises loudly); the single-pair files align.
- **CI**: `test_sacc_like.py` gates on `importorskip("cosmosis")` + `CSL_DIR`, skips cleanly in the image; the `[cosmosis]` extra is inert there (Dockerfile installs explicit extras).

Two builder deviations accepted as improvements: a `_realistic_sacc` fixture (cov ~ (0.1|ξ|)²) because PR-3's byte-compare fixtures have χ²~1e-9 — no teeth for χ² dynamics; real-data equality scoped ξ-only (the full 2pt-FITS's COVMAT_CELL/τ blocks trip `twopoint.from_fits`, and ξ± is the A_ia scope).

Adversarial review (5 opus lenses, 3 sonnet refuters/finding, 17 agents) confirmed 2 of 4 findings, both MEDIUM, both empirically reproduced and fixed pre-draft:

1. **save_theory/save_realization wrote radian θ tags** — the shim's in-place tag mutation leaked into upstream's save-by-copy paths (a consumer reading arcmin is 3437× off; re-ingestion double-converts to ~8.5e-8). Fix: `self.sacc_data` stays arcmin; theory extraction swaps in a radian-θ copy (`_sacc_data_rad`) around `super().extract_theory_points` in try/finally. Regression test pins arcmin tags in saved output + re-execution χ² stability.
2. **Ini templates were rule `params`, not `input`s** — no DAG edge, so template edits never regenerated configs. Fix: templates bound as inputs, anchored on the running checkout (`INFERENCE_TEMPLATE_DIR`) with generated configs still landing in `COSMO_INFERENCE`; dry-run guard asserts the edge.

Killed by refuters (correctly): boundary-semantics test-coverage nit; N=5-vs-20 fixture-size nit.

Final observed numbers: shim ≡ 2pt_like at machine zero — synthetic χ² = 1.18829133107 (Δχ²=0, Δtheory=0), real SP_v1.4.6_leak_corr ξ-only χ² = 420310.753739 (N=40, Δχ²=0); tripwire: raw upstream ~631× off. Test surfaces: 8 sacc_like + 5 generator + 3 dry-run guards = 16 passed (independently re-run by the orchestrator); fast suite 202 passed / 1 pre-existing unrelated failure (config-paths-on-candide); ruff clean.
