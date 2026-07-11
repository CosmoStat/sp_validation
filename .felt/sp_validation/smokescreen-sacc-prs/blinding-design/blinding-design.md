---
id: 01KX5T7GDXVHPTK7HEB920T15D
name: PR 6 blinding architecture (rulings)
tags:
    - decision
    - blinding
created-at: 2026-07-10T12:47:31.005842093+02:00
updated-at: 2026-07-10T12:47:31.005842093+02:00
outcome: Own seeded RNG draws (dS8,dOm); Smokescreen gets deterministic absolute {sigma8, Omega_c} (coarse xi only); pseudo-Cl + fine grid via direct CCL theory difference; COSEBIs/pure-EB re-derived through pipeline estimators guarded by a round-trip identity test; 4(b) custody = commit sha256(seed), encrypt seed+truth, strip seed_smokescreen.
---

The architecture for [[sp_validation/smokescreen-sacc-prs]] PR 6 (issue #252, PRD #241 §3/§4), fixed before the build from the verified facts in [[sp_validation/smokescreen-sacc-prs/smokescreen-api-facts]] and the [[sp_validation/smokescreen-sacc-prs/sacc-layout-contract]]. Four rulings:

**Randomness is ours, Smokescreen executes.** The PRD's uniform (S8, Ωm) box is inexpressible in Smokescreen's shift dict (keys must be CCL params; tuple = absolute-bounds draw via *global* `np.random.seed`). So `draw_hidden_cosmology(seed, config)` is a pure function — sha256(seed) → `default_rng` → (ΔS8, ΔΩm) in the ±0.075/±0.1 box → hidden `TheoryConfig` via `from_overrides` — and Smokescreen receives deterministic absolute `{sigma8, Omega_c}` (its replace-not-add semantics never bite on single values). Reproducibility contract for §4(b): seed → shift is exact, forever.

**Two shift paths, one hidden cosmology, pinned together by test.** Coarse ξ± goes through `ConcealDataVector` proper (extract coarse sub-SACC → conceal → verify → apply → merge back; `_verify_sacc_consistency` passes by construction because the likelihood reads its data vector from the same sub-SACC). Pseudo-Cℓ (W @ ΔCℓ_EE on the stored BandpowerWindows; ΔBB=ΔEB≡0, shift is pure E) and fine-grid ξ± (Δξ± at the fine θ) are direct CCL theory differences at the same two cosmologies — the fine grid cannot pass through firecrown (ConstGaussian densifies even a diagonal cov; contract ruling), and the PR-5 likelihood models ξ± only. A cross-path consistency test (Smokescreen factor vs direct-CCL diff at coarse θ) pins the paths; the P(k) recipe (CAMB HMCode2020, halofit_version string) must match — load-bearing, proven in PR 5.

**Derived statistics are re-derived, never shifted.** COSEBIs and pure-E/B values in the analysis file are replaced by re-running the *same pipeline estimators* (`b_modes.calculate_cosebis`, `calculate_pure_eb_correlation`) on blinded ξ± — no re-implemented kernels. The seam is guarded by a round-trip identity test: unshifted re-derivation must reproduce the analysis-file values exactly. Covariances never change (blinding hides the vector, not the uncertainty).

**Custody = §4(b) hash commitment.** Blind CLI: OS-entropy seed → blind immediately → publish sha256(seed) (commitment JSON, repo-committed) → encrypt seed + true vectors (`smokescreen.encryption`), delete plaintexts → strip `seed_smokescreen`, stamp `concealed=True`, `blind` label, `blind_commitment` (public hash ties file→commitment). Unblind verifies the hash before subtracting. PR-4's `blind=A` provenance naming reconciles here: the label in blinded metadata is the same axis.

Snakemake wiring is deliberately out of scope (PR 6 stacks on PR 5, not PR 4); it follows once both land.

## As landed (PR #253, review-hardened)

Four amendments earned during build + adversarial review (16 findings → 10 confirmed → all fixed pre-draft):

- **The Smokescreen path needs `systm_dict`, not just cosmo.** Its theory overlay keeps any firecrown default absent from `cosmo.to_dict()` — IA amplitude rode along at 0.5 vs our fiducial 0.0, polluting the coarse factor ~7% at low θ (invisible to round-trips; caught by demanding the realized-factor-vs-direct test). All non-cosmological fiducial params now pinned via `systm_dict=theory.ia_params()`; regression test guards it.
- **Direct paths are two-tracer.** Cross-pairs get `angular_cl(tracer_i, tracer_j)` with each bin's own n(z) — the auto-n(z) approximation was killed by review (2-bin Smokescreen-vs-direct observed 2.5e-10).
- **Commitment binds (seed, config).** Blinded metadata + commitment JSON carry a config digest (envelope + fiducial); unblind verifies both, so a wrong envelope can't silently subtract a wrong shift.
- **Pure-EB seam follows the pipeline's edge-based integration bounds** (identity vs `calculate_pure_eb_correlation`: rel diff 0.0); boundary-degenerate reporting points are NaN exactly as the pipeline's own estimator — production reporting grids are strict sub-ranges so it never fires on real files.

Acceptance (observed, container): ΔBₙ/Bₙ = 6.5e-6; Δξ⁺_B = 0.7%, Δξ⁻_B = 2.3% of an injected 2e-5 B-mode; the ΔB is fixed-absolute (E-shift leakage through the estimator), independent of B amplitude.
