# PRD — Native SACC likelihood (PR-7)

## Purpose

sp_validation's data vectors live in SACC files, but the CosmoSIS inference chain reads a DES-style "2pt FITS" file produced by a converter. This PR lets CosmoSIS consume the SACC file **natively**, so the inference chain no longer depends on the FITS converter for the fiducial (IA-only, ξ±) pipeline. CosmoSIS ships a native SACC likelihood (`likelihood/sacc/sacc_like.py`, `SaccClLikelihood`), but its ξ± path evaluates a radian-θ theory spline at arcmin data tags (theory ≈ 0, χ² silently wrong) and assumes a theory↔data ordering it never enforces — so it cannot be used as-is. This PR adopts it through a thin sp_validation-owned subclass that fixes those two defects (a third — an undefined-name bug in the unused `keep_tracers` path — is deliberately left untouched), and proves χ² equality against the converter path to machine precision.

## Desired end state

- An sp_validation-owned CosmoSIS module, `SaccLikeUnions(SaccClLikelihood)`, that reads a `{version}.sacc` file and produces a ξ± likelihood **numerically identical** to the existing 2pt-FITS + `2pt_like` path.
- The inference-prep step emits, from one assembled `{version}.sacc`, two CosmoSIS ini files: the legacy 2pt-FITS + `2pt_like` path (validating/legacy) and a `sacc_like_unions` path pointing directly at the SACC file (native).
- n(z) is read directly from the SACC NZ tracers — no 2pt-FITS is required anywhere on the native path.
- A test suite asserts χ² / theory-vector / post-cut-count equality between the two paths, and asserts that the ordering guard raises on a misordered file. The suite is skipped cleanly where CosmoSIS is unavailable.
- The shim is self-retiring: a tripwire test fails the day upstream fixes the unit bug, signaling the subclass can be deleted.

## Scope

**In scope:** the fiducial A_ia (IA-only, ξ±) inference pipeline. The subclass, the inference-prep rewiring, the n(z)-from-SACC path, and the equality validation.

**Out of scope (dormant, unchanged):** the PSF / xi_sys pipeline variant and the glass-mock rules stay on the existing FITS-assembly path (`cosmosis_fitting.py`); the PR body must state this explicitly rather than leave it silent.

## Interfaces / contracts

### The CosmoSIS module

An sp_validation-owned Python module file usable as a CosmoSIS likelihood, subclassing the upstream `SaccClLikelihood` and overriding only what the two hazards require:

```python
class SaccLikeUnions(SaccClLikelihood):
    def build_data(self):
        # 1. super().build_data() — scale cuts applied in arcmin
        #    (matches ini ergonomics and 2pt_like's angle_range convention)
        # 2. loud ordering guard: replay SaccClLikelihood's own theory loop
        #    verbatim and require the resulting row order equals the
        #    get_mean() insertion order, or raise ValueError. (See below.)
        ...
```

Contract points:

- **Angular units (the contract).** CosmoSIS theory θ is in radians; the SACC contract stores θ in arcmin. The subclass must guarantee three things: (a) theory extraction sees **radian** θ, so the upstream spline is evaluated on its own grid; (b) `self.sacc_data` and every save-by-copy path (`save_theory`, `save_realization`) emit **arcmin** θ, so a consumer re-ingesting saved output sees arcmin and does not double-convert; (c) scale cuts are applied in arcmin (matching ini ergonomics and `2pt_like`'s `angle_range` convention). The reference implementation achieves (a)+(b) by confining a radian-θ copy of the SACC data to the theory-extraction call and leaving `self.sacc_data` in arcmin — the how is the implementer's, but all three guarantees are contractual.
- **Ordering guard.** The upstream data vector is `get_mean()` (SACC insertion order); the upstream theory vector is built by replaying its own loop over data types × tracer combinations × points, which is aligned to the data *only when the file is grouped type-major-then-tracer in exactly that loop order*. The guard reconstructs the row provenance by replaying that same loop **verbatim** — do not sort, dedup-reorder, or otherwise reinterpret it; reproduce `SaccClLikelihood`'s literal iteration (`for dt in sacc_data.get_data_types(): for tc in sacc_data.get_tracer_combinations(dt): for each point ...`), whatever iteration order those upstream methods yield — and require the reconstructed order to equal the `get_mean()` insertion order, raising `ValueError` otherwise. Because both the guard and the theory extraction derive from the identical replayed loop, the guard is correct regardless of whether `get_data_types()` iterates sorted or insertion-order. A file grouped pair-major (e.g. a tomographic 2-bin file laid out tracer-major) breaks the alignment and must raise.
- **`like_name = 2pt_like`.** Set in the ini so chain post-processing and `extra_output` see identical DataBlock keys regardless of which likelihood engine ran.
- **Everything else rides upstream unmodified:** scale-cut grammar, Sellentin/Hartlap corrections, `save_theory`, Cℓ bandpower windows. Do not touch the `keep_tracers` removal path (it references undefined names upstream; the fiducial pipeline never exercises it).

### n(z) loading

The native path reads n(z) through CosmoSIS's `number_density/load_nz_sacc`, which reads NZ tracers named `source_i` (zero-based, contiguous) into the `nz_source` section. No 2pt-FITS n(z) file is produced or consumed on the native path.

### Inference-prep output

The inference-prep step consumes the assembled `{version}.sacc` and emits both:

1. **Legacy/validating path** — 2pt-FITS (via the SACC→2pt-FITS converter) + a `2pt_like` ini. On the fiducial native pipeline, inference-prep no longer *calls* `cosmosis_fitting.py`'s hand-rolled FITS assembly for its fiducial output; it routes through the converter instead. `cosmosis_fitting.py` itself is retained, not removed — it remains the byte-compare oracle for the converter PR and the assembly path for the dormant PSF/glass-mock variants.
2. **Native path** — a `sacc_like_unions` ini pointing at the `{version}.sacc` file.

Inference-prep is the (revived) `inference_prep` Snakemake rule, which consumes the assembled `{version}.sacc` and writes both inis into the inference output directory (`COSMO_INFERENCE`). Its ini **templates must be bound as DAG `input:`s, not rule `params`**, anchored on the running checkout via `INFERENCE_TEMPLATE_DIR`, so that editing a template forces a regeneration; a dry-run guard asserts that DAG edge exists.

## Acceptance criteria

Validation is **in-process module equality**, not full-pipeline: load both likelihood modules standalone (`build_module()` setup/execute) on a hand-built DataBlock carrying identical synthetic theory in `shear_xi_plus`/`shear_xi_minus`, feed the same underlying data through SACC (native) vs converter-FITS (legacy), and check:

1. **χ² / log-likelihood equal** across the two paths to machine zero (Δχ² = 0 at tight rtol), on both a synthetic realistic-covariance fixture (covariance ~ (0.1|ξ|)², to give χ² dynamics real teeth) and on real ξ-only data (N = 40). The claim is path-equality; the absolute χ² is not a target (it depends on the fixture seed and catalogue version). For reference, the implementation that produced this spec observed synthetic χ² = 1.18829133107 and real χ² = 420310.753739 — informative, not normative.
2. **Theory vectors equal** element-by-element (Δtheory = 0).
3. **Post-cut point counts equal** across both paths for the same scale cuts, including cut values placed off data points (edge semantics verified: no divergence between the two engines' cut behavior on tested ranges).
4. **Ordering guard raises** `ValueError` on a concretely-misaligned fixture: a 2-bin file whose rows are laid out **tracer-major** (all data types for pair 0, then all for pair 1) rather than the type-major-then-tracer order the upstream loop reconstructs. This layout is guaranteed to break alignment under the replayed loop (see the ordering-guard contract), so the fixture reliably triggers the guard.
5. **Unit-regression tripwire:** the raw upstream class (no shim), run on the same input, produces a χ² **~631× off** — the observed likelihood ratio from evaluating a radian-θ spline at arcmin tags (theory collapses to ≈ 0, so χ² → dᵀC⁻¹d). This test documents the upstream bug and fails the day upstream converts units correctly, signaling the shim can be retired. (The separate ~3437× figure is the radian↔arcmin θ ratio — the *cause* of the collapse, not a χ² factor; it belongs to the units contract above, not to this assertion.)
6. **Save-path regression:** saved theory/realization output carries **arcmin** θ tags, and re-executing on the saved output leaves χ² stable (no double-conversion).
7. **Skips cleanly without CosmoSIS:** the test module gates on `pytest.importorskip("cosmosis")` plus a CosmoSIS-standard-library directory env var (`CSL_DIR`); it skips in an image lacking CosmoSIS rather than failing. Real-data equality is additionally gated to the machine holding the catalogue.

## Non-goals

- **No upstream fix.** The upstream CosmoSIS unit bug and the undefined-name hazard in `keep_tracers` are not fixed here; they may be recommended as follow-ups in the PR body. This PR is a downstream shim, and it is designed to be deleted once upstream converts units correctly.
- **No reimplementation or fork of the likelihood.** Adoption is by subclass only; scale cuts, covariance corrections, and Cℓ windows ride the upstream implementation.
- **No tomographic support.** The fiducial pipeline is single-pair ξ±; the ordering guard exists precisely to fail loudly rather than silently mis-handle a pair-major tomographic file. Tomographic support is separate future work.
- **No change to the PSF / xi_sys or glass-mock pipelines,** which remain on the existing FITS-assembly path.
- **No CAMB or compiled CosmoSIS modules** in validation — the equality test uses pure-Python likelihood files plus a pip-installed `cosmosis`.
