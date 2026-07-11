# PRD — Smokescreen blinding wiring

Repo: `CosmoStat/sp_validation`. Depends on the `UNIONS-WL/Smokescreen` fork exposing the theory-backend concealment entry point and the CCL-native draw described under *External contract*.

## Purpose

The tomographic data vector must be blinded before analysis so that no one can read S8 off the measured statistics until the collaboration agrees to unblind. This PR wires the measured SACC data vector through Smokescreen: it shifts ξ± and pseudo-Cℓ by a difference of theory vectors between the fiducial cosmology and a hidden cosmology drawn inside a fixed amplitude envelope, re-derives the dependent statistics from the blinded ξ±, and manages the seed under a public hash-commitment scheme so both the blind and the unblind are verifiable without a trusted keyholder.

## Desired end state

- A **blind** CLI takes a master SACC file (the full `sacc_io` product) and produces a blinded master SACC file plus an encrypted seed bundle and a repo-committable commitment JSON. The plaintext true vector is deleted at blinding time.
- The blinded file carries shifted ξ± (coarse and fine θ grids) and shifted pseudo-Cℓ; its COSEBIs and pure-E/B blocks are re-derived by the pipeline's own estimators from the blinded ξ±; its covariance and its ρ/τ PSF diagnostics are byte-identical to the input.
- An **unblind** CLI takes a blinded file and the encrypted bundle, verifies the commitment, recomputes the shift, and subtracts it to restore the true vector.
- The shift is a deterministic, reproducible function of `(seed, blinding config)`: re-blinding an updated catalogue needs no human round-trip, and unblinding is mechanical.
- B-mode null-test estimators return the same answer on a blinded file as on the true file, to the estimator's numerical floor.
- The CCL theory backend that computes the shift is cross-validated against an independent CAMB stack, so the shift means what it is intended to mean (see *Tests*).

## External contract (provided by the fork, consumed here)

This PR calls the fork through its concealment entry point and does not reach around it. The signatures below are the fork's authoritative contract; this PR does not restate their internals, only the two sp_validation-specific pieces it supplies: the three theory callables and the SACC assembly. The hidden cosmology is drawn by the fork's own CCL-native draw — there is no injectable draw callable.

**Conceal.** The one entry point:

```python
ConcealDataVector(
    fiducial_params,    # Mapping[str, float] — the fiducial CCL cosmology dict (== TheoryConfig.ccl_params())
    shifts_dict,        # Mapping[str, float] — per-parameter uniform half-width (CCL-native); keys ⊆ fiducial_params
    sacc_data,          # sacc.Sacc — exactly the rows theory_fn spans
    *,
    seed,
    theory_fn,          # Callable[[Mapping[str, float]], np.ndarray]
)
```

This PR always supplies its own `theory_fn` explicitly (the three master-layout backends below), overriding the fork's built-in default CCL backend — the default is a general cosmic-shear convenience, whereas blinding needs the exact master ξ± / pseudo-Cℓ row layout with sp_validation's IA configuration. `fiducial_params` is the fiducial cosmology this PR passes explicitly (there is no `cosmo`/`likelihood` pair and no `systm_dict`); it **is** `TheoryConfig.ccl_params()`. The concealing factor the fork computes is `theory_fn(concealed_params) − theory_fn(fiducial_params)`, where `concealed_params = fiducial_params` overlaid with the fork's internal draw of `shifts_dict` under `seed`. Every `shifts_dict` key must name a key present in `fiducial_params` (the overlaid keys — `sigma8`, `Omega_c` — are a subset of the `ccl_params()` keys); an unknown key is a config bug.

**Draw (in the fork, not here).** The fork draws the hidden cosmology internally from `shifts_dict` and `seed` using a local `numpy.random.default_rng`: order-independent, no global seed. `shifts_dict` keys are **CCL-native primitives** (e.g. `sigma8`, `Omega_c`); each value is a per-parameter uniform half-width `h`, and the fork draws that key **independently** as `U(fiducial_params[key] − h, fiducial_params[key] + h)`. Independent per-key uniform draws are the fork's draw law; the (S8, Ωm) → (sigma8, Omega_c) calibration below relies on it. There is no injectable draw callable and no (S8, Ωm) parameter space inside the fork — the fork speaks CCL. This PR's responsibility is to supply a `shifts_dict` in CCL-native primitives whose envelope is **calibrated to an equivalent S8 amplitude** (see *Envelope calibration* below).

**Theory backend.** A callable aligned element-for-element to the SACC rows of the block being blinded:

```
theory_fn(cosmo_params: Mapping[str, float]) -> np.ndarray
```

Any callable satisfies the protocol; the fork imports no theory backend at module load (its default CCL backend imports pyccl only when constructed). This PR does not use that default — it supplies three of its own callables (coarse ξ±, fine ξ±, pseudo-Cℓ), each a direct CCL computation from `sp_validation.cosmology`, matching the master SACC layout. The theory this PR computes is exactly its fiducial: the fiducial cosmology and its non-cosmological parameters are the ones this PR passes, nothing rides along uninvited.

**SACC assembly and encryption.** The concealment call returns a shifted `.mean` array; this PR assembles the blinded SACC itself from that array via `sp_validation`'s own SACC writers (`sacc_io`), which own the master layout and row order. Encryption helpers (`smokescreen.encryption.encrypt_file` / `decrypt_file`, Fernet) are used as-is for the seed bundle.

## Interfaces and contracts

### Theory config and backends

The fiducial cosmology and the two ξ± theory paths (CCL-native and independent-CAMB) live in `sp_validation.cosmology`. The blinding backend imports the frozen `TheoryConfig` from there; its defaults **are** the blinding fiducial. `TheoryConfig` imports only `numpy` at module load — CCL and CAMB are imported inside the functions that need them — so importing it never drags in a theory backend.

`TheoryConfig` carries the full fiducial cosmology parametrised by the blind axes `S8` and `Omega_m`, plus the rest (`Omega_b`, `h`, `n_s`, `m_nu` as Σm_ν, `w0`, `wa`, `mass_split`), the nonlinear-model tokens, and NLA intrinsic-alignment fields. It exposes:

- `sigma8() = S8 / sqrt(Omega_m / 0.3)` — the standard weak-lensing definition; at the fiducial (`S8=0.80`, `Omega_m=0.30`), `σ8 = 0.80`.
- `omega_c() = Omega_m − Omega_b − Ω_ν`, with `Ω_ν h² = Σm_ν / 93.14 eV` (`m_nu` is the sum, distributed under `mass_split`), so the total matter density is exactly `Omega_m`.
- `ccl_params()` → exactly `Omega_c, Omega_b, h, n_s, sigma8, m_nu, mass_split, w0, wa, Neff, T_CMB` and no other keys — no CCL default rides along. `Neff` and `T_CMB` are fixed constants of the fiducial (`Neff = 3.046`, `T_CMB = 2.7255 K`), not user-facing `TheoryConfig` fields; they are load-bearing for the CAMB↔CCL amplitude match and are emitted explicitly rather than left to either stack's default.
- `from_overrides(overrides)` → applies overrides onto the fiducial; **raises on any unknown key** (fail-fast).

**Nonlinear-model token mapping (load-bearing).** CCL and CAMB name the same HMCode2020+feedback recipe with *different* strings: CCL takes `mead2020_feedback` in `extra_parameters['camb']['halofit_version']`; `camb.set_halofit_version` takes `mead2020`. `TheoryConfig` carries **two** tokens (`ccl_halofit_version`, `camb_halofit_version`) denoting one recipe; each stack is fed its own. Passing a single string to both APIs is a silent stack disagreement (`mead2020` vs `mead2020_feedback` differ several % at k≳1). The defaults mirror the CosmoSIS v1.4.6.3 IA-fiducial (`logT_AGN=7.5`; `S8=0.80`, `Omega_m=0.30`, `h=0.70`, `n_s=0.96`, `m_nu=0.06`).

### Envelope calibration: from S8 amplitude to CCL-native `shifts_dict`

The blinding intent is an amplitude smear of a chosen S8 half-width (`ΔS8` config; nominal `|ΔS8| ≤ 0.075`, `|ΔΩm| ≤ 0.1`). The fork draws in CCL-native primitives, so this PR maps the intended (S8, Ωm) box to a `shifts_dict` in `{sigma8, Omega_c}` half-widths at the fiducial:

```
sigma8 = S8 / sqrt(Omega_m / 0.3)      Omega_c = Omega_m − Omega_b − Ω_ν
```

The mapping is evaluated at the fiducial to set the half-widths handed to the fork; the calibration is exact enough for a blinding smear because the target is a *characteristic amplitude*, not a precise (S8, Ωm) posterior. The half-widths are **config, not code** (the group may resize the envelope). The same `seed` yields the same hidden cosmology across all three block passes, so the three shifted blocks are mutually consistent by construction.

### Theory backends: three callables, one hidden cosmology

All three are direct CCL theory differences at the same two cosmologies (fiducial and hidden), sharing one P(k) recipe (CAMB HMCode2020; `ccl_halofit_version` from `TheoryConfig`, matched to the inference stack). They are two-tracer: cross-pairs use `angular_cl(tracer_i, tracer_j)` with each bin's own n(z) — no auto-n(z) approximation.

**n(z) and row alignment.** Each backend reads the per-bin n(z) directly from the tracers of the sub-SACC block it is handed (the same block extracted for concealment) — `WeakLensingTracer(cosmo, dndz=(z, nz))` per bin, IA from `TheoryConfig`'s NLA fields. The output vector is laid out to match that block's SACC rows element-for-element: iterate the block's data points in stored order, and for each `(bin_i, bin_j, θ or ℓ)` emit the corresponding theory value. The alignment is the block's own row order, not an assumed pairing; the fork's length guard (below) enforces length agreement, and AC #4's order-preservation assertion guards the merge back into the master.

| Backend | Rows | Computation |
|---|---|---|
| coarse ξ± | analysis θ grid | ξ± at coarse θ |
| fine ξ± | fine θ grid | ξ± at fine θ |
| pseudo-Cℓ | bandpower rows | `W @ ΔCℓ_EE` on the stored `BandpowerWindows`; `ΔBB = ΔEB ≡ 0` (shift is pure E-mode) |

Each is passed to `ConcealDataVector` as its `theory_fn` for the matching sub-SACC block.

### Blinding: extract → conceal → merge, per block

The master SACC file carries more rows than any single blindable block (COSEBIs, ρ/τ). The fork's extract-then-blind contract (with its length guard — there is no value/covariance consistency check) requires the SACC handed to it to contain exactly the rows the theory callable spans. Therefore, per block:

1. Extract the block (ξ± coarse / ξ± fine / pseudo-Cℓ) with its covariance sub-block into a temporary SACC, recording each extracted row's index in the master.
2. `ConcealDataVector(fiducial_params, shifts_dict, sub_sacc, seed=seed, theory_fn=theory_fn)` → concealed `.mean` for the block.
3. Write the shifted values back into the master at their recorded indices, preserving row order exactly. Row-order preservation is asserted in tests (AC #4), not assumed — a mis-merge that scrambles ξ± ordering would silently corrupt the COSEBIs re-derivation.

COSEBIs and pure-E/B are then re-derived by re-running the pipeline estimators (`b_modes.calculate_cosebis`, `b_modes.calculate_pure_eb_correlation`) on the blinded ξ± — never re-implemented, never shifted independently. Covariance and ρ/τ blocks are untouched.

### Custody: hash commitment (no keyholder)

Blind CLI:

1. Draw an OS-entropy seed.
2. Blind immediately (all three blocks).
3. Write `commitment.json` (repo-committable): `sha256(seed)` + a **config digest** binding the blinding config (envelope half-widths + fiducial cosmology + the two fixed `halofit_version` tokens). The digest is `sha256` of a canonical serialization: JSON with sorted keys, floats formatted via `repr()` (round-trip-exact), the full ordered field set fixed by this PR — so two runs of the same config produce byte-identical digests. Both `sha256(seed)` and the config digest are checked at unblind, so a wrong envelope or a mismatched P(k) recipe cannot silently subtract a wrong shift.
4. Encrypt `seed` + the true vectors into a bundle (`smokescreen.encryption`, Fernet); delete plaintexts.
5. Stamp blinded-file metadata: `concealed = True`; `blind_commitment = sha256(seed)` (public hash ties file → commitment). Unblind clears `concealed` back to `False` and removes `blind_commitment`. **Strip `seed_smokescreen`** — Smokescreen writes the raw seed into metadata; it must not survive into the blinded file.

Unblind CLI: decrypt the bundle → recompute `sha256(seed)` and the config digest, verify both against `commitment.json` → recompute the three shifts from `seed` → subtract → restore the true vector. Verification precedes subtraction.

### Metadata keys (blinded file)

| Key | Value |
|---|---|
| `concealed` | `True` (reset to `False` at unblind) |
| `blind_commitment` | `sha256(seed)` hex string (public; removed at unblind) |
| `seed_smokescreen` | **absent** (stripped) |

### Module layout

- `sp_validation/cosmology.py` — `TheoryConfig`, the CCL-native ξ± path, and the independent-CAMB ξ± path (numpy-only module-level imports; CCL/CAMB function-local).
- `sp_validation/blinding.py` — envelope calibration, the three theory backends, extract/conceal/merge orchestration, re-derivation, custody. Imports `TheoryConfig` from `sp_validation.cosmology`.
- Blind / unblind CLIs (entry points).
- Tests under `src/sp_validation/tests/`.

## Tests

### Acceptance criteria — blinding

Checkable, run inside the container:

1. **Idempotence / plumbing (zero-shift).** Blinding with a zero shift (hidden cosmology equal to fiducial) reproduces the input file's ξ±, COSEBIs, and pure-E/B values exactly. This exercises the extract/merge/re-derivation plumbing only — a broken `theory_fn` also passes, since `theory_fn(fiducial) − theory_fn(fiducial) = 0` for any callable. The backend itself is tested by AC #2.
2. **On-file shift equals the intended theory difference (central correctness).** For a fixed non-zero seed on a two-bin fixture, the per-row shift actually present in the blinded file (`blinded.mean − true.mean` at each shifted block's rows) equals `theory_fn(hidden_params) − theory_fn(fiducial_params)` — where `hidden_params` is recovered by re-running the fork's draw on that seed — to ~1e-10. This closes the loop AC #1 leaves open (a wrong theory backend fails here) and asserts the merge places the shift at the right rows. Run for all three blocks.
3. **Cross-backend consistency (reference-independent).** For the same hidden cosmology, the concealment-path shift for a block is compared against an **independently written** direct-CCL reference — a self-contained `angular_cl`+`correlation` call in the test file that does not import `sp_validation.blinding`'s backend, reading the same n(z) and θ/ℓ from the fixture — and they agree to ~1e-10 on a two-bin fixture. The reference is independent code, so agreement is not tautological; this pins the backend's projection against a from-scratch CCL computation.
4. **B-mode invariance.** On mocks, the ΔBₙ induced by blinding is bounded by the fixed E→B leakage floor of the estimator and is **independent of the injected B amplitude** (the shift is pure E-mode, so its leakage into B is a fixed absolute offset, not a fractional one). The test injects two different B amplitudes at fixed shift and asserts the absolute ΔBₙ is identical between them (to ~1e-10). The comparand — the shift-induced leakage — is defined concretely as `blind(true_file) B-block − true_file B-block` re-derived through the pipeline estimator on the blinded ξ±; the test asserts this vector is the same for both injected B amplitudes. The magnitude of the leakage floor is not asserted against a fixed number; it is measured by the run and reported (see note below). The criterion this AC enforces — B-amplitude independence — is exact by construction.
5. **Covariance, ρ/τ, and row order untouched.** The blinded file's covariance block and ρ/τ diagnostic rows are byte-identical to the input, and every shifted block's rows land back at their original master indices (order-preservation assertion, guarding COSEBIs re-derivation against a mis-merge).
6. **Custody verifiability.** Blind → commit → encrypt → strip leaves no plaintext seed and no `seed_smokescreen` key on disk in the blinded file; unblind fails closed if either `sha256(seed)` or the config digest mismatches the commitment.
7. **Reproducibility.** Two blind runs with the same seed and config produce identical shifts; unblinding after a re-blind of an updated catalogue needs no external state beyond the encrypted bundle.
8. **End-to-end integration.** The blind CLI blinds a real sp_validation `sacc_io` master file end-to-end and the unblind CLI restores it bit-for-bit; run on a fixture master file in the container.
9. **Pure-EB boundary behavior.** Re-derived pure-E/B follows the pipeline's edge-based integration bounds. The test runs re-derivation on a fixture whose θ grid deliberately includes a boundary-degenerate reporting point (the first point of the integration range, where the edge-based estimator has no interior support) and asserts that point is NaN in both the true-file and blinded-file re-derivations — i.e. NaN-parity with the pipeline's own estimator, never a spurious finite value. Production reporting grids are strict sub-ranges of the integration range, so this degeneracy never fires on real files; the fixture forces it to exercise the parity.

*Note on the ΔBₙ leakage floor.* The absolute size of the shift-induced E→B leakage is a measured output, not a target: it is reported from the run that produces it (the mock B-mode invariance test), not carried as an unsourced constant in this PRD. The acceptance criterion enforces that this leakage is independent of the injected B amplitude and reproducible — both exact by construction — with the observed magnitude recorded alongside the test.

### Acceptance criteria — CAMB↔CCL theory cross-check

The shift is a difference of CCL theory vectors; downstream inference runs CAMB. The shift only means what it is intended to mean if CCL and CAMB predict the same ξ± at a fixed cosmology on our θ grid. This test asserts that agreement and settles the one convention subtlety that would otherwise make the two stacks silently disagree: the fiducial fixes σ8 for CCL but A_s for CAMB, and a nominal `A_s = 2.1e-9` leaves CAMB's σ8 ≈3% off target — enough to blow a ξ± comparison to ~9–10%.

Two independent ξ± paths, both in `sp_validation.cosmology`, at one cosmology and n(z):

- **Path A — CCL native.** CCL builds the nonlinear P(k) through its Boltzmann-CAMB HMCode2020 route (`matter_power_spectrum='camb'` + HMCode2020 `extra_parameters`, `ccl_halofit_version`, `HMCode_logT_AGN`) and projects to ξ± via its own Limber (`angular_cl`) + FFTLog (`correlation`). Bare `WeakLensingTracer(dndz=(z,nz))` — IA off in this cross-check.
- **Path B — independent CAMB P(k) → CCL projection.** A direct `pycamb` run produces the HMCode2020 `P(k, z)` (using `camb_halofit_version` and `hmcode_logT_AGN`), wrapped in `ccl.Pk2D` and projected through the same CCL Limber + FFTLog machinery, same bare tracer.

Because both route their nonlinear P(k) through CAMB's HMCode2020 and both project through CCL, a common Limber+FFTLog bug cancels: this test validates the **P(k) recipe** and the **σ8/A_s amplitude convention**, not the projection. Stated once in the module docstring.

**Amplitude reconciliation.** A single closed-form rescale sets CAMB's `A_s` to reproduce the CCL σ8 target: `A_s = A_s_seed * (sigma8_target / sigma8_seed)**2`, exact because linear σ8² ∝ A_s — one CAMB linear-σ8 evaluation, one rescale, no iteration. Path B's CAMB params are set from the **same `TheoryConfig` fields** CCL sees — `w0`, `wa` (via `set_dark_energy`), `Neff`, `T_CMB`, `m_nu`/`mass_split` — not just `A_s` plus the HMCode P(k) tokens; otherwise an unmatched dark-energy or `Neff`/`T_CMB` value reintroduces exactly the stack-disagreement bias AC #12 is designed to exclude. The CAMB `make_params` builder pins `set_matter_power(hubble_units=False, k_hunit=False)` so `P(k,z)` comes out in CCL's native `1/Mpc` / `Mpc³`; Path B applies **no `·h` / `/h³` conversion** (applying one double-counts an `h³` amplitude error). Before constructing `ccl.Pk2D`, arrange both axes ascending: `lk = log(k)` ascending, scale factor `a` ascending (CAMB returns `z` ascending → reverse). θ passed to `ccl.correlation` in degrees (`theta_arcmin / 60`), `type="GG+"` / `"GG-"`.

**Fixture.** A single Gaussian source bin, `n(z) ∝ exp(−½((z−0.7)/0.2)²)` on `z ∈ [0.01, 3.0]`, trapezoid-normalised; θ grid `np.geomspace(5.0, 250.0, 12)` arcmin. Deterministic, self-contained — no catalogue.

Criteria (test at `src/sp_validation/tests/test_camb_ccl_crosscheck.py`):

10. **σ8/A_s reconciliation.** Against `σ8_target = 0.80`: (a) a nominal `A_s = 2.1e-9` leaves CAMB's σ8 offset by `abs(σ8_nominal/σ8_target − 1) > 0.02` (observed ≈0.03 — the convention offset is real); (b) the closed-form rescale lands CAMB's σ8 on target to `< 1e-4` (checks CAMB linear-σ8 reproducibility; the rescale is exact).
11. **ξ± agreement at the fiducial.** Path A and Path B agree within `XIP_RTOL = 0.005` (ξ+) and `XIM_RTOL = 0.010` (ξ−) over the 12-point θ grid. **ξ− caveat:** ξ− crosses zero on this grid; the relative assertion applies only where `|ξ−| > XIM_FLOOR` (absolute floor from the fixture's peak |ξ−|), with an absolute-agreement assertion elsewhere. ξ+ is finite and positive; ξ− is not sign-definite.
12. **ξ± agreement off the fiducial.** The same tolerances hold at a representative in-envelope offset — `from_overrides({"S8": 0.80 + 0.075, "Omega_m": 0.30 − 0.05})`, a point inside the (`|ΔS8| ≤ 0.075`, `|ΔΩm| ≤ 0.1`) envelope, not its edge — so the *shift* — a difference of two theory vectors — does not inherit a stack-disagreement bias.
13. **halofit token match.** The blinding backend's `ccl_halofit_version` equals the inference config's; a test asserts it directly against the inference config value. All three blinding backends share one recipe by construction and would agree with each other while jointly diverging from the inference stack (mead2020 vs mead2020_feedback is a several-percent trap), so this cannot be caught by AC #3 and is asserted independently here.
14. **Fast smoke variant.** A non-`slow` test runs both paths at coarse resolution (few θ, reduced ℓ grid) and asserts finite, positive, few-percent-agreeing ξ+ plus a σ8-matched `A_s` in `(1e-9, 3e-9)` — broken wiring caught in the fast suite. The precision assertions (AC #11, #12) carry `@pytest.mark.slow`.

*Context (informational, not a tolerance): on the shared-venv build the observed agreement floor is ξ+ ≈ 0.21% / ξ− ≈ 0.10% — headroom against the 0.5%/1.0% tolerances above, measured once; version bumps move it and that is not a regression.*

The full suite passes inside the container (`pytest src/sp_validation/tests/`), and CI runs it in the freshly built image before publish.

## Non-goals

- **No theory-backend refactor of Smokescreen.** The `theory_fn` protocol and the CCL-native draw live in the fork and are a precondition, not part of this PR.
- **No independent Limber/FFTLog validation.** Both cross-check paths project through CCL; a common projection bug cancels and is out of scope. The cross-check validates the P(k) recipe and amplitude convention only.
- **No Snakemake wiring.** Integrating the blind step into the pipeline DAG is deferred; this PR delivers the step and its CLIs.
- **No covariance blinding.** Blinding hides the vector, not the uncertainty; covariances never change.
- **No unblinding-criteria policy.** Which tests must pass at what thresholds before unblinding is a team-process decision, not code.
- **No change to the fiducial-cosmology group decision.** The `TheoryConfig` defaults mirror the CosmoSIS v1.4.6.3 fiducial; a later named group choice changes config values, not code.
