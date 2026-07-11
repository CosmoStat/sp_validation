# PRD: theory_fn protocol + default CCL backend + fixed draw + fork-test migration

Repo: `UNIONS-WL/Smokescreen`.

## Purpose

Smokescreen blinds a data vector by adding a theory difference, `d → d + t(hidden) − t(fiducial)`. The theory source is a single `theory_fn(cosmo_params) -> np.ndarray` protocol, so any callable — the built-in CCL backend, a ΔΣ emulator, a synthetic fixture — can drive blinding. This PR establishes that protocol as the theory contract, ships a built-in default CCL backend so a casual user can blind a standard cosmic-shear SACC file without writing a callable, and draws the hidden shift with a local, order-independent RNG over CCL-native parameter names.

The theory backend is either the shipped CCL default or one the caller supplies. `pyccl` is a declared runtime dependency, imported by the default-backend module (not at top-level `import smokescreen`). Firecrown is a different case: it survives as an **optional, lazy-imported** path inherited from upstream — never imported at module level, never declared as a dependency, not installed, tested, supported, or maintained in this fork. "CCL-native parameter names" (`sigma8`, `Omega_c`, …) are the parameter keys the default backend and draw speak. The package's own test suite runs on a synthetic backend with firecrown absent.

Packaging, install identity, `pyccl` declaration, fork CI wiring, and install docs are a **separate sub-PR** and are out of scope here (see Non-goals).

## Desired end state

- `import smokescreen` imports no theory backend at module load: `pyccl` is imported only inside the default-backend module when that backend is constructed, and `firecrown` is never imported at module level (lazy only, inside the inherited optional path).
- Firecrown is **not deleted** but is fully sidelined: no `import firecrown` / `from firecrown …` at module scope, no `ParamsMap` on the default path, no `patch_firecrown` / NumCosmo import surgery, no `numpy<2.5` ceiling anywhere in the fork. The firecrown path remains as inherited-upstream code behind a lazy import, unsupported and untested here.
- The fork ships a **default CCL `theory_fn`**: a built-in backend that computes a cosmic-shear theory vector from CCL for a standard SACC file. Power users override it with their own callable.
- `ConcealDataVector` obtains both theory vectors (fiducial, hidden) by calling a `theory_fn` — the default CCL backend when none is supplied, else the caller's. There is no `likelihood` path, no `_create_concealed_cosmo` step, no `systm_dict` inside the concealing computation.
- The hidden shift is drawn by a pure, local-RNG function over a free-form, order-independent parameter mapping keyed by CCL-native names (e.g. `sigma8`, `Omega_c`); no process-global `np.random.seed`.
- The package's own test suite exercises the full blinding machinery through a synthetic `theory_fn`, with firecrown absent.

## Interfaces and contracts

### Theory-function protocol

The single theory entry point. Any callable satisfying this signature is a valid backend; the fork ships a default CCL implementation and accepts a caller-supplied override.

```python
def theory_fn(cosmo_params: Mapping[str, float]) -> np.ndarray:
    """Return the theory data vector aligned element-for-element to the
    SACC rows being concealed, evaluated at `cosmo_params`."""
```

- `cosmo_params` is a plain mapping of cosmological-parameter name → value. The protocol places no constraint on the parameter names; interpreting them is the backend's job.
- The return is a 1-D `np.ndarray` whose length and row order match `sacc_data.mean` for the rows to be concealed. Alignment is the backend's contract, not Smokescreen's.
- The callable is pure with respect to its argument: two calls with equal `cosmo_params` return equal vectors.

### Default CCL backend

The fork ships a built-in `theory_fn` so a standard cosmic-shear SACC file blinds out of the box:

- It lives in its own module (e.g. `smokescreen.backends.ccl`) that imports `pyccl`. That import happens when the backend is constructed, **not** at `import smokescreen` / `import smokescreen.datavector` — top-level import stays backend-free.
- Given a `sacc_data` carrying weak-lensing tracers with n(z), it builds a CCL cosmic-shear `theory_fn`: for each `cosmo_params` mapping (CCL-native names), construct the `pyccl.Cosmology`, `WeakLensingTracer` per bin, and return the ξ± / Cℓ vector aligned to `sacc_data`'s rows.
- The parameter names it accepts are CCL-native (`sigma8`, `Omega_c`, `Omega_b`, `h`, `n_s`, …). Interpreting them is this backend's job; the protocol itself is name-agnostic.
- It is the value `ConcealDataVector` uses when `theory_fn=None`. It is exercised by the fork test suite only where pyccl is available; the core blinding-mechanics tests use a synthetic backend and do not require it.

### `ConcealDataVector`

```python
ConcealDataVector(
    fiducial_params,    # Mapping[str, float]
    shifts_dict,        # Mapping[str, float | tuple[float, float]]
    sacc_data,          # sacc.sacc.Sacc
    *,
    seed,               # int | str; no default — see note
    theory_fn=None,     # Callable[[Mapping[str, float]], np.ndarray]; None → default CCL backend
    shift_distr="flat",
)
```

- `theory_fn` defaults to the shipped CCL backend (below) when `None`; supplying a callable overrides it. The default backend is constructed from `sacc_data` (it reads the tracers' n(z) and the row layout to build a cosmic-shear vector) — passing `theory_fn=None` with a SACC the default backend cannot interpret raises a clear error at construction.
- `fiducial_params` replaces stock Smokescreen's `cosmo` + `likelihood` constructor pair as the description of the fiducial point. It is a plain mapping of cosmological-parameter name → value over a free-form parameter space (CCL-native names, e.g. `sigma8`, `Omega_c`); it is not a `pyccl.Cosmology` and is not constrained to CCL constructor names. Interpreting the names is the `theory_fn`'s job. The hidden point is `fiducial_params` with the drawn shifts overlaid, a mapping of the same shape — there is no `pyccl.Cosmology` object and no `_create_concealed_cosmo` step.
- `sacc_data` is retained for output assembly and for the row layout; it must contain exactly the rows the `theory_fn` returns (the length/order the concealing factor is added onto). The user owns this alignment.
- No `systm_dict` parameter. Systematics, if a backend needs them, are closed over inside that backend's `theory_fn`; they never transit `ConcealDataVector`.
- No `draw_fn` parameter. The shift draw is the built-in pure function below; `shifts_dict` and `shift_distr` are its specification.
- `seed` has **no default** and must be supplied explicitly. Blinding custody depends on the seed being a deliberate, secret choice; a defaulted blinding seed would silently blind against a public constant. `seed` accepts an `int` or a `str` (see Shift draw for the string→int normalization).
- `shifts_dict` maps a parameter name (a key of `fiducial_params`) to its shift envelope. **The envelope is a delta about zero, not an absolute value or absolute bound**: the drawn number is added to `fiducial_params[k]` (see Shift draw). A `float` `h` is a symmetric delta envelope `(-h, +h)`; a `(lo, hi)` tuple is a delta box, the drawn delta lies in `[lo, hi]` (typically straddling zero). `shift_distr` selects `"flat"` (uniform over the delta envelope) or `"gaussian"` (delta drawn from a zero-mean Gaussian with σ = the `float` half-width). `"gaussian"` accepts only the symmetric `float` form; a `(lo, hi)` tuple with `shift_distr="gaussian"` raises.

### Shift draw

A pure, module-level function is the sole shift source:

```python
def draw_param_shifts(
    shifts_dict: Mapping[str, float | tuple[float, float]],
    seed,               # int | str
    shift_distr: str = "flat",
) -> dict[str, float]:
    """Draw a delta for each named parameter from a local RNG seeded by
    `seed`. Each returned value is a shift to be *added* to the fiducial
    value, not an absolute parameter value. Order-independent: the delta
    for a given key depends only on (key, seed, shift_distr), not on dict
    iteration order."""
```

- **Deltas, not absolute values.** Every returned number is a shift to overlay on `fiducial_params` (below), never a replacement value and never an absolute bound. This is the sole draw semantics; there is no absolute-replacement path.
- **Seed normalization.** `seed` may be an `int` or a `str`. A string is reduced to an integer seed deterministically (stable hash → int, e.g. SHA-256 of the UTF-8 bytes reduced to a 64-bit int) before constructing the RNG; an int is used as-is. The mapping from a given string to its integer seed is fixed and reproducible across runs and machines.
- Uses a **local** `numpy.random.default_rng(normalized_seed)`; it never calls `np.random.seed` and never perturbs process-global RNG state.
- **Order-independent.** The delta drawn for a parameter must not depend on the iteration order of `shifts_dict`. Concretely: iterate keys in a canonical order (sorted) and/or derive each key's draw from an RNG spawned deterministically per key, so that permuting `shifts_dict` yields identical per-key deltas.
- `"flat"` draws uniformly over the delta envelope (`(-h, +h)` for a `float` `h`, `[lo, hi]` for a tuple). `"gaussian"` draws from a zero-mean Gaussian with σ = the `float` half-width and accepts only the `float` form.
- Returns a mapping in `fiducial_params` space: `{param_name: drawn_delta}`, keys a subset of `fiducial_params` keys, values plain floats. No `pyccl` validation, no CCL-primitive dict, no `pyccl.Cosmology` construction, no import of pyccl.

The concealed parameter set is `fiducial_params` overlaid with these deltas: `concealed_params[k] = fiducial_params[k] + shift[k]` for `k` in the drawn shift, `fiducial_params[k]` otherwise.

### Concealing computation

`calculate_concealing_factor(factor_type="add")` computes:

```
t_fid     = theory_fn(fiducial_params)
t_conceal = theory_fn(concealed_params)          # concealed_params = fiducial + drawn shift
factor    = t_conceal - t_fid                     # "add"; t_conceal / t_fid for "mult"
```

Both branches are element-wise over the aligned theory vectors: `factor` is a 1-D array the length of `sacc_data.mean`, and for `"mult"` `factor = t_conceal / t_fid` is a per-row ratio.

`apply_concealing_to_likelihood_datavec()` returns `sacc_data.mean + factor` (`"add"`) or `sacc_data.mean * factor` (`"mult"`, element-wise). The additive amplitude-shift envelope is the path sp_validation blinding uses; the multiplicative branch is preserved stock Smokescreen behavior, offered for callers that want it and not exercised by any sp_validation path. The default concealing path invokes the `theory_fn` directly: no firecrown `get_default_params_map`, `modify_default_params`, `tools.update/prepare`, or `likelihood.update/compute_theory_vector` calls are on it. The inherited firecrown likelihood path is not deleted, but it is off the default flow, lazily imported, and unsupported.

### Save path

`save_concealed_datavector` writes the blinded vector back to a SACC file through SACC's own API, with no firecrown import and no `likelihood`:

- The rows to conceal are exactly the rows the caller placed in `sacc_data`, in `sacc_data.mean` order — the same rows `theory_fn` returns (extract-then-blind: the caller hands in a SACC containing only the to-be-blinded block). The concealing factor is added over the full mean. There is no wider-SACC / explicit-index path; scoping to a sub-block is the caller's job before construction.
- Writing: deep-copy `sacc_data`, overwrite its mean with the concealed vector (`concealed_sacc.mean = concealed_data_vector`), copy metadata, then `save_fits`/`save_hdf5`. **The copied SACC's covariance is carried over unchanged** — blinding shifts the mean and never touches the covariance. No `get_sacc_indices`, no `save_to_sacc`, no firecrown import.
- The metadata stamping/stripping (concealed flag, creator, timestamp, seed strip) is custody-owned and unchanged; this PRD does not alter it.

### Consistency check

There is **no value/covariance consistency check**. Row-to-theory alignment is the backend's contract (stated in the protocol), not Smokescreen's to verify; with no likelihood there is no second internal vector to compare a SACC mean against. Any `_verify_sacc_consistency`-style comparison of `sacc_data.mean` against a likelihood-derived vector (and its `allclose` covariance check) is absent from the fork — dropped, not re-expressed. This is a deliberate, non-behavior-preserving change.

The sole guard is a **length check**: on construction, verify `len(theory_fn(fiducial_params)) == len(sacc_data.mean)` and surface a clear error on mismatch. This catches gross shape errors; it does **not** catch a `theory_fn` that returns a correctly-sized but wrong-cosmology or wrong-row-order vector. That is an accepted consequence of moving alignment into the backend contract, recorded here so the guard is not mistaken for behavior-preserving.

### Test-suite migration

- The blinding-mechanics tests (`ConcealDataVector` construction, shift draw, concealing factor, apply, save/metadata) run against a **synthetic `theory_fn`** — a closure returning a fixed-length vector as a deterministic function of `cosmo_params` — without importing firecrown, and without requiring the default CCL backend (they inject the synthetic callable).
- Every test, fixture, and helper that required a firecrown likelihood or asserted firecrown-specific behavior is removed from the fork's exercised suite: firecrown is not installed, tested, or maintained here, so no test depends on it. The inherited firecrown source stays in the tree behind its lazy import; the fork simply does not test it.
- The draw is tested directly: (i) a fixed `(shifts_dict, seed)` reproduces a fixed shift; (ii) permuting the key order of `shifts_dict` yields identical per-key shifts (order-independence); (iii) the draw does not perturb `np.random`'s global state (draw, then assert a subsequent `np.random.random()` matches a run with no draw); (iv) the returned value is a **delta** — with a fiducial value `v` and envelope `h`, the concealed parameter is `v + drawn_delta` with `|drawn_delta| ≤ h`, never an absolute value in the envelope's range; (v) a string `seed` and its normalized integer produce the same draw, confirming string→int normalization.

## Acceptance criteria

1. In an environment with `pyccl` present but `firecrown` and `numcosmo_py` absent: `import smokescreen` succeeds, a `ConcealDataVector` built from a synthetic `theory_fn` blinds a SACC vector end-to-end (`calculate_concealing_factor` → `apply_concealing_to_likelihood_datavec` → `save_concealed_datavector`) with the default backend never constructed, and a `ConcealDataVector` on `theory_fn=None` blinds a standard cosmic-shear SACC fixture through the default CCL backend — both producing a blinded vector that differs from the input and leaves the covariance untouched.
2. Import discipline, verified by test and grep:
   - `import smokescreen` and `import smokescreen.datavector` import neither `pyccl` nor `firecrown` at module load (assert absence from `sys.modules` after a fresh import in a subprocess). `pyccl` enters `sys.modules` only once the default backend is constructed.
   - No `import firecrown` / `from firecrown` at module scope **anywhere** in the package; the inherited firecrown path imports firecrown lazily, inside the function that uses it. `import numcosmo_py` appears nowhere at module scope.
   - No `numpy<2.5` constraint and no `patch_firecrown` module anywhere. None of the removed default-path symbols `_verify_sacc_consistency`, `get_data_vector`, `get_sacc_indices`, `save_to_sacc`, `_create_concealed_cosmo`, `systm_dict` appear on the default concealing flow.
3. The concealing factor equals `theory_fn(concealed_params) - theory_fn(fiducial_params)` exactly (bit-for-bit) for the synthetic backend, and `apply_…` returns `sacc_data.mean + factor`. The `factor_type="mult"` branch (`t_conceal / t_fid`, applied multiplicatively) is exercised by the same synthetic backend and returns `sacc_data.mean * factor`.
4. `ConcealDataVector` construction raises a clear error when `len(theory_fn(fiducial_params)) != len(sacc_data.mean)`.
5. The shift draw uses a local RNG: after `draw_param_shifts(...)`, `np.random.random()` returns the same value it would with no prior draw (global state untouched). Permuting the key order of `shifts_dict` produces an identical per-key shift mapping. A fixed `(shifts_dict, seed, shift_distr)` reproduces a fixed shift across runs. The drawn value is a **delta** overlaid on the fiducial (`concealed_params[k] == fiducial_params[k] + drawn_delta[k]`, `|drawn_delta[k]|` within the envelope), not an absolute replacement. A string `seed` and its normalized integer yield the same draw.
6. The full exercised test suite passes in the supported firecrown-free environment (pyccl present — pyccl is the shipped default backend); no test is skipped for a missing firecrown. Only the synthetic-backend mechanics subset is additionally required to pass with pyccl absent; the suite as a whole is not required to run pyccl-absent.

## Non-goals

- **Packaging, install identity, dependency declaration, CI wiring, and install docs** — declaring `pyccl` in the fork `pyproject`, settling the fork's pinned `git+https` install identity, re-pointing fork CI gates, writing pip-only install docs. Owned by the packaging sub-PR.
- **A production, survey-specific theory backend.** This PR ships a general default CCL `theory_fn` for standard cosmic-shear SACC files; the sp_validation-specific backends (coarse/fine ξ± and pseudo-Cℓ against the master layout, IA-configured) are downstream blinding-wiring work, not part of this PR.
- **Firecrown support.** The inherited firecrown path is retained lazily but is not installed, tested, maintained, or documented as supported here.
- **Arbitrary-parameter-space draw indirection.** The draw is a fixed pure function over CCL-native parameter names; an injectable `draw_fn` abstraction over arbitrary parameter spaces is explicitly deferred.
- **Any change to the custody/encryption scheme** (seed handling, hash commitment, plaintext deletion, metadata stamping/stripping). Untouched here.
- **Any PR against DESC upstream.** All changes target the `UNIONS-WL/Smokescreen` fork.
