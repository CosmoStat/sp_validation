> **RETIRED (pass 2, unrevived by pass 3): superseded by prd-fork-protocol.md** — the injectable `draw_fn` abstraction over an arbitrary parameter space is explicitly deferred (protocol PRD Non-goals; blinding PRD: "no injectable draw callable"); the pure order-independent local-RNG draw itself lives in the protocol PRD as the built-in `draw_param_shifts`. Kept for reference only. **Note (pass 3):** this doc's "no `pyccl` object in scope" / "validates keys against `pyccl.Cosmology`" framing is stale — pass 3 rules that the fork ships a default CCL backend that *uses* pyccl at construction; the draw itself remains RNG-only, but pyccl is a used runtime dependency of the fork, not something being removed.

# PRD — PR-0.1a: Injectable pure `draw_fn` (fork)

Repo: `UNIONS-WL/Smokescreen`. Sub-PR under the "fork Smokescreen and make it theory-backend independent" issue.

This PR and the theory-protocol sub-PR both edit `ConcealDataVector.__init__`. They are **sequenced, not order-free**: this draw PR lands first against the stock signature (adding `draw_fn` + a pinned pure draw); the theory-protocol PR then rewrites the constructor wholesale (dropping `cosmo`/`likelihood`/`systm_dict`, adding `theory_fn`/`fiducial_params`) and carries `draw_fn` forward. The interfaces below are stated as they stand after both land, with the constructor shown in its final `theory_fn` form so the two PRDs agree on one target.

## Purpose

Smokescreen draws a hidden cosmology from a seed and shifts the data vector by the theory difference between that cosmology and the fiducial. The draw must be a **pure, reproducible function of `(seed, shift spec)` alone** — this is the load-bearing property of the blinding custody scheme: at unblind time the published seed is fed back through the same draw and must reproduce the identical shift to verify the hash commitment, possibly on a different checkout, machine, or process. A draw that depends on global RNG state, on dict iteration order, or on the fiducial cosmology's parameter ordering cannot serve as that verifiable function. This PR establishes the pure draw and makes it injectable, so a caller can supply its own draw over an arbitrary parameter space without subclassing.

## Desired end state

- A single pure function `draw_flat_param_shifts(shift_dict, seed)` backs flat shifts: it uses a local `numpy.random.Generator` (`default_rng`), touches no global RNG state, validates no keys against any cosmology, and draws in a canonical key order.
- The draw is **order-independent**: returned values depend only on `(seed, per-key spec)`, never on dict insertion order or on any cosmology's parameter ordering. Two shift dicts with the same keys and bounds in different order produce identical draws for a given seed.
- The draw operates over an **arbitrary parameter space**: keys are free-form strings. A box such as `{"S8": (-0.075, 0.075), "Omega_m": (-0.1, 0.1)}` is drawn natively. The draw makes no claim that its keys are cosmology parameters.
- The flat path calls this pure function. The stock flat path routes through `draw_flat_or_deterministic_param_shifts` (which seeds the global RNG, validates keys against `pyccl.Cosmology`, and iterates in `to_dict()` order); that function is **deleted** and `_load_shifts`'s flat branch is repointed to `draw_flat_param_shifts`. This repoint is the central change — the flat path does not currently call the pure function.
- `ConcealDataVector` accepts an **injectable** `draw_fn`. When supplied it is the sole source of shifts; the built-in flat/gaussian dispatch is bypassed. When omitted, behavior defaults to the built-in pure flat draw.
- The gaussian draw migrates to the same local-RNG, order-independent, arbitrary-key contract, keeping its `(mean, std)` tuple semantics.

## Interfaces / contracts

### `smokescreen/param_shifts.py`

```python
def draw_flat_param_shifts(
    shift_dict: dict[str, float | tuple[float, float]],
    seed: int | str,
) -> dict[str, float]:
    ...
```

- `seed`: an `int` used directly, or a `str` hashed to an `int` via the existing `smokescreen.utils.string_to_seed` (guarded by `type(seed) is str`, so an `int` seed is passed to `default_rng` raw, never re-hashed). `np.random.default_rng` accepts the resulting int.
- RNG: `rng = np.random.default_rng(seed)`. No call to `np.random.seed`.
- Per key: a 2-tuple `(lo, hi)` → `rng.uniform(lo, hi)` (absolute bounds); a scalar `w` → `rng.uniform(-w, w)` (symmetric half-width about zero). Tuple-vs-scalar is decided per key, not once for the whole dict.
- **Pinned order-independence mechanism** (not implementer's choice — it is the custody-verification contract and must be stable across versions and checkouts): iterate keys in `sorted(shift_dict)` and issue **one scalar `rng.uniform` call per key in that order**. Do not vectorize: a vectorized `rng.uniform(lo_arr, hi_arr)` consumes the bit stream differently and would yield different values for the same seed, silently breaking hash reproduction against any file blinded under the scalar-loop version. The scalar per-sorted-key loop is the frozen definition.
- Returns `{key: drawn_value}` for every key in `shift_dict`.
- No dependence on any `pyccl` object; no key validation against a cosmology.

```python
def draw_gaussian_param_shifts(
    shift_dict: dict[str, tuple[float, float]],
    seed: int | str,
) -> dict[str, float]:
    ...
```

- Same local-RNG, sorted-key, one-scalar-call-per-key, arbitrary-key contract.
- Every value must be a 2-tuple `(mean, std)`; a non-tuple raises `ValueError`. Draw is `rng.normal(mean, std)` per key in sorted order.
- The `cosmo` parameter and the `cosmo._params` / `cosmo.to_dict()` validation are removed. Key legality is enforced downstream at cosmology construction, not here.

`draw_flat_or_deterministic_param_shifts` is **deleted**. Its flat (tuple) behavior is subsumed by `draw_flat_param_shifts`; its deterministic (scalar-as-absolute-replacement) behavior is not on the blinding path and is dropped. A caller needing deterministic shifts supplies them through `draw_fn`.

### `smokescreen/datavector.py` — injectable draw

`ConcealDataVector.__init__` (final form after the theory-protocol PR; this PR adds the `draw_fn` parameter and the `_load_shifts` dispatch, the protocol PR supplies `theory_fn`/`fiducial_params`):

```python
def __init__(self, theory_fn, fiducial_params, shifts_dict, sacc_data, *,
             seed="2112", draw_fn=None, shift_distr="flat"):
    ...
```

- `draw_fn: Callable[[int | str], dict[str, float]] | None`.
- `shift_distr` becomes a **named parameter** with default `"flat"`, replacing the stock `**kwargs`-gated read (`if 'shift_distr' in kwargs: ...`). `_load_shifts`'s signature is `_load_shifts(self, seed, draw_fn=None, shift_distr="flat")` and dispatches:
  - `draw_fn is not None` → returns `draw_fn(seed)`; neither `shifts_dict` nor `shift_distr` is consulted.
  - `draw_fn is None`, `shift_distr == "flat"` → `draw_flat_param_shifts(shifts_dict, seed)`.
  - `draw_fn is None`, `shift_distr == "gaussian"` → `draw_gaussian_param_shifts(shifts_dict, seed)`.
- **Output contract for the shift dict.** The dict `_load_shifts` returns is applied by overlaying its keys onto `fiducial_params` to form the concealed parameter set, which is then handed to `theory_fn`. So the shift keys must be keys the chosen `theory_fn` understands. Two obligations follow, and they are **caller obligations, not guarantees this PR can enforce**:
  - The built-in flat/gaussian draws return their input keys verbatim. Using them means `shifts_dict` keys must already be keys the fiducial+theory path accepts (e.g. CCL primitives for a CCL `theory_fn`). The built-in draws do **not** translate parameter spaces.
  - An injected `draw_fn` may draw in an arbitrary space (`S8`, `Omega_m`, …) **and must itself map to the theory backend's key space before returning**, since its output is overlaid onto `fiducial_params` directly. The `S8`/`Omega_m` → CCL-primitive translation is the caller's `draw_fn` composition (draw-in-derived-space, then map), living in the consuming blinding step — not in this module. This is how the "arbitrary parameter space" capability reaches the default path: through an injected `draw_fn`, not through the built-in draw.

### Purity as a caller obligation

For the built-in draws this PR **guarantees**: for a fixed `seed`, the pinned scalar-per-sorted-key mechanism returns identical values across processes, machines, and reordered `shifts_dict` inputs. For an injected `draw_fn`, Smokescreen cannot enforce purity or order-independence; it **requires** them as a precondition the caller must meet (`draw_fn` must be a pure, order-independent function of `seed`). The custody scheme's reproducibility rests on this precondition holding for whatever draw is in use.

## Acceptance criteria

- `draw_flat_param_shifts(sd, seed)` returns identical output for two dicts differing only in key insertion order (assert equality against a shuffled-key copy).
- Draws correctly for non-CCL keys (`{"S8": (-0.075, 0.075), "Omega_m": (-0.1, 0.1)}`) with no `pyccl` object in scope.
- Calling the flat or gaussian draw leaves `np.random.get_state()` unchanged (guards against *legacy global* mutation; does not by itself prove a `default_rng` isn't reseeded from a global — covered by the cross-process equality check below).
- Same seed → equal dicts; different seeds → unequal dicts (checkable on a fixed pair).
- A value drawn under the pinned scalar-per-sorted-key mechanism is asserted against a hardcoded expected number for a fixed `(seed, key, bounds)`, freezing the mechanism against silent drift (e.g. a future vectorization).
- `ConcealDataVector(..., draw_fn=f)` uses `f(seed)` and ignores `shifts_dict`/`shift_distr`; without `draw_fn` it reproduces the built-in flat draw.
- A synthetic `draw_fn` returning fixed backend-key shifts drives a full blind end-to-end (concealed params formed, concealing factor computed) without a firecrown import.
- The fork's `param_shifts` test suite is updated to the new signatures and passes.

## Non-goals

- The theory-backend protocol (`theory_fn`), firecrown-adapter extraction, and the wholesale `__init__` rewrite — the sequenced theory-protocol sub-PR (this PR only adds `draw_fn` + the pinned pure draw).
- Packaging, install identity, `pyccl` declaration, fork CI — a separate sub-PR.
- The `S8`/`Omega_m` → backend-key translation and the custody/encryption flow — they live in the consuming sp_validation blinding step, which supplies a `draw_fn`; this PR guarantees only the injection point and the pinned pure-draw contract.
- The concealing-factor computation and any systematics handling.
