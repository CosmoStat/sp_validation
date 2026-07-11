> **RETIRED (pass 2): folded into prd-smokescreen-blinding.md** — the cross-check is no longer its own PR; its verified technical content lives in the blinding PRD's test section. Kept for reference only.

# PRD — CAMB↔CCL theory-consistency cross-check

Repo: `CosmoStat/sp_validation`.

## Purpose

Data-vector blinding shifts the measured ξ± by a difference of theory vectors, `t(hidden) − t(fiducial)`, computed with CCL; cosmological inference downstream runs CAMB. That shift only means what we intend if CCL and CAMB predict the same ξ± at a fixed cosmology on our θ grid. This test asserts that agreement to a stated tolerance, and settles the one convention subtlety that would otherwise make the two stacks silently disagree: our fiducial fixes σ8 for CCL but A_s for CAMB, and a nominal `A_s = 2.1e-9` leaves CAMB's σ8 ≈3% off the target — enough to blow a ξ± comparison to ~9–10%.

## Desired end state

A standalone theory-consistency test compares ξ± computed two independent ways at one cosmology and n(z):

- **Path A — CCL native.** CCL builds the nonlinear matter power spectrum through its Boltzmann-CAMB HMCode2020 route and projects to ξ± via its own Limber (`angular_cl`) + FFTLog (`correlation`).
- **Path B — independent CAMB P(k) → CCL projection.** A direct `pycamb` run produces the HMCode2020 `P(k, z)`, wrapped in a `ccl.Pk2D` and projected to ξ± through the same CCL Limber + FFTLog machinery.

The two paths must agree to sub-percent on our θ grid. The fiducial cosmology and model configuration live in a small, dependency-light config object (`TheoryConfig`) that names one choice — HMCode2020 nonlinear via CAMB, NLA intrinsic alignments, normal-hierarchy neutrinos — shared by both paths and, crucially, by the blinding backend (see *Shared config* below).

Because both paths route their nonlinear P(k) through CAMB's HMCode2020 and both project through CCL, a common bug in CCL's Limber+FFTLog cancels between them: this test validates the **P(k) recipe** and the **σ8/A_s amplitude convention**, not the projection. This scoping is stated once in the module docstring.

## Interfaces and contracts

### Shared config — this is a cross-PR interface, not a free choice

`TheoryConfig` is imported by the blinding backend (`sp_validation.blinding_likelihood`), whose `TheoryConfig()` defaults *are* the blinding fiducial. Two consequences bind this PRD:

- **Module home is fixed: `sp_validation.cosmology`.** The blinding work imports `TheoryConfig` from there; a different home breaks that import. (`sp_validation.cosmology` is already the CAMB/CCL theory module.)
- **The config carries the full fiducial**, including IA (`ia_*`) and neutrino fields the blinding backend pins into Smokescreen's `systm_dict` — so those fields are load-bearing for the *shared* fiducial even though this cross-check's two ξ± paths run IA-free (they build bare `WeakLensingTracer(dndz=(z,nz))`; see *The two ξ± paths*). This test does not exercise IA; it does not own the IA fields either.

### `TheoryConfig` (no firecrown, no CCL/CAMB construction at import)

A frozen dataclass carries the fiducial cosmology and model choices, parametrised by the blind axes `S8` and `Omega_m`. It imports only `numpy` at module load — CCL/CAMB are imported inside the functions that need them — so importing it never drags in a theory backend.

```python
@dataclasses.dataclass(frozen=True)
class TheoryConfig:
    # Blind axes + the rest of the cosmology.
    S8: float = 0.80
    Omega_m: float = 0.30
    Omega_b: float = 0.0469
    h: float = 0.70
    n_s: float = 0.96
    m_nu: float = 0.06          # SUM of neutrino masses, eV (Σm_ν)
    w0: float = -1.0
    wa: float = 0.0
    mass_split: str = "normal"  # hierarchy for distributing Σm_ν; CCL token
    # Nonlinear model — the SAME recipe, named per stack (see below).
    ccl_halofit_version: str = "mead2020_feedback"  # CCL extra_parameters token
    camb_halofit_version: str = "mead2020"          # camb.set_halofit_version token
    hmcode_logT_AGN: float = 7.5
    # Intrinsic alignments — NLA; off in this cross-check, pinned by blinding.
    ia_bias: float = 0.0
    ia_z_piv: float = 0.62
    ia_alphaz: float = 0.0
    Neff: float = 3.046
    T_CMB: float = 2.7255

    def sigma8(self) -> float:   # S8 / sqrt(Omega_m / 0.3)
    def omega_c(self) -> float:  # Omega_m - Omega_b - Omega_nu(Sigma m_nu, h)
    def ccl_params(self) -> dict:
        # EXACTLY: Omega_c, Omega_b, h, n_s, sigma8, m_nu, mass_split,
        # w0, wa, Neff, T_CMB. No other keys; nothing rides CCL defaults.

    @classmethod
    def from_overrides(cls, overrides: dict) -> "TheoryConfig":
        # Applies overrides onto the fiducial. Unknown keys RAISE (fail-fast).
```

Contracts:
- `sigma8()` returns `S8 / sqrt(Omega_m / 0.3)` — the standard weak-lensing definition. At the fiducial (`S8=0.80`, `Omega_m=0.30`) this is `σ8 = 0.80`.
- `omega_c()` returns `Omega_m − Omega_b − Ω_ν`, with `Ω_ν h² = Σm_ν / 93.14 eV` (`m_nu` is the **sum**, distributed under `mass_split`), so the *total* matter density is exactly `Omega_m`.
- `ccl_params()` returns exactly the keys listed above and no others: every CCL cosmology parameter is set explicitly, so no CCL default silently rides along.
- `from_overrides` applies the given fields onto the fiducial and **raises on any unknown key** — a mistyped override is a bug, surfaced loudly.

**Nonlinear-model token mapping (the load-bearing subtlety).** CCL and CAMB name the same HMCode2020+feedback recipe with *different* strings: CCL takes `mead2020_feedback` in `extra_parameters['camb']['halofit_version']`; `camb.set_halofit_version` takes `mead2020`. `TheoryConfig` therefore carries **two** tokens (`ccl_halofit_version`, `camb_halofit_version`) that denote one recipe, and each stack is fed its own. Passing a single string to both APIs is exactly the silent stack-disagreement this test exists to catch (`mead2020` vs `mead2020_feedback` differ by several % at k≳1).

The defaults mirror the CosmoSIS v1.4.6.3 IA-fiducial (`logT_AGN=7.5`; `S8=0.80`, `Omega_m=0.30`, `h=0.70`, `n_s=0.96`, `m_nu=0.06`).

### Amplitude reconciliation

```python
def _camb_As_for_sigma8(config, sigma8_target, ...) -> tuple[float, callable]:
    """CAMB A_s reproducing sigma8_target, plus a CAMBparams builder.

    Single closed-form rescale: A_s = A_s_seed * (sigma8_target / sigma8_seed)**2,
    exact because LINEAR sigma8^2 is proportional to A_s. One CAMB linear-sigma8
    evaluation, one rescale — no iteration loop. So the CAMB stack shares sigma8
    with CCL before any projection. Returns (A_s, make_params), where
    make_params(A_s, nonlinear) builds a CAMBparams at the fiducial background.
    """
```

`make_params` pins CAMB's power-spectrum output convention explicitly — `set_matter_power(hubble_units=False, k_hunit=False)` — so `P(k,z)` comes out in CCL's native `1/Mpc` / `Mpc³` and Path B's unit handling is unambiguous (see *The two ξ± paths*). The nonlinear model uses `config.camb_halofit_version` and `config.hmcode_logT_AGN`; `mnu`/`neutrino_hierarchy` come from `config.m_nu` / `config.mass_split`.

### The two ξ± paths

```python
def _ccl_native_xi(z, nz, theta_arcmin, config) -> tuple[np.ndarray, np.ndarray]:
    """Path A: CCL Cosmology with matter_power_spectrum='camb' + HMCode2020
    extra_parameters, WeakLensingTracer(dndz=(z,nz)), angular_cl + correlation."""

def _independent_camb_xi(z, nz, theta_arcmin, config, ...) -> tuple[np.ndarray, np.ndarray, float]:
    """Path B: direct pycamb HMCode2020 P(k,z) -> ccl.Pk2D -> angular_cl +
    correlation. Returns (xi_plus, xi_minus, A_s)."""
```

Both paths build a CCL `Cosmology` from `config.ccl_params()` (`sigma8 = config.sigma8()`) with the HMCode2020 `extra_parameters` (`ccl_halofit_version`, `HMCode_logT_AGN`); both use a bare `WeakLensingTracer(dndz=(z,nz))` — IA is off in this cross-check. Path B additionally supplies its independent CAMB P(k) as `p_of_k_a`.

Path B unit / ordering handling: because `make_params` pins `hubble_units=False, k_hunit=False`, CAMB already returns `k` in `1/Mpc` and `P` in `Mpc³` — **no `·h` / `/h³` conversion is applied** (applying one here would double-count and inject a silent `h³` amplitude error). Before constructing `ccl.Pk2D`, arrange the arrays so both axes ascend: `lk = log(k)` ascending, and scale factor `a` ascending (CAMB returns `z` ascending → reverse to make `a` ascend).

θ is passed to `ccl.correlation` in degrees (`theta_arcmin / 60`), `type="GG+"` / `"GG-"`.

### Fixture

A single Gaussian source bin, `n(z) ∝ exp(−½((z−0.7)/0.2)²)` on `z ∈ [0.01, 3.0]`, trapezoid-normalised. θ grid: `np.geomspace(5.0, 250.0, 12)` arcmin. Deterministic and self-contained — no catalogue, no on-disk data.

### File layout

- `TheoryConfig` and the two ξ± path functions live in **`sp_validation.cosmology`** (numpy-only module-level imports; CCL/CAMB function-local; no firecrown). The blinding backend imports `TheoryConfig` from here.
- The test lives at `src/sp_validation/tests/test_camb_ccl_crosscheck.py`.

## Acceptance criteria

- **No firecrown.** `grep -r firecrown` over the test file and `sp_validation.cosmology` returns nothing. Importing `sp_validation.cosmology` does not import firecrown, pyccl, or camb (all function-local).
- **σ8/A_s reconciliation asserted.** A test proves, against the named fiducial (`σ8_target = 0.80`): (a) a nominal `A_s = 2.1e-9` leaves CAMB's σ8 offset by `abs(σ8_nominal/σ8_target − 1) > 0.02` (observed ≈0.03 — the convention offset is real, not negligible); and (b) the closed-form rescale lands CAMB's σ8 on the target to `< 1e-4` (this checks CAMB's linear-σ8 reproducibility, since the rescale is exact).
- **ξ± agreement at the fiducial.** Path A and Path B agree within `XIP_RTOL = 0.005` (ξ+) and `XIM_RTOL = 0.010` (ξ−) over the 12-point θ grid. **ξ− caveat:** ξ− crosses zero on this grid, where relative tolerance is ill-defined; the ξ− assertion applies only where `|ξ−| > XIM_FLOOR` (an absolute floor set from the fixture's peak |ξ−|), and asserts an absolute agreement elsewhere. ξ+ is finite and positive (physical); ξ− is not sign-definite and is not asserted positive.
- **ξ± agreement off the fiducial.** The same tolerances hold at a blind-sized offset (`from_overrides({"S8": 0.80 + 0.075, "Omega_m": 0.30 − 0.05})`), so the *shift* — a difference of two theory vectors — does not inherit a stack-disagreement bias.
- **Fast smoke variant.** A non-`slow` test runs both paths at coarse resolution (few θ, reduced ℓ grid) and asserts finite, positive, few-percent-agreeing ξ+ plus a σ8-matched `A_s` in `(1e-9, 3e-9)` — so broken wiring is caught in the fast suite. The precision assertions carry `@pytest.mark.slow`.
- The suite passes inside the container (`pytest src/sp_validation/tests/test_camb_ccl_crosscheck.py`), and CI runs it in the freshly built image before publish.

*Context (informational, not a tolerance): on the shared-venv build the observed agreement floor is ξ+ ≈ 0.21% / ξ− ≈ 0.10% — headroom against the tolerances above, measured once; version bumps move it and that is not a regression.*

## Non-goals

- **No firecrown, no likelihood.** Nothing in the required path imports firecrown or builds a likelihood object; the blinding theory engine is CCL called directly, delivered by the blinding-wiring work.
- **No independent Limber/FFTLog validation.** Both paths project through CCL; a common projection bug cancels and is out of scope.
- **No blinding, seed, or Smokescreen wiring** — this is purely a theory-stack consistency check.
- **No tomographic multi-bin fixture** — one source bin exercises the amplitude and P(k) consistency. The path functions are written generically over `(z, nz)`; cross-tracer pairs and the two-tracer amplitude convention are validated by the blinding backend's own tests, not here.
- **No change to the fiducial-cosmology group decision.** The defaults are reasonable stand-ins mirroring the CosmoSIS v1.4.6.3 fiducial; a later named group choice changes config values, not code.
