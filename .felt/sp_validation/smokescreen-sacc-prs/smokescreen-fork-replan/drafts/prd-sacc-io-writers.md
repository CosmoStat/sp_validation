# PRD — `sacc_io` writers

## Purpose

sp_validation produces weak-lensing data products (ξ±, pseudo-Cℓ, COSEBIs, pure-E/B, ρ/τ PSF diagnostics, n(z), covariance) scattered across TreeCorr `.txt`, custom FITS, `.npz`, and ASCII files. This PR introduces `src/sp_validation/sacc_io.py`: a single module that writes every product into self-describing [SACC](https://sacc.readthedocs.io) files in one fixed, self-consistent layout. It is pure format work — it defines the on-disk contract that the converter, migration, and blinding stages all consume, and nothing about it depends on where theory comes from.

## Desired end state

A new module `src/sp_validation/sacc_io.py` exposes per-statistic builder functions, mirror reader functions, a covariance assembler, and thin save/load helpers. Round-trip tests prove that every product's numeric payload survives write→read bit-exact and that insertion order, covariance alignment, and n(z) all reconstruct exactly.

Each catalogue version is serialized as **two SACC files**, not one:

- **`{version}.sacc`** — the analysis vector: NZ tracers, coarse ξ±, pseudo-Cℓ (EE/BB/EB) with bandpower windows, COSEBIs at the fiducial scale cut, pure-E/B, ρ/τ, and a `FullCovariance` assembled from per-statistic blocks with zero cross-blocks.
- **`{version}_xi_fine.sacc`** — the COSEBIs/pure-EB integration input: the same NZ tracers (self-describing), fine ξ± (10000 bins), and a `DiagonalCovariance` built from TreeCorr `varxip`/`varxim`.

The two-file split is mandatory, not stylistic: sacc 2.4 requires `covariance.size == len(mean)` and serializes a `BlockDiagonalCovariance` as one FITS table per block (`storage_type = ONE_OBJECT_MULTIPLE_TABLES`). A 10000-bin fine grid cannot coexist with a full analysis covariance in one file — the dense fine block alone is 3.2 GB single-bin and grows to a ~720 GB dense-equivalent tomographically (300k points at 5 bins, 300000² × 8 bytes). Measured reference: 4070 points with a dense-block covariance = 129–133 MB.

The API is tomography-native throughout: bin pairs are addressed as `bins=(i,j)`; the single-bin case is `(0,0)`. NZ tracers are named `source_i` (the suffix CosmoSIS's SACC reader expects).

## Interfaces and contracts

### Module surface (`src/sp_validation/sacc_io.py`)

All builders **mutate `s` in place and return `None`** (the converters stage relies on this convention when it chains readers into `assemble_covariance`). All arrays are `np.ndarray`; `bins` is a `tuple[int, int]` normalized to `i ≤ j`.

Builders:

```python
new_sacc(nz: dict[int, tuple[np.ndarray, np.ndarray]],
         metadata: dict) -> sacc.Sacc
    # nz maps bin index -> (z, nz); iteration order fixes tracer order (source_0, source_1, …).
    # Adds every NZ tracer first, stamps metadata, returns the fresh Sacc.

add_xi(s, bins, theta, xip, xim, *, grid, npairs, weight, theta_nom) -> None
    # Emits [ξ+ ; ξ−], θ ascending. One data point per (θ, sign).

add_pseudo_cl(s, bins, ell, cl_ee, cl_bb, cl_eb, window) -> None
    # Wraps THREE sacc.Sacc.add_ell_cl calls — one each for
    #   galaxy_shear_cl_ee, _bb, _eb — all sharing the single
    #   BandpowerWindow(window) passed in. "One shared window per call"
    #   means the same window object indexes all three sub-statistics.

add_cosebis(s, bins, n, En, Bn, *, theta_min, theta_max) -> None
add_pure_eb(s, bins, theta, values_by_key, *, theta_min=None, theta_max=None) -> None
    # values_by_key: dict keyed by the pure-E/B keys; see "pure-E/B ordering" below.
add_rho(s, k, theta, xip, xim) -> None          # k = 0..5
add_tau(s, bins, k, theta, xip, xim) -> None    # k in {0, 2, 5}

assemble_covariance(s, blocks: list[np.ndarray]) -> None
    # blocks is an ORDERED list; block order = insertion order of the data
    # vector it tiles. Calls s.add_covariance(scipy.linalg.block_diag(*blocks)).
```

Readers mirror each builder (`get_xi(s, bins, grid=…)`, `get_pseudo_cl`, `get_cosebis`, `get_pure_eb`, `get_rho`, `get_tau`, `get_nz`); each returns the arrays its builder consumed, in insertion order.

Helpers:

```python
extract(s, *, bins=None, data_type=None, grid=None, **tag_filters) -> sacc.Sacc
    # s.copy() then s.keep_selection(...) on the given selectors; the retained
    # sub-covariance and the tag-filtered points compose (intersection).
save(s, path) -> None
load(path) -> sacc.Sacc
```

**pure-E/B ordering.** The pure-E/B key set and their canonical order are defined by, and imported from, `b_modes._EB_KEYS` as the single source of truth — `add_pure_eb` writes and `get_pure_eb` reads in exactly that sequence. Do not hardcode a parallel list here or in the converters.

### Data types, tracers, point tags

| Product | Type string | Tracers | Point tags |
|---|---|---|---|
| ξ± coarse/fine | `galaxy_shear_xi_plus` / `…_minus` (standard) | (`source_i`, `source_j`) | `theta` (=TreeCorr `meanr`, **arcmin**), `theta_nom` (=`rnom`), `npairs`, `weight`, `grid` = `coarse`\|`fine` |
| pseudo-Cℓ | `galaxy_shear_cl_ee` / `…_bb` / `…_eb` (standard) | (`source_i`, `source_j`) | `ell` (effective), `window`/`window_ind` — one shared `BandpowerWindow(ells, W(nell, nbp))` per `add_pseudo_cl` call, indexing all three `add_ell_cl` (ee/bb/eb) sub-calls |
| COSEBIs | `galaxy_shear_cosebi_ee` / `…_bb` (standard; ee=Eₙ, bb=Bₙ) | (`source_i`, `source_j`) | `n` (mode, 1-based), `theta_min`, `theta_max` (arcmin scale cut) |
| pure E/B | `galaxy_shear_xiPureE_plus`/`…_minus`, `…xiPureB_…`, `…xiPureAmb_…` (custom) | (`source_i`, `source_j`) | `theta`, plus `theta_min`/`theta_max` if cut |
| ρₖ (PSF autos) | `psf_rho{k}_xi_plus` / `…_minus`, k=0…5 (custom) | (`psf_stars`, `psf_stars`) Misc tracer | `theta` |
| τₖ (gal×PSF) | `galaxyPsf_tau{k}_xi_plus` / `…_minus`, k∈{0,2,5} (custom) | (`source_i`, `psf_stars`) | `theta` |
| n(z) | — | NZ tracer `source_i` (`z`, `nz` round-trip bit-exact) | — |

Custom type strings (pure-E/B, ρ, τ) are all 3–4 underscore-separated parts so they parse under `sacc.parse_data_type_name`, even though sacc does not enforce the grammar. Pure-E/B writers are part of this module's surface (not merely a downstream convenience): the pure-E/B products are derived from ξ± and consumed by the blinding acceptance test, so they need a canonical home here.

### File layout

- `{version}.sacc` — analysis file (contents above), `FullCovariance`.
- `{version}_xi_fine.sacc` — fine ξ± + NZ tracers, `DiagonalCovariance`.

Naming stability is scoped to exactly these two terminal files. Any per-statistic intermediate files are DAG internals owned by the migration stage, not part of this contract; consumers bind them through workflow path helpers, never by hardcoding names.

Metadata keys on every file: `catalogue_version`, `sp_validation_version`, `created`, `npatch`.

Two fixed conventions: **θ is stored in arcminutes** (SACC stores θ unitless; downstream theory hard-codes arcmin). **n(z) carries no normalization convention** — every consumer normalizes internally.

### Ordering and covariance assembly

Insertion order is preserved exactly through save/load (as an index sequence — `s.indices` after reload equals `s.indices` before save) and covariance rows align to it by construction. Three rules make alignment hold and must all be enforced:

1. **Writers enforce strictly ascending θ/ℓ grids** — a non-ascending grid raises a loud `ValueError`.
2. **Readers return `s.indices` insertion order and never re-sort.** Reader-side sorting is exactly what lets data, covariance sub-blocks, and window columns silently diverge.
3. **Bin pairs normalize to i ≤ j.**

Canonical order is **pair-major**: within each `add_xi` call `[ξ+; ξ−]` with θ ascending (matching TreeCorr's covariance layout), then pseudo-Cℓ (EE, BB, EB), COSEBIs (all Eₙ then all Bₙ), pure-E/B in `b_modes._EB_KEYS` order (that constant is the single source of truth — see "pure-E/B ordering"), then ρ, then τ (ρ before τ; τ for k∈{0,2,5}).

This is the block order the covariance byte-compare in the converters stage depends on: within ξ the sub-order is `[ξ+ | ξ−]`, and τ (when present) follows ρ as `TAU_0 | TAU_2 | TAU_5`. The converters stage consumes only τ0/τ2 and treats this PRD as the authority for that order; any change here is a contract change there.

`assemble_covariance(s, blocks)` builds the analysis-file `FullCovariance` from per-statistic dense blocks with zero cross-blocks — the same independence assumption the current `np.block` assembly makes, which the converter's byte-compare needs preserved. A tomographic ξ covariance with cross-pair correlations is supplied as ONE contiguous block spanning consecutive `add_xi` calls; type-major consumers permute via `s.indices`. The assembler validates that the supplied blocks are contiguous-ascending, tile `0..len(s.mean)` exactly, and are each square and size-matched — loud `ValueError` otherwise.

### sacc 2.4 constraints the implementation must respect

- Tag filters are plain kwargs: `s.indices(dt, tracers, grid='fine')`. The `tags={'grid':'fine'}` form silently returns `[]` with only a `UserWarning` — never use it.
- `get_theta_xi` accepts no tag filters; readers go through `indices` / `get_mean` / `get_tag`.
- `get_ell_cl(..., return_windows=True)` does not exist in 2.4; use `s.get_bandpower_windows(indices)`.
- Tracers must be added before any data point that references them (`new_sacc` adds NZ tracers first; do not use the `tracers_later=True` escape).
- `add_covariance` with a 1-D array yields `DiagonalCovariance`; with `np.diag(...)` it yields `FullCovariance`. Pass the fine file its variances as a 1-D array (dense is wasteful).

## Acceptance criteria

- `src/sp_validation/sacc_io.py` exists and exposes every builder, reader, and helper above.
- Every product round-trips write→read: ξ± (coarse and fine), pseudo-Cℓ with windows, COSEBIs, pure-E/B, ρ (k=0…5), τ (k∈{0,2,5}), and n(z). The round-trip test asserts **bit-exact** equality (`np.array_equal`) on the numeric payload fields — `z`, `nz`, and every `mean` value — proving no float32 downcast, unit round-trip, or dtype coercion occurred; the test's job is to prove this, not assume it.
- A catalogue version serializes to exactly two files, `{version}.sacc` and `{version}_xi_fine.sacc`, each self-describing (NZ tracers present in both).
- After save→load, `s.indices` returns points in the canonical pair-major order, and a covariance recovered by index aligns element-for-element with the data vector.
- `assemble_covariance` raises `ValueError` on blocks that are non-contiguous, non-tiling, non-square, or size-mismatched; on valid input the resulting `FullCovariance` has `size == len(s.mean)`.
- Writers raise `ValueError` on a non-ascending θ/ℓ grid.
- `extract(s, *, bins=…, data_type=…, grid=…, **tags)` returns a sub-SACC whose covariance and tag-filtered points are exactly the selected subset (the retained covariance is the corresponding sub-matrix).
- Custom type strings parse under `sacc.parse_data_type_name` **into their intended `(sacc_quantity, …)` decomposition** — the test asserts the parsed tuple, not merely that parsing succeeds (any underscore string parses; the decomposition is what matters).
- Round-trip tests pass inside the container.

## Non-goals

- **No conversion to other formats.** SACC→2pt-FITS and SACC↔OneCovariance glue are a separate stage.
- **No migration of existing write sites.** `cosmo_val` mixins and the Snakemake workflow keep their current writers; re-pointing them is a separate stage. This module only defines the format and its I/O.
- **No blinding, theory, or cosmology.** No theory backend, no shift, no encryption — this module is theory-agnostic on-disk format only.
- **No regeneration of old products.** New products are born in this format; nothing existing is rewritten.
- **No fine-grid analysis covariance.** The fine file carries only a diagonal (TreeCorr variance) covariance; no full fine-grid covariance estimate exists or is produced here.
