---
id: 01KX4PKRY0KX848SNW2WZ5TZB0
name: SACC layout contract (PR 2, grounds PRs 3/4/6)
tags:
    - decision
    - sacc
created-at: 2026-07-10T02:25:04.192529552+02:00
updated-at: 2026-07-11T00:05:05.61612283+02:00
outcome: ONE SACC file per catalogue version (re-ruled 2026-07-11 — the 10k-bin premise was a one-time convergence check; production fine grid is 1000 bins, and Paper II's dense CosmoCov fine covariance exists and is load-bearing). Fine ξ± rides as grid='fine' points with a dense per-pair block. Tags carry grid/scale-cut/mode; custom types for rho/tau/pure-EB; insertion-order covariance assembly.
---

The file-layout contract for [[sp_validation/smokescreen-sacc-prs]] PR 2 (`sacc_io`), fixed 2026-07-10 from empirical probes of sacc 2.4 (probe scripts: `sp_validation-worktrees/env-probe/sacc-probe/`) and the write-site inventory. PRs 3 (converter), 4 (migration), and 6 (blinding) consume this contract; change it only with a reason strong enough to re-open all three.

## One file per catalogue version — the two-file split is retired (re-ruled 2026-07-11)

The 2026-07-10 ruling split `{version}_xi_fine.sacc` out because a 10000-bin fine grid couldn't coexist with the analysis covariance (`covariance.size == len(mean)`; dense fine block = 3.2 GB single-bin). Cail: **10k was a one-time convergence check, never a production point.** Paper II production fine grid is **1000 bins, 0.5–300′** (`analyses/shear_2d/bmodes_2d/config/config.yaml` L34–36; the paper's "0.5–500" is a typo), where a dense per-pair fine block is 2000² ≈ 32 MB — unremarkable.

Moreover the fine-grid covariance is not a placeholder: **Paper II's dense Gaussian CosmoCov covariance is computed *at* the 1000-bin integration binning and is the input to the derived-statistic covariances** — pure-E/B errors via 2000 MC draws through the Schneider transform, COSEBIs via matrix transform (`b_modes.py` L171–235 / L258–327; jackknife on the fine grid explicitly rejected as too noisy, `bmodes_2d/config/covariance.md`). Storing it is load-bearing for downstream error propagation.

- **`{version}.sacc`** — everything. NZ tracers `source_i`; coarse ξ±, pseudo-Cℓ (EE/BB/EB) with `BandpowerWindow`s, COSEBIs at the fiducial scale cut, pure-E/B, ρ/τ; **and fine ξ± as `grid='fine'` tagged points**. Covariance is `BlockDiagonal`: the analysis block(s) as before (zero cross-blocks — the same independence assumption today's `cosmosis_fitting.py` `np.block` assembly makes), plus a dense per-pair fine block from the CosmoCov integration covariance when it exists, `varxip`/`varxim` diagonal as the explicit degraded fallback. Insertion order: analysis points contiguous first, then fine points, so blocks tile the mean.
- Extreme grids (a future 10k-style check) degrade to a diagonal fine block; they don't fork the file format.
- PR 6 unaffected in design: Smokescreen's `_verify_sacc_consistency` already forces per-block extract→blind→merge passes, so blinding the fine block is its own pass inside the same file; the fine-grid shift stays a direct CCL theory difference, not a likelihood pass.

This restores the PRD §1 letter (one self-describing file per catalogue version); note the reversal on #241 and in the PR #245 body. Run-time access to "just the usual data vectors" is a Snakemake-parts concern, not a file-layout one: the terminal file is a pure gather; anyone wanting only coarse ξ targets the coarse part rule. (COSEBIs/pure-E/B depend on fine ξ in *any* layout, so a split never shortened the critical path.)

**Naming stability is scoped to the terminal file above.** The per-statistic *parts* (`{version}_xi_coarse_…​.sacc`, tagged `pseudo_cl_…​.sacc`, `{version}_cosebis.sacc`, …) are internal DAG intermediates; their filenames carry the producing rule's binning/tag wildcards where Snakemake's one-wildcard-set-per-rule constraint demands it (PR-4 design ruling, adversarially reviewed). Consumers bind parts through the workflow's path helpers, never by hardcoding part names.

## Data types and tracers

| Product | Type string | Tracers | Point tags |
|---|---|---|---|
| ξ± coarse/fine | `galaxy_shear_xi_plus/minus` (standard) | (`source_i`,`source_j`) | `theta` (=TreeCorr `meanr`, **arcmin**), `theta_nom` (=`rnom`), `npairs`, `weight`, `grid`=`coarse`\|`fine` |
| pseudo-Cℓ | `galaxy_shear_cl_ee/bb/eb` (standard) | (`source_i`,`source_j`) | `ell` (effective), `window`/`window_ind` (one shared `BandpowerWindow(ells, W(nell,nbp))` per `add_ell_cl` call) |
| COSEBIs | `galaxy_shear_cosebi_ee/bb` (standard; ee=Eₙ, bb=Bₙ) | (`source_i`,`source_j`) | `n` (mode, 1-based), `theta_min`, `theta_max` (arcmin scale cut) |
| pure E/B | `galaxy_shear_xiPureE_plus/minus`, `…xiPureB_…`, `…xiPureAmb_…` (custom, grammar-valid) | (`source_i`,`source_j`) | `theta`, `theta_min`, `theta_max` if cut |
| ρₖ (PSF autos) | `psf_rho{k}_xi_plus/minus`, k=0…5 (custom) | (`psf_stars`,`psf_stars`) Misc | `theta` |
| τₖ (gal×PSF) | `galaxyPsf_tau{k}_xi_plus/minus`, k∈{0,2,5} (custom) | (`source_i`,`psf_stars`) | `theta` |
| n(z) | — | NZ tracer `source_i` (z, nz round-trip bitwise) | — |

Pure-E/B writers are a deliberate addition beyond the PRD row-2 list: §3 derives them from blinded ξ± and the PR-6 acceptance test measures them; they need a home. Custom type strings all parse under `sacc.parse_data_type_name` (3–4 underscore parts) even though sacc doesn't enforce it.

## Ordering and covariance assembly

Insertion order is preserved bitwise through save/load (probed; `save_fits` never reorders) and covariance rows align to it. **Alignment holds by construction** (hardened by adversarial review before PR #245 opened): writers *enforce* strictly ascending θ/ℓ grids (loud ValueError), readers return `s.indices` insertion order (never re-sort — reader-side sorting is exactly what lets data, covariance sub-blocks, and window columns silently diverge), and bin pairs normalize to i≤j. Canonical order is **pair-major**: per `add_xi` call `[ξ+; ξ−]` (θ ascending, matching TreeCorr's cov layout), then Cℓ EE/BB/EB, COSEBIs (Eₙ then Bₙ), pure-E/B in `b_modes._EB_KEYS` order (verified: b_modes.py L19, sliced L392), ρ, τ. A tomographic ξ covariance with cross-pair correlations is supplied to `assemble_covariance` as ONE contiguous block spanning consecutive `add_xi` calls; type-major consumers (DES 2pt-FITS, PR 3) permute via `s.indices`. The assembler validates blocks contiguous-ascending, tiling `0..len(s.mean)` exactly, square and size-matched — loud ValueError otherwise.

## API surface (module `src/sp_validation/sacc_io.py`)

Builders `new_sacc(nz, metadata)` / `add_xi` / `add_pseudo_cl` / `add_cosebis` / `add_pure_eb` / `add_rho` / `add_tau` / `assemble_covariance(s, blocks)`; readers `get_xi(s, bins, grid=…)` etc. mirroring each writer (+ `get_nz`); `extract(s, …)` = `copy()`+`keep_selection` wrapper (probed: sub-covariance comes out exactly right, tag filters compose); thin `save`/`load`. Tomography-native `bins=(i,j)`; single-bin = `(0,0)`. Metadata: `catalogue_version`, `sp_validation_version`, `created`, `npatch`.

## sacc 2.4 gotchas (probed)

- Tag filters are **plain kwargs**: `s.indices(dt, tracers, grid='fine')`. The tempting `tags={'grid':'fine'}` silently returns `[]` with only a UserWarning.
- `get_theta_xi` accepts **no** tag filters — readers go through `indices`/`get_mean`/`get_tag`.
- `get_ell_cl(..., return_windows=True)` doesn't exist in 2.4; use `s.get_bandpower_windows(indices)`.
- Tracers must be added before any data point referencing them (`tracers_later=True` escape exists; don't use it).
- `add_covariance` with a 1-D array → `DiagonalCovariance`; with `np.diag(...)` → `FullCovariance` (wasteful for a diagonal fine fallback — pass the 1-D).
