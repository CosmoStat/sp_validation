# PRD: cosmo_val migration to SACC

## Purpose

The cosmology-validation code computes every two-point data product — ξ± on coarse and fine θ grids, pseudo-Cℓ, COSEBIs, pure-E/B, ρ/τ PSF diagnostics, n(z) — but each mixin writes its own bespoke format (TreeCorr `.txt`, custom FITS, `.npz`), and a separate assembly step stitches them into an inference file. In the end state SACC is the sole native output format of the validation pipeline: every product is written through the `sacc_io` builders into SACC parts, and the Snakemake workflow assembles those parts into two terminal SACC files per catalogue version. No bespoke writer remains on the data-vector path.

## Desired end state

For each catalogue version the workflow produces exactly two terminal files:

- **`{version}.sacc`** — the analysis vector: n(z) tracers, coarse ξ±, pseudo-Cℓ (EE/BB/EB) with bandpower windows, COSEBIs, pure-E/B, ρ/τ, and a `FullCovariance` assembled from per-statistic blocks with zero cross-blocks.
- **`{version}_xi_fine.sacc`** — the fine ξ± grid (10000 bins, npatch=1) with a `DiagonalCovariance` from TreeCorr variances, the integration input for COSEBIs and pure-E/B.

Every per-statistic computation writes its result through a `sacc_io` builder into a single-statistic SACC **part** file. A dedicated assembly rule reads the parts, concatenates them in canonical order, attaches the assembled covariance, and writes the terminal file. The bespoke `.txt`/custom-FITS/`.npz` writers on the data-vector path are deleted; nothing downstream reads them.

## Interfaces and contracts

### `sacc_io` is a fixed dependency

The `sacc_io` module (`src/sp_validation/sacc_io.py`) supplies the builders, readers, covariance assembler, and save/load helpers, in one fixed on-disk layout. Its API is a landed contract and is **the single source of truth for every builder signature** — this document does not restate those signatures, because a paraphrased API surface is a second source of truth that drifts. Implementers call the builders as `sacc_io` defines them (`new_sacc`, `add_xi(..., grid=…)`, `add_pseudo_cl`, `add_cosebis`, `add_pure_eb`, `add_rho`, `add_tau`, `assemble_covariance`, `save`, `load`, `extract`); consult `sacc_io` for exact arguments, including the tomographic `bins=(i,j)` addressing and the keyed pure-E/B series.

Metadata keys required on every file: `catalogue_version`, `sp_validation_version`, `created`, `npatch`.

### Compute-path contract — statistics write SACC parts

Each `cosmo_val` statistic's write step is replaced by a call to the matching `sacc_io` builder, producing a **part** file that holds exactly one statistic (plus the self-describing n(z) tracers it references):

| Producer | Statistic | Builder | Part |
|---|---|---|---|
| `RealSpaceMixin.calculate_2pcf` (rule `xi`, script `run_2pcf.py`) | coarse ξ± | `add_xi(..., grid="coarse")` | `{version}_xi_coarse_….sacc` |
| `RealSpaceMixin.calculate_2pcf` (rule `xi_highres`, script `run_2pcf_highres.py`) | fine ξ± | `add_xi(..., grid="fine")` | `{version}_xi_fine_….sacc` |
| `pseudo_cl.py` | pseudo-Cℓ | `add_pseudo_cl` | `pseudo_cl_….sacc` |
| `cosebis.py` | COSEBIs | `add_cosebis` | `{version}_cosebis_….sacc` |
| `pure_eb.py` | pure-E/B | `add_pure_eb` | `{version}_pure_eb_….sacc` |
| `psf_systematics.py` | ρ/τ | `add_rho` / `add_tau` | `{version}_rho_tau_….sacc` |

A producer writes a part with a single `save(s, path)` call; it never assembles a terminal file and never emits a bespoke format alongside. TreeCorr variances (`varxip`/`varxim`) flow into the fine part as a 1-D array (yielding `DiagonalCovariance`); pseudo-Cℓ, COSEBIs, and ρ/τ covariances flow in as dense blocks for the analysis assembly step.

### Point-tag and ordering contract (from the SACC layout contract, not re-decided here)

- θ is stored in arcmin; grids are distinguished by the `grid` point tag (`coarse`|`fine`).
- Writers enforce strictly ascending θ/ℓ grids (loud `ValueError`); bin pairs normalize to `i ≤ j`.
- Canonical insertion order is pair-major: per `add_xi` call `[ξ+; ξ−]`, then Cℓ (EE/BB/EB), COSEBIs (Eₙ then Bₙ), pure-E/B, ρ, τ. Covariance rows align to insertion order by construction; readers never re-sort.

### Snakemake contract — parts, then assembly

Two rule tiers replace the current per-statistic FITS/txt rules:

1. **Part rules** — one per statistic, each wrapping a producer and emitting a `.sacc` part. Each rule carries a single wildcard set: the producing statistic's own binning/scale-cut wildcards, plus the `blind` wildcard that already threads the current data-vector rules (e.g. `pseudo_cl`). The `blind` wildcard flows through part filenames and is preserved onto the terminal files unchanged; this migration does not draw, encrypt, or otherwise interpret blinding — it only carries the existing wildcard through. Snakemake's one-wildcard-set-per-rule constraint is what forces part filenames to carry these wildcards. Part filenames are DAG-internal intermediates and are **not** a stability surface.
2. **Assembly rules** — one per terminal file:
   - `assemble_analysis`: inputs are the coarse-ξ±, pseudo-Cℓ, COSEBIs, pure-E/B, and ρ/τ parts for a version/blind; output is `{version}.sacc` (blind-tagged as the parts are). It loads each part, appends its data points in canonical order into one `Sacc`, calls `assemble_covariance` with the per-statistic dense blocks (zero cross-blocks), and saves.
   - `assemble_xi_fine`: input is the fine-ξ± part; output is `{version}_xi_fine.sacc`.

Consumers and downstream rules bind parts through the workflow's existing path helpers, never by hardcoding part filenames. The terminal filenames `{version}.sacc` and `{version}_xi_fine.sacc` (blind-tagged) are the only stable names.

## Scope of deletion

These are the data-vector writers this migration removes; each is superseded by a `sacc_io` part builder, and after the migration no code path reads the removed output:

- `RealSpaceMixin.calculate_2pcf`'s `save_fits` TreeCorr-FITS / `.txt` ξ± writer (both grids).
- The pseudo-Cℓ FITS writer behind rule `pseudo_cl` (`pseudo_cl_{version}_….fits`).
- The COSEBIs and pure-E/B bespoke writers (`cosebis.py`, `pure_eb.py`).
- The ρ/τ writers behind `rho_tau_stats` / `cv_rho_tau_fits`.

Explicitly **out of scope** (retained, not data-vector writers): the CosmoCov covariance products under `data/covariance/…` (rule family `covariance_*`), `pseudo_cl_cov_*.fits`, jackknife-sample `.npz` intermediates, and the `cosmo_val.smk` plot rules (`cv_plot_*`) that consume `.txt`/`.fits`/`.npz`. Plot rules that currently read a bespoke data-vector file are repointed to read the SACC part/terminal file instead; their own outputs (PDFs) are unaffected.

## Acceptance criteria

- Running the two-point workflow for a catalogue version produces `{version}.sacc` and `{version}_xi_fine.sacc`, and none of the writers named in *Scope of deletion* produce output.
- `{version}.sacc` loads via `sacc_io.load` and contains: n(z) tracers `source_i`; coarse ξ± (`grid="coarse"`); pseudo-Cℓ EE/BB/EB with bandpower windows; COSEBIs Eₙ/Bₙ; pure-E/B; ρ (k=0..5) and τ (k∈{0,2,5}); a `FullCovariance` with `covariance.size == len(mean)`.
- `{version}_xi_fine.sacc` contains fine ξ± (10000 bins, `grid="fine"`, npatch=1) with a `DiagonalCovariance` matching TreeCorr `varxip`/`varxim`.
- Every terminal file carries the four metadata keys with correct values and the `blind` label of its parts.
- Each part rule declares a single wildcard set; the DAG resolves from raw catalogues to both terminal files with no dangling targets pointing at a deleted writer.
- The writers named in *Scope of deletion* are removed from the codebase, and no remaining code path reads their outputs.
- Load-then-reload of each terminal file is bitwise-stable, and a covariance recovered by `s.indices` aligns element-for-element with the data vector — confirming the assembly step preserves insertion order and covariance alignment. (The write→read bitwise identity of each individual product is covered by `sacc_io`'s own round-trip tests and is not re-tested here.)

## Non-goals

- **No changes to `sacc_io`.** Builder, reader, and assembler signatures and the two-file layout are fixed inputs; if a builder proves insufficient, that is a defect in `sacc_io`, out of scope here.
- **No converters.** Producing DES 2pt-FITS from SACC, and the OneCovariance glue, are separate work.
- **No blinding logic.** The `blind` wildcard is carried through unchanged; drawing shifts, encrypting truths, and stamping concealed-file metadata belong to the blinding wiring, not here.
- **No science change.** Every product written is one the pipeline already computes at its existing binning; no new statistic, binning scheme, theory, or covariance is produced. Covariance blocks are the same arrays the pipeline computes today, re-homed into SACC, with the zero-cross-block independence assumption inherited unchanged. Existing bespoke-format outputs are left in place — new products are born in SACC.
