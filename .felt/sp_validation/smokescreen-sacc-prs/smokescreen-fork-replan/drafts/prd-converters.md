# PRD — Converters

Repo: `CosmoStat/sp_validation`. Format work; theory-agnostic.

## Purpose

The SACC file is the standard container for every sp_validation data product, but two consumers do not read SACC: the CosmoSIS inference chain reads a DES-style "2pt-FITS" file, and OneCovariance neither emits nor ingests SACC. This PR supplies the two glue layers those consumers need — a SACC→2pt-FITS converter, and n(z)→OneCovariance-input plus OneCovariance-output→SACC-covariance glue. Nothing here computes theory or blinds anything.

## Desired end state

- A converter turns a `{version}.sacc` analysis file into a 2pt-FITS file with the exact HDU layout the CosmoSIS chain (including the PSF ρ/τ likelihood modules in the cosmosis-standard-library fork) consumes. The chain runs against the converter's output and produces results identical to running against a directly-assembled file, with no pipeline change.
- Two glue functions bridge OneCovariance: sp_validation n(z) tracers → OneCovariance's per-bin two-column text input (plus its config), and OneCovariance's flat output covariance → a SACC covariance block aligned to the analysis file's insertion order.
- Both directions are tested: the 2pt-FITS converter against a real reference file and a synthetic fixture; the OneCovariance glue round-trips through a fixture.

## Target file: the 2pt-FITS layout

The output is a multi-extension FITS file with **11 HDUs**, in this order:

| # | EXTNAME | Type | Contents |
|---|---------|------|----------|
| 0 | `PRIMARY` | Primary | empty |
| 1 | `COVMAT` | Image | real-space covariance, blocked `XI_PLUS \| XI_MINUS` and (when present) `TAU_0_PLUS \| TAU_2_PLUS \| CELL_EE`, with `NAME_i`/`STRT_i` offset headers |
| 2 | `COVMAT_CELL` | Image | pseudo-Cℓ covariance, single `CELL_EE` block (`NAME_0=CELL_EE`, `STRT_0=0`) |
| 3 | `NZ_SOURCE` | BinTable | one n(z) per tomographic bin |
| 4 | `XI_PLUS` | BinTable | coarse-grid ξ₊ |
| 5 | `XI_MINUS` | BinTable | coarse-grid ξ₋ |
| 6 | `CELL_EE` | BinTable | pseudo-Cℓ EE |
| 7 | `CELL_BB` | BinTable | pseudo-Cℓ BB |
| 8 | `TAU_0_PLUS` | BinTable | τ₀₊ PSF statistic |
| 9 | `TAU_2_PLUS` | BinTable | τ₂₊ PSF statistic |
| 10 | `RHO_STATS` | BinTable | ρ statistics table (copied through) |

The executable definition of every HDU's columns and header cards is `cosmo_inference/scripts/cosmosis_fitting.py` — its `nz_to_fits`, `_create_2pt_hdu`, `covdat_to_fits`, `cov_cl_to_fits`, `tau_to_fits`, `rho_to_fits`. Read those for the exact column formats and header keys; the load-bearing details:

- **BinTable 2-point HDUs** (`_create_2pt_hdu`): columns `BIN1, BIN2, ANGBIN, VALUE, ANG`; header carries `2PTDATA=T`, `QUANT1`, `QUANT2`, `KERNEL_1=NZ_SOURCE`, `KERNEL_2=NZ_SOURCE`, `WINDOWS=SAMPLE`. The `(QUANT1, QUANT2)` pair labels the statistic: `XI_PLUS`/`XI_MINUS` use `(G+R, G+R)`; `TAU_0_PLUS` uses `(G+R, P+R)`; `TAU_2_PLUS` uses `(G+R, SR+R)`. There is no single `QUANT` key.
- **`NZ_SOURCE`** is the n(z) extname (not `NZDATA`; `NZDATA=T` is a header card *inside* the `NZ_SOURCE` HDU). The 2-point HDUs reference it via `KERNEL_1/2="NZ_SOURCE"`.
- **`COVMAT`** block order and offsets follow `covdat_to_fits`: `NAME_0=XI_PLUS, STRT_0=0`; `NAME_1=XI_MINUS, STRT_1=nbins`; then, when tau is present, `TAU_0_PLUS`/`TAU_2_PLUS`; then, when pseudo-Cℓ is present, `CELL_EE`. The τ covariance is truncated from 3·nbins to 2·nbins (drops the τ₅ block). `COVMAT_CELL` is a separate image HDU.
- **`RHO_STATS`** is copied through from the input ρ-statistics table (`rho_to_fits`).

## Interfaces and contracts

New module `src/sp_validation/converters.py`.

### SACC → 2pt-FITS

```python
def sacc_to_2pt_fits(sacc_path: str, out_path: str) -> None:
    """Write a DES-style 2pt-FITS file (11-HDU layout above) from a
    {version}.sacc analysis file, structurally matching cosmosis_fitting.py's
    assembly of the same measurements."""
```

The converter reads SACC points in the file's insertion order and permutes to the 2pt-FITS type-major layout via `sacc.Sacc.indices`; it never re-sorts by value. The real-space blocks of the SACC covariance are extracted as slices — the SACC file's `FullCovariance` is assembled with zero cross-statistic blocks (the zero-cross-block guarantee is the `sacc_io` PR's `assemble_covariance` contract for `{version}.sacc`; this converter depends on it), matching the block-diagonal `np.block` structure of `covdat_to_fits`. The pseudo-Cℓ covariance populates `COVMAT_CELL`.

Load the SACC file with `sp_validation.sacc_io` readers; do not hand-parse FITS.

### OneCovariance glue

```python
def nz_to_onecovariance(sacc_path: str, out_dir: str) -> dict:
    """Write per-bin two-column (z, nz) text files for OneCovariance and return
    the config dict it needs."""

def onecovariance_to_sacc_cov(oc_output_path: str, s: "sacc.Sacc") -> np.ndarray:
    """Read OneCovariance's flat output and return the covariance matrix aligned
    to s's insertion order, ready for sacc_io.assemble_covariance."""
```

- `nz_to_onecovariance` writes one whitespace-delimited two-column file per tomographic bin, no header line, column order `(z, nz)`, ascending z. The returned dict must enumerate exactly the fields OneCovariance's config consumes: at minimum the per-bin file paths (in bin order), the number of tomographic bins, and the z-column/nz-column conventions. Pin the exact key names in the docstring against OneCovariance's config parser — an unspecified key set lets two implementations diverge.
- `onecovariance_to_sacc_cov` reuses `sp_validation.statistics.cov_from_one_covariance` for parsing the flat table. Note the column convention there: `index_value = 10 if gaussian else 9`, i.e. **column 10 is the Gaussian-only covariance and column 9 is Gaussian+non-Gaussian** (source: `statistics.py`). This PR adds only the SACC-alignment wrapper around that reshape, plus the n(z)→text input side. It must not touch or import `scratch/guerrini/one_covariance.py` (reserved).

## Acceptance criteria

Primary contract is **semantic equality**, not byte-identity: the converter output and a directly-assembled reference must agree HDU-for-HDU on extension-name set and order, per-HDU data arrays, and every load-bearing header card (`QUANT1/2`, `KERNEL_*`, `2PTDATA`, `COVDATA`, `NAME_i`/`STRT_i` offsets). Byte-identity across two independent FITS assembly paths depends on astropy version, column/card insertion order, and padding, and is not required. (The reference file carries no `DATE`/`CHECKSUM`, so a byte-compare is not defeated by timestamps and may be run as an opportunistic extra check, but it is not the gate.)

- **Real reference:** semantic-equality check against `cosmo_inference/data/SP_v1.4.6_leak_corr_A_minsep=1.0_maxsep=250.0_nbins=20_npatch=1/cosmosis_SP_v1.4.6_leak_corr_A_….fits` (11 HDUs). Convert the corresponding `{version}.sacc` and compare.
- **Synthetic:** `src/sp_validation/tests/test_cosmosis_fitting.py` provides deterministic synthetic inputs and per-HDU structural/value assertions (extension-name sets, data-vector shapes, `STRT_i` offsets, `np.array_equal` on data). Build a SACC file from those same inputs, convert, and assert the same structural/value equalities against the direct assembly. (This is net-new harness code; the existing test does not perform a byte-compare and none is required.)
- **CosmoSIS run** (manual gate, run inside the container; not CI): the chain runs to completion on a converter-produced file with the PSF ρ/τ modules loaded and produces a data vector / χ² matching a run on the directly-assembled file, with no pipeline change.
- **OneCovariance round-trip:** n(z) tracers → text input (correct per-bin files and config dict), and a sample OneCovariance flat output → a SACC covariance block whose ordering matches `assemble_covariance`'s contiguous-ascending tiling of `0..len(s.mean)`.
- No import of a theory backend (pyccl, CAMB, firecrown, Smokescreen) anywhere in the module.

## Non-goals

CosmoSIS keeps reading 2pt-FITS; adopting its native SACC likelihood is separate later work. The converter reproduces the current 2pt-FITS layout exactly and adds no statistics, binning, or covariance assumptions of its own; anything beyond the 11-HDU layout is out of scope. Converters operate on whatever vector is in the SACC file (blinded or not) and compute no theory or blinding. This PR supplies the glue at OneCovariance's two edges, not a runner that invokes it.
