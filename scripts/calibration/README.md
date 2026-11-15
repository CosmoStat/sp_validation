# Calibration scripts

The calibrate stage: the scripts that turn a final `ShapePipe` output
catalogue into a science-ready, metacalibrated shear catalogue. Run them in
order. See `docs/source/post_processing.md` for the full prose.

**ShapePipe v2** (one snapshot of every tile, with mask columns):

| Step | Script | Does |
|------|--------|------|
| 1 | `calibrate_sample.py` | The calibrated shear sample from a snapshot's shapes: selection, weights, metacalibration, HATS catalogue ([below](#the-calibrated-sample-v2)). |
| 2 | `calibrate_nz.py` | The tomographic bins and their n(z) ([below](#redshift-distribution-step-5)). |
| 3 | `bin_responses.py` | The response of each tomographic bin, into `catalogs/response.json` ([below](#the-response-of-each-bin)). |

`compare_samples.py` compares two samples object by object.

**ShapePipe v1 and the image simulations** (per-patch catalogues):

| Step | Script | Does |
|------|--------|------|
| 1 | `extract_info.py` | Extract metacal + diagnostic info for one campaign; create pre-calibration shear catalogues. Configured via `params.py`. |
| 2 | `create_joint_comprehensive_cat.py` | Merge the per-campaign comprehensive catalogues into one joint catalogue (front-end of `catalog_builders.JointCat`). |
| 3 | `demo_apply_hsp_masks.py` | Add the structural and coverage (HealSparse) masks. |
| 4 | `calibrate_comprehensive_cat.py` | Galaxy selection + metacalibration (`sp_validation.response.calibrate_catalogue`). Uses the mask configs in `config/calibration/`, whose `metacal` block must declare `weight_scheme: des` or `iv`. |
| 5 | `calibrate_nz.py` | SOM direct calibration: the tomographic n(z) file, each row's `TOM_BIN_ID` and n(z) realisations. |

Step 4 writes the cut catalogue `shape_catalog_cut_<shape>.fits` and beside it:

- `<stem>_response.json`: R, its shear, selection and joint parts, `c_e`,
  `c_g`, the branch populations, and (with `TILE_ID`) the tile-jackknife
  covariance;
- `<stem>_sums.npz`: the per-tile response sums (`response.Sums`), from which
  `scripts/compute_m_bias_image_sims.py` measures image-simulation m;
- `<stem>_des_weights.fits`: the DES weight table (`response.write_table`),
  for `weight_scheme: des`.

A `metacal.weight_table` path makes step 4 weight with that table instead of
fitting one (image simulations use the data's table).

`params.py` is the shared parameter template (paths, column names, survey
constants) imported by `extract_info.py`; copy and edit it per run.

## The calibrated sample (v2)

The sample is one rule of the Snakemake workflow, driven by a recipe. The
recipe for v2 is `config/calibration/sample/v2.yaml`: the ShapePipe snapshot
(the directory holding `MANIFEST.json`), its shapes (the v2 collection built
by `scripts/assembly/`), the selection config
(`config/calibration/mask_v2.0.yaml`) and the release directory to write
under, with data roots (`config/data_roots.yaml`). The choices and their
alternatives are recorded as decisions in
`scratch/cdaley/unions_astra/analyses/dr6_catalogue/astra.yaml`.

To build it you need this checkout, `apptainer` and `snakemake` on the host
(`uv tool install --python 3.12 snakemake`, or `uvx --python 3.12 snakemake`
in place of `snakemake` below to run it without installing), the image
(`spv-container pull`, see `workflow/README.md`), and the snapshot and its
collection under the `unions_v2` data root. From the repository root:

```bash
snakemake --profile workflow/profiles/default -s workflow/calibrate/Snakefile -j 32 \
    -d /path/to/a/working/directory \
    --apptainer-args "--cleanenv --bind /home,/project,/scratch" \
    --config calibrate='{recipe: config/calibration/sample/v2.yaml, output: /path/to/a/new/directory, reference: /path/to/catalogs/calibrated}' \
        container=/path/to/sp_validation.sif
```

- `-j` is the number of cores. The build is one job that uses all of them:
  3 minutes on 60 cores and 6.5 minutes on 16 for v2, comparison included,
  with about 2.5 GiB per core.
- On a cluster, the whole command runs inside an allocation, since the profile
  runs jobs where Snakemake runs:
  `srun --jobid=<allocation> -n 1 -c 32 bash -c 'module load apptainer && snakemake ... -j 32'`.
- `-d` keeps Snakemake's state (`.snakemake/`) out of the checkout. `recipe`
  stays relative to the repository root.
- `--bind` lists the disks that hold the data, the checkout and the output,
  after `/home`.
- `output` is where the products go. Without it they go to the recipe's own
  `output`, the release directory `${unions_v2}/v2.0.0`. The build refuses to
  write over an existing `catalogs/calibrated`, a partial one left by an
  interrupted build included: remove it yourself to build there again.
- `reference` is optional: a catalogue the sample should reproduce. The two
  are compared object by object into `provenance/calibrated_comparison.json`,
  and the run fails unless its `agree` is true (the report is then kept as
  `calibrated_comparison.json.failed`).
- `container` is optional: an image other than your own
  (`~/.cache/sp_validation/sp_validation.sif`).

What it does (`sp_validation.sample.pipeline`): reads the shapes one HEALPix
pixel at a time, in parallel; cuts tile overlaps against every tile of the
snapshot; applies the selection; reduces each pixel to per-(tile, cell) sums
(`sp_validation.response`, `docs/source/response_calibration.md`), whose total
gives the DES weight table, the response `R = R_shear + R_selection` and the
additive bias `c_e` of the whole sample, and their delete-one-tile jackknife;
then writes each pixel with `w_des` and `e = R⁻¹(g − c_e)`. The products, laid out as a release:

- `catalogs/calibrated/`: HATS catalogue of `RA`, `Dec`, `e1`, `e2`, `w_des`,
  `e1_PSF`, `e2_PSF`, `SIZE_RATIO`, `SNR`, `N_EPOCH`, `TILE_KEY`, `TILE_ID`,
  `NUMBER` and `OBJECT_ID`, the DR6 object ID that joins it to the collection.
  It has no `TOM_BIN_ID`: the n(z) step assigns bins, by `OBJECT_ID`.
- `catalogs/response.json`, `catalogs/des_weights.fits`: the response and its
  parts, `c_e`, the branch populations and their jackknife covariance
  (`response.record`); the weight table (`response.read_table` reads it).
- `catalogs/response_sums.npz`: the sums, grouped by `TILE_KEY`
  (`response.Sums.load` reads them). The response, `c_e` and jackknife of
  any table on the same cells (unit weights, a refitted or reweighted table,
  a restricted S/N range) follow from them without the catalogue.
- `catalogs/survey.json`, `masks/footprint-nside4096-ring.fits`: area, n_eff
  and σ_e over the occupied pixels, and those pixels.
- `catalogs/config_mask.yaml`, `provenance/calibrated_recipe.yaml`: the
  selection config and the recipe, byte for byte.
- `provenance/calibrated_tile_accounting.json`: rows of every snapshot tile
  after each cut, and for each tile without galaxies the cut that empties it.

**The snapshot is checked twice.** The build stops unless the shapes name the
manifest's SHA-256 as their source and hold exactly the manifest's object
count of every tile. The catalogue then records the snapshot (name, tile
count, manifest SHA-256) in its HATS properties, and a
`cosmo_val/cat_config.yaml` entry names the snapshot it expects under
`shear.snapshot`; cosmo_val stops at start when the catalogue records another
(`sp_validation.sample.snapshot`).

Outside the workflow, the same build is
`python scripts/calibration/calibrate_sample.py <recipe> --output <directory> --workers N`
inside the container, and the comparison
`python scripts/calibration/compare_samples.py <new> <reference> --output comparison.json --workers N`.

## The response of each bin

A tomographic bin has its own response: the sample's estimator on the bin's
rows, with the sample's cuts, cells and weight table held fixed
(`docs/source/response_calibration.md`). `sp_validation.sample.bins`
measures them after the n(z) stage has put its bins in the release, as the
rule `bin_responses`:

```bash
snakemake --profile workflow/profiles/default -s workflow/calibrate/Snakefile -j 96 \
    -d /path/to/a/working/directory \
    --apptainer-args "--cleanenv --bind /home,/project,/scratch" \
    bin_responses
```

or, inside the container,
`python scripts/calibration/bin_responses.py config/calibration/sample/v2.yaml --workers 96`
(`--output DIRECTORY` for a sample built elsewhere). It reads, under the
release directory, the sample's `catalogs/response.json`, `des_weights.fits`
and `response_sums.npz`, and the n(z) stage's `nz/tom_bin.parquet` and
`provenance/nz_som_model.npz`, `nz_calibration_arrays.npz` and
`nz_recipe.yaml`. One pass over the shapes takes about 3 minutes on 96 cores
and 40 GiB plus 2.5 GiB per core.

The selection response needs the rows a sheared metacal branch selects and
the no-shear branch does not, which have no `TOM_BIN_ID`. So every
preselected row gets its bin from its PhotoPipe photometry with the n(z)
stage's saved model, as that stage assigns it (a row without an object ID has
no photometry and is in no bin), and keeps that bin on every branch. The rows
are reduced to sums per (tile, bin) and cell, and the run stops unless

- every galaxy of `nz/tom_bin.parquet` with an object ID is selected with
  its `TOM_BIN_ID`, and the selected galaxies it does not name by ID are as
  many as its rows without one (those rows are in its bin 0, as the n(z)
  stage joins photometry by ID; here they keep the bin of their photometry,
  recorded as `unnamed_per_bin`), and
- the bins' sums, bin 0 included, add to `catalogs/response_sums.npz`.

Each bin's R, its shear and selection parts, `c_e`, the branch populations
and their delete-one-tile jackknife covariance (`response.record`) go into
`catalogs/response.json` under `bins`:
`{"column": "TOM_BIN_ID", "records": {"1": {...}, ..., "4": {...}}}`; the
rest of the file is the whole sample's record, unchanged.
`provenance/bin_response_checks.json` records what the checks compared and is
written last. The response of the bin assignment itself is not included:
the photometry has no sheared versions.

Rerun it whenever the bins, the cuts or the weights change: after every n(z)
run, and after every rebuild of the sample (which rewrites `response.json`
without `bins`). Snakemake reruns it when the sample or the n(z) products are
newer than its checks.

## Redshift distribution (step 5)

The n(z) file is this stage's output. A `cosmo_val/cat_config.yaml` entry
names it in `shear.redshift_path` and its bin column in `tomo_bin_col`;
cosmo_val only reads it. The file has z in column zero and one density per
bin, each integrating to that bin's retained shear-weight fraction; their sum
has unit integral.

The recipe for v2 is `config/calibration/nz/v2.yaml`: the inputs, every
choice of the calibration, and the output directory, written with data roots
(`config/data_roots.yaml`). The choices and their alternatives are recorded as
decisions in `scratch/cdaley/unions_astra/analyses/nz/astra.yaml`. Run inside
the container on an allocation:

```bash
python scripts/calibration/calibrate_nz.py config/calibration/nz/v2.yaml --workers 32
```

The products go to the recipe's `output`, the release's `nz/`; `--output
DIRECTORY` writes them elsewhere. They are (`sp_validation.nz.pipeline`)
`nz_tomo.txt`, `nz_nontomo.txt` (their sum), `tom_bin.parquet` (`OBJECT_ID`,
`TOM_BIN_ID` and `ZBAND`, row-aligned with the catalogue; `TOM_BIN_ID` is 1 to
4 for the four bins and 0 for a galaxy in no bin, and cosmo_val drops IDs ≤ 0),
`realisations.npz`, `calibrators.npz`, `calibration_arrays.npz`,
`som_model.npz`, `summary.json` and `recipe.yaml`, the recipe file byte for
byte.

A run redefines the bins, so its `nz_tomo.txt` describes its own
`tom_bin.parquet` and no earlier `TOM_BIN_ID`. Write that column into the
catalogue (the rows are aligned) whenever the n(z) a `cat_config` entry reads
is replaced.

**Sources** are the rows of the released calibrated shear catalogue, with its
`w_des`. Sources and calibrators share one photometry: PhotoPipe's run on
Stephen Gwyn's DR6 detections (dereddened GAaP ugriz, z2 and BPZ), used as
delivered. Each row takes its photometry from the v2 collection (the shapes
with their PhotoPipe extension, built by `scripts/assembly/`) by its DR6
object ID, `OBJECT_ID` (`sp_validation.nz.inputs.sources`); no positions are
compared. `summary.json` reports, under `match`, the shear weight of rows
with no ID, with an ID the collection lacks, in a tile without PhotoPipe,
absent from PhotoPipe in a tile it covers, and undetected in each band.

**Calibrators** are a selection on the same collection's spec-z extension
(`specz/`, PhotoPipe's matches to its own objects, joined on `OBJECT_ID`;
`sp_validation.nz.inputs.calibrators`): spectra from the recipe's surveys
(DEEP2 and DEEP3) whose position lies in the recipe's field (the EGS; DEEP2's
Field 2 at RA 252° was colour-preselected to z > 0.7 and stays out), one row
per spectrum, with complete ugriz after the z2 fill. Nothing is staged
separately.

**Method.** A 101×101, 60-epoch batch SOM of all ten ugriz colours plus r,
standardized per feature and trained on the calibrators, grouped into 2,000
Ward groups. The four bins are intervals of DR6 BPZ between 0.10 and 1.20,
lower edges included, holding equal shear weight: each run places the three
inner edges at the quartiles of the weight of its sources with complete
photometry and BPZ in range, before the gold selection, and records them in
`summary.json`. Equal weight gives the bins comparable signal-to-noise, and
the range is where BPZ on ugriz is usable (the KiDS range). Within a bin, a group is gold when it holds in-bin
calibrators and their mean spec-z agrees with its sources' weighted mean BPZ
within max(0.4, 5 NMAD of spec-z − BPZ); the test runs on every source of the
catalogue. Each spectrum carries its group's source shear weight divided by
the group's number of in-bin calibrators. The spectra are deposited linearly
onto the recipe's grid ([0, 5] in steps of 0.05), which conserves each bin's
mass and mean and broadens it by about step²/6 in variance. The step sets the
delivered resolution: finer grids resolve the line-of-sight structure of the
single calibration field, which is not a property of the source population.
Against a 0.01 grid, 0.05 moves the ξ± theory (12–83′) by Δχ² < 0.01 without
intrinsic alignments and 0.06 with A_IA = 1, and leaves ⟨z⟩ unchanged.
Canonical z takes precedence; detected z2
fills missing z after the recipe's offset, for sources and calibrators alike.

**Realisations.** With an `ensemble` section in the recipe, the run also
writes `realisations.npz`: `z` and `realisations` (n_draw, n_bin, n_z), each
bin unit-integral, every bin drawn together, plus `ensemble_inputs.npz`, what
the draws need from the sources. `python -m sp_validation.nz.ensemble <run
directory> -o realisations.npz [--settings ensemble.yaml]` draws from an
existing run. A draw perturbs the calibration's inputs and recalibrates: each
in-bin spectrum of a gold group carries the group's source weight S_kg times
q_i / Σ_group q_j (`sp_validation.nz.ensemble`). Each key of the `ensemble`
section switches one term on.

Statistical terms, through q_i = m_i (1 + δ(z_i)):

- `bootstrap`: m_i are multinomial bootstrap counts of the spectra. Bins are
  calibrated by disjoint spectra (split by BPZ), so this term is independent
  between bins.
- `density`: δ(z) is a lognormal galaxy density along the line of sight,
  per grid cell, with covariance from the nonlinear matter power seen through
  a 2° × 0.25° footprint (the DEEP2/3 strip) times bias² (b = 1.3). It is
  shared by every bin. Within-group renormalisation cancels δ wherever a
  group's spectra share a redshift, which is how the SOM suppresses sample
  variance.

Systematic terms:

- `zero_points`: a zero-point error of EGS against the wide field. The
  spectra trained the SOM, and training is translation-equivariant, so
  correcting their photometry would move the cells with them: the error acts
  as an offset of every source against the SOM, which moves the catalogue's
  weight between groups, and as a shift of each spectrum's BPZ along its
  local BPZ gradient (a plane through its 128 nearest spectra), which can move
  it between bins. Sizes come from stellar-locus regression on the 46,700
  stars (spec-z < 0.002, 17 < r < 21) of PhotoPipe's DR6 spec-z match: the
  colour offsets of 3.4 deg² patches (about the EGS footprint) at |b| > 30°,
  less each patch's own noise, have σ = 0.022, 0.019, 0.012, 0.021 mag in
  u−g, g−r, r−i, i−z, with correlations −0.59 (u−g, g−r) and −0.49 (r−i, i−z),
  the signature of single-band errors. The grey term, 0.01 mag, is the
  Pan-STARRS 3π uniformity the DR6 griz calibration inherits. Offsets along
  the stellar locus are invisible to this regression, so that component is
  unmeasured. EGS itself sits at +0.009, +0.035, −0.013 and +0.055
  (± 0.009, 0.005, 0.004, 0.005) off the survey locus; its i−z is redder than
  99% of patches. The stars are spectroscopic targets (colour-selected SDSS
  stars over most of the sky, DEEP2/3's faint stars in EGS), so a population
  difference could mimic part of that offset.
- `z2_offset`: an error in the constant that maps z2 onto z, applied to the z
  band of every source measured through z2 (a third of the gold weight; the
  spectra have none). σ = 0.02 mag: stars in z2-only areas sit 0.006–0.017 mag
  off the canonical i−z locus after the mapping, and galaxies may carry a
  colour term the stars don't.
- `specz_failure`: each spectrum's redshift is, with probability 0.01,
  replaced by the redshift of a random spectrum. PhotoPipe's compilation
  (2026-10-19) gives every DEEP2/3 redshift confidence 0.99; Newman et al.
  (2013) measure 0.75% (Q = 3) and 0.15% (Q = 4) failures from repeats, so
  0.01 is conservative. Failures pull a bin toward the spectra's mean
  redshift, so this term is a one-sided bias more than a scatter: the raw
  ensemble mean moves bin 1 by +0.004, and the delivered n(z), which already
  holds the real failures, is biased by about as much the other way. The
  recentred draws keep only the scatter.
- `nmad_factor`: the gold-QC factor drawn uniformly in [4, 6] around the
  fiducial 5 (threshold 0.40–0.56 with the 0.4 floor); gold is re-derived at
  that factor, so the bins' source selection moves with it. The other terms
  keep the catalogue's gold selection.

The zero-point and z2 offsets enter to first order: each draw adds half the
difference between the recalibrations at +v and −v. That keeps the
catalogue's response linear (its source tables are; central differences at
±0.02 mag, checked against exact reassignment at 0.05 mag) and cancels what is
even in the offsets: a group whose last in-bin spectrum leaves the bin loses
its sources' weight at either sign. Every term but `nmad_factor` keeps the
catalogue's selection, its gold sources wherever the offsets move them. The
draws are recentred on the calibration's n(z); the raw ensemble mean moves by
up to 0.004 in ⟨z⟩, mostly from spec-z failures.

Held fixed: the SOM and its groups. Left out: spectroscopic selection and
incompleteness (DEEP2's [OII] leaves the spectral window above z ≈ 1.4, so
spectra at higher redshift are missing, not wrong), method bias calibrated on
simulations, SOM training stochasticity (on v2, seeds move ⟨z⟩ by up to
0.0015 in bin 4), redshift-space smearing of δ, and bias evolution.

**Limits.** The calibrators are about 9,900 EGS DEEP2/3 spectra: no
field-to-field variance and no model of spectroscopic selection, so the n(z)
is for validation, null tests and mocks, not parameter constraints. It
describes only `TOM_BIN_ID > 0` rows; a non-tomographic measurement pairs
`nz_nontomo.txt` with those rows, not with the whole catalogue.
