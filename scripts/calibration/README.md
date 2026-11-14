# Calibration scripts

The calibrate stage: the scripts that turn a final `ShapePipe` output
catalogue into a science-ready, metacalibrated shear catalogue, and give its
selection a redshift distribution. Run them in order. See
`docs/source/post_processing.md` for the full prose.

**ShapePipe v2** (one snapshot of every tile, with mask columns):

| Step | Script | Does |
|------|--------|------|
| 1 | `calibrate_sample.py` | The calibrated shear sample from a snapshot's shapes: selection, weights, metacalibration, HATS catalogue ([below](#the-calibrated-sample-v2)). |
| 2 | `calibrate_nz.py` | SOM direct calibration: the tomographic n(z) file, each row's `TOM_BIN_ID` and n(z) realisations. |

`compare_samples.py` compares two samples object by object.

**ShapePipe v1 and the image simulations** (per-patch catalogues):

| Step | Script | Does |
|------|--------|------|
| 1 | `extract_info.py` | Extract metacal + diagnostic info for one campaign; create pre-calibration shear catalogues. Configured via `params.py`. |
| 2 | `create_joint_comprehensive_cat.py` | Merge the per-campaign comprehensive catalogues into one joint catalogue (front-end of `catalog_builders.JointCat`). |
| 3 | `demo_apply_hsp_masks.py` | Add the structural and coverage (HealSparse) masks. |
| 4 | `calibrate_comprehensive_cat.py` | Galaxy selection + metacalibration. Uses the mask configs in `config/calibration/`. |

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
  write over an existing `catalogs/calibrated`.
- `reference` is optional: a catalogue the sample should reproduce. The two
  are compared object by object into `provenance/calibrated_comparison.json`,
  and the run fails unless its `agree` is true.
- `container` is optional: an image other than your own
  (`~/.cache/sp_validation/sp_validation.sif`).

What it does (`sp_validation.sample.pipeline`): reads the shapes one HEALPix
pixel at a time, in parallel; cuts tile overlaps against every tile of the
snapshot; applies the selection; sums, over all pixels, the per-cell moments
that give the DES weight table and the response `R = R_shear + R_selection`
and additive bias `c_e` of the whole sample; then writes each pixel with
`w_des` and `e = R⁻¹(g − c_e)`. The products, laid out as a release:

- `catalogs/calibrated/`: HATS catalogue of `RA`, `Dec`, `e1`, `e2`, `w_des`,
  `e1_PSF`, `e2_PSF`, `SIZE_RATIO`, `SNR`, `N_EPOCH`, `TILE_KEY`, `TILE_ID`,
  `NUMBER` and `OBJECT_ID`, the DR6 object ID that joins it to the collection.
  It has no `TOM_BIN_ID`: the n(z) step assigns bins, by `OBJECT_ID`.
- `catalogs/response.json`, `catalogs/des_weights.fits`: the response and its
  parts, `c_e`, the branch populations; the weight table, readable as a
  `des_weight_product`.
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

## Redshift distribution

The n(z) file is this stage's output. A `cosmo_val/cat_config.yaml` entry
names it in `shear.redshift_path` and its bin column in `tomo_bin_col`;
cosmo_val only reads it. The file has z in column zero and one density per
bin, each integrating to that bin's retained shear-weight fraction; their sum
has unit integral.

The recipe for v2 is `config/calibration/nz/v2.yaml`, written with data roots
(`config/data_roots.yaml`). Run inside the container on an allocation:

```bash
python scripts/calibration/calibrate_nz.py config/calibration/nz/v2.yaml --prepare --workers 32
```

`--prepare` builds the two inputs (a minute on 32 cores); later runs reuse
them. `--output DIRECTORY` redirects the products and `--figures DIRECTORY`
renders diagnostics. The products (`sp_validation.nz_pipeline`) are
`nz_tomo.txt`, `nz_nontomo.txt` (their sum), `tom_bin.parquet` (`TOM_BIN_ID`
and `ZBAND` row-aligned with the catalogue; `TOM_BIN_ID` is 1 to 4 for the
four bins and 0 for a galaxy in no bin, and cosmo_val drops IDs ≤ 0), `realisations.npz`,
`calibration_arrays.npz`, `som_model.npz` and `summary.json`.

**Sources** are the rows of the catalogue cosmo_val measures, with its `w_des`.
Sources and calibrators share one photometry: PhotoPipe's run on Stephen
Gwyn's DR6 detections (`UNIONS.<tile>_ugriz_photoz_ext.cat`, dereddened GAaP
ugriz, z2 and BPZ). Each catalogue row is found in the ShapePipe HATS
catalogue by exact position, and the PhotoPipe HATS catalogue beside it
(built by `scripts/assembly/`) is joined row for row on (tile, `NUMBER` = `SeqNr`) (`sp_validation.nz_inputs`).

**Calibrators** are the DEEP2 and DEEP3 spectra in the EGS field that
PhotoPipe matched to its own objects (`UNIONS.<tile>_ugriz_photoz_ext_specz.cat`,
surveys named by the redshift compilation beside it), one row per spectrum,
with complete ugriz. Stage them from
`arc:projects/unions/catalogues/unions/GAaP_photometry/UNIONS_DR6/UNIONS.<tile>/`
for the tiles around RA 215°, Dec 52.8°.

**Method.** A 101×101, 60-epoch batch SOM of all ten ugriz colours plus r,
standardized per feature and trained on the calibrators, grouped into 2,000
Ward groups. Bins are DR6 BPZ intervals [0.10, 0.37, 0.55, 0.76, 1.20), lower
edges included. Within a bin, a group is gold when it holds in-bin
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
the draws need from the sources. `python -m sp_validation.nz_ensemble <run
directory> -o realisations.npz [--settings ensemble.yaml]` draws from an
existing run. A draw perturbs the calibration's inputs and recalibrates: each
in-bin spectrum of a gold group carries the group's source weight S_kg times
q_i / Σ_group q_j (`sp_validation.nz_ensemble`). Each key of the `ensemble`
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
