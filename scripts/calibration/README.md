# Calibration scripts

The calibrate stage: the scripts that turn a final `ShapePipe` output
catalogue into a science-ready, metacalibrated shear catalogue. Run them in
order. See `docs/source/post_processing.md` for the full prose.

**ShapePipe v2** (one snapshot of every tile, with mask columns):

| Step | Script | Does |
|------|--------|------|
| 1 | `calibrate_sample.py` | The calibrated shear sample from a snapshot's shapes: selection, weights, metacalibration, HATS catalogue ([below](#the-calibrated-sample-v2)). |

`compare_samples.py` compares two samples object by object.

**ShapePipe v1 and the image simulations** (per-patch catalogues):

| Step | Script | Does |
|------|--------|------|
| 1 | `extract_info.py` | Extract metacal + diagnostic info for one campaign; create pre-calibration shear catalogues. Configured via `params.py`. |
| 2 | `create_joint_comprehensive_cat.py` | Merge the per-campaign comprehensive catalogues into one joint catalogue (front-end of `catalog_builders.JointCat`). |
| 3 | `demo_apply_hsp_masks.py` | Add the structural and coverage (HealSparse) masks. |
| 4 | `calibrate_comprehensive_cat.py` | Galaxy selection + metacalibration (`sp_validation.response.calibrate_catalogue`). Uses the mask configs in `config/calibration/`, whose `metacal` block must declare `weight_scheme: des` or `iv`. |

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
