# Assembly scripts

The assemble stage: the scripts that bring ShapePipe and PhotoPipe outputs
together into the catalogues calibrate reads. See `docs/source/stages.md`.

## PhotoPipe join (ShapePipe v2)

Gives every row of the v2 shape HATS catalogue its PhotoPipe DR6 row, as a
second HATS catalogue beside the shapes: a PhotoPipe rerun replaces this
catalogue and leaves the shapes alone.
The n(z) recipe (`config/calibration/nz/v2.yaml`) reads it as `photometry`.

How the two catalogues relate:

1. **Join on the object key within the pixel.** Both catalogues have the same
   HATS pixels, and an object is in the same pixel in both; that much HATS
   guarantees. The key is `_source_row`, or equally (`TILE_ID`, `NUMBER`), the
   DR6 object ID. HATS says nothing about row position inside a pixel file.
2. **Row-for-row alignment is our convention on top.** This build writes each
   pixel file with the rows of the matching shape file, in the same order, so
   reading the same pixel from both gives aligned rows with no join. A file
   rewritten by another tool keeps (1) and may lose (2).

| Step | Script | Runs on | Does |
|------|--------|---------|------|
| 1 | `extract_photopipe.py` | CANFAR, beside the tile tree | Read the science columns of every `UNIONS.<tile>_ugriz_photoz_ext.cat` (and `MASK` from `_finalmask`) into one Parquet shard per RA prefix, with a manifest of the files read. Unchanged shards are skipped. |
| 2 | `join_photopipe.py` | a Slurm job, in the image | Join the extract to the shapes on (tile, `NUMBER` = `SeqNr`) and write `catalogue/`, `photopipe_sources.parquet`, `match_by_tile.parquet` and `BUILD.json` under the output. |

`join_photopipe.sh` runs both steps and the copy between them from a login
node, in about 100 minutes from cold:

```bash
ACCOUNT=<slurm account> CANFAR_USER=<canfar user> scripts/assembly/join_photopipe.sh
```

It needs a CADC proxy at `~/.ssl/cadcproxy.pem`. The shapes and the output
default to `${unions_data}/v2.x/run1-snap3-hats/catalogue` and
`${unions_data}/photopipe-dr6`; a new ShapePipe or PhotoPipe version is a
rerun with `--shapes`/`--output` changed in step 2.

The join is exact because ShapePipe v2 and this PhotoPipe run both number
objects with Stephen Gwyn's DR6 IDs. The `_SP_ugriz` PhotoPipe files ran on
other detections, with their own numbering, and are not used.
`PHOTOPIPE_STATUS` is 0 for a joined row, 1 when the tile has no row with that
ID, 2 when the tile has no PhotoPipe file; `MATCH_SEP_ARCSEC` records each
joined pair's separation as a check.

## HATS collection (ShapePipe v2)

Turns a catalogue holding the shape and PhotoPipe columns in each pixel file
(the v2.0.0 release's `catalogs/comprehensive`) into one HATS collection:

```
comprehensive/
├── shapes/                 primary: the shape catalogue's columns and OBJECT_ID
├── shapes_OBJECT_ID/       index: OBJECT_ID -> pixel, the default index
├── photopipe/              extension: the PhotoPipe columns, with OBJECT_ID,
│                           _healpix_29 and the coordinates (a collection of its own)
├── photopipe.properties    the extension, joined to the primary on OBJECT_ID
└── collection.properties
```

`OBJECT_ID` is the DR6 object ID in one int64,
`10**6 * tile + NUMBER` with the tile as the integer `XXXYYY`
(194.248 and `NUMBER` 32402 give 194248032402), null where `NUMBER` < 0
(`sp_validation.object_id`). The PhotoPipe join writes it too.

| Step | Script | Runs in | Does |
|------|--------|---------|------|
| 3 | `add_object_id.py` | the image | Rewrite every pixel file with `OBJECT_ID` after `NUMBER`, rows and row groups unchanged, to `WORK/merged/shapes`. |
| 4 | `index_object_id.py` | `uv run` | Build the `OBJECT_ID` index with `hats_import` and make `WORK/merged` a collection with it as the default index. |
| 5 | `split_photopipe.py` | `uv run` | Split `WORK/merged` with `hats_import`'s extension split: the shape catalogue's columns stay in the primary, the others go to the extension, and the index follows `OBJECT_ID` into the primary. Writes `WORK/comprehensive`. |

Each step runs on a Slurm allocation; steps 4 and 5 need `hats-import`, which
the image lacks, so they run under `uv run` (dependencies in the script header):

```bash
WORK=$SCRATCH/v2-collection
apptainer exec -B /project,/scratch --env PYTHONPATH=$PWD/src $SIF \
    python scripts/assembly/add_object_id.py $WORK --workers 32
PYTHONPATH=$PWD/src uv run scripts/assembly/index_object_id.py $WORK --workers 32
PYTHONPATH=$PWD/src uv run scripts/assembly/split_photopipe.py $WORK --workers 32
```

`WORK/merged` can be deleted once step 5 is done. The split keeps each pixel
file's rows in order, so the same pixel of `shapes` and `photopipe` holds the
same objects in the same order, as in the PhotoPipe join.

Reading: `lsdb.open_catalog(WORK/comprehensive)` opens the primary, and
`catalog.id_search(values={"OBJECT_ID": [...]})` reads only the pixels holding
those IDs. lsdb 0.11.1 does not read extensions (`lsdb.read_hats` on
`photopipe.properties` raises `NotImplementedError`); `photopipe/` opens with
`lsdb.open_catalog` as a catalogue of its own.
