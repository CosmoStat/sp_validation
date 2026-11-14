# Assembly scripts

The assemble stage: the scripts that bring ShapePipe and PhotoPipe outputs
together into the catalogues calibrate reads. See `docs/source/stages.md`.

## The v2 collection (ShapePipe v2 + PhotoPipe DR6)

The v2 catalogue is one HATS collection: the shapes as the primary catalogue,
an index on the DR6 object ID, and PhotoPipe as an extension. A PhotoPipe
rerun replaces `photopipe/` and leaves the shapes alone.

```
COLLECTION/
├── shapes/                 primary: the shape catalogue with OBJECT_ID
├── shapes_OBJECT_ID/       index: OBJECT_ID -> pixel, the default index
├── photopipe/              extension data, a collection of its own
│   ├── photopipe/          one row per shape row: _healpix_29, OBJECT_ID, the
│   │                       shape coordinates and every PhotoPipe column
│   ├── collection.properties
│   └── BUILD.json, match_by_tile.parquet, photopipe_sources.parquet
├── photopipe.properties    the extension, joined to the primary on OBJECT_ID
└── collection.properties
```

`OBJECT_ID` is the DR6 object ID in one int64,
`10**6 * tile + NUMBER` with the tile as the integer `XXXYYY`
(194.248 and `NUMBER` 32402 give 194248032402), null where `NUMBER` < 0
(`sp_validation.object_id`).

How the shapes and PhotoPipe relate:

1. **Join on the object key within the pixel.** Both catalogues have the same
   HATS pixels, and an object is in the same pixel in both; that much HATS
   guarantees. The key is `OBJECT_ID`, which `photopipe.properties` names.
   HATS says nothing about row position inside a pixel file.
2. **Row-for-row alignment is our convention on top.** The join writes each
   PhotoPipe pixel file with the rows of the matching shape file, in the same
   order, so reading the same pixel from both gives aligned rows with no join.
   `sp_validation.io` reads the collection this way and checks `OBJECT_ID`
   on both sides. A file rewritten by another tool keeps (1) and may lose (2).

| Step | Script | Runs on | Does |
|------|--------|---------|------|
| 1 | `extract_photopipe.py` | CANFAR, beside the tile tree | Read every column of every `UNIONS.<tile>_ugriz_photoz_ext.cat` (and `MASK` from `_finalmask`) into one Parquet shard per RA prefix, with a manifest of the files read. Unchanged shards are skipped. |
| 2 | `join_photopipe.py` | a Slurm job, in the image | Join the extract to the shapes on (tile, `NUMBER` = `SeqNr`) and write `COLLECTION/photopipe/`: the catalogue `photopipe/` and its provenance. |
| 3 | `add_object_id.py` | a Slurm job, in the image | Rewrite every shape pixel file with `OBJECT_ID` after `NUMBER`, rows and row groups unchanged, as `COLLECTION/shapes`. |
| 4 | `build_collection.py` | a Slurm job, in the image | Build the `OBJECT_ID` index with `hats_import` and write the properties files that make `COLLECTION` a collection with PhotoPipe as its extension. |

Steps 2 and 3 are independent; step 4 needs both. The layout is the one
`hats_import`'s extension split writes. `join_photopipe.sh` runs steps 1 and
2 and the copy between them from a login node, in about 100 minutes from cold:

```bash
ACCOUNT=<slurm account> CANFAR_USER=<canfar user> scripts/assembly/join_photopipe.sh
```

It needs a CADC proxy at `~/.ssl/cadcproxy.pem`. The shapes default to
`${unions_data}/v2.x/run1-snap3-hats/catalogue` and the collection to
`${unions_data}/v2.x/run1-snap3-collection`; a new ShapePipe or PhotoPipe
version is a rerun with `--shapes`/`--output`/`--collection` changed.

The join is exact because ShapePipe v2 and this PhotoPipe run both number
objects with Stephen Gwyn's DR6 IDs. The `_SP_ugriz` PhotoPipe files ran on
other detections, with their own numbering, and are not used.
`PHOTOPIPE_STATUS` is 0 for a joined row, 1 when the tile has no row with that
ID, 2 when the tile has no PhotoPipe file; `MATCH_SEP_ARCSEC` records each
joined pair's separation as a check. PhotoPipe's `SeqNr` becomes
`PHOTOPIPE_SEQNR`, and the PhotoPipe columns whose names the shapes already
use take the prefix `PHOTOPIPE_` (`MAG_AUTO`, `MAGERR_AUTO`, `FLUX_RADIUS`).

Reading: `sp_validation.io.read_catalogue(COLLECTION, columns)` takes shape
and PhotoPipe columns by name. `lsdb.open_catalog(COLLECTION)` opens the
primary, and `catalog.id_search(values={"OBJECT_ID": [...]})` reads only the
pixels holding those IDs (open it with `OBJECT_ID` among the columns). lsdb
0.11.1 does not read extensions (`lsdb.read_hats` on `photopipe.properties`
raises `NotImplementedError`); `photopipe/` opens with `lsdb.open_catalog` as
a catalogue of its own.
