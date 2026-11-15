# Assembly scripts

The assemble stage: the scripts that bring ShapePipe and PhotoPipe outputs
together into the catalogues calibrate reads. See `docs/source/stages.md`.

## The v2 collection (ShapePipe v2 + PhotoPipe DR6)

The v2 catalogue is one HATS collection: the shapes as the primary catalogue,
an index on the DR6 object ID, and two extensions, PhotoPipe's photometry and
the spectroscopic redshifts PhotoPipe matched to DR6 objects. A PhotoPipe
rerun replaces `photopipe/` and `specz/` and leaves the shapes alone.

```
COLLECTION/
├── shapes/                 primary: the shape catalogue with OBJECT_ID
├── shapes_OBJECT_ID/       index: OBJECT_ID -> pixel, the default index
├── photopipe/              extension data, a collection of its own
│   ├── photopipe/          one row per shape row: _healpix_29, OBJECT_ID, the
│   │                       shape coordinates and every PhotoPipe column
│   ├── collection.properties
│   └── BUILD.json, match_by_tile.parquet, photopipe_sources.parquet
├── specz/                  extension data
│   ├── specz/              one row per shape object with a spectrum: the keys,
│   │                       Z_SPEC, Z_SPEC_ERR, Z_SPEC_CONFIDENCE, Z_SPEC_SURVEY,
│   │                       Z_SPEC_RA, Z_SPEC_DEC
│   ├── collection.properties
│   └── BUILD.json
├── photopipe.properties    the extensions, each joined to the primary on OBJECT_ID
├── specz.properties
└── collection.properties
```

`OBJECT_ID` is the DR6 object ID in one int64,
`10**6 * tile + NUMBER` with the tile as the integer `XXXYYY`
(194.248 and `NUMBER` 32402 give 194248032402), null where `NUMBER` < 0
(`sp_validation.object_id`).

How the shapes and an extension relate:

1. **Join on the object key within the pixel.** Each extension file is on a
   pixel of the shapes, and an object is in the same pixel in both; that much
   HATS guarantees. The key is `OBJECT_ID`, which `<extension>.properties`
   names, with the join style `left`: a shape row without an extension row
   takes nulls. `sp_validation.io` and `lsdb` both join this way.
2. **PhotoPipe is dense, spec-z sparse.** A PhotoPipe pixel file holds every
   row of the shape file, in the same order. A spec-z file holds only the
   objects with a spectrum, in the shapes' order, and a pixel without one has
   no file: 4.7M rows against the shapes' 773M. Written dense, spec-z would
   repeat the four key columns on every shape row to carry a few columns on
   0.6% of them.

| Step | Script | Runs on | Does |
|------|--------|---------|------|
| 1 | `extract_photopipe.py` | CANFAR, beside the tile tree | Read every column of every `UNIONS.<tile>_ugriz_photoz_ext.cat` (and `MASK` from `_finalmask`) into one Parquet shard per RA prefix, with a manifest of the files read. Unchanged shards are skipped. |
| 2 | `join_photopipe.py` | a Slurm job, in the image | Join the extract to the shapes on (tile, `NUMBER` = `SeqNr`) and write `COLLECTION/photopipe/`: the catalogue `photopipe/` and its provenance. |
| 3 | `add_object_id.py` | a Slurm job, in the image | Rewrite every shape pixel file with `OBJECT_ID` after `NUMBER`, rows and row groups unchanged, as `COLLECTION/shapes`. |
| 4 | `join_specz.py` | a Slurm job, in the image | Find each row of PhotoPipe's spec-z match in `COLLECTION/photopipe/` and write `COLLECTION/specz/`. |
| 5 | `build_collection.py` | a Slurm job, in the image | Build the `OBJECT_ID` index with `hats_import` and write the properties files that make `COLLECTION` a collection with PhotoPipe and spec-z as its extensions. |

Steps 2 and 3 are independent; step 4 needs step 2, and step 5 needs the
others. The layout is the one `hats_import`'s extension split writes.
`join_photopipe.sh` runs steps 1 and 2 and the copy between them from a login
node, in about 100 minutes from cold:

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

**Spec-z.** PhotoPipe matched DR6 objects to a compilation of spectra on
every tile; `${unions_data}/photopipe-dr6-specz-all/` holds the result, the
per-tile `UNIONS.<tile>_ugriz_photoz_ext_specz.cat` files concatenated in tile
order into `UNIONS_DR6_ugriz_specz.cat` (4,746,631 rows, the PhotoPipe columns
of each object and the spectrum's position and redshift), and the compilation
itself, `asc/UNIONS.<tile>_redshifts-20261019.asc` (`ra dec zspec zspec_err
wspec ref`). The table has no tile column and its `SeqNr` renumbers rows; the
object's DR6 number is `Pair_0_spec`. Step 4 finds each row in the PhotoPipe
extension by that number and its exact position, which gives `OBJECT_ID`, and
takes the survey (`ref`) and confidence (`wspec`) from the compilation by the
spectrum's exact position. Rows pairing several objects (`N_0` > 1) or several
spectra (`N_1` > 1) average them and are left out (5,469). Of the rest,
4,733,015 are shape objects; a spectrum in a tile overlap is on the object in
each tile (4,515,100 distinct spectra). Selecting `Z_SPEC_SURVEY` in
`deep2_final`, `deep3_final` with the n(z) photometry cuts gives the EGS
calibrators.

Reading: `sp_validation.io.read_catalogue(COLLECTION, columns)` takes shape,
PhotoPipe and spec-z columns by name; a spec-z column is NaN (None for
`Z_SPEC_SURVEY`) where the object has no spectrum. `lsdb.open_catalog(COLLECTION)`
opens the primary, `catalog.load_extension("specz")` (or `"photopipe"`) joins
an extension, and `catalog.id_search(values={"OBJECT_ID": [...]})` reads only
the pixels holding those IDs (open it with `OBJECT_ID` among the columns).
`lsdb.read_hats` on an extension's properties file raises
`NotImplementedError`; `photopipe/` and `specz/` also open with
`lsdb.open_catalog` as catalogues of their own.
