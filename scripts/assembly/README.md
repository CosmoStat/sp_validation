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
