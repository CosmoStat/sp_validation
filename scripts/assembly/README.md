# Assembly scripts

The assemble stage: the scripts that bring ShapePipe and PhotoPipe outputs
together into the catalogues calibrate reads. See `docs/source/stages.md`.

## PhotoPipe join (ShapePipe v2)

Builds the comprehensive catalogue: every column of the v2 shape HATS catalogue
and every column of PhotoPipe DR6, one row per shape row, on the same pixels in
the same order. PhotoPipe's `SeqNr` becomes `PHOTOPIPE_SEQNR`, and the PhotoPipe
columns whose names the shapes already use take the prefix `PHOTOPIPE_`
(`MAG_AUTO`, `MAGERR_AUTO`, `FLUX_RADIUS`). The n(z) recipe
(`config/calibration/nz/v2.yaml`) reads it as `photometry`.

| Step | Script | Runs on | Does |
|------|--------|---------|------|
| 1 | `extract_photopipe.py` | CANFAR, beside the tile tree | Read every column of every `UNIONS.<tile>_ugriz_photoz_ext.cat` (and `MASK` from `_finalmask`) into one Parquet shard per RA prefix, with a manifest of the files read. Unchanged shards are skipped. |
| 2 | `join_photopipe.py` | a Slurm job, in the image | Join the extract to the shapes on (tile, `NUMBER` = `SeqNr`) and write the comprehensive `catalogue/`, `photopipe_sources.parquet`, `match_by_tile.parquet` and `BUILD.json` under the output. |

`join_photopipe.sh` runs both steps and the copy between them from a login
node, in about 100 minutes from cold:

```bash
ACCOUNT=<slurm account> CANFAR_USER=<canfar user> scripts/assembly/join_photopipe.sh
```

It needs a CADC proxy at `~/.ssl/cadcproxy.pem`. The shapes and the output
default to `${unions_data}/v2.x/run1-snap3-hats/catalogue` and
`${unions_data}/v2.x/run1-snap3-comprehensive`; a new ShapePipe or PhotoPipe version is a
rerun with `--shapes`/`--output` changed in step 2.

The join is exact because ShapePipe v2 and this PhotoPipe run both number
objects with Stephen Gwyn's DR6 IDs. The `_SP_ugriz` PhotoPipe files ran on
other detections, with their own numbering, and are not used.
`PHOTOPIPE_STATUS` is 0 for a joined row, 1 when the tile has no row with that
ID, 2 when the tile has no PhotoPipe file; `MATCH_SEP_ARCSEC` records each
joined pair's separation as a check.
