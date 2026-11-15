# Catalogue-match tile selection

Decision `dr6_catalogue.catalogue_match_tile_selection` keeps the external paired fraction diagnostic-only and applies no tile cut.
Its rationale, alternatives with their measured costs, and `Values` pins live in the [ASTRA record](../scratch/cdaley/unions_astra/analyses/dr6_catalogue/astra.yaml); this page holds the audit coverage and tile-level evidence behind it.
The optional audit records published match counts, warnings and catalogue row counts without removing objects.
Missing metadata remains distinct from a measured empty external catalogue.

## Evidence and limits

The census covers the v2.0.0 release: 22,901 run-1 tiles and its selected sample of 191,016,178 galaxies under `config/calibration/mask_v2.0.yaml` (overlap, `(FLAGS & ~16) <= 2`, the four standard masks, `N_EPOCH >= 2`, valid fits, 10 < S/N < 500, 0.707 < T/T_PSF < 9).
ShapePipe's published catalogue keeps exactly the paired objects, so the external paired fraction is its row count over the DR6 external (PhotoPipe) catalogue's row count.
The release provenance (`match_by_tile.parquet`, `photopipe_sources.parquet`) gives both on 22,678 tiles, and on the 434 tiles also measured from published manifests or logs both counts agree exactly.
Join counts recovered from logs add four tiles absent from the PhotoPipe extract, and empty tile 177.257 has a measured empty external catalogue: 22,683 tiles in all.
The remaining 218 tiles, with 1,373,435 selected galaxies (0.72%), have no external count on Fir.
Of the 22,901 manifests, 27 carry `catalogue_match`.

All 18 measured tiles below 0.98 pairing yield zero selected galaxies; the lowest-pairing tile with selected galaxies is 611.236 at 0.9811.
Only 207.238 keeps objects past `N_EPOCH >= 2` (three, lost at the fit cut); 592.237 (0.839, 52 objects) is emptied by the overlap cut and the rest by the epoch cut.
A 0.99 cut would remove 198 tiles and 675,307 selected galaxies (0.354% of the sample, 0.367% of its shear weight); a 0.995 cut, 3,072 tiles and 10.92%.

The two published warning tiles, 058.220 and 207.238, are sparse coverage edges.
Their positive-weight image areas are approximately 0.00955 and 0.04191 deg².
The unmatched objects have median DR6 catalogue S/N of 1.96 and 2.81.
The standard masks cover 71/71 and 147/171 unmatched positions; star-halo masks cover 71/71 and 165/171.
These observations support star-related low-S/N detections in sparse fields as the explanation for these losses.
They do not establish the pixel-level cause of each missed detection.

The rejection census covers actual logged external-only guard failures, not every tile below 0.98 pairing.
Before #975, rejection required both more than 20 external-only objects and more than 2% of the external catalogue.
The #975 guard adds a 50-object allowance for nonempty joins, so some tiles below 0.98 pass.
Published batches did run #975: Fir batch `fir-b0093` was rerun on `aad8cbce`, and the Nibi chair records #975 resumes for the earlier eight diagnostic rejections.
The retained Nibi batch logs identify 11 rejected tiles; the Fir chair's exhaustive scan of 285 live and superseded trees identifies three nonempty rejections and one empty-tile crash.
The union is 14 pairing rejections plus empty tile 177.257.
All 15 have a published catalogue and tile manifest in the complete `latest` inventory; each is among the 18 tiles below 0.98 and yields zero selected galaxies.
The Nibi log scan adds 038.229 and 039.221 to the chair's notes, and the Fir scan adds 134.283 and 131.285 to the initial diagnostic sample.
There are no observed guard-rejected tiles still unpublished or omitted from this comparison.
The other three tiles below 0.98 (592.237, 210.238, 228.238) have no logged rejection.

The warning-tile spatial checks and per-exposure stellar PSF residuals come from a 439-tile sample measured under the earlier selection (`FLAGS <= 2`, T/T_PSF < 3); those two tiles yield zero galaxies under both.
Per-tile mean calibrated ellipticities and PSF residuals are diagnostics, not a shear-calibration bound.
Nominal rejected tile area is 0.25 deg² per tile; this is not masked survey area.

## Audit

Set `catalogue_match_manifests` in the calibration parameters to an explicit list of published `tile_make_cat.json` paths belonging to the input catalogue.
The extraction script writes `catalogue_match_audit.csv` with per-tile row counts, join counts and warnings.
Include zero-row manifests explicitly; a catalogue cannot supply those tile IDs.
Conflicting metadata for the same tile raises an error, so the caller must resolve publication versions before auditing.
Missing metadata is reported without rejecting its objects.
