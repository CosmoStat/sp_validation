# Catalogue-match tile selection

Decision `dr6_catalogue.catalogue_match_tile_selection` keeps the external paired fraction diagnostic-only and applies no tile cut.
Its rationale, alternatives with their measured costs, and `Values` pins live in the [ASTRA record](../scratch/cdaley/unions_astra/analyses/dr6_catalogue/astra.yaml); this page holds the audit coverage and tile-level evidence behind it.
The optional audit records published match counts, warnings and catalogue row counts without removing objects.
Missing metadata remains distinct from a measured empty external catalogue.

## Evidence and limits

The publication census contains 22,901 tile manifests; 27 carry `catalogue_match`.
Join counts recovered from existing logs extend the measurement to 439 distinct published tiles.
The comparison uses `config/calibration/mask_v2.0.yaml`, including its overlap, mask, epoch, fit, S/N and size selection.
The sampled catalogues yield 2,321,814 selected galaxies.

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
All 15 have a published catalogue and tile manifest in the complete `latest` inventory; each yields zero selected galaxies.
The Nibi log scan adds 038.229 and 039.221 to the chair's notes, and the Fir scan adds 134.283 and 131.285 to the initial diagnostic sample.
There are no observed guard-rejected tiles still unpublished or omitted from this comparison.

The count-available sample is not random, although the complete observed guard-rejection population is included.
Mean uncalibrated ellipticities and per-exposure stellar PSF residuals are diagnostics, not a shear-calibration bound.
Nominal rejected tile area is 0.25 deg² per tile; this is not masked survey area.

## Audit

Set `catalogue_match_manifests` in the calibration parameters to an explicit list of published `tile_make_cat.json` paths belonging to the input catalogue.
The extraction script writes `catalogue_match_audit.csv` with per-tile row counts, join counts and warnings.
Include zero-row manifests explicitly; a catalogue cannot supply those tile IDs.
Conflicting metadata for the same tile raises an error, so the caller must resolve publication versions before auditing.
Missing metadata is reported without rejecting its objects.
