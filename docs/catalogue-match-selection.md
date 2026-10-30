# Catalogue-match tile selection

Decision `catalogue_match.tile_selection`: use the external paired fraction as a diagnostic and apply no tile cut.
The optional audit records published match counts, warnings and catalogue row counts without removing objects.
Missing metadata remains distinct from a measured empty external catalogue.

## Evidence and limits

The publication census contains 22,901 tile manifests; 27 carry `catalogue_match`.
Join counts recovered from existing logs extend the measurement to 439 distinct published tiles.
The comparison uses `config/calibration/mask_v2.0.yaml`, including its overlap, mask, epoch, fit, S/N and size selection.
The sampled catalogues yield 2,321,814 selected galaxies.
All 15 sampled tiles below 0.98 pairing, including one empty tile, yield zero selected galaxies.
Rejecting below 0.99 would remove 11,002 selected galaxies (0.474% of this sample); rejecting below 0.995 would remove 424,394 (18.28%).
These thresholds are sensitivity checks, not selection recommendations.

The two published warning tiles, 058.220 and 207.238, are sparse coverage edges and yield no selected galaxies.
Their positive-weight image areas are approximately 0.00955 and 0.04191 deg².
The unmatched objects have median DR6 catalogue S/N of 1.96 and 2.81.
The standard masks cover 71/71 and 147/171 unmatched positions; star-halo masks cover 71/71 and 165/171.
These observations support star-related low-S/N detections in sparse fields as the explanation for these losses.
They do not establish the pixel-level cause of each missed detection.

The retained Nibi batch logs identify 11 rejected tiles; the Fir chair's exhaustive scan of 285 live and superseded trees identifies three nonempty rejections and one empty-tile crash.
The union is 14 pairing rejections plus empty tile 177.257.
All 15 have a published catalogue and tile manifest in the complete `latest` inventory; each yields zero selected galaxies.
The Nibi log scan adds 038.229 and 039.221 to the chair's notes, and the Fir scan adds 134.283 and 131.285 to the initial diagnostic sample.
There are no observed guard-rejected tiles still unpublished or omitted from this comparison.

The current sample supports no additional fraction cut because it would either duplicate existing selections or discard galaxies without a measured improvement.
It does not establish that future low-match tiles are safe.
No selected galaxies remain below 0.98 to test their shear.
The count-available sample is not random, although the complete observed guard-rejection population is included.
Mean uncalibrated ellipticities and per-exposure stellar PSF residuals are diagnostics, not a shear-calibration bound.
Nominal rejected tile area is 0.25 deg² per tile; this is not masked survey area.
The no-cut decision removes zero additional objects or area.
Revisit it if low-match tiles retain a usable galaxy population or show a coherent quality anomaly.

Alternatives are a fixed paired-fraction threshold, rejecting ShapePipe warnings, or masking unmatched positions.
A fixed threshold has no supported operating point in these measurements.
Warnings budget join discrepancies rather than catalogue quality.
Unmatched objects have no shapes; their absence alone does not justify masking retained neighbours.

## Audit

Set `catalogue_match_manifests` in the calibration parameters to an explicit list of published `tile_make_cat.json` paths belonging to the input catalogue.
The extraction script writes `catalogue_match_audit.csv` with per-tile row counts, join counts and warnings.
Include zero-row manifests explicitly; a catalogue cannot supply those tile IDs.
Conflicting metadata for the same tile raises an error, so the caller must resolve publication versions before auditing.
Missing metadata is reported without rejecting its objects.

## Decision pin

Values: MIN_EXTERNAL_PAIRED_FRACTION = None.

```python
MIN_EXTERNAL_PAIRED_FRACTION = None
```

The policy-pin test compares the record with the module and injects a 0.98 threshold to check that drift fails.
The decision is also recorded in the PR and worker fiber while the analysis ASTRA home is being chosen.
