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
This decision depends on the current `N_EPOCH >= 2` requirement.
No objects survive through that cut on the rejected tiles except 207.238, which retains three detections and loses them at the fit cut.
A one-epoch selection would admit a different population and needs a new assessment.
Rejecting below 0.99 would remove 11,002 selected galaxies (0.474% of this sample); rejecting below 0.995 would remove 424,394 (18.28%).
These thresholds are sensitivity checks, not selection recommendations.

The two published warning tiles, 058.220 and 207.238, are sparse coverage edges and yield no selected galaxies.
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

The current sample supports no additional fraction cut because it would either duplicate existing selections or discard galaxies without a measured improvement.
It does not establish that future low-match tiles are safe.
No selected galaxies remain below 0.98 to test their shear.
The count-available sample is not random, although the complete observed guard-rejection population is included.
Mean uncalibrated ellipticities and per-exposure stellar PSF residuals are diagnostics, not a shear-calibration bound.
Nominal rejected tile area is 0.25 deg² per tile; this is not masked survey area.
The no-cut decision removes zero additional objects or area.
Revisit this decision if the `N_EPOCH >= 2` cut is relaxed or any tile with `external_paired_fraction < 0.98` contributes a selected galaxy.
Check that condition by auditing the selected catalogue's `TILE_ID` column and testing for nonzero `catalogue_rows` where the paired fraction is below 0.98.
A coherent quality anomaly also warrants reassessment.

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

Values: MIN_EXTERNAL_PAIRED_FRACTION = None; mask_v2.0.yaml#N_EPOCH = 2.

The pin checker compares the recorded no-cut value and epoch requirement with the module and current selection YAML.
Negative checks change the actual module policy and parsed epoch value and require the same checker to reject both changes.
The `@sc` citations name this decision but do not resolve to an ASTRA record: sp_validation has no agreed analysis ASTRA home for it.
That record gap remains open with the steward; the local document and pin checks do not replace ASTRA record integrity.
The decision is also recorded in the PR and worker fiber.
