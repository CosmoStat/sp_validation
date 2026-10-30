# DR6 catalogue-match evidence for sp_validation #446

The study covers 439 distinct published tiles, including the entire observed historical failure population: 14 external-only pairing rejections and one empty-tile crash.
The complete retained Nibi batch-log scan identifies 11 pairing-rejected tiles.
The Fir chair scanned all 285 live and superseded batch trees and identifies three pairing rejections plus the empty tile.
Every one of these 15 cases is published in latest and selects zero galaxies under mask_v2.0.
The Nibi scan adds 038.229 and 039.221 to the chair notes; the Fir scan adds 134.283 and 131.285 to the initial measurement.
No observed rejected population remains unpublished or absent from the comparison.

The 439-tile current-mask sample yields 2,321,814 galaxies; all 15 tiles below 98% pairing yield zero.
A 99% cut removes 11,002 galaxies (0.474%); no threshold is applied.
The figures show uncalibrated ellipticity proxies and one-point stellar PSF diagnostics, not cosmological statistics or a shear-bias calibration.
The ordinary count-available controls are not a representative survey sample.

`tile-quality.json` contains per-tile measurements identified by campaign and tile, with count provenance.
`guard-rejected-tiles.json` and `guard-reject-publication-check.tsv` cover all 15 historical failure cases.
`report.html` is self-contained and includes all plots.
The selection base is sp_validation develop 2b5bede5 with config/calibration/mask_v2.0.yaml; scientific reads used the sp_validation-develop-20261024 container on Nibi.
Run `python figures.py` in that container to regenerate the distribution, quality-comparison and PSF-residual plots from the measurements.
