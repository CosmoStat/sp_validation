# DR6 catalogue-match evidence for sp_validation #446

The study covers 437 distinct published tiles, including all 13 currently identified historical failure cases (12 pairing rejections and one empty-tile crash).
The Nibi chair’s complete failure-log reconciliation determines whether any further cases exist.
The current mask_v2.0 selection yields 2,321,814 galaxies; all 13 tiles below 98% pairing yield zero.
Figures show raw uncalibrated ellipticity proxies and one-point stellar PSF diagnostics, not cosmological statistics or a shear-bias calibration.
`tile-quality.json` holds per-tile measurements identified by campaign and tile.
The code base for selection is sp_validation develop 2b5bede5; scientific reads used the sp_validation-develop-20261024 container on Nibi.
No threshold is applied; the dotted thresholds show sensitivity costs.

Run `python figures.py` in the sp_validation container to regenerate the distribution, quality-comparison and PSF-residual plots from the accompanying measurements.
