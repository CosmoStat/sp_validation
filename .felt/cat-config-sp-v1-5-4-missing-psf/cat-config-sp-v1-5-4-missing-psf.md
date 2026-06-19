---
name: 'cat_config: SP_v1.5.4 missing PSF file'
status: closed
tags:
    - data
    - catalog
created-at: 2026-06-03T10:59:39.118309714+02:00
closed-at: 2026-06-04T00:47:15.280965453+02:00
outcome: 'Dropped SP_v1.5.4 from cat_config.yaml. Root cause: the shared v1.5.a PSF catalog was never finalized — only unions_shapepipe_psf_2024_v1.5.a_prev.fits exists on disk (the sibling v1.6.a was regenerated Jan 11, v1.5.a was not). Shear catalog + v1.5.4/ dir are fine; only PSF missing. Non-fiducial version (we''re v1.4.6.3); only consumed by plot_footprints.py (builds paths directly, doesn''t read cat_config) and stale plot_comparison.ipynb output cells — no live consumer breaks. test_catalog_paths_exist now passes on cluster.'
---
