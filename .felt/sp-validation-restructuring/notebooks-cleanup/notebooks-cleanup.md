---
id: 01KV8CY7C8J17GRT9BCQTVMAFA
name: 'Notebooks: prune the tree, tutorial → docs'
status: closed
tags:
    - reorg
    - sp-validation
    - docs
created-at: 2026-06-16T14:21:34.984834681Z
updated-at: 2026-06-16T15:30:54.109812476Z
outcome: |-
    Blanket .ipynb deletion reversed per Cail. The one user-facing notebook,
    tutorial_UNIONS_SP_v1.0, became docs/source/using_the_catalogues.md —
    wired into the User Guide toctree, updated for the HDF5 catalogue format
    (≥ v1.4.1) and pointed at calibration.get_calibrated_m_c /
    get_calibrate_e_from_cat, which now automate its hand-rolled metacal steps.
    main_set_up left deleted: its set-up walkthrough is already covered by
    run_validation.md + extract_info.py. The other 13 notebooks (validation
    walkthrough series + exploratory one-offs + 2 junk files) stay deleted,
    recoverable from git @88f2134. The 19 .py moves to scripts/examples/ stand.
---

The reorg removed the top-level `notebooks/` directory. The 19 `.py` helpers
moved to `scripts/examples/`; the 16 notebooks were deleted outright until Cail
pushed back on losing all of them.

The notebooks split cleanly: one genuinely user-facing tutorial on *consuming*
the released catalogues (open base/extended FITS, plot, TreeCorr `GGCorrelation`,
hand-applied metacalibration), versus the old internal validation walkthrough
(`main_set_up`, `metacal_*`, `maps*`, `psf_leakage`, …) that the `scripts/` +
`cosmo_val/` workflow and the existing docs pages now supersede. Only the
tutorial earned a home in the live Sphinx docs; everything else stays in git
history. The conventions the tutorial documents (`e1_uncal`, `R_g{ij}`, header
`R_S{ij}`, `R = R_shear + R_select`) were verified still current against
`src/sp_validation/calibration.py` before publishing.
