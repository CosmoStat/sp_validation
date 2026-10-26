# Response weighted like the shear mean (#395): reference scripts

The image-sim tests of issue #395 that define case d (DES table W fitted once
on the no-shear branch, evaluated on each branch's own SNR and size). The
production code in `src/sp_validation/response.py` and `calibration.py`
reproduces it; these scripts are kept as the reference it was checked against.

They ran on candide in October 2026 against `feat/spv-launcher` (766cc568) in
the image-sim container. `diff_jk.py` and `table.py` use the old m-bias API
(`image_sims._jackknife_std`, `results["joint"]`) and do not run on the current
code. Paths are hard-coded to candide.

- `wdes_response.py`: recalibrate one grid's branches with the weightings iv,
  des_c, des_d, iv_d, iv_box; writes one cut catalogue per weighting.
- `mbias_config.py`: m-bias config per weighting, from the workflow's config.
- `diff_jk.py`: tile-jackknife error of the m difference between weightings.
- `table.py`: grid table of m1, m2 and the differences.
- `job_wdes.sh`: SLURM array over the four grids.

Outputs: `/n23data1/mkilbing/imsim_wdes_response/grid_*/<weighting>/`, read by
`scripts/validation/round2_weights.py`, `v2_data_table.py` and
`validate_response.py`.
