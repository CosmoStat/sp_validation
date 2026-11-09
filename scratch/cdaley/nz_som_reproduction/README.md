# Full-area SOM n(z) reproduction

Run inside the sp_validation container on an allocated candide compute node:

```bash
python scratch/cdaley/nz_som_reproduction/reproduce.py config/calibration/nz/v2_bound9.yaml scratch/nz-reproduction --workers 16
python scratch/cdaley/nz_som_reproduction/reproduce.py config/calibration/nz/v2_bound9.yaml scratch/nz-rebuild --workers 16 --rebuild-model
```

Both commands compare the output against `reference.yaml` and assert zero `TOM_BIN_ID` differences across every source row and per-bin mean redshift differences below 1e-4.
The rebuild command also checks exact equality of the SOM code, feature normalization, groups, edges and gold masks.
`verify.py reference.yaml OUTPUT` can check an existing output independently.
The small cosmo_val reader and SACC round-trip check is in `test_nz_calibration.py`.
