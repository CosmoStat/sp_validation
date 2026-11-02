"""Build one sparse count map per catalogue; no figures."""

from cv_runner import make_cv, verify_outputs

from sp_validation.cosmo_val.diagnostic_products import save_footprint

cv = make_cv(snakemake)
for (_, count_map), path in zip(
    cv.footprint_maps(), snakemake.output.maps, strict=True
):
    save_footprint(count_map, path)
verify_outputs(snakemake)
