"""Rule xi_patches: the jackknife patch centres of one base catalogue.

TreeCorr's k-means over the catalogue's positions and weights, written once to
the declared output; every ξ± measurement of the catalogue and its variants
splits at these centres.
"""

from cv_runner import _unbuffer_streams

from sp_validation.cosmo_val import CosmologyValidation

_unbuffer_streams()
p = snakemake.params
cv = CosmologyValidation(
    versions=[p["catalogue"]],
    catalog_config=p["cat_config"],
    output_dir=p["output_dir"],
)
cv.write_patch_centers(p["catalogue"], int(p["npatch"]), snakemake.output["patches"])
