"""Cache regression coefficients and alpha summaries, without rendering."""

import json

from cv_runner import make_cv, prepare_cached_psf, verify_outputs

cv = make_cv(snakemake)
prepare_cached_psf(cv)
cv.calculate_objectwise_leakage(tomography=False)
cv.calculate_alpha_leakage_summaries(tomography=False)
coefficients = {
    ver: {
        bin_id: {k: [v.nominal_value, v.std_dev] for k, v in coeff.items()}
        for bin_id, coeff in bins.items()
    }
    for ver, bins in cv.leakage_coeff.items()
}
with open(snakemake.output.coefficients, "w") as stream:
    json.dump(coefficients, stream, indent=2)
verify_outputs(snakemake)
