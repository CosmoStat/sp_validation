"""Cache weight-weighted bin sums and bin edges; no figures."""

import numpy as np
from cv_runner import make_cv, verify_outputs

cv = make_cv(snakemake)
data = {"versions": np.asarray(cv.versions)}
for i, (counts, edges) in enumerate(cv.weight_histograms().values()):
    data[f"counts_{i}"] = counts
    data[f"edges_{i}"] = edges
np.savez(snakemake.output.histograms, **data)
verify_outputs(snakemake)
