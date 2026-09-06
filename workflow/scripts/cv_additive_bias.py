"""Rule cv_additive_bias: weighted mean ellipticity (c1, c2) per version.

calculate_additive_bias holds c1/c2 in memory only — the original code never
persists them. This rule writes them to a small JSON so the DAG has a real
output to track, and so the 2pcf rule (which subtracts c1/c2) has an explicit
upstream dependency rather than recomputing silently. Reads each version's
shear catalog.
"""

import json

from cv_runner import _unbuffer_streams, make_cv, verify_outputs

_unbuffer_streams()
cv = make_cv(snakemake)
cv.calculate_additive_bias()
additive_bias = {
    ver: {"c1": float(cv.c1[ver]), "c2": float(cv.c2[ver])} for ver in cv.versions
}
with open(snakemake.output["additive_bias"], "w") as f:
    json.dump(additive_bias, f, indent=2)
verify_outputs(snakemake)
