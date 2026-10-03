"""Rule cv_weights: weighted shear-weight histograms across versions.

Leaf rule. Reads each version's shear weights; writes weight_hist.png at a
fixed path under the output dir (declared output).
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_weights()
verify_outputs(snakemake)
