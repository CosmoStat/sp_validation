"""Rule cv_weights: weighted shear-weight histograms across versions.

Leaf rule. Reads each version's shear weights; writes weight_hist.png at a
fixed path under the output dir (declared output).
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_weights()
verify_outputs(snakemake)
