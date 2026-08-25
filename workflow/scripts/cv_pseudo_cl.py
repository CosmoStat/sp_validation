"""Rule cv_pseudo_cl: harmonic-space pseudo-Cl B-mode spectra.

plot_pseudo_cl triggers calculate_pseudo_cl, which writes pseudo_cl_{version}.fits
for every version (the BB spectrum cv_summarize_bmodes reads) and the cell_ee.png
figure. The per-version FITS files are the declared outputs.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_pseudo_cl()
verify_outputs(snakemake)
