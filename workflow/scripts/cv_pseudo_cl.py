"""Rule cv_pseudo_cl: harmonic-space pseudo-Cl B-mode spectra.

plot_pseudo_cl triggers calculate_pseudo_cl, which writes pseudo_cl_{version}.fits
for every version (the BB spectrum cv_summarize_bmodes reads) and the cell_ee.png
figure. The per-version FITS files are the declared outputs.
"""

from snakemake.script import snakemake

from cv_runner import make_cv, verify_outputs, _unbuffer_streams

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_pseudo_cl()
verify_outputs(snakemake)
