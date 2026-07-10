"""Rule cv_pseudo_cl: harmonic-space pseudo-Cl B-mode spectra.

plot_pseudo_cl triggers calculate_pseudo_cl, which writes the born-as-SACC
pseudo_cl_{version}.sacc part for every version (EE/BB/EB with the shared
bandpower window — the BB spectrum cv_summarize_bmodes reads) and the
cell_ee.png figure. The per-version SACC parts are the declared outputs and
feed both cv_summarize_bmodes and the assemble_sacc rule.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_pseudo_cl()
verify_outputs(snakemake)
