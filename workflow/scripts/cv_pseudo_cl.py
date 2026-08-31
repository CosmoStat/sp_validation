"""Rule cv_pseudo_cl: harmonic-space pseudo-Cl B-mode spectra.

plot_pseudo_cl triggers calculate_pseudo_cl, which writes one SACC part per
version (EE/BB/EB with the shared bandpower window) and the cell_ee.png figure.
"""

from cv_runner import _unbuffer_streams, make_cv, verify_outputs
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_pseudo_cl()
verify_outputs(snakemake)
