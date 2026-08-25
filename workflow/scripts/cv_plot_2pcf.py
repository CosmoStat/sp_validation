"""Rule cv_plot_2pcf: n_pairs / xi± overlay across versions.

Reads each version's xi txt (declared inputs, produced by cv_2pcf); calls
plot_2pcf, which re-reads the existing txt files rather than recomputing.
Writes figures under the output dir. Sentinel-tracked: plot_2pcf emits several
figures whose names are internal.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

# `snakemake` is injected as a module global by Snakemake's `script:` preamble
# before this file runs; no import is needed (and `from snakemake.script
# import snakemake` is IDE-hint-only -- snakemake.script has no such runtime
# attribute and raises ImportError if actually executed).
_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_2pcf()
touch_sentinels(snakemake)
