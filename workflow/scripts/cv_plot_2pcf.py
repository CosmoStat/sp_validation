"""Rule cv_plot_2pcf: n_pairs / xi± overlay across versions.

Reads each version's xi txt (declared inputs, produced by cv_2pcf); calls
plot_2pcf, which re-reads the existing txt files rather than recomputing.
Writes figures under the output dir. Sentinel-tracked: plot_2pcf emits several
figures whose names are internal.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels
from snakemake.script import snakemake

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_2pcf()
touch_sentinels(snakemake)
