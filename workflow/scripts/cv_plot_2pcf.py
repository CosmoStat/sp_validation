"""Rule cv_plot_2pcf: xi± overlay across versions.

Reads each version's xi txt (declared inputs, produced by the xi rule); calls
plot_2pcf, which reads the existing txt files back rather than recomputing, for
the non-tomographic ("all", "all") pair. Writes figures under the output dir.
Sentinel-tracked: plot_2pcf emits several figures whose names are internal.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_2pcf(tomography=False, show=False)
touch_sentinels(snakemake)
