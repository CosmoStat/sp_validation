"""Rule cv_plot_2pcf: n_pairs / xi± overlay across versions.

Draws each version's reporting ξ± part (the declared inputs). Writes figures
under the output dir. Sentinel-tracked: plot_2pcf emits several figures whose
names are internal.
"""

from cv_runner import _unbuffer_streams, make_cv, take_reporting_parts, touch_sentinels

_unbuffer_streams()
cv = make_cv(snakemake)
take_reporting_parts(cv, snakemake.input["xi"])
cv.plot_2pcf()
touch_sentinels(snakemake)
