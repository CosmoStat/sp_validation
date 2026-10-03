"""Rule cv_plot_2pcf: n_pairs / xi± overlay across versions.

Reads each version's xi text (declared inputs from the generic `xi` rule in
`workflow/rules/twopoint.smk`) and calls plot_2pcf, which re-reads the files.
Writes figures under the output dir. Sentinel-tracked: plot_2pcf emits several
figures whose names are internal.
"""

from cv_runner import _unbuffer_streams, make_cv, touch_sentinels

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_2pcf()
touch_sentinels(snakemake)
