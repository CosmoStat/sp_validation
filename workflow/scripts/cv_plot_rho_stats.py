"""Rule cv_plot_rho_stats: rho-statistics overlay across versions.

Reads rho_stats_{base}.fits for every version (declared inputs); writes
rho_stats.png into the leakage output dir. Sentinel-tracked because the figure
lands beside the first version's leakage products, not at a fixed path.
"""

from snakemake.script import snakemake

from cv_runner import make_cv, touch_sentinels, _unbuffer_streams

_unbuffer_streams()
cv = make_cv(snakemake)
cv.plot_rho_stats()
touch_sentinels(snakemake)
